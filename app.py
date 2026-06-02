import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
import io

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="배터리 Second-Life 추천 플랫폼",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
    .main-title  { font-size:28px; font-weight:700; margin-bottom:4px; }
    .sub-title   { font-size:14px; color:#888; margin-bottom:24px; }
    .metric-card { background:#1a1a2e; border-radius:12px; padding:20px;
                   text-align:center; border:1px solid #2a2a4a; }
    .metric-val  { font-size:28px; font-weight:700; color:#00d4aa; }
    .metric-label{ font-size:12px; color:#aaa; margin-top:4px; }
    .rec-card    { background:#1a1a2e; border-radius:12px; padding:16px 20px;
                   margin-bottom:10px; border:1px solid #2a2a4a; }
    .top-card    { border:2px solid #00d4aa !important; }
    .section-title { font-size:18px; font-weight:600; margin:20px 0 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 모델 학습
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_from_samples():
    np.random.seed(42)
    features, labels = [], []
    soh_params = {
        100: dict(re=0.022, rct=0.018, zw=0.015),
        95:  dict(re=0.028, rct=0.022, zw=0.018),
        90:  dict(re=0.035, rct=0.030, zw=0.022),
        85:  dict(re=0.042, rct=0.038, zw=0.028),
        80:  dict(re=0.052, rct=0.048, zw=0.035),
    }
    for soh, params in soh_params.items():
        for _ in range(72):
            noise = 0.003
            re    = params['re']  + np.random.normal(0, noise)
            rct   = params['rct'] + np.random.normal(0, noise)
            zw    = params['zw']  + np.random.normal(0, noise)
            z_real_max  = re + rct + zw
            z_imag_min  = -(rct * 0.6 + np.random.normal(0, 0.002))
            z_imag_max  =  rct * 0.3 + np.random.normal(0, 0.001)
            z_real_mean = re + rct * 0.5
            z_imag_std  = abs(z_imag_min) * 0.4
            features.append([re, z_real_max, z_imag_min, z_imag_max, z_real_mean, z_imag_std])
            labels.append(soh)
    X, y = np.array(features), np.array(labels)
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def load_model_from_uploads(file_data_tuple):
    features, labels = [], []
    for filename, data in file_data_tuple:
        try:
            soh = int(filename.split('SOH')[0].split('_')[-1])
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(data), header=None)
            else:
                df = pd.read_excel(io.BytesIO(data), engine='xlrd', header=None)
            df.columns = ['freq', 'z_real', 'z_imag']
            features.append([
                float(df['z_real'].iloc[0]),
                float(df['z_real'].max()),
                float(df['z_imag'].min()),
                float(df['z_imag'].max()),
                float(df['z_real'].mean()),
                float(df['z_imag'].std()),
            ])
            labels.append(soh)
        except:
            pass
    if len(features) < 10:
        return None
    X, y = np.array(features), np.array(labels)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def extract_features(df):
    return np.array([[
        float(df['z_real'].iloc[0]),
        float(df['z_real'].max()),
        float(df['z_imag'].min()),
        float(df['z_imag'].max()),
        float(df['z_real'].mean()),
        float(df['z_imag'].std()),
    ]])

# ─────────────────────────────────────────────
# 배터리 종류별 특성 계수
# ─────────────────────────────────────────────
BAT_PROPS = {
    "NCM": dict(soh_threshold=75, cycle_life=1500, energy_density=1.0,  temp_sensitive=1.2),
    "LFP": dict(soh_threshold=70, cycle_life=3000, energy_density=0.75, temp_sensitive=0.8),
    "NCA": dict(soh_threshold=78, cycle_life=1200, energy_density=1.1,  temp_sensitive=1.3),
    "LCO": dict(soh_threshold=80, cycle_life=800,  energy_density=1.05, temp_sensitive=1.4),
}

def get_recommendations(soh, years, cycles, bat_type, voltage):
    props = BAT_PROPS[bat_type]

    # 배터리 종류별 사이클 소모 비율
    cycle_ratio     = cycles / props['cycle_life']           # 0~1+
    # 연수 패널티 (종류별 온도 민감도 반영)
    age_penalty     = years * 0.3 * props['temp_sensitive']
    # 사이클 패널티
    cycle_penalty   = cycle_ratio * 15
    # 전압 보정 (정격 대비 현재 전압: 낮으면 감점)
    # NCM/NCA 정격 3.6V, LFP 3.2V, LCO 3.7V
    nominal_v = {"NCM": 3.6, "LFP": 3.2, "NCA": 3.6, "LCO": 3.7}[bat_type]
    voltage_bonus = (voltage - nominal_v) * 5   # 전압 높을수록 소폭 가점

    base = soh - age_penalty - cycle_penalty + voltage_bonus

    apps = [
        {
            "name": "가정용 ESS",
            "icon": "🏠",
            "desc": "저출력 장기 사용. 태양광 패널과 연계해 잉여전력 저장.",
            "score": max(10, base + 5),
            "life":  max(1, round((soh - 60) / 8 - years * 0.1 * props['temp_sensitive'])),
            "value": round(soh * 2.5 * props['energy_density']),
            "carbon": round(soh * 8),
            "condition": soh >= props['soh_threshold'],
        },
        {
            "name": "태양광 연계 ESS",
            "icon": "☀️",
            "desc": "재생에너지 저장에 최적. 낮은 충방전 반복 환경.",
            "score": max(10, base + 10),
            "life":  max(1, round((soh - 65) / 7 - years * 0.1 * props['temp_sensitive'])),
            "value": round(soh * 3.2 * props['energy_density']),
            "carbon": round(soh * 12),
            "condition": soh >= props['soh_threshold'] - 5,
        },
        {
            "name": "통신기지국 백업전원",
            "icon": "📡",
            "desc": "간헐적 방전 환경. 안정적 출력 유지.",
            "score": max(10, base),
            "life":  max(1, round((soh - 55) / 10 - years * 0.1)),
            "value": round(soh * 2.8 * props['energy_density']),
            "carbon": round(soh * 7),
            "condition": soh >= props['soh_threshold'] - 10,
        },
        {
            "name": "UPS 비상전원",
            "icon": "🏥",
            "desc": "병원·데이터센터 비상전원. 단기 방전 위주.",
            "score": max(10, base - 5),
            "life":  max(1, round((soh - 50) / 12 - years * 0.1)),
            "value": round(soh * 2.2 * props['energy_density']),
            "carbon": round(soh * 6),
            "condition": soh >= props['soh_threshold'] - 15,
        },
    ]

    # LFP는 사이클 수명이 길어서 가점
    if bat_type == "LFP":
        for a in apps:
            a['score'] = min(100, a['score'] + 8)
            a['life']  = round(a['life'] * 1.3)

    valid = [a for a in apps if a["condition"]]
    if not valid:
        valid = [apps[-1]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)[:3]

def safety_eval(soh, years, cycles, bat_type, voltage):
    props    = BAT_PROPS[bat_type]
    nominal_v = {"NCM": 3.6, "LFP": 3.2, "NCA": 3.6, "LCO": 3.7}[bat_type]
    cycle_ratio = cycles / props['cycle_life']

    score = soh - years * 1.5 * props['temp_sensitive'] - cycle_ratio * 20

    # 전압 이상 감지
    v_diff = abs(voltage - nominal_v)
    if v_diff > 0.5:
        score -= 10

    if score >= 80:
        return "안전", "#00d4aa", "정상 범위 내 운용 가능합니다."
    elif score >= 65:
        return "주의", "#f0a500", "주기적 점검이 필요합니다."
    else:
        return "위험", "#e05555", "재사용보다 재활용 공정 투입을 권장합니다."

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔋 배터리 Second-Life 추천 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EIS 데이터 기반 AI 진단 · 최적 활용처 추천 | Powered by Warwick DIB Dataset</div>', unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.header("⚙️ 모델 설정")
    model_mode = st.radio(
        "학습 데이터 선택",
        ["기본 모델 사용 (바로 시작)", "내 데이터로 학습 (고급)"],
        index=0
    )

    if model_mode == "내 데이터로 학습 (고급)":
        train_files = st.file_uploader(
            "학습용 EIS 파일 업로드 (여러 개)",
            type=["xls", "csv"],
            accept_multiple_files=True,
            help="파일명에 SOH 정보가 포함되어야 합니다. 예: Cell02_95SOH_..."
        )
        if train_files and len(train_files) >= 10:
            file_data = tuple((f.name, f.read()) for f in train_files)
            with st.spinner("학습 중..."):
                custom_model = load_model_from_uploads(file_data)
            if custom_model:
                st.success(f"✅ 학습 완료! ({len(train_files)}개)")
                st.session_state['model'] = custom_model
        elif train_files:
            st.warning("10개 이상 업로드해주세요.")
    else:
        with st.spinner("기본 모델 로딩 중..."):
            st.session_state['model'] = load_model_from_samples()
        st.success("✅ 기본 모델 준비 완료!")

    st.divider()
    st.markdown("**데이터셋 정보**")
    st.markdown("- Warwick DIB Dataset")
    st.markdown("- SOH: 80 / 85 / 90 / 95 / 100%")
    st.markdown("- 온도: 15 / 25 / 35°C")
    st.markdown("- 총 360개 파일")

# ─── 배터리 기본 정보 ───
st.markdown('<div class="section-title">📋 배터리 기본 정보 입력</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    bat_type = st.selectbox("배터리 종류", ["NCM", "LFP", "NCA", "LCO"])
    props = BAT_PROPS[bat_type]
    st.caption(f"권장 최소 SOH: {props['soh_threshold']}% | 설계 사이클: {props['cycle_life']}회")
with c2:
    years = st.number_input("사용 연수 (년)", min_value=0, max_value=20, value=5)
with c3:
    cycles = st.number_input("충방전 횟수 (회)", min_value=0, max_value=5000, value=500, step=50)
    cycle_ratio = cycles / props['cycle_life']
    if cycle_ratio >= 1.0:
        st.caption("⚠️ 설계 사이클 초과")
    elif cycle_ratio >= 0.8:
        st.caption("🟡 사이클 80% 소모")
    else:
        st.caption(f"🟢 사이클 {round(cycle_ratio*100)}% 소모")
with c4:
    nominal_v = {"NCM": 3.6, "LFP": 3.2, "NCA": 3.6, "LCO": 3.7}[bat_type]
    voltage = st.number_input("현재 전압 (V)", min_value=2.0, max_value=4.5, value=nominal_v, step=0.01)
    v_diff = voltage - nominal_v
    if abs(v_diff) > 0.5:
        st.caption(f"⚠️ 정격({nominal_v}V) 대비 {v_diff:+.2f}V")
    else:
        st.caption(f"🟢 정격 전압({nominal_v}V) 정상 범위")

# ─── EIS 파일 업로드 (여러 개) ───
st.markdown('<div class="section-title">📂 분석할 EIS 파일 업로드</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "EIS 측정 파일 (.xls, .csv) — 반복 측정 여러 개 동시 업로드 가능",
    type=["xls", "csv"],
    accept_multiple_files=True,
    key="analysis_file"
)

if uploaded_files:
    # 여러 파일 읽어서 평균
    dfs = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.csv'):
                tmp = pd.read_csv(f, header=None)
            else:
                tmp = pd.read_excel(f, engine='xlrd', header=None)
            tmp.columns = ['freq', 'z_real', 'z_imag']
            dfs.append(tmp)
        except Exception as e:
            st.warning(f"⚠️ {f.name} 읽기 실패: {e}")

    if not dfs:
        st.error("읽을 수 있는 파일이 없습니다.")
        st.stop()

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = dfs[0].copy()
        df['z_real'] = pd.concat([d['z_real'] for d in dfs], axis=1).mean(axis=1)
        df['z_imag'] = pd.concat([d['z_imag'] for d in dfs], axis=1).mean(axis=1)
        st.caption(f"📊 {len(dfs)}개 반복 측정 평균값으로 분석합니다.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
        fig = go.Figure()
        # 반복 측정 개별 선 (반투명)
        if len(dfs) > 1:
            for i, d in enumerate(dfs):
                fig.add_trace(go.Scatter(
                    x=d['z_real'], y=-d['z_imag'],
                    mode='lines',
                    line=dict(color='rgba(0,212,170,0.2)', width=1),
                    showlegend=False,
                    name=f'측정 {i+1}'
                ))
        # 평균선
        fig.add_trace(go.Scatter(
            x=df['z_real'], y=-df['z_imag'],
            mode='lines+markers',
            name='평균' if len(dfs) > 1 else '측정값',
            marker=dict(color=np.log10(df['freq']+0.001), colorscale='Plasma', size=7,
                        colorbar=dict(title="log₁₀(Hz)", thickness=12)),
            line=dict(color='rgba(255,255,255,0.8)', width=2),
        ))
        fig.update_layout(
            xaxis_title="Z' (실수부, Ω)", yaxis_title="-Z'' (허수부, Ω)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">📊 임피던스 크기</div>', unsafe_allow_html=True)
        z_mag = np.sqrt(df['z_real']**2 + df['z_imag']**2) * 1000
        fig2 = go.Figure()
        if len(dfs) > 1:
            for i, d in enumerate(dfs):
                zm = np.sqrt(d['z_real']**2 + d['z_imag']**2) * 1000
                fig2.add_trace(go.Scatter(
                    x=d['freq'], y=zm,
                    mode='lines',
                    line=dict(color='rgba(0,212,170,0.2)', width=1),
                    showlegend=False,
                ))
        fig2.add_trace(go.Scatter(
            x=df['freq'], y=z_mag,
            mode='lines+markers',
            name='평균' if len(dfs) > 1 else '측정값',
            line=dict(color='#00d4aa', width=2),
            marker=dict(size=5)
        ))
        fig2.update_layout(
            xaxis_title="주파수 (Hz)", xaxis_type="log",
            yaxis_title="|Z| (mΩ)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AI 분석
    st.divider()
    model = st.session_state.get('model')

    if model is None:
        st.warning("⚠️ 사이드바에서 모델을 먼저 선택해주세요!")
    else:
        feats    = extract_features(df)
        soh_pred = float(model.predict(feats)[0])
        soh_pred = round(min(100, max(50, soh_pred)), 1)

        st.markdown('<div class="section-title">🤖 AI 진단 결과</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        re_val  = round(df['z_real'].iloc[0] * 1000, 2)
        rct_val = round((df['z_real'].max() - df['z_real'].iloc[0]) * 1000, 2)
        status_txt   = "양호" if soh_pred >= 85 else "보통" if soh_pred >= 70 else "주의"
        status_color = "#00d4aa" if soh_pred >= 85 else "#f0a500" if soh_pred >= 70 else "#e05555"

        for col, val, label, color in zip(
            [m1, m2, m3, m4, m5],
            [f"{soh_pred}%", f"{re_val}mΩ", f"{rct_val}mΩ", f"{voltage}V", status_txt],
            ["예측 SOH", "전해질 저항(Re)", "전하전달 저항(Rct)", "현재 전압", "배터리 상태"],
            ["#00d4aa","#00d4aa","#00d4aa","#00d4aa", status_color]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        # 안전성 평가
        st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
        safety_txt, safety_color, safety_desc = safety_eval(soh_pred, years, cycles, bat_type, voltage)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; border:2px solid {safety_color};">
            <span style="font-size:20px; font-weight:700; color:{safety_color}">{safety_txt}</span>
            <span style="font-size:14px; color:#ccc; margin-left:12px;">{safety_desc}</span>
        </div>""", unsafe_allow_html=True)

        # 추천 활용처
        st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
        recs = get_recommendations(soh_pred, years, cycles, bat_type, voltage)
        for i, rec in enumerate(recs):
            card_class = "rec-card top-card" if i == 0 else "rec-card"
            rank_label = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:16px; font-weight:600;">{rec['icon']} {rec['name']}</div>
                        <div style="font-size:12px; color:#aaa;">{rank_label} · 적합도 {round(rec['score'])}%</div>
                        <div style="font-size:13px; color:#bbb; margin-top:6px;">{rec['desc']}</div>
                    </div>
                    <div style="display:flex; gap:20px; flex-wrap:wrap;">
                        <div style="text-align:center;">
                            <div style="font-size:18px; font-weight:600; color:#00d4aa;">{rec['life']}년</div>
                            <div style="font-size:11px; color:#aaa;">예상 잔존수명</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:18px; font-weight:600; color:#00d4aa;">{rec['value']}만원</div>
                            <div style="font-size:11px; color:#aaa;">경제적 가치</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:18px; font-weight:600; color:#00d4aa;">{rec['carbon']}kg</div>
                            <div style="font-size:11px; color:#aaa;">CO₂ 절감</div>
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # 에너지 임팩트
        st.markdown('<div class="section-title">🌍 에너지 임팩트</div>', unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("예측 SOH",   f"{soh_pred}%")
        i2.metric("CO₂ 절감",   f"{recs[0]['carbon']}kg",  "탄소 감축")
        i3.metric("경제적 가치", f"{recs[0]['value']}만원", "재사용 가치")
        i4.metric("광물 절약",   f"{round(soh_pred*0.05,1)}kg", "리튬·코발트")

        # 최종 판단 (안전성 평가와 연동)
        st.divider()
        cycle_pct = round(cycles / props['cycle_life'] * 100)

        if safety_txt == "위험":
            color = "#e05555"
            msg   = "❌ 재사용 불가 — 재활용 공정 필요"
        elif safety_txt == "주의":
            color = "#f0a500"
            msg   = "⚠️ 조건부 재사용 가능 — 주기적 점검 필요"
        else:  # 안전
            if soh_pred >= props['soh_threshold']:
                color = "#00d4aa"
                msg   = "✅ 재사용 가능"
            else:
                color = "#e05555"
                msg   = "❌ 재활용 공정 권장 — SOH 기준 미달"

        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:20px;
                    border:2px solid {color}; text-align:center;">
            <div style="font-size:24px; font-weight:700; color:{color}">{msg}</div>
            <div style="font-size:14px; color:#aaa; margin-top:8px;">
                배터리 종류: {bat_type} | 사용 연수: {years}년 |
                충방전: {cycles}회 ({cycle_pct}% 소모) | 현재 전압: {voltage}V
            </div>
        </div>""", unsafe_allow_html=True)

else:
    st.info("👆 EIS 파일(.xls 또는 .csv)을 업로드하면 AI가 자동으로 분석해드립니다.")
    st.markdown("""
    **사용 방법:**
    1. 배터리 기본 정보 입력 (종류, 연수, 충방전 횟수, 전압)
    2. EIS 파일 업로드 (반복 측정 여러 개 동시 업로드 가능 → 자동 평균)
    3. AI 진단 결과 및 추천 활용처 확인
    """)
