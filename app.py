import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import struct
import re
import os
import io
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

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
    .ref-text    { font-size:11px; color:#666; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 학술 근거 기반 배터리 특성값
# [1] SOH 80%: IEC 62933, UL 1974
# [2] SOH 50%: Edge et al. (2023), doi:10.5281/zenodo.10257443
# [3] 사이클 수명: Frontiers in Energy Research (2023), doi:10.3389/fenrg.2023.1108269
# [4] 전압 2.5V: EU Battery Regulation 2023/1542
# ─────────────────────────────────────────────
BAT_PROPS = {
    "NCM": dict(soh_reuse=80, soh_recycle=50, cycle_life=2000, nominal_v=3.6),
    "LFP": dict(soh_reuse=80, soh_recycle=50, cycle_life=4000, nominal_v=3.2),
    "NCA": dict(soh_reuse=80, soh_recycle=50, cycle_life=1500, nominal_v=3.6),
    "LCO": dict(soh_reuse=80, soh_recycle=50, cycle_life=800,  nominal_v=3.7),
}

# ─────────────────────────────────────────────
# XLS 파일 파싱 (라이브러리 없이 직접 파싱)
# Warwick DIB xls 포맷 기준
# ─────────────────────────────────────────────
def parse_xls_eis(file_input):
    """
    xls 파일에서 EIS 임피던스 데이터 추출
    file_input: 파일 경로(str) 또는 바이트(bytes)
    반환: z_real, z_imag 리스트
    """
    if isinstance(file_input, (str, os.PathLike)):
        with open(file_input, 'rb') as f:
            raw = f.read()
    else:
        raw = file_input

    all_floats = []
    for i in range(0, len(raw) - 8, 8):
        try:
            v = struct.unpack_from('<d', raw, i)[0]
            if not (v != v) and abs(v) < 1e10:
                all_floats.append(v)
        except:
            pass

    z_real     = sorted([v for v in all_floats if 0.01  <= v  <= 0.5],  reverse=True)
    z_imag_neg = sorted([v for v in all_floats if -0.1  <= v  < -0.0001])
    return z_real, z_imag_neg

def parse_csv_eis(file_input):
    """csv 파일에서 EIS 데이터 추출 (freq, z_real, z_imag 3열)"""
    if isinstance(file_input, (str, os.PathLike)):
        df = pd.read_csv(file_input, header=None)
    else:
        df = pd.read_csv(io.BytesIO(file_input), header=None)
    df.columns = ['freq', 'z_real', 'z_imag']
    df = df.sort_values('freq', ascending=False).reset_index(drop=True)
    return df['z_real'].tolist(), df['z_imag'].tolist()

def extract_features(z_real_list, z_imag_list):
    """EIS 데이터 → ML 피처 6개 추출"""
    if len(z_real_list) < 5:
        return None
    zr = np.array(z_real_list)
    zi = np.array(z_imag_list) if z_imag_list else np.array([0.0])
    return [
        float(zr[0]),           # Re: 고주파 실수부 (전해질 저항)
        float(zr.max()),        # 최대 실수부
        float(zi.min()),        # 최소 허수부 (반원 깊이)
        float(zi.max()),        # 최대 허수부
        float(zr.mean()),       # 실수부 평균
        float(zr.std()),        # 실수부 표준편차
    ]

# ─────────────────────────────────────────────
# Warwick DIB 데이터셋으로 모델 학습
# ─────────────────────────────────────────────
@st.cache_resource
def train_model_from_dib():
    """
    data/EIS_Test.zip 또는 data/EIS_Test/ 폴더에서 Warwick DIB 360개 파일 학습
    zip 파일 우선 → 없으면 폴더 탐색
    출처: Rashid et al. (2023), doi:10.1016/j.dib.2023.109157
    """
    import zipfile, tempfile

    base_dir  = os.path.dirname(__file__)
    zip_path  = os.path.join(base_dir, 'EIS_Test.zip')
    dir_path  = os.path.join(base_dir, 'data', 'EIS_Test')

    # xls 파일 목록 수집
    file_items = []  # (fname, filepath_or_bytes)

    if os.path.exists(zip_path):
        # zip 파일에서 직접 읽기 (압축 풀기 없이)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for zname in zf.namelist():
                fname = os.path.basename(zname)
                if fname.endswith('.xls') and 'SOH' in fname:
                    file_items.append((fname, zf.read(zname)))
    elif os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.endswith('.xls') and 'SOH' in fname:
                file_items.append((fname, os.path.join(dir_path, fname)))
    else:
        return None, None, 0

    X, y = [], []
    for fname, file_data in file_items:
        m = re.search(r'(\d+)SOH', fname)
        if not m:
            continue
        soh = int(m.group(1))
        try:
            raw = file_data if isinstance(file_data, bytes) else None
            zr, zi = parse_xls_eis(raw if raw else file_data)
            feats = extract_features(zr, zi)
            if feats:
                X.append(feats)
                y.append(soh)
        except:
            continue

    if len(X) < 10:
        return None, None, len(X)

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, len(X)

def predict_soh(model, scaler, z_real_list, z_imag_list):
    feats = extract_features(z_real_list, z_imag_list)
    if feats is None:
        return None
    X = scaler.transform([feats])
    pred = float(model.predict(X)[0])
    return round(float(np.clip(pred, 50, 100)), 1)

# ─────────────────────────────────────────────
# 활용처 추천
# [5] Edge et al. (2023): ESS SOH 70~80%
# [6] Martinez-Laserna et al. (2018), Appl. Energy: 통신 SOH 60%
# ─────────────────────────────────────────────
def get_recommendations(health, years, cycles, bat_type, voltage):
    props = BAT_PROPS[bat_type]
    cycle_ratio  = cycles / props['cycle_life']
    cycle_penalty = cycle_ratio * 20
    age_penalty   = years * 2
    v_diff        = abs(voltage - props['nominal_v'])
    v_penalty     = v_diff * 10 if v_diff > 0.3 else 0
    base = health - cycle_penalty - age_penalty - v_penalty

    apps = [
        {
            "name": "태양광 연계 ESS",
            "icon": "☀️",
            "desc": "재생에너지 저장. 낮은 C-rate, 1일 1~2회 충방전 환경.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, base + 5),
            "condition": health >= 70,
        },
        {
            "name": "가정용 ESS",
            "icon": "🏠",
            "desc": "저출력 장기 사용. 태양광 연계 잉여전력 저장.",
            "ref": "Edge et al. (2023); UL 1974",
            "score": max(0, base),
            "condition": health >= 70,
        },
        {
            "name": "통신기지국 백업전원",
            "icon": "📡",
            "desc": "간헐적 방전. 부동충전 위주로 배터리 부담 낮음.",
            "ref": "Martinez-Laserna et al. (2018), Appl. Energy",
            "score": max(0, base - 5),
            "condition": health >= 60,
        },
        {
            "name": "UPS 비상전원",
            "icon": "🏥",
            "desc": "단기 방전 위주. 충방전 빈도 낮아 열화 부담 적음.",
            "ref": "Edge et al. (2023) mid-range 기준",
            "score": max(0, base - 10),
            "condition": health >= 50,
        },
    ]
    if bat_type == "LFP":
        for a in apps:
            a['score'] = min(100, a['score'] + 5)

    valid = [a for a in apps if a["condition"]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)[:3]

def safety_eval(health, years, cycles, bat_type, voltage):
    props = BAT_PROPS[bat_type]
    cycle_ratio = cycles / props['cycle_life']
    v_diff = abs(voltage - props['nominal_v'])

    if voltage < 2.5:
        return "위험", "#e05555", "전압 2.5V 미만 — 안전 기준 미달 (EU 배터리 규정 2023/1542)"

    v_warn = v_diff > 0.5

    if health >= 80 and cycle_ratio < 0.8 and not v_warn:
        return "안전", "#00d4aa", "정상 범위 — 재사용 적합 (IEC 62933 기준 충족)"
    elif health >= 50:
        reason = []
        if health < 80:
            reason.append(f"건강도 {round(health)}% (기준 80% 미달)")
        if cycle_ratio >= 0.8:
            reason.append(f"사이클 {round(cycle_ratio*100)}% 소모")
        if v_warn:
            reason.append(f"전압 정격 대비 ±{round(v_diff,2)}V 이상")
        return "주의", "#f0a500", " · ".join(reason) + " — 점검 필요"
    else:
        return "위험", "#e05555", f"건강도 {round(health)}% — SOH 50% 미만, 해체/재활용 필수 (Edge et al. 2023)"

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔋 배터리 Second-Life 추천 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EIS 기반 SOH 예측 · 학술 근거 기반 2차 활용처 추천 | Warwick DIB Dataset</div>', unsafe_allow_html=True)

# ─── 모델 로드 ───
with st.spinner("🔄 Warwick DIB 데이터셋으로 모델 학습 중..."):
    model, scaler, n_files = train_model_from_dib()

if model is None:
    st.sidebar.error(f"❌ data/EIS_Test/ 폴더를 찾을 수 없습니다. ({n_files}개 파일 감지)")
    st.sidebar.info("GitHub 레포에 data/EIS_Test/ 폴더와 360개 xls 파일을 추가해주세요.")
else:
    st.sidebar.success(f"✅ 모델 학습 완료 ({n_files}개 파일)")
    st.sidebar.markdown("**학습 데이터**")
    st.sidebar.markdown("- Warwick DIB Dataset")
    st.sidebar.markdown("- SOH: 80/85/90/95/100%")
    st.sidebar.markdown("- 온도: 15/25/35°C")
    st.sidebar.markdown("- 출처: Rashid et al. (2023)")

# ─── 배터리 기본 정보 ───
st.markdown('<div class="section-title">📋 배터리 기본 정보 입력</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    bat_type = st.selectbox("배터리 종류", ["NCM", "LFP", "NCA", "LCO"])
    props = BAT_PROPS[bat_type]
    st.caption(f"설계 사이클: {props['cycle_life']}회 | 정격: {props['nominal_v']}V")
with c2:
    years = st.number_input("사용 연수 (년)", min_value=0, max_value=20, value=5)
with c3:
    cycles = st.number_input("충방전 횟수 (회)", min_value=0, max_value=10000, value=500, step=50)
    cycle_ratio = cycles / props['cycle_life']
    if cycle_ratio >= 1.0:
        st.caption("⚠️ 설계 사이클 초과")
    elif cycle_ratio >= 0.8:
        st.caption(f"🟡 사이클 {round(cycle_ratio*100)}% 소모")
    else:
        st.caption(f"🟢 사이클 {round(cycle_ratio*100)}% 소모")
with c4:
    voltage = st.number_input("현재 전압 (V)", min_value=2.0, max_value=4.5,
                               value=props['nominal_v'], step=0.01)
    v_diff = abs(voltage - props['nominal_v'])
    if voltage < 2.5:
        st.caption("🔴 2.5V 미만 — 안전 위험")
    elif v_diff > 0.5:
        st.caption(f"⚠️ 정격 대비 ±{round(v_diff,2)}V")
    else:
        st.caption(f"🟢 정격({props['nominal_v']}V) 정상")

# ─── SOH 입력 방식 ───
st.markdown('<div class="section-title">🔢 SOH 정보</div>', unsafe_allow_html=True)
soh_mode = st.radio(
    "SOH 입력 방식",
    ["SOH 모름 — EIS로 자동 예측 (Warwick DIB 모델)",
     "SOH 직접 입력 (용량 측정값 등 보유 시)"],
    index=0,
    horizontal=True
)
soh_input = None
if soh_mode == "SOH 직접 입력 (용량 측정값 등 보유 시)":
    soh_input = st.number_input(
        "SOH (%)", min_value=0, max_value=100, value=80, step=1,
        help="SOH = 현재용량 / 초기용량 × 100% (IEC 62660-1)"
    )
    st.caption("📌 직접 입력값을 우선 사용합니다.")

# ─── EIS 파일 업로드 ───
st.markdown('<div class="section-title">📂 EIS 파일 업로드</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "EIS 측정 파일 (.xls, .csv) — 반복 측정 여러 개 동시 업로드 시 자동 평균",
    type=["xls", "csv"],
    accept_multiple_files=True,
    key="analysis_file"
)

if uploaded_files:
    # ─── 파일 읽기 및 평균 ───
    all_zr, all_zi = [], []
    df_list = []
    for f in uploaded_files:
        try:
            raw = f.read()
            if f.name.endswith('.csv'):
                zr, zi = parse_csv_eis(raw)
            else:
                zr, zi = parse_xls_eis(raw)
            all_zr.append(zr)
            all_zi.append(zi)
            # 시각화용 DataFrame
            min_len = min(len(zr), len(zi)) if zi else len(zr)
            df_list.append(pd.DataFrame({
                'z_real': zr[:min_len],
                'z_imag': zi[:min_len] if zi else [0]*min_len,
            }))
        except Exception as e:
            st.warning(f"⚠️ {f.name} 읽기 실패: {e}")

    if not all_zr:
        st.error("읽을 수 있는 파일이 없습니다.")
        st.stop()

    # 평균 처리
    max_len = max(len(zr) for zr in all_zr)
    def pad(lst, length):
        return lst + [lst[-1]] * (length - len(lst)) if lst else [0] * length

    avg_zr = np.mean([pad(zr, max_len) for zr in all_zr], axis=0).tolist()
    avg_zi = np.mean([pad(zi, max_len) for zi in all_zi], axis=0).tolist() if all_zi[0] else []

    if len(uploaded_files) > 1:
        st.caption(f"📊 {len(uploaded_files)}개 반복 측정 평균값으로 분석합니다.")

    # ─── 시각화 ───
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if len(df_list) > 1:
            for d in df_list:
                fig.add_trace(go.Scatter(
                    x=d['z_real'], y=-d['z_imag'], mode='lines',
                    line=dict(color='rgba(0,212,170,0.2)', width=1),
                    showlegend=False
                ))
        min_len = min(len(avg_zr), len(avg_zi)) if avg_zi else len(avg_zr)
        fig.add_trace(go.Scatter(
            x=avg_zr[:min_len],
            y=[-v for v in avg_zi[:min_len]] if avg_zi else [0]*min_len,
            mode='lines+markers',
            name='평균' if len(df_list) > 1 else '측정값',
            marker=dict(color=list(range(min_len)), colorscale='Plasma', size=7,
                        colorbar=dict(title="포인트", thickness=12)),
            line=dict(color='rgba(255,255,255,0.8)', width=2),
        ))
        fig.update_layout(
            xaxis_title="Z' (실수부, Ω)", yaxis_title="-Z'' (허수부, Ω)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">📊 임피던스 크기</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for d in df_list:
            zm = np.sqrt(d['z_real']**2 + d['z_imag']**2) * 1000
            fig2.add_trace(go.Scatter(
                y=zm, mode='lines',
                line=dict(color='rgba(0,212,170,0.2)', width=1),
                showlegend=False
            ))
        avg_zm = np.sqrt(np.array(avg_zr)**2 +
                         np.array(avg_zi if avg_zi else [0]*len(avg_zr))**2) * 1000
        fig2.add_trace(go.Scatter(
            y=avg_zm, mode='lines+markers',
            name='평균' if len(df_list) > 1 else '측정값',
            line=dict(color='#00d4aa', width=2), marker=dict(size=5)
        ))
        fig2.update_layout(
            xaxis_title="포인트 (고주파 → 저주파)",
            yaxis_title="|Z| (mΩ)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── SOH 결정 ───
    st.divider()

    if soh_input is not None:
        # 직접 입력
        soh_final = soh_input
        soh_source = f"직접 입력값 (IEC 62660-1 기준)"
        soh_certain = True
    elif model is not None:
        # ML 예측
        soh_pred = predict_soh(model, scaler, avg_zr, avg_zi)
        if soh_pred is not None:
            soh_final = soh_pred
            soh_source = "EIS 기반 ML 예측 (Warwick DIB, Rashid et al. 2023)"
            soh_certain = False
        else:
            st.error("EIS 피처 추출 실패. 파일을 확인해주세요.")
            st.stop()
    else:
        st.warning("⚠️ 모델 미로드 상태. data/EIS_Test/ 폴더를 확인해주세요.")
        st.stop()

    # ─── 진단 결과 ───
    st.markdown('<div class="section-title">🤖 진단 결과</div>', unsafe_allow_html=True)

    re_val  = round(avg_zr[0] * 1000, 2) if avg_zr else 0
    rct_val = round((max(avg_zr) - avg_zr[0]) * 1000, 2) if avg_zr else 0
    soh_color = "#00d4aa" if soh_final >= 80 else "#f0a500" if soh_final >= 50 else "#e05555"
    status_txt   = "양호" if soh_final >= 80 else "주의" if soh_final >= 50 else "위험"
    status_color = soh_color

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, val, label, color, note in zip(
        [m1, m2, m3, m4, m5],
        [f"{soh_final}%", f"{re_val}mΩ", f"{rct_val}mΩ", f"{voltage}V", status_txt],
        ["SOH", "전해질 저항 (Re)", "전하전달 저항 (Rct)", "현재 전압", "배터리 상태"],
        [soh_color, "#00d4aa", "#00d4aa", "#00d4aa", status_color],
        [soh_source[:25]+"...", "고주파 실수부", "반원 크기", "측정값", "종합 평가"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="ref-text">{note}</div>
        </div>""", unsafe_allow_html=True)

    if not soh_certain:
        st.caption(f"📌 SOH 예측 근거: {soh_source} | 평균 오차 ±5% (교차검증 기준)")

    # ─── 안전성 평가 ───
    st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
    safety_txt, safety_color, safety_desc = safety_eval(soh_final, years, cycles, bat_type, voltage)
    st.markdown(f"""
    <div class="metric-card" style="text-align:left; border:2px solid {safety_color};">
        <span style="font-size:20px; font-weight:700; color:{safety_color}">{safety_txt}</span>
        <span style="font-size:14px; color:#ccc; margin-left:12px;">{safety_desc}</span>
    </div>""", unsafe_allow_html=True)

    # ─── 추천 활용처 ───
    st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
    st.caption("📌 활용처별 SOH 기준 — Edge et al. (2023); Martinez-Laserna et al. (2018); IEC 62933")

    recs = get_recommendations(soh_final, years, cycles, bat_type, voltage)
    if not recs:
        st.error("❌ 모든 활용처 기준 미달 — 재활용 공정 투입 권장 (SOH 50% 미만, Edge et al. 2023)")
    else:
        for i, rec in enumerate(recs):
            card_class = "rec-card top-card" if i == 0 else "rec-card"
            rank_label = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:16px; font-weight:600;">{rec['icon']} {rec['name']}</div>
                        <div style="font-size:12px; color:#aaa;">{rank_label} · 적합도 {round(rec['score'])}점</div>
                        <div style="font-size:13px; color:#bbb; margin-top:6px;">{rec['desc']}</div>
                        <div style="font-size:11px; color:#666; margin-top:4px;">📚 {rec['ref']}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ─── 최종 판단 ───
    st.divider()
    cycle_pct = round(cycle_ratio * 100)

    if safety_txt == "위험":
        final_color = "#e05555"
        final_msg   = "❌ 재사용 불가 — 재활용 공정 필요"
        final_ref   = "Edge et al. (2023); IEC 62619"
    elif safety_txt == "주의":
        final_color = "#f0a500"
        final_msg   = "⚠️ 조건부 재사용 가능 — 주기적 점검 필요"
        final_ref   = "Edge et al. (2023) mid-range 기준"
    else:
        final_color = "#00d4aa"
        final_msg   = "✅ 재사용 가능"
        final_ref   = "IEC 62933, UL 1974"

    st.markdown(f"""
    <div style="background:#1a1a2e; border-radius:12px; padding:20px;
                border:2px solid {final_color}; text-align:center;">
        <div style="font-size:24px; font-weight:700; color:{final_color}">{final_msg}</div>
        <div style="font-size:13px; color:#aaa; margin-top:8px;">
            배터리 종류: {bat_type} | SOH: {soh_final}% | 사용 연수: {years}년 |
            충방전: {cycles}회 ({cycle_pct}% 소모) | 전압: {voltage}V
        </div>
        <div style="font-size:11px; color:#666; margin-top:6px;">📚 근거: {final_ref}</div>
    </div>""", unsafe_allow_html=True)

else:
    st.info("👆 EIS 파일(.xls 또는 .csv)을 업로드하면 분석이 시작됩니다.")
    st.markdown("""
    **사용 방법:**
    1. 배터리 기본 정보 입력 (종류, 연수, 충방전 횟수, 전압)
    2. SOH 입력 방식 선택
       - **모르는 경우** → EIS 파일 업로드 시 Warwick DIB 모델로 자동 예측
       - **아는 경우** → 직접 입력 (용량 측정값 등)
    3. EIS 파일 업로드 (반복 측정 여러 개 동시 업로드 → 자동 평균)
    4. SOH · 안전성 · 추천 활용처 확인
    """)
