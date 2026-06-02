import streamlit as st
import pandas as pd
import numpy as np
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
    .ref-text    { font-size:11px; color:#666; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 학술 근거 기반 배터리 특성값
# ─────────────────────────────────────────────
# 출처:
# [1] SOH 80% 임계값: IEC 62933, UL 1974, Birmingham Univ. (2024)
# [2] SOH 50% 해체: Edge et al. (2023), DOI:10.5281/zenodo.10257443
# [3] 사이클 수명: Preger et al. (2020), Tran et al. (2021) — Frontiers in Energy Research (2023)
# [4] LFP 4000+회: Frontiers in Energy Research (2023), doi:10.3389/fenrg.2023.1108269
# [5] NCA 1500회, NCM 2000회: 동일 출처
# [6] 정격전압: 배터리 제조사 표준 스펙

BAT_PROPS = {
    # soh_reuse: 2차 활용 최소 SOH (IEC 62933, UL 1974 기준 80%) [1]
    # soh_recycle: 이 이하면 해체/재활용 (Edge et al. 2023 기준 50%) [2]
    # cycle_life: 설계 사이클 수명 (Frontiers 2023) [3][4][5]
    # nominal_v: 정격 전압 (제조사 표준) [6]
    "NCM": dict(soh_reuse=80, soh_recycle=50, cycle_life=2000, nominal_v=3.6),
    "LFP": dict(soh_reuse=80, soh_recycle=50, cycle_life=4000, nominal_v=3.2),
    "NCA": dict(soh_reuse=80, soh_recycle=50, cycle_life=1500, nominal_v=3.6),
    "LCO": dict(soh_reuse=80, soh_recycle=50, cycle_life=800,  nominal_v=3.7),
}

# ─────────────────────────────────────────────
# EIS 임피던스 지표 추출
# ─────────────────────────────────────────────
def extract_eis_indicators(df):
    """
    EIS 측정값에서 핵심 임피던스 지표 추출
    - Re  (전해질 저항): 고주파 실수부 시작점 [Ω]
    - Rct (전하전달 저항): 반원 크기 = 실수부 최댓값 - 시작점 [Ω]
    - |Z|_low: 저주파 임피던스 크기 (확산 저항 반영) [Ω]
    출처: Warwick DIB Dataset 논문 (Rashid et al. 2023), doi:10.1016/j.dib.2023.109157
    """
    # 주파수 내림차순 정렬 (고주파 → 저주파)
    df_sorted = df.sort_values('freq', ascending=False).reset_index(drop=True)

    re  = float(df_sorted['z_real'].iloc[0])           # 고주파 실수부 = 전해질 저항
    rct = float(df_sorted['z_real'].max() - re)        # 반원 크기 = 전하전달 저항
    z_low = float(np.sqrt(
        df_sorted['z_real'].iloc[-1]**2 +
        df_sorted['z_imag'].iloc[-1]**2
    ))  # 저주파 임피던스 크기

    return re, rct, z_low

def impedance_health_score(re, rct, z_low, bat_type):
    """
    EIS 임피던스 지표로 배터리 건강도 점수 산출 (0~100)
    - Re 증가 → 전해질 열화
    - Rct 증가 → 전극/계면 열화
    - 기준값: Warwick DIB 데이터셋의 SOH 100% 평균값 대비 비율
    출처: Niri et al. (2022), Journal of Energy Storage, doi:10.1016/j.est.2022.106295
    """
    # SOH 100% 기준 임피던스 (Warwick DIB 데이터 기반)
    baseline = {
        "NCM": dict(re=0.022, rct=0.018),
        "LFP": dict(re=0.020, rct=0.015),
        "NCA": dict(re=0.020, rct=0.016),
        "LCO": dict(re=0.025, rct=0.022),
    }
    b = baseline[bat_type]

    # Re, Rct 각각 기준 대비 증가 비율로 건강도 감점
    re_ratio  = re  / b['re']   # 1.0이면 새 배터리 수준
    rct_ratio = rct / b['rct']  # 클수록 열화

    # 건강도 점수: Re 40%, Rct 60% 가중 (Rct가 더 민감한 노화 지표)
    score = 100 - (re_ratio - 1) * 40 - (rct_ratio - 1) * 60
    return round(float(np.clip(score, 0, 100)), 1)

# ─────────────────────────────────────────────
# 활용처 추천 (EIS 기반)
# ─────────────────────────────────────────────
def get_recommendations(eis_score, soh, years, cycles, bat_type, voltage):
    """
    활용처별 SOH/EIS 기준:
    - 가정용/태양광 ESS: SOH 70~80% 구간 활용 가능 (Edge et al. 2023) [2]
    - 통신기지국: SOH 60% 이상 (Martinez-Laserna et al. 2018)
    - UPS: SOH 50% 이상 (Edge et al. 2023 mid-range 기준) [2]
    - 재활용: SOH 50% 미만 → 해체 필수 [2]
    """
    props = BAT_PROPS[bat_type]
    cycle_ratio = cycles / props['cycle_life']

    # SOH 모르면 EIS 점수로 대체
    health = soh if soh is not None else eis_score

    # 사이클 소모율 패널티 (설계 수명 대비)
    cycle_penalty = cycle_ratio * 20
    # 연수 패널티
    age_penalty = years * 2
    # 전압 이상 패널티
    v_diff = abs(voltage - props['nominal_v'])
    v_penalty = v_diff * 10 if v_diff > 0.3 else 0

    base = health - cycle_penalty - age_penalty - v_penalty

    apps = [
        {
            "name": "태양광 연계 ESS",
            "icon": "☀️",
            "desc": "재생에너지 저장. 낮은 C-rate, 1일 1~2회 충방전 환경에 적합.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, base + 5),
            "condition": health >= 70,  # Edge et al. 2023: 70~80% 구간 활용
        },
        {
            "name": "가정용 ESS",
            "icon": "🏠",
            "desc": "저출력 장기 사용. 태양광 패널 연계 잉여전력 저장.",
            "ref": "Edge et al. (2023); UL 1974",
            "score": max(0, base),
            "condition": health >= 70,
        },
        {
            "name": "통신기지국 백업전원",
            "icon": "📡",
            "desc": "간헐적 방전. 부동충전 위주 환경으로 배터리 부담 낮음.",
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

    # LFP는 사이클 수명 우수 → 가점 (Frontiers 2023)
    if bat_type == "LFP":
        for a in apps:
            a['score'] = min(100, a['score'] + 5)

    valid = [a for a in apps if a["condition"]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)[:3]

def safety_eval(eis_score, soh, years, cycles, bat_type, voltage):
    """
    안전성 평가 기준:
    - SOH > 80%: 재사용 적합 (IEC 62933, UL 1974) [1]
    - SOH 50~80%: 용도 변경 검토 (Edge et al. 2023) [2]
    - SOH < 50%: 해체/재활용 필수 (Edge et al. 2023) [2]
    - 전압 < 2.5V: 안전 위험 (EU 배터리 규정 2023/1542) [7]
    [7] EU Battery Regulation 2023/1542, Art. 10
    """
    props = BAT_PROPS[bat_type]
    health = soh if soh is not None else eis_score
    cycle_ratio = cycles / props['cycle_life']
    v_diff = abs(voltage - props['nominal_v'])

    # 전압 심각 이상 (2.5V 미만: EU 배터리 규정)
    if voltage < 2.5:
        return "위험", "#e05555", "전압 2.5V 미만 — 안전 기준 미달 (EU 배터리 규정)"

    # 전압 경고
    if v_diff > 0.5:
        v_warn = True
    else:
        v_warn = False

    # SOH/EIS 기반 판단
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
st.markdown('<div class="sub-title">EIS 임피던스 기반 진단 · 학술 근거 기반 활용처 추천</div>', unsafe_allow_html=True)

# ─── 배터리 기본 정보 ───
st.markdown('<div class="section-title">📋 배터리 기본 정보 입력</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    bat_type = st.selectbox("배터리 종류", ["NCM", "LFP", "NCA", "LCO"])
    props = BAT_PROPS[bat_type]
    st.caption(f"설계 사이클: {props['cycle_life']}회 | 정격전압: {props['nominal_v']}V")
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
        st.caption("🔴 2.5V 미만 — 안전 위험 (EU 배터리 규정)")
    elif v_diff > 0.5:
        st.caption(f"⚠️ 정격 대비 {v_diff:+.2f}V 이상")
    else:
        st.caption(f"🟢 정격 전압({props['nominal_v']}V) 정상")

# ─── SOH 입력 옵션 ───
st.markdown('<div class="section-title">🔢 SOH 정보</div>', unsafe_allow_html=True)
soh_mode = st.radio(
    "SOH 입력 방식",
    ["SOH 모름 — EIS 임피던스 지표로만 판단", "SOH 직접 입력 (용량 측정값 등 보유 시)"],
    index=0,
    horizontal=True
)

soh_input = None
if soh_mode == "SOH 직접 입력 (용량 측정값 등 보유 시)":
    soh_input = st.number_input(
        "SOH (%)", min_value=0, max_value=100, value=80, step=1,
        help="용량 측정법: SOH = 현재용량 / 초기용량 × 100%"
    )
    st.caption("📌 SOH 측정 기준: IEC 62660-1 (용량 측정법)")

# ─── EIS 파일 업로드 ───
st.markdown('<div class="section-title">📂 EIS 파일 업로드</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "EIS 측정 파일 (.xls, .csv) — 반복 측정 여러 개 동시 업로드 시 자동 평균",
    type=["xls", "csv"],
    accept_multiple_files=True,
    key="analysis_file"
)

if uploaded_files:
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

    # ─── 시각화 ───
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if len(dfs) > 1:
            for i, d in enumerate(dfs):
                fig.add_trace(go.Scatter(
                    x=d['z_real'], y=-d['z_imag'], mode='lines',
                    line=dict(color='rgba(0,212,170,0.2)', width=1),
                    showlegend=False
                ))
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
            for d in dfs:
                zm = np.sqrt(d['z_real']**2 + d['z_imag']**2) * 1000
                fig2.add_trace(go.Scatter(
                    x=d['freq'], y=zm, mode='lines',
                    line=dict(color='rgba(0,212,170,0.2)', width=1),
                    showlegend=False
                ))
        fig2.add_trace(go.Scatter(
            x=df['freq'], y=z_mag, mode='lines+markers',
            name='평균' if len(dfs) > 1 else '측정값',
            line=dict(color='#00d4aa', width=2), marker=dict(size=5)
        ))
        fig2.update_layout(
            xaxis_title="주파수 (Hz)", xaxis_type="log",
            yaxis_title="|Z| (mΩ)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── EIS 지표 분석 ───
    st.divider()
    re, rct, z_low = extract_eis_indicators(df)
    eis_score = impedance_health_score(re, rct, z_low, bat_type)

    # 최종 사용할 건강도 (SOH 직접 입력 우선, 없으면 EIS 점수)
    health = soh_input if soh_input is not None else eis_score
    health_label = "입력 SOH" if soh_input is not None else "EIS 건강도 점수"

    st.markdown('<div class="section-title">🔬 EIS 임피던스 진단</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    re_color  = "#00d4aa" if re  < 0.035 else "#f0a500" if re  < 0.055 else "#e05555"
    rct_color = "#00d4aa" if rct < 0.030 else "#f0a500" if rct < 0.050 else "#e05555"
    health_color = "#00d4aa" if health >= 80 else "#f0a500" if health >= 50 else "#e05555"

    for col, val, label, color, ref in zip(
        [m1, m2, m3, m4, m5],
        [f"{round(re*1000,2)}mΩ", f"{round(rct*1000,2)}mΩ",
         f"{round(z_low*1000,2)}mΩ", f"{round(health,1)}%", f"{voltage}V"],
        ["전해질 저항 (Re)", "전하전달 저항 (Rct)", "저주파 임피던스", health_label, "현재 전압"],
        [re_color, rct_color, "#00d4aa", health_color, "#00d4aa"],
        ["고주파 실수부", "반원 크기", "확산 저항 반영", "Niri et al. 2022", "측정값"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="ref-text">{ref}</div>
        </div>""", unsafe_allow_html=True)

    if soh_input is None:
        st.caption("📌 EIS 건강도 점수: Warwick DIB 데이터셋 SOH 100% 기준값 대비 Re·Rct 증가율로 산출 (Niri et al. 2022, doi:10.1016/j.est.2022.106295)")

    # ─── 안전성 평가 ───
    st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
    safety_txt, safety_color, safety_desc = safety_eval(eis_score, soh_input, years, cycles, bat_type, voltage)
    st.markdown(f"""
    <div class="metric-card" style="text-align:left; border:2px solid {safety_color};">
        <span style="font-size:20px; font-weight:700; color:{safety_color}">{safety_txt}</span>
        <span style="font-size:14px; color:#ccc; margin-left:12px;">{safety_desc}</span>
    </div>""", unsafe_allow_html=True)

    # ─── 추천 활용처 ───
    st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
    st.caption("📌 활용처별 SOH 기준 — Edge et al. (2023); Martinez-Laserna et al. (2018); IEC 62933")

    recs = get_recommendations(eis_score, soh_input, years, cycles, bat_type, voltage)

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
            배터리 종류: {bat_type} | 사용 연수: {years}년 |
            충방전: {cycles}회 ({cycle_pct}% 소모) | 전압: {voltage}V
        </div>
        <div style="font-size:11px; color:#666; margin-top:6px;">📚 근거: {final_ref}</div>
    </div>""", unsafe_allow_html=True)

else:
    st.info("👆 EIS 파일(.xls 또는 .csv)을 업로드하면 분석이 시작됩니다.")
    st.markdown("""
    **사용 방법:**
    1. 배터리 기본 정보 입력 (종류, 연수, 충방전 횟수, 전압)
    2. SOH 입력 방식 선택 (모르면 EIS 지표로 자동 판단)
    3. EIS 파일 업로드 (반복 측정 여러 개 동시 업로드 → 자동 평균)
    4. 임피던스 진단 결과 및 추천 활용처 확인
    """)
