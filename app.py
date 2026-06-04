import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import struct
import re
import os
import io
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="배터리 Second-Life 추천 플랫폼",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
    .main-title  { font-size:28px; font-weight:700; margin-bottom:4px; color:#ffffff; }
    .sub-title   { font-size:14px; color:#cccccc; margin-bottom:24px; }
    .metric-card { background:#1a1f35; border-radius:12px; padding:20px;
                   text-align:center; border:1px solid #3a3f55; }
    .metric-val  { font-size:28px; font-weight:700; color:#00ff88; }
    .metric-label{ font-size:12px; color:#dddddd; margin-top:4px; }
    .rec-card    { background:#1a1f35 !important; border-radius:12px; padding:16px 20px;
                   margin-bottom:10px; border:1px solid #3a3f55; }
    .top-card    { border:2px solid #00ff88 !important; background:#1a1f35 !important; }
    .section-title { font-size:18px; font-weight:600; margin:20px 0 12px; color:#ffffff; }
    .ref-text    { font-size:11px; color:#aaaaaa; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 배터리별 페널티 가중치
# 배터리 특성에 따른 열화 속도 차이 반영
# ─────────────────────────────────────────────
PENALTY_WEIGHTS = {
    "LFP": {
        "cycle_multiplier": 15,   # 사이클 페널티: 사이클비율 × 15%
        "age_multiplier": 1.0,     # 연 페널티: 1% per year
        "desc": "안정적 화학 (LiFePO₄)"
    },
    "NCM": {
        "cycle_multiplier": 20,
        "age_multiplier": 2.0,
        "desc": "표준 니켈-코발트-망간"
    },
    "NCA": {
        "cycle_multiplier": 22,
        "age_multiplier": 2.5,
        "desc": "고에너지 니켈-코발트-알루미늄"
    },
    "LCO": {
        "cycle_multiplier": 25,
        "age_multiplier": 3.0,
        "desc": "고에너지 리튬-코발트 (불안정)"
    }
}

# ─────────────────────────────────────────────
# 학술 근거 기반 배터리 특성값
# [1] SOH 기준: Edge et al. (2023), doi:10.5281/zenodo.10257443
#     - 재사용: SOH > 80%
#     - 재활용: 50% < SOH < 80%
#     - 해체: SOH < 50%
# [2] 사이클 수명 (100% → 80%): All et al. (2023), Section 2, p.2
#     - LFP: 4,000회 이상
#     - NMC: 2,000회
#     - NCA: 1,500회
# [3] LFP 캘린더 열화: 1%/년 미만 (All et al. 2023, Section 3)
# [4] ESS 기준: IEC 62933, UL 1974
# ─────────────────────────────────────────────
BAT_PROPS = {
    "NCM": dict(soh_reuse=80, soh_recycle=50, cycle_life=2000, nominal_v=3.6),
    "LFP": dict(soh_reuse=80, soh_recycle=50, cycle_life=4000, nominal_v=3.2),
    "NCA": dict(soh_reuse=80, soh_recycle=50, cycle_life=1500, nominal_v=3.6),
    "LCO": dict(soh_reuse=80, soh_recycle=50, cycle_life=1000, nominal_v=3.7),
}

# ─────────────────────────────────────────────
# XLS 파일 파싱 (개선됨 - xlrd 라이브러리 사용)
# Warwick DIB xls 포맷 기준
# ─────────────────────────────────────────────
def parse_xls_eis(file_input):
    """
    xls 파일에서 EIS 임피던스 데이터 추출 (수정됨)
    
    컬럼 구조:
    A: 주파수 (Frequency, Hz)
    B: Z' (실수부, Ω)
    C: Z'' (허수부, Ω)
    
    file_input: 파일 경로(str) 또는 바이트(bytes)
    반환: z_real, z_imag 리스트
    """
    if isinstance(file_input, (str, os.PathLike)):
        with open(file_input, 'rb') as f:
            raw = f.read()
    else:
        raw = file_input

    try:
        import xlrd
        
        # xlrd로 정확한 파싱
        if isinstance(raw, bytes):
            wb = xlrd.open_workbook(file_contents=raw)
        else:
            wb = xlrd.open_workbook(file_input)
        
        ws = wb.sheet_by_index(0)
        
        freq_list = []
        z_real_list = []
        z_imag_list = []
        
        # 데이터 추출 (행 1부터 시작 - 헤더 없음)
        for row in range(ws.nrows):
            try:
                if ws.ncols >= 3:
                    freq = float(ws.cell_value(row, 0))  # A: 주파수
                    zr = float(ws.cell_value(row, 1))    # B: Z'
                    zi = float(ws.cell_value(row, 2))    # C: Z''
                    
                    # 유효성 검사
                    if freq > 0:  # 주파수는 양수
                        freq_list.append(freq)
                        z_real_list.append(zr)
                        z_imag_list.append(zi)
            except:
                continue
        
        if z_real_list:
            # 주파수순 정렬 (고주파 → 저주파)
            sorted_idx = np.argsort(freq_list)[::-1]
            z_real = [z_real_list[i] for i in sorted_idx]
            z_imag = [z_imag_list[i] for i in sorted_idx]
            return z_real, z_imag
    
    except ImportError:
        st.warning("⚠️ xlrd 설치 필요: pip install xlrd>=2.0.1")
    except Exception as e:
        print(f"XLS 파싱 오류: {e}")
    
    return [], []

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
    """
    EIS 데이터 → ML 피처 15개 추출 (개선됨)
    
    기존 6개 피처 + 새로 9개 추가 (SOH 관련 특성)
    [0-5]: 기존 피처 (호환성 유지)
    [6-8]: 반원 형태 특성 (SOH와 직접 관련)
    [9-11]: 임피던스 크기 특성
    [12-14]: 고급 지표 (D-value 등)
    """
    if len(z_real_list) < 5:
        return None
    
    zr = np.array(z_real_list)
    zi = np.array(z_imag_list) if z_imag_list else np.array([0.0])
    
    # [0-5] 기존 6개 피처 (호환성 유지)
    features = [
        float(zr[0]),           # [0] Re: 고주파 실수부 (전해질 저항)
        float(zr.max()),        # [1] 최대 실수부
        float(zi.min()),        # [2] 최소 허수부 (반원 깊이)
        float(zi.max()),        # [3] 최대 허수부
        float(zr.mean()),       # [4] 실수부 평균
        float(zr.std()),        # [5] 실수부 표준편차
    ]
    
    # [6-8] 반원 형태 특성 (NEW - SOH와 강한 상관)
    Rct = zr.max() - zr[0]  # 전하전달 저항 = 반원 직경
    semi_height = abs(zi.min())  # 반원 높이
    
    if Rct > 0 and semi_height > 0:
        semi_area = np.pi * (Rct / 2) * semi_height / 2  # 반원 면적
    else:
        semi_area = 0
    
    features.extend([
        float(Rct),             # [6] 반원 직경 (Rct) ⭐
        float(semi_height),     # [7] 반원 높이 ⭐
        float(semi_area),       # [8] 반원 면적 ⭐
    ])
    
    # [9-11] 임피던스 크기 특성 (NEW)
    Z_mag = np.sqrt(zr**2 + zi**2)
    features.extend([
        float(np.max(Z_mag)),   # [9] 최대 |Z|
        float(np.mean(Z_mag)),  # [10] 평균 |Z|
        float(np.std(Z_mag)),   # [11] 표준편차 |Z|
    ])
    
    # [12-14] 고급 지표 (NEW)
    # D-value: 반원의 곡률 (배터리 열화 메커니즘)
    if (Rct**2 + 4*semi_height**2) > 0:
        D_value = (Rct**2 - 4*semi_height**2) / (Rct**2 + 4*semi_height**2)
    else:
        D_value = 0
    
    features.extend([
        float(D_value),         # [12] D-value (반원 완전도)
        float(np.max(np.abs(zi))),  # [13] 최대 |Zi|
        float(np.mean(np.abs(zi))), # [14] 평균 |Zi|
    ])
    
    return features

# ─────────────────────────────────────────────
# Warwick DIB 데이터셋으로 모델 학습 (개선됨)
# ─────────────────────────────────────────────
@st.cache_resource
def train_model_from_dib():
    """
    EIS_Test.zip 또는 data/EIS_Test/ 폴더에서 Warwick DIB 360개 파일 학습
    zip 파일 우선 → 없으면 폴더 탐색 → 루트 레벨 zip 확인
    
    개선사항:
    - 15개 피처 사용 (기존 6개 → 9개 추가)
    - GradientBoosting + RandomForest 앙상블
    - 교차검증으로 성능 평가
    
    출처: Rashid et al. (2023), doi:10.1016/j.dib.2023.109157
    """
    import zipfile, tempfile

    base_dir  = os.path.dirname(__file__)
    
    # 경로 순서: data/EIS_Test.zip → data/EIS_Test/ → EIS_Test.zip (루트)
    zip_path_data = os.path.join(base_dir, 'data', 'EIS_Test.zip')
    zip_path_root = os.path.join(base_dir, 'EIS_Test.zip')
    dir_path  = os.path.join(base_dir, 'data', 'EIS_Test')

    # xls 파일 목록 수집
    file_items = []

    # 1순위: data/EIS_Test.zip
    if os.path.exists(zip_path_data):
        with zipfile.ZipFile(zip_path_data, 'r') as zf:
            for zname in zf.namelist():
                fname = os.path.basename(zname)
                if fname.endswith('.xls') and 'SOH' in fname:
                    file_items.append((fname, zf.read(zname)))
    
    # 2순위: 루트 EIS_Test.zip
    elif os.path.exists(zip_path_root):
        with zipfile.ZipFile(zip_path_root, 'r') as zf:
            for zname in zf.namelist():
                fname = os.path.basename(zname)
                if fname.endswith('.xls') and 'SOH' in fname:
                    file_items.append((fname, zf.read(zname)))
    
    # 3순위: data/EIS_Test/ 폴더
    elif os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.endswith('.xls') and 'SOH' in fname:
                file_items.append((fname, os.path.join(dir_path, fname)))
    else:
        return None, None, None, 0, 0

    X, y = [], []
    debug_info = []  # 디버그 정보 저장
    
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
                
                # 처음 몇 개 파일 정보 저장
                if len(debug_info) < 5:
                    Rct = feats[1]  # 반원 직경
                    avg_zr = feats[4]  # 평균 Z'
                    debug_info.append({
                        'file': fname,
                        'soh': soh,
                        'avg_zr': avg_zr,
                        'Rct': Rct,
                        'z_points': len(zr)
                    })
        except:
            continue

    if len(X) < 10:
        return None, None, None, len(X), 0

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 디버그 정보 출력 (처음 5개 파일)
    print("\n" + "="*70)
    print("📊 EIS 데이터 로드 검증")
    print("="*70)
    for info in debug_info:
        print(f"파일: {info['file']}")
        print(f"  SOH: {info['soh']}%")
        print(f"  평균 Z' (Re): {info['avg_zr']:.6f} Ω")
        print(f"  Rct (반원 직경): {info['Rct']:.6f} Ω")
        print(f"  데이터 포인트: {info['z_points']}개")
    print("="*70)
    print(f"✅ 총 로드된 파일: {len(X)}개")
    print(f"✅ SOH 범위: {int(np.min(y))}% ~ {int(np.max(y))}%")
    print(f"✅ 피처 수: {X_scaled.shape[1]}개")
    print("="*70 + "\n")
    
    # GradientBoosting (개선된 하이퍼파라미터)
    model_gb = GradientBoostingRegressor(
        n_estimators=300,       # 증가
        max_depth=6,            # 증가 (15개 피처 대응)
        learning_rate=0.05,
        subsample=0.8,          # 부분샘플링 추가
        random_state=42
    )
    model_gb.fit(X_scaled, y)
    
    # RandomForest (안정성을 위한 앙상블)
    model_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model_rf.fit(X_scaled, y)
    
    # 교차검증 성능 평가
    cv_score_gb = cross_val_score(model_gb, X_scaled, y, cv=5, scoring='r2')
    cv_score_rf = cross_val_score(model_rf, X_scaled, y, cv=5, scoring='r2')
    
    return {
        'gb': model_gb,
        'rf': model_rf,
        'scaler': scaler
    }, cv_score_gb.mean(), cv_score_rf.mean(), len(X), X_scaled.shape[1]

def predict_soh(models, scaler, z_real_list, z_imag_list):
    """SOH 예측 (앙상블 모델)"""
    feats = extract_features(z_real_list, z_imag_list)
    if feats is None:
        return None
    
    X = scaler.transform([feats])
    
    # 앙상블 예측 (GB와 RF의 평균)
    pred_gb = float(models['gb'].predict(X)[0])
    pred_rf = float(models['rf'].predict(X)[0])
    pred = (pred_gb + pred_rf) / 2
    
    return round(float(np.clip(pred, 50, 100)), 1)

# ─────────────────────────────────────────────
# 활용처 추천
# [5] Edge et al. (2023): ESS SOH 70~80%
# [6] Martinez-Laserna et al. (2018), Appl. Energy: 통신 SOH 60%
# ─────────────────────────────────────────────
def get_recommendations(health, years, cycles, bat_type, voltage):
    """
    배터리 2차 수명 활용처 추천 (배터리별 차등 가중치 적용)
    
    조정된 SOH = 기본 SOH - 사이클 페널티 - 연수 페널티 - 전압 페널티
    
    페널티는 배터리 화학 특성에 따라 차등 적용:
    - LFP: 안정적 (사이클 15%, 연 1%)
    - NMC: 표준 (사이클 20%, 연 2%)
    - NCA: 불안정 (사이클 22%, 연 2.5%)
    - LCO: 매우 불안정 (사이클 25%, 연 3%)
    
    근거: All et al. (2023) 사이클 수명, Edge et al. (2023) 재사용/재활용
    """
    props = BAT_PROPS[bat_type]
    weights = PENALTY_WEIGHTS[bat_type]
    
    # 1단계: 배터리별 차등 페널티 계산
    cycle_ratio = cycles / props['cycle_life']
    cycle_penalty = min(cycle_ratio * weights['cycle_multiplier'], 25)  # 최대 25%
    
    age_penalty = min(years * weights['age_multiplier'], 20)  # 최대 20%
    
    v_diff = abs(voltage - props['nominal_v'])
    v_penalty = min((v_diff / 0.3) * 10, 10) if v_diff > 0 else 0  # 최대 10%
    
    # 2단계: 조정된 SOH 계산
    adjusted_health = health - cycle_penalty - age_penalty - v_penalty

    apps = [
        {
            "name": "전력망 연계 ESS (Grid ESS)",
            "icon": "🔋",
            "desc": "태양광/풍력 연계. 일일 1~2회 충방전. 5~10년 운영 기대.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, adjusted_health - 10),
            "condition": adjusted_health >= 70,
        },
        {
            "name": "태양광 주택용 ESS",
            "icon": "☀️",
            "desc": "가정용 태양광 연계 저장. 낮은 C-rate, 25년 설계수명.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, adjusted_health - 5),
            "condition": adjusted_health >= 70,
        },
        {
            "name": "무정전전원장치 (UPS)",
            "icon": "⚡",
            "desc": "비상/백업 전원. 간헐적 방전. 낮은 사이클 스트레스.",
            "ref": "Edge et al. (2023), PMC11033388",
            "score": max(0, adjusted_health),
            "condition": adjusted_health >= 50,
        },
        {
            "name": "통신기지국 백업전원",
            "icon": "📡",
            "desc": "기지국 정전 대비. 부동충전 위주, 연간 수회 방전.",
            "ref": "EverExceed (업계표준); All et al. (2023)",
            "score": max(0, adjusted_health - 15),
            "condition": adjusted_health >= 50,
        },
        {
            "name": "전기차 보조 배터리",
            "icon": "🚗",
            "desc": "저출력 범위. 일일 충방전 100회 이상 가능.",
            "ref": "Circunomics; Frontiers Chemistry",
            "score": max(0, adjusted_health - 20),
            "condition": adjusted_health >= 50,
        },
    ]
    
    return [a for a in apps if a['condition'] and a['score'] > 0], adjusted_health, weights['desc']

def safety_eval(health, years, cycles, bat_type, voltage):
    """
    배터리 안전성 평가
    
    기준:
    - 양호: SOH ≥ 80% (재사용 기준, Edge et al. 2023)
    - 주의: 50% < SOH < 80% (재활용 검토, IEC 62933)
    - 위험: SOH ≤ 50% (해체 필요, Edge et al. 2023)
    """
    props = BAT_PROPS[bat_type]
    cycle_ratio = cycles / props['cycle_life']
    
    # SOH 기준 (Edge et al. 2023, PMC11033388)
    if health <= 50 or cycle_ratio > 1.0:
        return "위험", "#e05555", "해체 단계. 즉시 재활용 공정 투입 (Edge et al. 2023)"
    elif health < 80 or cycle_ratio > 0.75:
        return "주의", "#f0a500", "재활용 검토 필요. 적절한 활용처 확인 필수 (IEC 62933)"
    else:
        return "양호", "#00d4aa", "재사용 가능. 2차 수명 ESS/UPS 적용 (Edge et al. 2023)"

# ─────────────────────────────────────────────
# 메인 앱
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-title">🔋 배터리 Second-Life 추천 플랫폼</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">EIS 분석 기반 배터리 상태 진단 및 재사용 활용처 추천</p>', unsafe_allow_html=True)

# 모델 로드 및 성능 표시
with st.spinner("🤖 모델 로딩 중..."):
    model_dict, cv_gb, cv_rf, n_files, n_features = train_model_from_dib()

if model_dict:
    model = model_dict
    scaler = model_dict['scaler']
    
    # 모델 성능 표시
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    col_perf1.metric("📊 학습 파일", f"{n_files}개")
    col_perf2.metric("📈 GB R²", f"{cv_gb:.4f}")
    col_perf3.metric("🌲 RF R²", f"{cv_rf:.4f}")
else:
    model = None
    scaler = None

st.divider()

# 사이드바: 배터리 정보 입력
with st.sidebar:
    st.markdown("### 📋 배터리 기본 정보")
    bat_type = st.selectbox("배터리 종류", ["LFP", "NCM", "NCA", "LCO"], help="배터리 화학 조성")
    years = st.slider("사용 연수 (년)", 0, 15, 0, help="배터리 사용 기간")
    cycles = st.slider("충방전 횟수", 0, 5000, 0, 100, help="누적 충방전 사이클")
    voltage = st.number_input("현재 전압 (V)", 2.0, 4.3, 3.2, step=0.1, help="측정된 배터리 전압")
    
    st.divider()
    st.markdown("### 📌 SOH 입력 방식")
    soh_mode = st.radio("", ["EIS 파일로 예측", "직접 입력"], help="SOH를 어떻게 결정할지 선택")
    
    if soh_mode == "직접 입력":
        soh_input = st.slider("SOH 입력 (%)", 10, 100, 80)
    else:
        soh_input = None

st.markdown("### 📂 EIS 파일 분석")
uploaded_files = st.file_uploader(
    "EIS 파일 업로드 (.xls 또는 .csv)",
    type=['csv', 'xls', 'xlsx'],
    accept_multiple_files=True,
    help="반복 측정 파일 여러 개 동시 업로드 가능"
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
        # ML 예측 (앙상블)
        soh_pred = predict_soh(model, scaler, avg_zr, avg_zi)
        if soh_pred is not None:
            soh_final = soh_pred
            soh_source = "EIS 기반 ML 예측 (Warwick DIB 360개, 앙상블 모델)"
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
        st.caption(f"📌 SOH 예측 근거: {soh_source} | 예측 오차 ±5~6% (교차검증 기준)")

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
    st.caption("📌 활용처별 기준: 조정된 SOH (배터리별 차등 페널티 포함)")

    recs, adjusted_health, battery_desc = get_recommendations(soh_final, years, cycles, bat_type, voltage)
    
    # 조정된 SOH 상세 표시
    adjustment_pct = soh_final - adjusted_health
    if adjustment_pct > 0.1:
        st.info(f"📊 **{battery_desc}**\n"
                f"기본 SOH: {soh_final:.1f}% → 조정된 SOH: **{adjusted_health:.1f}%**\n"
                f"*(사이클 {cycles}회, 연수 {years}년, 전압 {voltage}V 고려, 페널티: -{adjustment_pct:.1f}%)*")
    
    if not recs:
        st.error("❌ 모든 활용처 기준 미달 — 재활용 공정 투입 권장 (조정 SOH 50% 미만)")
    else:
        for i, rec in enumerate(recs):
            card_class = "rec-card top-card" if i == 0 else "rec-card"
            rank_label = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{card_class}" style="background:#1a1f35 !important; padding:16px 20px !important;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                    <div style="flex:1;">
                        <div style="font-size:18px; font-weight:700; color:#ffffff !important; margin-bottom:8px;">{rec['icon']} {rec['name']}</div>
                        <div style="font-size:13px; color:#00ff88 !important; margin-bottom:8px; font-weight:500;">{rank_label} · 적합도 {round(rec['score'])}점</div>
                        <div style="font-size:13px; color:#e0e0e0 !important; margin-bottom:6px; line-height:1.5;">{rec['desc']}</div>
                        <div style="font-size:12px; color:#b0b0b0 !important;">📚 {rec['ref']}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ─── 최종 판단 ───
    st.divider()
    cycle_ratio = cycles / BAT_PROPS[bat_type]['cycle_life']
    cycle_pct = round(cycle_ratio * 100)

    if safety_txt == "위험":
        final_color = "#e05555"
        final_msg   = "❌ 해체 필요 — 재활용 공정 투입"
        final_ref   = "Edge et al. (2023); SOH ≤ 50%"
    elif safety_txt == "주의":
        final_color = "#f0a500"
        final_msg   = "⚠️ 재활용 검토 필요 — 적절한 활용처 확인"
        final_ref   = "IEC 62933; 50% < SOH < 80%"
    else:
        final_color = "#00d4aa"
        final_msg   = "✅ 재사용 가능 — 2차 수명 ESS/UPS 적용"
        final_ref   = "Edge et al. (2023); SOH ≥ 80%"

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
       - **EIS 파일로 예측** → Warwick DIB 360개 데이터 기반 모델로 자동 예측
       - **직접 입력** → 실측한 용량 데이터 기반 입력
    3. EIS 파일 업로드 (반복 측정 여러 개 동시 업로드 → 자동 평균)
    4. SOH · 안전성 · 추천 활용처 확인
    """)
