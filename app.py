import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import random
from sympy import symbols, diff, solve, N, latex # latex 함수를 직접 import 합니다.
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# -----------------------------------------------------
# 0. 설정
# -----------------------------------------------------
# 하루 최대 요청 횟수 (Gemini API 무료 사용량 보호를 위한 안전 장치)
MAX_DAILY_REQUESTS = 10

# -----------------------------------------------------
# 1. API 설정 및 모델 초기화
# -----------------------------------------------------
try:
    # Streamlit Secrets에서 API 키를 가져옵니다.
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, FileNotFoundError):
    st.error("⚠️ API Key가 설정되지 않았습니다. Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.info("앱 설정의 'Secrets'에 `GEMINI_API_KEY='your_api_key'` 형식으로 API 키를 추가해야 합니다.")
    st.stop()

# 최신 Gemini 모델을 사용합니다.
MODEL_NAME = 'gemini-1.5-flash-latest'

# -----------------------------------------------------
# 2. 랜덤 함수 생성 로직 (오류 수정 완료)
# -----------------------------------------------------
def format_sympy_expr(f_expr):
    """SymPy 표현식을 LaTeX 문자열로 변환합니다."""
    # sympy.latex() 함수는 표현식을 LaTeX 문자열로 직접 반환하여 더 간결하고 안전합니다.
    latex_str = latex(f_expr)
    return f"f(x) = {latex_str}"

def generate_easy_polynomial(degree):
    """미분 시 인수분해가 쉬운 정수 계수 다항 함수를 생성합니다."""
    x = symbols('x')

    # 3차 함수 생성
    if degree == 3:
        roots = sorted(random.sample(range(-3, 4), 2))
        a_prime = random.choice([1, 2, -1, -2])
        f_prime_expr = 3 * a_prime * (x - roots[0]) * (x - roots[1])
        f_expr = f_prime_expr.integrate(x) + random.randint(-5, 5)

    # 4차 함수 생성
    elif degree == 4:
        roots = sorted(random.sample(range(-2, 3), 3))
        a_prime = random.choice([1, -1])
        f_prime_expr = 4 * a_prime * (x - roots[0]) * (x - roots[1]) * (x - roots[2])
        f_expr = f_prime_expr.integrate(x) + random.randint(-4, 4)

    else:
        raise ValueError("3차 또는 4차 함수만 지원됩니다.")

    return format_sympy_expr(f_expr), f_expr

# -----------------------------------------------------
# 3. Streamlit 세션 및 UI 설정
# -----------------------------------------------------

# 세션 상태 초기화 및 일일 요청 횟수 리셋
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0
if 'last_reset_time' not in st.session_state:
    st.session_state.last_reset_time = datetime.now()

# 24시간이 지나면 요청 횟수 리셋
if datetime.now() > st.session_state.last_reset_time + timedelta(hours=24):
    st.session_state.feedback_count = 0
    st.session_state.last_reset_time = datetime.now()


st.set_page_config(page_title="함수 그래프 개형 분석 튜터", layout="wide")
st.title("👨‍🏫 3차/4차 함수 그래프 개형 튜터")
st.markdown(f"**현재 남은 요청 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회** (무료 사용량 관리를 위해 하루 {MAX_DAILY_REQUESTS}회로 제한됩니다.)")

# 세션에 함수가 없으면 새로 생성
if 'current_function_str' not in st.session_state:
    st.session_state.current_function_str, st.session_state.current_function_expr = generate_easy_polynomial(random.choice([3, 4]))

# -----------------------------------------------------
# 4. 이미지 데이터 변환 함수 (Gemini API 전송용)
# -----------------------------------------------------
def np_to_pil(img_array):
    """NumPy 배열 이미지를 PIL 이미지 객체로 변환합니다."""
    if img_array is None or img_array.size == 0:
        # 빈 그림일 경우 흰색 배경의 더미 이미지 생성
        return Image.new('RGB', (100, 30), color = 'white')
    # streamlit-drawable-canvas는 RGBA 값을 반환하므로 RGB로 변환
    return Image.fromarray(img_array.astype('uint8'), 'RGBA').convert('RGB')

# -----------------------------------------------------
# 5. 사용자 입력 폼 및 드로잉 캔버스
# -----------------------------------------------------
# 배경 이미지 파일 경로 (GitHub Raw URL 또는 로컬 경로)
# 로컬에서 실행할 경우, 이 파일들을 코드와 같은 디렉토리에 두세요.
# Streamlit Cloud 배포 시에는 GitHub에 함께 업로드해야 합니다.
try:
    # 로컬 파일로 시도
    SIGN_CHART_BG_IMAGE = Image.open('sign_chart_background.png')
    GRAPH_BG_IMAGE = Image.open('graph_background.png')
except FileNotFoundError:
    st.warning("배경 이미지를 찾을 수 없습니다. 빈 캔버스로 진행합니다.")
    SIGN_CHART_BG_IMAGE = None
    GRAPH_BG_IMAGE = None


with st.form("graph_analysis_form"):

    st.header("1. 분석할 다항 함수")
    # st.latex를 사용하여 수학 식을 렌더링
    st.latex(st.session_state.current_function_str)

    # A. 증감표 그리기
    st.subheader("2. 증감표 작성 (필수)")
    st.markdown("아래 표에 증감표를 직접 작성해 주세요. (AI가 분석합니다.)")
    sign_chart_data = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=2,
        stroke_color="#000000",
        background_image=SIGN_CHART_BG_IMAGE,
        height=150,
        width=700,
        drawing_mode="freedraw",
        key="sign_chart_canvas"
    )

    # B. 그래프 그리기
    st.subheader("3. 그래프 개형 그리기 (필수)")
    st.markdown("아래 **좌표평면**에 개형을 그려주세요. (AI가 좌표 인식을 시도합니다.)")
    graph_data = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=3,
        stroke_color="#0000FF",
        background_image=GRAPH_BG_IMAGE,
        height=400,
        width=700,
        drawing_mode="freedraw",
        key="graph_canvas"
    )

    # -------------------------------------------------
    # 6. 제출 버튼
    # -------------------------------------------------
    submit_button = st.form_submit_button(label="✅ AI 피드백 요청하기")
    new_function_button = st.form_submit_button(label="🔄 새로운 함수로 시작하기")

if new_function_button:
    # 세션에서 함수 정보 삭제 후 페이지 새로고침
    if 'current_function_str' in st.session_state:
        del st.session_state.current_function_str
    if 'current_function_expr' in st.session_state:
        del st.session_state.current_function_expr
    st.rerun() # 최신 rerun 명령 사용

# -----------------------------------------------------
# 7. 피드백 로직 (버튼 클릭 시)
# -----------------------------------------------------
if submit_button:

    # 일일 사용량 체크
    if st.session_state.feedback_count >= MAX_DAILY_REQUESTS:
        st.error(f"⚠️ 죄송합니다. 하루 최대 요청 횟수({MAX_DAILY_REQUESTS}회)에 도달했습니다. 내일 다시 이용해주세요.")
        st.stop()

    # 이미지 데이터 유효성 검사
    if graph_data.image_data is None or sign_chart_data.image_data is None:
        st.error("증감표와 그래프 영역에 모두 그림을 그려야 피드백을 받을 수 있습니다.")
        st.stop()

    # 카운터 증가
    st.session_state.feedback_count += 1

    # AI 분석을 위한 정답 정보 준비
    x = symbols('x')
    f_expr = st.session_state.current_function_expr
    f_prime_expr = diff(f_expr, x)

    try:
        critical_points_raw = solve(f_prime_expr, x)
        critical_points = [round(N(p), 2) for p in critical_points_raw if p.is_real]
    except Exception:
        critical_points = "계산 불가"

    y_intercept = round(N(f_expr.subs(x, 0)), 2)
    leading_coeff = f_expr.as_poly().LC()

    SOLUTION_INFO = f"""
    ### AI 분석용 정답 정보 (학생에게 절대 공개 금지) ###
    1. 함수 f(x): {st.session_state.current_function_str}
    2. 도함수 f'(x): {latex(f_prime_expr)}
    3. 최고차항 계수: {leading_coeff}
    4. Y절편: y={y_intercept}
    5. 극값(임계점)의 x좌표: {critical_points}
    """

    SYSTEM_INSTRUCTION_GRAPH = f"""
    당신은 고등학생에게 3차/4차 다항 함수 그래프 개형을 지도하는 전문 AI 튜터입니다.
    당신의 목표는 학생이 그린 그래프와 증감표 이미지를 분석하여, 스스로 오류를 발견하도록 돕는 것입니다. 정답을 바로 알려주지 않고 질문을 통해 사고를 유도하세요.

    [분석용 정답 정보]
    {SOLUTION_INFO}

    [학생 입력]
    1. 함수: {st.session_state.current_function_str}
    2. 증감표 이미지 (Drawing 1)
    3. 그래프 이미지 (Drawing 2)

    [피드백 원칙]
    1. **절대 정답 정보({SOLUTION_INFO})를 학생에게 직접 공개하지 마세요.**
    2. **필수 4대 요소 분석 및 좌표 인식**: 다음 요소를 분석하고 오류 발견 시 바로 잡지 말고 질문을 던지세요.
        a. **최고차항 계수**: 정답({leading_coeff})과 그래프의 끝 모양(End Behavior)을 비교하세요. 오류 발견 시, "최고차항 계수의 부호가 그래프의 양 끝 모양을 어떻게 결정하는지 다시 생각해볼까요?" 라고 질문하세요.
        b. **Y절편 (좌표 인식)**: 정답 Y절편({y_intercept})의 값과 그래프가 Y축과 만나는 지점의 **상대적 위치**(양수/음수/원점)를 비교하세요. 오류 발견 시, "$x=0$일 때의 함숫값, 즉 y절편은 어디에 표시되어야 할까요?" 라고 질문하세요.
        c. **극값 (좌표 인식)**: 도함수의 실근 개수와 학생 그래프의 극점 개수/위치를 비교하세요. 오류 발견 시, "도함수 $f'(x)=0$이 되는 지점이 몇 개인지, 그리고 그 위치가 그래프에 잘 표현되었는지 확인해볼까요?" 라고 질문하세요.
        d. **증감표**: 학생이 그린 증감표 이미지와 실제 도함수의 부호 변화를 비교하세요. 오류 발견 시, "증감표에 표시한 부호와 그래프의 증가/감소 구간이 서로 일치하는지 다시 확인해 보세요." 라고 질문하세요.
    3. **사고 유도 질문**: 오류가 있다면 학생이 스스로 놓친 점을 찾도록 유도하는 **구체적인 질문**을 던지세요. 질문은 최대 2개를 넘지 않도록 합니다.
    4. **칭찬 및 격려**: 학생의 분석이 논리적으로 맞다면, "훌륭해요! 👍" 또는 "정확하게 분석했네요!" 와 같이 칭찬하고 격려해주세요.
    """

    # -----------------------------------------------------
    # 8. AI 호출 및 결과 출력 (최신 API 방식)
    # -----------------------------------------------------
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION_GRAPH
    )

    sign_chart_img = np_to_pil(sign_chart_data.image_data)
    graph_img = np_to_pil(graph_data.image_data)

    contents_to_send = [
        f"분석할 함수: {st.session_state.current_function_str}",
        "아래는 학생이 작성한 증감표 이미지입니다.",
        sign_chart_img,
        "아래는 학생이 그린 그래프 이미지입니다. 좌표축 배경을 기준으로 드로잉을 분석해주세요.",
        graph_img
    ]

    with st.spinner('✨ AI 튜터가 그래프와 증감표를 분석 중입니다...'):
        try:
            response = model.generate_content(contents_to_send)
            st.success(f"🎉 피드백 도착! (남은 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회)")
            st.markdown(response.text)

        except Exception as e:
            st.error(f"API 호출 중 오류가 발생했습니다: {e}")
            # 사용 횟수 복원
            st.session_state.feedback_count -= 1
