import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import random
from sympy import symbols, diff, solve, N, latex
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# -----------------------------------------------------
# 0. 설정
# -----------------------------------------------------
MAX_DAILY_REQUESTS = 10

# -----------------------------------------------------
# 1. API 설정 및 모델 초기화
# -----------------------------------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except (KeyError, FileNotFoundError):
    st.error("⚠️ API Key가 설정되지 않았습니다. Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.info("앱 설정의 'Secrets'에 `GEMINI_API_KEY='your_api_key'` 형식으로 API 키를 추가해야 합니다.")
    st.stop()

MODEL_NAME = 'gemini-1.5-flash-latest'

# -----------------------------------------------------
# 2. 랜덤 함수 생성 로직
# -----------------------------------------------------
def format_sympy_expr(f_expr):
    latex_str = latex(f_expr)
    return f"f(x) = {latex_str}"

def generate_easy_polynomial(degree):
    x = symbols('x')
    if degree == 3:
        roots = sorted(random.sample(range(-3, 4), 2))
        a_prime = random.choice([1, 2, -1, -2])
        f_prime_expr = 3 * a_prime * (x - roots[0]) * (x - roots[1])
        f_expr = f_prime_expr.integrate(x) + random.randint(-5, 5)
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
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0
if 'last_reset_time' not in st.session_state:
    st.session_state.last_reset_time = datetime.now()

if datetime.now() > st.session_state.last_reset_time + timedelta(hours=24):
    st.session_state.feedback_count = 0
    st.session_state.last_reset_time = datetime.now()

st.set_page_config(page_title="함수 그래프 개형 분석 튜터", layout="wide")
st.title("👨‍🏫 3차/4차 함수 그래프 개형 튜터")
st.markdown(f"**현재 남은 요청 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회**")

if 'current_function_str' not in st.session_state:
    st.session_state.current_function_str, st.session_state.current_function_expr = generate_easy_polynomial(random.choice([3, 4]))

# -----------------------------------------------------
# 4. 이미지 데이터 변환 함수
# -----------------------------------------------------
def np_to_pil(img_array):
    if img_array is None or img_array.size == 0:
        return Image.new('RGB', (100, 30), color='white')
    return Image.fromarray(img_array.astype('uint8'), 'RGBA').convert('RGB')

# -----------------------------------------------------
# 5. 사용자 입력 폼 및 드로잉 캔버스 (오류 수정)
# -----------------------------------------------------
try:
    # 이미지를 불러온 후, .convert('RGB')를 통해 투명도 값을 제거하여 오류를 방지합니다.
    SIGN_CHART_BG_IMAGE = Image.open('sign_chart_background.png').convert('RGB')
    GRAPH_BG_IMAGE = Image.open('graph_background.png').convert('RGB')
except FileNotFoundError:
    st.warning("배경 이미지 파일을 찾을 수 없어 빈 캔버스로 표시됩니다.")
    SIGN_CHART_BG_IMAGE = None
    GRAPH_BG_IMAGE = None

with st.form("graph_analysis_form"):
    st.header("1. 분석할 다항 함수")
    st.latex(st.session_state.current_function_str)

    st.subheader("2. 증감표 작성 (필수)")
    sign_chart_data = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", stroke_width=2, stroke_color="#000000",
        background_image=SIGN_CHART_BG_IMAGE, height=150, width=700,
        drawing_mode="freedraw", key="sign_chart_canvas"
    )

    st.subheader("3. 그래프 개형 그리기 (필수)")
    graph_data = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", stroke_width=3, stroke_color="#0000FF",
        background_image=GRAPH_BG_IMAGE, height=400, width=700,
        drawing_mode="freedraw", key="graph_canvas"
    )

    submit_button = st.form_submit_button(label="✅ AI 피드백 요청하기")

if st.button("🔄 새로운 함수로 시작하기"):
    if 'current_function_str' in st.session_state:
        del st.session_state.current_function_str
    if 'current_function_expr' in st.session_state:
        del st.session_state.current_function_expr
    st.rerun()

# -----------------------------------------------------
# 7. 피드백 로직
# -----------------------------------------------------
if submit_button:
    if st.session_state.feedback_count >= MAX_DAILY_REQUESTS:
        st.error(f"⚠️ 하루 최대 요청 횟수({MAX_DAILY_REQUESTS}회)에 도달했습니다.")
        st.stop()

    if graph_data.image_data is None or sign_chart_data.image_data is None:
        st.error("증감표와 그래프를 모두 그려야 피드백을 받을 수 있습니다.")
        st.stop()

    st.session_state.feedback_count += 1

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

    SOLUTION_INFO = f"정답 정보: f'(x)={latex(f_prime_expr)}, 최고차항 계수={leading_coeff}, y절편={y_intercept}, 극점 x좌표={critical_points}"
    SYSTEM_INSTRUCTION_GRAPH = f"당신은 학생의 다항 함수 그래프를 분석하는 AI 튜터입니다. 정답({SOLUTION_INFO})과 학생이 그린 증감표, 그래프를 비교하여 잘못된 부분을 질문으로 유도하세요. 정답을 직접 알려주지 마세요. 최고차항, y절편, 극점, 증감표-그래프 일치 여부를 중심으로 분석하세요. 잘했다면 칭찬해주세요."

    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM_INSTRUCTION_GRAPH)
    sign_chart_img = np_to_pil(sign_chart_data.image_data)
    graph_img = np_to_pil(graph_data.image_data)

    contents_to_send = [
        f"분석할 함수: {st.session_state.current_function_str}",
        "학생이 작성한 증감표:", sign_chart_img,
        "학생이 그린 그래프:", graph_img
    ]

    with st.spinner('✨ AI 튜터가 분석 중입니다...'):
        try:
            response = model.generate_content(contents_to_send)
            st.success(f"🎉 피드백 도착! (남은 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회)")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"API 호출 중 오류: {e}")
            st.session_state.feedback_count -= 1
