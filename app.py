import streamlit as st
from google import genai
from google.genai import types
import os
from datetime import datetime, timedelta
import random
from sympy import symbols, diff, solve, N
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import numpy as np

# -----------------------------------------------------
# 0. 설정 및 안전 장치
# -----------------------------------------------------
MAX_DAILY_REQUESTS = 5 # 하루 최대 요청 횟수 (무료 한도 안전 장치)
API_KEY = "AIzaSyAU1iwa-OFdgFyiookp8Rcwez6rlNXajm4" # 실제 키 입력

# -----------------------------------------------------
# 1. API 설정 및 모델 초기화
# -----------------------------------------------------
# API 키는 Secrets 또는 코드에서 가져옵니다.
try:
    api_key = st.secrets.get("GEMINI_API_KEY", API_KEY)
except AttributeError:
    api_key = API_KEY

if not api_key:
    st.error("⚠️ API Key가 설정되지 않았습니다. Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = 'gemini-2.5-flash' 

# -----------------------------------------------------
# 2. 랜덤 함수 생성 로직
# -----------------------------------------------------
def format_sympy_expr(f_expr):
    """SymPy 식을 LaTeX 형식으로 보기 좋게 변환합니다."""
    f_str = str(f_expr).replace('**', '^').replace('*', '')
    
    # 계수 1 정리
    f_str = f_str.replace('1x', 'x').replace('-1x', '-x')
    
    # 덧셈 부호 정리
    f_str = f_str.replace('+ -', ' - ')
    
    # LaTeX 형식으로 최종 변환
    f_str = f_str.replace('^', '^{') + '}'
    f_str = f_str.replace('x^{', 'x^{')
    
    # 분수 처리 (예시로 단순화)
    # 실제로는 더 복잡한 분수 처리가 필요할 수 있으나, 이 예시에서는 생략
    
    return f"f(x) = {f_str}"

def generate_easy_polynomial(degree):
    """미분 시 인수분해가 쉬운 정수 계수 다항 함수를 생성합니다."""
    x = symbols('x')
    
    # 3차 함수
    if degree == 3:
        roots = sorted(random.sample(range(-3, 4), 2))
        a_prime = random.choice([1, 2, -1, -2])
        f_prime_expr = 3 * a_prime * (x - roots[0]) * (x - roots[1])
        f_expr = f_prime_expr.integrate(x) + random.randint(-5, 5) 
        
    # 4차 함수
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

# 세션 상태 초기화 및 일일 요청 횟수 리셋 로직
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0
if 'last_reset_time' not in st.session_state:
    st.session_state.last_reset_time = datetime.now()
    
if datetime.now() > st.session_state.last_reset_time + timedelta(hours=24):
    st.session_state.feedback_count = 0
    st.session_state.last_reset_time = datetime.now()


st.set_page_config(page_title="함수 그래프 개형 분석 튜터 챗봇", layout="wide")
st.title("👨‍🏫 3차/4차 함수 그래프 개형 튜터 챗봇")
st.markdown(f"**현재 남은 요청 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회** (무료 사용량 관리를 위해 하루 {MAX_DAILY_REQUESTS}회로 제한됩니다.)")

if 'current_function_str' not in st.session_state:
    st.session_state.current_function_str, st.session_state.current_function_expr = generate_easy_polynomial(random.choice([3, 4]))

# -----------------------------------------------------
# 4. 이미지 데이터 변환 함수 (Gemini API 전송용)
# -----------------------------------------------------
def np_to_bytes(img_array):
    """NumPy 배열 이미지를 PNG 바이트 데이터로 변환합니다."""
    # 배열이 None이거나 비어있으면 빈 이미지를 생성하여 오류 방지
    if img_array is None or img_array.size == 0:
        img_array = np.zeros((10, 10, 3), dtype=np.uint8) 
    
    # R, G, B 채널만 사용
    img = Image.fromarray(img_array[:, :, 0:3].astype('uint8'))
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    return byte_io.getvalue()

# -----------------------------------------------------
# 5. 사용자 입력 폼 및 드로잉 캔버스 구현
# -----------------------------------------------------
with st.form("graph_analysis_form"):
    
    st.header("1. 분석할 다항 함수")
    # 1. 지수 표현 개선 (LaTeX 적용)
    st.markdown(f"### **${st.session_state.current_function_str}$**") 
    
    # -------------------------------------------------
    # A. 증감표 그리기 유도 (Drawing Canvas 1) - 2번, 4번 반영
    # -------------------------------------------------
    st.subheader("2. 증감표 작성 (필수)")
    st.markdown("아래 표에 증감표를 직접 작성해 주세요. (AI가 분석합니다.)")
    
    # ⚠️ 증감표 배경 이미지 파일 경로 (GitHub에 'sign_chart_background.png' 업로드 필요)
    SIGN_CHART_BG_IMAGE = 'sign_chart_background.png'
    
    sign_chart_data = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=2,
        stroke_color="#000000",
        background_image=SIGN_CHART_BG_IMAGE, 
        height=150,
        width=700,
        drawing_mode="freedraw",
        key="sign_chart_canvas"
    )

    # -------------------------------------------------
    # B. 그래프 그리기 (Drawing Canvas 2) - 3번, 4번 반영
    # -------------------------------------------------
    st.header("3. 그래프 개형 그리기 (필수)")
    st.markdown("아래 **좌표평면**에 개형을 그려주세요. (AI가 좌표 인식을 시도합니다.)")
    
    # ⚠️ 좌표평면 배경 이미지 파일 경로 (GitHub에 'graph_background.png' 업로드 필요)
    GRAPH_BG_IMAGE = 'graph_background.png'
    
    graph_data = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=3,
        stroke_color="#000000",
        background_image=GRAPH_BG_IMAGE, 
        height=400,
        width=700,
        drawing_mode="freedraw",
        key="graph_canvas"
    )

    col_submit, col_new = st.columns(2)
    
    with col_submit:
        submit_button = st.form_submit_button(label="AI 피드백 요청하기")
        
    with col_new:
        if st.form_submit_button(label="새로운 함수로 시작하기"):
            del st.session_state.current_function_str
            del st.session_state.current_function_expr
            st.experimental_rerun()
        
# -----------------------------------------------------
# 6. 피드백 로직 (버튼 클릭 시)
# -----------------------------------------------------
if submit_button:
    
    # 1. 안전 장치 체크 (일일 사용량)
    if st.session_state.feedback_count >= MAX_DAILY_REQUESTS:
        st.error(f"⚠️ 죄송합니다. 무료 사용량 관리를 위해 하루 최대 요청 횟수({MAX_DAILY_REQUESTS}회)에 도달했습니다. 내일 다시 이용해주세요.")
        st.stop()
        
    # 2. 이미지 데이터 유효성 검사
    if graph_data.image_data is None:
        st.error("그래프 영역에 그림을 그려야 AI 피드백을 받을 수 있습니다.")
        st.stop()
        
    # 3. 카운터 증가 (API 호출 전에 증가시켜야 함)
    st.session_state.feedback_count += 1
    
    # 4. AI 비전 API 호출을 위한 데이터 준비
    x = symbols('x')
    f_expr = st.session_state.current_function_expr
    f_prime_expr = diff(f_expr, x)
    
    critical_points_raw = solve(f_prime_expr, x)
    critical_points = [N(p) for p in critical_points_raw if p.is_real]
    y_intercept = N(f_expr.subs(x, 0))
    
    SOLUTION_INFO = f"""
    ### AI 분석용 정답 정보 (절대 학생에게 공개하지 마세요) ###
    1. 함수 f(x): {st.session_state.current_function_str}
    2. 도함수 f'(x): {f_prime_expr}
    3. 최고차항 계수: {f_expr.as_poly().LC()}
    4. Y절편: y={y_intercept}
    5. 극값(임계점)의 x좌표: {critical_points} 
    """
    
    SYSTEM_INSTRUCTION_GRAPH = f"""
    당신은 고등학교 2학년 학생에게 3차/4차 다항 함수 그래프 개형을 지도하는 전문 AI 튜터입니다.
    당신의 목표는 학생이 그린 그래프 이미지와 작성한 증감표 이미지를 분석하여, 스스로 오류를 발견하도록 돕는 것입니다. 정답을 바로 알려주지 않고 질문을 통해 사고를 유도하세요.

    [분석용 정답 정보]
    {SOLUTION_INFO}

    [학생 입력]
    1. 함수: {st.session_state.current_function_str}
    2. 증감표 이미지 (Drawing 1)
    3. 그래프 이미지 (Drawing 2)

    [당신의 피드백 원칙]
    1. **절대 정답 정보({SOLUTION_INFO})를 학생에게 직접 공개하지 마세요.**
    2. **필수 4대 요소 분석 및 좌표 인식 강화**: 다음 요소를 분석하고 오류 발견 시 바로 잡지 말고 질문을 던지세요.
        a. **최고차항 계수**: 정답 정보와 그래프의 끝 모양(End Behavior)을 비교하여 오류 발견 시, "최고차항 계수의 부호에 따라 그래프의 양 끝이 어떻게 되는지 확인했나요?" 질문.
        b. **Y절편 (좌표 인식)**: 정답 Y절편({y_intercept})의 값과 그래프가 Y축과 만나는 지점의 **상대적 위치** (양수/음수/원점)를 비교하여 오류를 지적. 오류 발견 시, "$x=0$일 때의 함숫값은 어디에 찍혀야 하나요?" 질문.
        c. **극값 (좌표 인식)**: 도함수의 근 개수와 학생 그래프의 극점 개수/위치를 비교하여 오류 발견 시, "도함수 $f'(x)$가 0이 되는 지점을 정확히 찾았고, 극점이 몇 개여야 할까요?" 질문.
        d. **증감표**: 학생이 그린 증감표 이미지와 실제 도함수의 부호 변화를 비교하여 오류 발견 시, "증감표의 부호와 그래프의 증가/감소 구간이 일치하는지 다시 확인해 보세요." 질문.
    3. **사고 유도 질문**: 오류가 있다면 학생이 스스로 놓친 점을 찾도록 유도하는 **구체적 질문**을 던지세요. 질문은 최대 2개를 넘지 않도록 합니다.
    4. **칭찬 및 보강**: 그래프 개형이 거의 완벽하다면 "논리적으로 완벽해요!👏"라고 칭찬하고, 그 그래프의 특징을 추가로 설명해 주세요.
    """
    
    # -----------------------------------------------------
    # 7. AI 호출 및 결과 출력 (비전 API)
    # -----------------------------------------------------
    
    graph_image_bytes = np_to_bytes(graph_data.image_data)
    sign_chart_image_bytes = np_to_bytes(sign_chart_data.image_data)

    contents_to_send = [
        {"mime_type": "text/plain", "text": f"함수: {st.session_state.current_function_str}"},
        {"mime_type": "image/png", "data": sign_chart_image_bytes},
        {"mime_type": "text/plain", "text": "위는 학생이 작성한 증감표입니다. 아래는 학생이 그린 그래프입니다."},
        {"mime_type": "image/png", "data": graph_image_bytes},
    ]

    with st.spinner('✨ AI 튜터가 그래프와 증감표를 분석 중입니다... (이미지 분석은 시간이 조금 걸립니다.)'):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents_to_send,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION_GRAPH
                )
            )

            st.success(f"🎉 피드백이 도착했습니다! (남은 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회)")
            st.markdown(response.text)

        except Exception as e:
            st.error(f"API 호출 중 오류가 발생했습니다. 키 설정을 다시 확인하거나, 잠시 후 다시 시도해 주세요. (오류: {e})")
