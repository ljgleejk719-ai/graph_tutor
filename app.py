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

# -----------------------------------------------------
# 0. 설정 및 안전 장치
# -----------------------------------------------------
# ⚠️ Google Sheets 연동 코드는 제외되었습니다.
MAX_DAILY_REQUESTS = 5 # 하루 최대 요청 횟수 (무료 한도 안전 장치)

# -----------------------------------------------------
# 1. API 설정 및 모델 초기화
# -----------------------------------------------------
# API 키는 Secrets에서 가져옵니다.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # 이전에 발급받은 실제 키로 대체하여 로컬 테스트 가능
    api_key = "YOUR_GEMINI_API_KEY" 

if not api_key or api_key == "YOUR_GEMINI_API_KEY":
    st.error("⚠️ API Key가 설정되지 않았습니다. Secrets에 GEMINI_API_KEY를 설정해주세요.")
    st.stop()

client = genai.Client(api_key=api_key)
# 그래프 분석에는 이미지(Vision) 기능이 필요합니다.
MODEL_NAME = 'gemini-2.5-flash' 

# -----------------------------------------------------
# 2. 랜덤 함수 생성 로직
# -----------------------------------------------------
def generate_easy_polynomial(degree):
    """미분 시 인수분해가 쉬운 정수 계수 다항 함수를 생성합니다."""
    x = symbols('x')
    
    # 3차 함수: f'(x) = 3a(x-r1)(x-r2) 형태를 목표
    if degree == 3:
        roots = sorted(random.sample(range(-3, 4), 2)) # -3부터 3 사이의 서로 다른 두 근
        a_prime = random.choice([1, 2, -1, -2]) # 최고차항 계수
        f_prime_expr = 3 * a_prime * (x - roots[0]) * (x - roots[1])
        f_expr = f_prime_expr.integrate(x) + random.randint(-5, 5) # y절편 c
        
    # 4차 함수: f'(x) = 4a(x-r1)(x-r2)(x-r3) 형태를 목표
    elif degree == 4:
        roots = sorted(random.sample(range(-2, 3), 3)) # -2부터 2 사이의 서로 다른 세 근
        a_prime = random.choice([1, -1]) 
        f_prime_expr = 4 * a_prime * (x - roots[0]) * (x - roots[1]) * (x - roots[2])
        f_expr = f_prime_expr.integrate(x) + random.randint(-4, 4)
        
    else:
        # 이 코드는 실행되지 않지만 안전을 위해 추가
        raise ValueError("3차 또는 4차 함수만 지원됩니다.")

    # SymPy 객체를 예쁜 문자열로 변환 (예: x^3 + 2x^2 - 5)
    f_str = str(f_expr).replace('**', '^').replace('*', '')
    f_str = f_str.replace('1x', 'x').replace('-1x', '-x')
    
    return f"f(x) = {f_str}", f_expr

# -----------------------------------------------------
# 3. Streamlit 세션 및 UI 설정
# -----------------------------------------------------

# 세션 상태 초기화 및 일일 요청 횟수 리셋 로직
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0
if 'last_reset_time' not in st.session_state:
    st.session_state.last_reset_time = datetime.now()
    
# 24시간이 지났는지 확인하고 카운터 리셋
if datetime.now() > st.session_state.last_reset_time + timedelta(hours=24):
    st.session_state.feedback_count = 0
    st.session_state.last_reset_time = datetime.now()


st.set_page_config(page_title="함수 그래프 개형 분석 튜터 챗봇", layout="wide")
st.title("👨‍🏫 3차/4차 함수 그래프 개형 튜터 챗봇")
st.markdown(f"**현재 남은 요청 횟수: {MAX_DAILY_REQUESTS - st.session_state.feedback_count}회** (무료 사용량 관리를 위해 하루 {MAX_DAILY_REQUESTS}회로 제한됩니다.)")

# 함수가 세션에 없거나, 재시작 버튼이 눌렸을 때 새 함수 생성
if 'current_function_str' not in st.session_state:
    st.session_state.current_function_str, st.session_state.current_function_expr = generate_easy_polynomial(random.choice([3, 4]))

# -----------------------------------------------------
# 4. 사용자 입력 폼 및 드로잉 캔버스 구현
# -----------------------------------------------------
with st.form("graph_analysis_form"):
    
    st.header("1. 분석할 다항 함수")
    st.markdown(f"### **{st.session_state.current_function_str}**")
    
    # -------------------------------------------------
    # A. 증감표 그리기 유도 (Drawing Canvas 1)
    # -------------------------------------------------
    st.header("2. 증감표 작성 (필수)")
    st.markdown("아래 **흰색 영역**에 3행 8열 형태로 증감표 (x, f'(x), f(x) 포함)를 직접 작성해 주세요. (AI가 분석합니다.)")
    
    sign_chart_data = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=2,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=150,
        drawing_mode="freedraw",
        key="sign_chart_canvas"
    )

    # -------------------------------------------------
    # B. 그래프 그리기 (Drawing Canvas 2)
    # -------------------------------------------------
    st.header("3. 그래프 개형 그리기 (필수)")
    st.markdown("아래 좌표평면 영역에 위 함수에 대한 그래프 **개형**을 직접 그려주세요. (주요 절편, 극값 위치를 대략적으로 표시)")
    
    # 
    graph_data = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=3,
        stroke_color="#000000",
        background_image=None, 
        background_color="#F0F0FF", 
        height=400,
        width=700,
        drawing_mode="freedraw",
        key="graph_canvas"
    )

    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button(label="AI 피드백 요청하기")
    with col2:
        if st.form_submit_button(label="새로운 함수로 시작하기"):
            # 세션에서 함수 정보 삭제 후 페이지 새로고침
            del st.session_state.current_function_str
            del st.session_state.current_function_expr
            st.experimental_rerun()
        
# -----------------------------------------------------
# 5. 피드백 로직 (버튼 클릭 시)
# -----------------------------------------------------
if submit_button:
    
    # 1. 안전 장치 체크 (일일 사용량)
    if st.session_state.feedback_count >= MAX_DAILY_REQUESTS:
        st.error(f"⚠️ 죄송합니다. 무료 사용량 관리를 위해 하루 최대 요청 횟수({MAX_DAILY_REQUESTS}회)에 도달했습니다. 내일 다시 이용해주세요.")
        st.stop()
        
    # 2. 이미지 데이터 유효성 검사 (아무것도 안 그렸을 때)
    # Streamlit Canvas는 빈 그림에도 이미지 데이터를 반환할 수 있으므로, 
    # 여기서는 간단하게 None 체크만 합니다.
    if graph_data.image_data is None:
        st.error("그래프를 그려야 AI 피드백을 받을 수 있습니다.")
        st.stop()
        
    # 3. 카운터 증가 (API 호출 전에 증가시켜야 함)
    st.session_state.feedback_count += 1
    
    # 4. AI 비전 API 호출을 위한 데이터 준비
    
    # SymPy를 사용하여 함수의 정확한 정보 계산
    x = symbols('x')
    f_expr = st.session_state.current_function_expr
    f_prime_expr = diff(f_expr, x)
    
    # 극값(임계점) 계산 (실수 근만 추출)
    critical_points_raw = solve(f_prime_expr, x)
    critical_points = [N(p) for p in critical_points_raw if p.is_real]
    
    # y절편 계산
    y_intercept = N(f_expr.subs(x, 0))
    
    # AI에게 전달할 정답 정보 및 피드백 프롬프트
    SOLUTION_INFO = f"""
    ### AI 분석용 정답 정보 (절대 학생에게 공개하지 마세요) ###
    1. 함수 f(x): {st.session_state.current_function_str}
    2. 도함수 f'(x): {f_prime_expr}
    3. 최고차항 계수: {f_expr.as_poly().LC()}
    4. Y절편: y={y_intercept}
    5. 극값(임계점)의 x좌표: {critical_points} 
    6. 이 그래프는 최대 {len(critical_points)}개의 극점을 가질 수 있습니다.
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
    2. **필수 4대 요소 분석 및 사고 유도 질문**: 다음 요소를 분석하고 오류 발견 시 바로 잡지 말고 질문을 던지세요.
        a. **최고차항 계수**: 정답 정보와 그래프의 끝 모양(End Behavior)을 비교하여 오류 발견 시, "최고차항 계수의 부호에 따라 그래프의 양 끝이 어떻게 결정되는지 다시 한번 떠올려 보세요."와 같이 질문.
        b. **Y절편**: 정답 정보와 그래프의 y축 교차점을 비교하여 오류 발견 시, "$x=0$을 함수식에 대입했을 때 $y$의 값은 무엇이 되어야 하나요?"와 같이 질문.
        c. **극값**: 도함수의 근 개수와 학생이 그린 그래프의 극점 개수/위치를 비교하여 오류 발견 시, "도함수 $f'(x)$가 0이 되는 지점을 정확히 찾았고, 그 지점에서 $f'(x)$의 부호가 바뀌는지 확인했나요? 극점이 몇 개여야 할까요?"와 같이 질문.
        d. **증감표**: 학생이 그린 증감표 이미지와 실제 도함수의 부호 변화를 비교하여 오류 발견 시, "증감표에서 $f'(x)$의 부호가 변하는 지점과 그래프의 극점이 일치하는지 다시 확인해 보세요."와 같이 질문.
    3. **칭찬 및 보강**: 그래프 개형이 거의 완벽하다면 "논리적으로 완벽해요!👏"라고 칭찬하고, 그 그래프의 특징(예: 대칭성, 변곡점의 의미)을 추가로 설명해 주세요.
    """
    
    # -----------------------------------------------------
    # 6. AI 호출 및 결과 출력 (비전 API)
    # -----------------------------------------------------
    
    # 이미지 데이터를 BytesIO 객체로 변환 (Gemini API에 맞게 처리)
    def np_to_bytes(img_array):
        img = Image.fromarray(img_array.astype('uint8'))
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        return byte_io.getvalue()

    # 이미지 데이터 준비 (R, G, B 채널만 사용)
    graph_image_bytes = np_to_bytes(graph_data.image_data[:, :, 0:3])
    sign_chart_image_bytes = np_to_bytes(sign_chart_data.image_data[:, :, 0:3])

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
            st.error(f"API 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. (오류: {e})")