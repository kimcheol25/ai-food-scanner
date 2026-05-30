import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI 식재료 마스터", page_icon="🥦", layout="centered")
st.markdown("""
    <style>
    .safe {background-color: #d4edda; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .caution {background-color: #fff3cd; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .danger {background-color: #f8d7da; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

food_db = {
    "A등급 (신선함)": {"condition": "본연의 색과 단단함을 유지함.", "guide": "알맞은 방법(냉장/냉동)으로 보관하여 신선도를 유지하세요."},
    "B등급 (주의/데드라인)": {"condition": "색이 변하거나 겉이 마르기 시작함.", "guide": "⚠️ 데드라인: 가급적 오늘 내로 가열하는 요리(볶음, 찌개 등)에 활용해 소비하세요."},
    "C등급 (위험/부패)": {"condition": "점액질, 곰팡이, 악취가 발생함.", "guide": "🚨 식중독 위험: 절대 섭취 금지. 즉시 음식물 쓰레기로 폐기하세요."}
}

st.title("📸 AI 식재료 자동 분석기")
st.write("구글의 AI가 식재료를 파악하고, 최적의 행동 가이드를 제시합니다.")

img_file = st.camera_input("식재료 촬영하기")

if img_file is not None:
    image = Image.open(img_file).convert('RGB')
    
    with st.spinner("수백만 장의 데이터를 바탕으로 분석 중입니다..."):
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        
        preds = model.predict(img_preprocessed)
        results = decode_predictions(preds, top=3)[0]
        
    st.success("✅ 분석 완료!")
    st.markdown(f"### 🤖 AI 인식 결과: **{results[0][1]}** (확률: {int(results[0][2]*100)}%)")
    
    st.markdown("---")
    st.subheader("💡 현재 식재료 상태를 선택해 행동 가이드를 확인하세요!")
    
    st.markdown("<div class='safe'>", unsafe_allow_html=True)
    st.write("**🟢 " + food_db["A등급 (신선함)"]["condition"] + "**")
    st.write("👉 " + food_db["A등급 (신선함)"]["guide"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='caution'>", unsafe_allow_html=True)
    st.write("**🟡 " + food_db["B등급 (주의/데드라인)"]["condition"] + "**")
    st.write("👉 " + food_db["B등급 (주의/데드라인)"]["guide"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='danger'>", unsafe_allow_html=True)
    st.write("**🔴 " + food_db["C등급 (위험/부패)"]["condition"] + "**")
    st.write("👉 " + food_db["C등급 (위험/부패)"]["guide"])
    st.markdown("</div>", unsafe_allow_html=True)
