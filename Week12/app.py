import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter, ImageOps

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="My AI Studio (Final)", layout="wide")

st.sidebar.title("🤖 AI Studio")
mode = st.sidebar.radio("Choose Mode:", ["💬 Chat Mode", "🎨 Art Mode"])
st.sidebar.markdown("---")
st.sidebar.write("Final Project Submission")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_chat_model():
    # T5 runs okay on CPU
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_art_model():
    # AUTOMATICALLY DETECT HARDWARE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Select precision (float16 for GPU, float32 for CPU to prevent errors)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    st.info(f"⏳ Loading Art Model on: {device.upper()}... (This may be slow)")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)
    
    # Enable memory saving tricks for CPU
    if device == "cpu":
        pipe.enable_attention_slicing() 
        
    return pipe

# --- 3. FILTER FUNCTION ---
def apply_filter(image, filter_name):
    if filter_name == "Grayscale":
        return ImageOps.grayscale(image)
    elif filter_name == "Blur":
        return image.filter(ImageFilter.GaussianBlur(5))
    elif filter_name == "Edge Detection":
        return image.convert("L").filter(ImageFilter.FIND_EDGES)
    elif filter_name == "Contour":
        return image.filter(ImageFilter.CONTOUR)
    return image

# --- 4. CHAT MODE ---
if mode == "💬 Chat Mode":
    st.title("💬 AI Chatbot")
    
    user_input = st.text_input("Your Question:", "")
    if st.button("Send"):
        if user_input:
            with st.spinner("Thinking..."):
                chatbot = load_chat_model()
                response = chatbot(
                    user_input, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.7
                )
                st.success(response[0]['generated_text'])

# --- 5. ART MODE ---
elif mode == "🎨 Art Mode":
    st.title("🎨 AI Art Generator")
    
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None

    prompt = st.text_input("Describe your image:", "A futuristic city")
    
    if st.button("Generate Art"):
        with st.spinner("Painting... (On CPU this might take 5-10 minutes!)"):
            try:
                art_pipe = load_art_model()
                image = art_pipe(prompt).images[0]
                st.session_state.generated_image = image
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.generated_image is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.generated_image, caption="Original", use_container_width=True)
        with col2:
            filter_choice = st.selectbox("Apply Filter:", ["None", "Grayscale", "Blur", "Edge Detection"])
            if filter_choice != "None":
                filtered_image = apply_filter(st.session_state.generated_image, filter_choice)
                st.image(filtered_image, caption=f"Filter: {filter_choice}", use_container_width=True)