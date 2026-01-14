import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter, ImageOps
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Generative AI Studio", layout="wide")

st.sidebar.title("🤖 AI Studio")
st.sidebar.info("Week 12: Final Submission")
selected_mode = st.sidebar.radio(
    "Select Mode:",
    ("🏠 Home", "💬 Chat Mode", "🎨 Art Mode", "🖼️ Image Filter Mode")
)

# --- 1. HOME MODE ---
if selected_mode == "🏠 Home":
    st.title("✨ Generative AI Studio")
    st.markdown("""
    Welcome to the Final Project! This app demonstrates three AI capabilities:
    
    * **💬 Chat Mode:** Talk to a language model (Flan-T5).
    * **🎨 Art Mode:** Generate images from text (Stable Diffusion).
    * **🖼️ Image Filters:** Apply classic image processing filters.
    """)
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)

# --- 2. CHAT MODE ---
elif selected_mode == "💬 Chat Mode":
    st.title("💬 Chatbot")
    st.write("Powered by `google/flan-t5-base`")

    # Cache the model so it loads fast
    @st.cache_resource
    def load_chat_model():
        return pipeline("text2text-generation", model="google/flan-t5-base")

    user_input = st.text_input("Your Question:", "")
    
    if st.button("Send"):
        if user_input:
            with st.spinner("Thinking..."):
                chatbot = load_chat_model()
                # T5 runs fine on CPU, no special logic needed here
                response = chatbot(user_input, max_length=500, do_sample=True)
                st.success(response[0]['generated_text'])

# --- 3. ART MODE (THE FIX IS HERE) ---
elif selected_mode == "🎨 Art Mode":
    st.title("🎨 Art Generator")
    st.write("Powered by `stable-diffusion-v1-5`")
    
    # Check hardware availability immediately
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        st.warning("⚠️ Running on **CPU**. Image generation will be slow (2-5 mins).")
    else:
        st.success("🚀 Running on **GPU**. Generation will be fast!")

    @st.cache_resource
    def load_art_model():
        # FRIEND'S LOGIC: Detect Device First
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=dtype
        )
        pipe.to(device)
        return pipe

    image_prompt = st.text_input("Enter your image prompt:", "A futuristic city")

    if st.button("Generate Image"):
        with st.spinner("Generating... (Please be patient)"):
            try:
                art_pipeline = load_art_model()
                image = art_pipeline(image_prompt).images[0]
                st.image(image, caption=image_prompt)
            except Exception as e:
                st.error(f"Error: {e}")

# --- 4. IMAGE FILTER MODE ---
elif selected_mode == "🖼️ Image Filter Mode":
    st.title("🖼️ Image Filters")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
    
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original", width=300)
        
        filter_choice = st.selectbox("Choose a Filter:", ["None", "Grayscale", "Blur", "Edge Detection"])
        
        if filter_choice == "Grayscale":
            st.image(original_image.convert("L"), width=300)
        elif filter_choice == "Blur":
            st.image(original_image.filter(ImageFilter.GaussianBlur(5)), width=300)
        elif filter_choice == "Edge Detection":
            st.image(original_image.convert("L").filter(ImageFilter.FIND_EDGES), width=300)