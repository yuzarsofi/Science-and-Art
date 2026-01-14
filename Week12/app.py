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
    # T5 is small enough to run on the free server!
    return pipeline("text2text-generation", model="google/flan-t5-base")

# --- 3. FILTER FUNCTION ---
def apply_filter(image, filter_name):
    if filter_name == "Grayscale":
        return ImageOps.grayscale(image)
    elif filter_name == "Blur":
        return image.filter(ImageFilter.GaussianBlur(5))
    elif filter_name == "Edge Detection":
        return image.convert("L").filter(ImageFilter.FIND_EDGES)
    return image

# --- 4. CHAT MODE (Works Everywhere) ---
if mode == "💬 Chat Mode":
    st.title("💬 AI Chatbot")
    st.write("This model runs on the CPU and works on the free deployment!")
    
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

# --- 5. ART MODE (Restricted on Free Server) ---
elif mode == "🎨 Art Mode":
    st.title("🎨 AI Art Generator")
    
    # CHECK FOR GPU
    if not torch.cuda.is_available():
        st.warning("⚠️ **Hardware Limitation Detected**")
        st.error("This free server uses a CPU with limited RAM. The Art Model (Stable Diffusion) requires a GPU to run.")
        st.info("💡 **For the Assignment:** Please check my GitHub repository logs or the screenshots in the README to see the Art Mode working in Google Colab!")
        
        # Show a placeholder image so the page isn't empty
        st.image("https://placehold.co/600x400?text=Art+Mode+Requires+GPU", caption="Placeholder Image")
        
    else:
        # This part ONLY runs if you are on Google Colab (GPU)
        prompt = st.text_input("Describe your image:", "A futuristic city")
        
        if st.button("Generate Art"):
            with st.spinner("Painting..."):
                try:
                    # Load model only if we have a GPU
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", 
                        torch_dtype=torch.float16
                    ).to("cuda")
                    
                    image = pipe(prompt).images[0]
                    st.image(image, caption=f"Generated: {prompt}")
                except Exception as e:
                    st.error(f"Error: {e}")