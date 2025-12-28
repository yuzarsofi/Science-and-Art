import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter, ImageOps

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="My AI Studio (Week 11)", layout="wide")

st.sidebar.title("🤖 AI Studio")
mode = st.sidebar.radio("Choose Mode:", ["💬 Chat Mode", "🎨 Art Mode"])
st.sidebar.markdown("---")
st.sidebar.write("Week 11: Now with Filters!")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_chat_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_art_model():
    # Load Stable Diffusion (GPU optimized)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    return pipe

# --- 3. FILTER FUNCTION (WEEK 11 TASK) ---
def apply_filter(image, filter_name):
    if filter_name == "Grayscale":
        return ImageOps.grayscale(image)
    elif filter_name == "Blur":
        return image.filter(ImageFilter.GaussianBlur(5))
    elif filter_name == "Edge Detection":
        # Convert to grayscale first for better edges, then find edges
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

# --- 5. ART MODE (WITH FILTERS) ---
elif mode == "🎨 Art Mode":
    st.title("🎨 AI Art Generator + Filters")
    
    # Initialize Session State to remember the image
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None

    prompt = st.text_input("Describe your image:", "A futuristic city in watercolor style")
    
    if st.button("Generate Art"):
        with st.spinner("Painting..."):
            try:
                art_pipe = load_art_model()
                image = art_pipe(prompt).images[0]
                # Save to session state so it doesn't vanish
                st.session_state.generated_image = image
            except Exception as e:
                st.error(f"Error: {e}")

    # If an image exists in memory, show filter options
    if st.session_state.generated_image is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(st.session_state.generated_image, use_container_width=True)
            
        with col2:
            st.subheader("Apply Filter (Week 11)")
            filter_choice = st.selectbox(
                "Choose a style:", 
                ["None", "Grayscale", "Blur", "Edge Detection", "Contour"]
            )
            
            if filter_choice != "None":
                filtered_image = apply_filter(st.session_state.generated_image, filter_choice)
                st.image(filtered_image, caption=f"Filter: {filter_choice}", use_container_width=True)
            else:
                st.info("Select a filter to see the magic!")
