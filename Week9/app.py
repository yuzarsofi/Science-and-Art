import streamlit as st
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="My AI Studio", layout="wide")

# Sidebar Menu
st.sidebar.title("🤖 AI Studio")
mode = st.sidebar.radio("Choose Mode:", ["💬 Chat Mode", "🎨 Art Mode"])

st.sidebar.markdown("---")
st.sidebar.write("Running on Google Colab GPU")

# --- 2. LOAD MODELS (Cached for Speed) ---

@st.cache_resource
def load_chat_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_art_model():
    # Load Stable Diffusion in float16 (Fast & Low Memory)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda") # Send to GPU
    return pipe

# --- 3. CHAT MODE ---
if mode == "💬 Chat Mode":
    st.title("💬 AI Chatbot")
    st.write("Ask me anything! (Powered by Flan-T5)")
    
    # Chat Input
    user_input = st.text_input("Your Question:", "")
    
    if st.button("Send"):
        if not user_input:
            st.warning("Please type something!")
        else:
            with st.spinner("Thinking..."):
                chatbot = load_chat_model()
                response = chatbot(
                    user_input, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.7
                )
                st.success(response[0]['generated_text'])

# --- 4. ART MODE ---
elif mode == "🎨 Art Mode":
    st.title("🎨 AI Art Generator")
    st.write("Describe an image and I will paint it. (Powered by Stable Diffusion)")
    
    # Art Input
    prompt = st.text_input("Describe your image:", "A futuristic city in watercolor style")
    
    if st.button("Generate Art"):
        with st.spinner("Painting... (This takes about 10 seconds)"):
            try:
                # Load art model only when needed
                art_pipe = load_art_model()
                
                # Generate
                image = art_pipe(prompt).images[0]
                
                # Display
                st.image(image, caption=f"Generated: {prompt}", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
