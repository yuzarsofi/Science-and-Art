import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import requests
from PIL import Image, ImageFilter, ImageEnhance
import io
from diffusers import StableDiffusionPipeline
import os # Import os for environment variables

st.set_page_config(
    page_title="Generative AI Studio",
    page_icon="✨",
    layout="wide" 
)

st.sidebar.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50) 
st.sidebar.title("Navigation")

selected_mode = st.sidebar.radio(
    "Select Mode:",
    ("🏠 Home", "💬 Chat Mode", "🎨 Art Mode", "🖼️ Image Filter Mode")
)

st.sidebar.markdown("---")
st.sidebar.info("This project was developed by Caner Sivri.")

if selected_mode == "🏠 Home":
    st.title("✨ Generative AI Studio")
    st.markdown("""
    ### Welcome to the World of AI!
    This application is designed for you to experience modern Generative AI models.
    You can start by selecting one of the modes below.
    """)
    
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("💬 Chatbot")
        st.markdown("Chat with the Google Flan-T5 model. Ask questions, generate text.")
        st.info("Select 'Chat Mode' from the sidebar.")

    with col2:
        st.header("🎨 Image Generation")
        st.markdown("Create images from your text prompts using Stable Diffusion model.")
        st.success("Select 'Art Mode' from the sidebar.")

    with col3:
        st.header("🖼️ Filters")
        st.markdown("Upload your images and apply various image processing filters.")
        st.warning("Select 'Image Filter Mode' from the sidebar.")

    st.divider()

    st.caption("Powered by Streamlit, Hugging Face & PyTorch")

# --- Chat Mode Logic ---
elif selected_mode == "💬 Chat Mode":
    st.title("Chat Mode")
    st.write("Talk with a chatbot")

    @st.cache_resource
    def load_chat_model():
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        return tokenizer, model

    chat_tokenizer, chat_model = load_chat_model()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.text_input("You:", value=message, key=f"user_hist_{i}", disabled=True, label_visibility="collapsed")
        else:
            st.text_input("Bot:", value=message, key=f"bot_hist_{i}", disabled=True, label_visibility="collapsed")

    with st.form(key='chat_form', clear_on_submit=True):
        user_chat_input = st.text_input("Type your message here for chat:", key="user_chat_message_form_input")
        submitted = st.form_submit_button("Send")

        if submitted and user_chat_input:
            # Append user input to text history for display
            st.session_state.chat_history.append(user_chat_input)

            # Construct the full conversation context for Flan-T5
            formatted_conversation = ""
            # Iterate up to the current user input (last item in chat_history is the current user_chat_input)
            for i, msg in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    formatted_conversation += f"User: {msg}\n"
                else:
                    formatted_conversation += f"Bot: {msg}\n"

            # For Flan-T5, we create a single prompt from the entire conversation history
            # The last line should imply the model needs to generate the bot's response
            prompt_text = formatted_conversation.strip() + "\nBot:"

            # Encode the prompt for the model
            input_ids = chat_tokenizer(prompt_text, return_tensors="pt").input_ids

            # Generate a response with sampling parameters
            with torch.no_grad():
                # For Seq2Seq models, `generate` directly produces the target sequence.
                output_ids = chat_model.generate(
                    input_ids,
                    max_length=500,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    num_return_sequences=1
                )

            # Decode the bot's response. For Seq2Seq models, no slicing is typically needed.
            bot_response = chat_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.session_state.chat_history.append(bot_response)

            st.rerun()

# --- Art Mode Logic ---
elif selected_mode == "🎨 Art Mode":
    st.title("Art Mode")
    st.write("Generate images from text prompts using a locally loaded model.")
    st.warning("Generating image may take a while, please wait.")

    @st.cache_resource
    def load_art_model():

        hf_token = os.getenv('HF_TOKEN')

        if hf_token is None:
            st.warning("Hugging Face token not found in environment variables. Attempting to load model without explicit token, but it might fail for gated models like 'runwayml/stable-diffusion-v1-5'. Please ensure 'HF_TOKEN' is set in the environment where Streamlit is launched.")
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        else:
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=hf_token)

        # Check for CUDA availability and move model accordingly
        if torch.cuda.is_available():
            pipe.to("cuda") # Move model to GPU if available
        else:
            pipe.to("cpu") # Fallback to CPU if no CUDA device is found
        return pipe

    art_pipeline = load_art_model()

    image_prompt = st.text_input("Enter your image prompt here:", key="image_prompt")

    if st.button("Generate Image"):
        if not image_prompt:
            st.error("Please enter a prompt to generate an image.")
        else:
            with st.spinner("Generating image..."):
                try:
                    # Generate image using the local pipeline
                    image = art_pipeline(image_prompt).images[0]
                    st.image(image, caption=image_prompt)
                except Exception as e:
                    st.error(f"Error generating image: {e}")

elif selected_mode == "🖼️ Image Filter Mode":
    st.title("Image Filter Mode")
    st.write("Upload an image and apply various filters.")

    # Image filter functions
    def apply_grayscale(image):
        return image.convert("L")

    def apply_blur(image, radius=2):
        return image.filter(ImageFilter.GaussianBlur(radius))

    def apply_color_shift(image, r_factor=1.0, g_factor=1.0, b_factor=1.0):
        # Split into R, G, B bands
        r, g, b = image.split()
        # Apply enhancement factors
        r = ImageEnhance.Brightness(r).enhance(r_factor)
        g = ImageEnhance.Brightness(g).enhance(g_factor)
        b = ImageEnhance.Brightness(b).enhance(b_factor)
        # Merge back
        return Image.merge('RGB', (r, g, b))

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

        st.subheader("Apply Filters")
        selected_filter = st.selectbox(
            "Select a filter",
            ("None", "Grayscale", "Blur", "Color Shift")
        )

        filtered_image = None

        if selected_filter == "Grayscale":
            filtered_image = apply_grayscale(original_image)
        elif selected_filter == "Blur":
            blur_radius = st.slider("Blur Radius", 0.0, 10.0, 2.0, 0.1)
            filtered_image = apply_blur(original_image, blur_radius)
        elif selected_filter == "Color Shift":
            r_factor = st.slider("Red Channel Factor", 0.0, 2.0, 1.0, 0.05)
            g_factor = st.slider("Green Channel Factor", 0.0, 2.0, 1.0, 0.05)
            b_factor = st.slider("Blue Channel Factor", 0.0, 2.0, 1.0, 0.05)
            filtered_image = apply_color_shift(original_image, r_factor, g_factor, b_factor)

        if filtered_image:
            st.subheader(f"Filtered Image ({selected_filter})")
            st.image(filtered_image, use_container_width=True)
        elif selected_filter != "None":
            st.info("Select a filter to apply.")
