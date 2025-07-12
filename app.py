import streamlit as st
import os
import torch
from diffusers import StableDiffusionPipeline

@st.cache_resource
def load_model():
    model_path = "model"  # local path after download
    if not os.path.exists(model_path):
        st.write("📥 Downloading model from Google Drive...")
        os.system("gdown https://drive.google.com/uc?id=1ErCyGDdmZl8056BiBsfWbDj02zA_sgC- --output model.safetensors")
        # Convert or move as needed
    pipe = StableDiffusionPipeline.from_single_file(
        "model.safetensors",
        torch_dtype=torch.float16,
        revision="fp16"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

st.title("🎨 Neura AI Image Generator")

prompt = st.text_input("Enter your image prompt")

if st.button("Generate"):
    with st.spinner("Generating..."):
        pipe = load_model()
        image = pipe(prompt).images[0]
        st.image(image, caption=prompt)
