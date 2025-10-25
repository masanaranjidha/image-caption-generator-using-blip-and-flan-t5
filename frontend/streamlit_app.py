# frontend/streamlit_app.py
import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Image Caption + Audio", layout="centered")
st.header("🖼️ Image Captioning with Style and Audio")

# ------------------ Session State ------------------
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "top_captions" not in st.session_state:
    st.session_state.top_captions = []
if "language" not in st.session_state:
    st.session_state.language = "english"
if "style" not in st.session_state:
    st.session_state.style = "Normal"

# ------------------ Upload Image ------------------
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
language = st.selectbox(
    "Select language",
    ["english", "tamil", "hindi", "french", "german"],
    index=0
)
style = st.selectbox(
    "Select Caption Style",
    ["Normal", "Cheerful", "Sarcastic", "GenZ", "Instagram", "Romantic", "Motivational"],
    index=0
)

if uploaded:
    st.session_state.uploaded_image = uploaded
    st.session_state.language = language
    st.session_state.style = style
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

# ------------------ Generate Captions ------------------
if st.button("Generate Captions"):
    if st.session_state.uploaded_image is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Generating captions..."):
            files = {
                "file": (
                    st.session_state.uploaded_image.name,
                    st.session_state.uploaded_image.getvalue(),
                    st.session_state.uploaded_image.type
                )
            }
            data = {
                "language": st.session_state.language,
                "style": st.session_state.style
            }
            try:
                resp = requests.post("http://localhost:8000/caption", files=files, data=data, timeout=180)
            except Exception as e:
                st.error(f"Backend not reachable: {e}")
                st.stop()

            if resp.status_code == 200:
                js = resp.json()
                st.session_state.top_captions = js.get("top_captions", [])
                styled_caption = js.get("best_caption")
                audio_filename = js.get("audio_filename")

                st.markdown("### 🏆 Top-3 factual captions (confidence):")
                for i, item in enumerate(st.session_state.top_captions, start=1):
                    st.write(f"{i}. **{item['caption']}** — {item['confidence']}%")

                st.markdown("---")
                st.markdown(f"### 🎨 {st.session_state.style} Style Caption:")
                st.info(styled_caption)


                audio_resp = requests.get(f"http://localhost:8000/audio?filename={audio_filename}", timeout=60)
                if audio_resp.status_code == 200:
                    st.audio(audio_resp.content, format="audio/mp3")
                else:
                    st.error("Failed to load audio.")
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                st.error(f"Server error: {err}")
