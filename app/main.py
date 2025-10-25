# app/main.py
import io
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator
from functools import lru_cache
import uvicorn
import traceback
import torch

app = FastAPI()

# ---------------------- Middleware ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Paths ----------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------------------- Models ----------------------
@lru_cache()
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@lru_cache()
def load_style_model():
    # Small and fast paraphrasing model
    return pipeline("text2text-generation", model="google/flan-t5-base")

processor, model = load_caption_model()
style_rewriter = load_style_model()

LANG_CODES = {
    "english": "en",
    "tamil": "ta",
    "hindi": "hi",
    "french": "fr",
    "german": "de"
}


# ---------------------- Caption Generation ----------------------
@app.post("/caption")
async def caption_image(
    file: UploadFile = File(...),
    language: str = Form("english"),
    style: str = Form("Normal")
):
    try:
        language_key = language.lower()
        if language_key not in LANG_CODES:
            return JSONResponse({"error": "Unsupported language"}, status_code=400)
        lang_code = LANG_CODES[language_key]

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # BLIP Caption Generation
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=3,
            max_length=50,
            output_scores=True,
            return_dict_in_generate=True
        )

        sequences = outputs.sequences
        scores = outputs.sequences_scores
        captions_en = [processor.decode(seq, skip_special_tokens=True) for seq in sequences]
        confidences = torch.nn.functional.softmax(scores, dim=0) * 100
        confidences = confidences.tolist()

        caption_conf_list = [
            {"caption": cap, "confidence": round(conf, 2)}
            for cap, conf in zip(captions_en, confidences)
        ]

        # Best factual caption
        best_caption_en = caption_conf_list[0]["caption"]

        # Apply style transformation
        # ---------------------- Apply Style Transformation ----------------------
        if style.lower() != "normal":
            style_prompts = {
                "cheerful": "Make this caption sound cheerful, positive, and full of joy:",
                "sarcastic": "Rewrite this caption with a humorous and sarcastic twist:",
                "genz": "Rewrite this caption like a Gen Z Instagram post using emojis and slang:",
                "instagram": "Make this caption aesthetic and suitable for Instagram:",
                "romantic": "Rewrite this caption with a romantic and dreamy tone:",
                "motivational": "Make this caption sound inspiring and motivational:",
            }

            style_prompt = style_prompts.get(style.lower(), f"Rewrite this caption in a {style.lower()} tone:")

            prompt = f"{style_prompt} {best_caption_en}"

            styled_text = style_rewriter(
                prompt,
                max_length=60,
                num_return_sequences=1,
                temperature=0.9,   # adds slight creative randomness
                top_p=0.95,
            )

            best_caption_styled = styled_text[0]["generated_text"].strip()
        else:
            best_caption_styled = best_caption_en


        # Translate if needed
        if lang_code != "en":
            try:
                best_caption_styled = GoogleTranslator(source="auto", target=lang_code).translate(best_caption_styled)
                caption_conf_list = [
                    {
                        "caption": GoogleTranslator(source="auto", target=lang_code).translate(c["caption"]),
                        "confidence": c["confidence"]
                    } for c in caption_conf_list
                ]
            except Exception:
                pass

        # Generate TTS
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        tts = gTTS(text=best_caption_styled, lang=lang_code)
        tts.save(audio_path)

        return JSONResponse({
            "top_captions": caption_conf_list,
            "best_caption": best_caption_styled,
            "audio_filename": audio_filename,
            "style": style
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)

# ---------------------- Caption Audio Only ----------------------
@app.post("/caption_audio")
async def generate_caption_audio(caption: str = Form(...), language: str = Form("english")):
    try:
        lang_code = "en" if language.lower() == "english" else "ta"
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        tts = gTTS(text=caption, lang=lang_code)
        tts.save(audio_path)
        return JSONResponse({"audio_filename": audio_filename, "caption": caption})
    except Exception as e:
        return JSONResponse({"error": f"Failed to generate audio: {e}"}, status_code=500)

# ---------------------- Serve Audio ----------------------
@app.get("/audio")
async def get_audio(filename: str):
    audio_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg", filename=filename)
    return JSONResponse({"error": "file not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
