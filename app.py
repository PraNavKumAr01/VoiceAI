from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from AGENT import get_llm_response
from TTS import text_to_speech
from STT import audio_to_text

app = FastAPI()

origins = [
    "http://localhost:3000/session",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextBody(BaseModel):
    text: str

class AudioBody(BaseModel):
    audio: bytes

@app.get("/start-session/")
async def start_session():
    return {"message": "Server session started"}

@app.post("/text-to-llm/")
async def text_to_llm(payload: TextBody):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text input is empty")

    llm_response = get_llm_response(payload.text)
    return {"llm_response": llm_response}

@app.post("/text-to-ai-voice/")
async def text_to_ai_voice(payload: TextBody):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text input is empty")

    llm_response = get_llm_response(payload.text)
    ai_audio_bytes = text_to_speech(llm_response)

    return Response(content=ai_audio_bytes, media_type="audio/mpeg")

@app.post("/audio-to-ai-voice/")
async def audio_to_ai_voice(payload: AudioBody):
    if not payload.audio:
        raise HTTPException(status_code=400, detail="Audio input is empty")

    transcript = audio_to_text(payload.audio)
    llm_response = get_llm_response(transcript)
    ai_audio_bytes = text_to_speech(llm_response)

    return {"audio_data": ai_audio_bytes}
