from fastapi import FastAPI, HTTPException
from AGENT import get_llm_response
from TTS import text_to_speech
from STT import audio_to_text

app = FastAPI()

@app.get("/start-session/")
async def start_session():
    return {"message": "Server session started"}

@app.post("/text-to-llm/")
async def text_to_llm(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text input is empty")

    llm_response = get_llm_response(text)
    return {"llm_response": llm_response}

@app.post("/text-to-ai-voice/")
async def text_to_ai_voice(text: str):

    llm_response = get_llm_response(text)
    ai_audio_bytes = text_to_speech(llm_response)

    return {"audio_data": ai_audio_bytes}

@app.post("/audio-to-ai-voice/")
async def audio_to_ai_voice(audio_data: bytes):

    transcript = audio_to_text(audio_data)
    llm_response = get_llm_response(transcript)
    ai_audio_bytes = text_to_speech(llm_response)

    return {"audio_data": ai_audio_bytes}
