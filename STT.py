import os
from dotenv import load_dotenv
import logging, verboselogs
import json
from datetime import datetime
import httpx
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions,
    FileSource,
)
from dotenv import load_dotenv

load_dotenv()

def audio_to_text(buffer_data):

    os.environ["DEEPGRAM_API_KEY"] = os.environ.get('DEEPGRAM_API_KEY')

    try:
        
        config: DeepgramClientOptions = DeepgramClientOptions()
        deepgram: DeepgramClient = DeepgramClient("", config)

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model="nova",
            smart_format=True,
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )

        response = response.to_json(indent=4)
        data = json.loads(response)
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']

        return transcript

    except Exception as e:
        print(f"Exception: {e}")
        return None