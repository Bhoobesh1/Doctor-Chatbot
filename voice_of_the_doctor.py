from openai import OpenAI
import wave
import os

client = OpenAI()

def text_to_speech_openai(text):
    output_file = "doctor_voice.wav"

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(output_file)

    # VALIDATE FILE
    if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
        return output_file
    return None
