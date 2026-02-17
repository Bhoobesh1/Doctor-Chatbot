from openai import OpenAI

client = OpenAI()

def transcribe_with_openai(audio_path):
    with open(audio_path, "rb") as audio:
        result = client.audio.transcriptions.create(
            file=audio,
            model="gpt-4o-mini-transcribe"
        )
    return result.text.strip()
