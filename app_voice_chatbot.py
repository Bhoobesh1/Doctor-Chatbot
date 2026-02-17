import gradio as gr
import asyncio
import sys
import os
from openai import OpenAI

# ---------------- WINDOWS FIX ----------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------- OPENAI CLIENT ----------------
client = OpenAI()

# ---------------- IMPORT YOUR EXISTING MODULES ----------------
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_openai
from voice_of_the_doctor import text_to_speech_openai

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are a highly experienced doctor with decades of clinical practice.
Respond only for learning and awareness purposes.
Suggest likely causes and simple home level remedies.
Speak calmly and naturally like a real doctor talking to a patient.
Keep the response short clear and reassuring.
Do not use numbers symbols or special characters.
Avoid medical jargon and do not exaggerate.
Start the response directly without greetings or introductions.
"""

# ---------------- MEMORY ----------------
last_medical_issue = ""

# ---------------- AI MEDICAL VALIDATION (NO KEYWORDS) ----------------
def is_medical_query_ai(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Reply with only YES or NO. "
                    "Answer YES only if the user's question is related to health, "
                    "medical conditions, symptoms, disease, diagnosis, treatment, or medicine. "
                    "Otherwise reply NO."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=3
    )

    answer = response.choices[0].message.content.strip().upper()
    return answer == "YES"

# ---------------- TEXT ONLY DOCTOR ----------------
def analyze_text_only(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# ---------------- CORE LOGIC ----------------
def process_inputs(audio, image):
    global last_medical_issue

    if not audio:
        return "", "Please speak your health concern.", None

    # Speech to text
    patient_text = transcribe_with_openai(audio)
    text_lower = patient_text.lower()

    # INTRO
    if any(x in text_lower for x in ["who are you", "introduce yourself", "your name"]):
        doctor_response = (
            "I am an AI doctor created to help you understand health concerns "
            "for learning and awareness purposes."
        )

    # THANKS
    elif any(x in text_lower for x in ["thank you", "thanks"]):
        doctor_response = "You are welcome take care and feel free to ask anytime."

    # IMAGE PRESENT
    elif image and os.path.exists(image):
        last_medical_issue = patient_text if patient_text.strip() else "Please analyze the medical image."

    # If user says vague words, auto-generate a proper medical prompt
        if not is_medical_query_ai(patient_text):
            image_prompt = SYSTEM_PROMPT + " Analyze the medical image and explain any visible health related concerns."
        else:
            image_prompt = SYSTEM_PROMPT + " " + patient_text

        doctor_response = analyze_image_with_query(
            image_prompt,
            encode_image(image)
    )


    # TEXT ONLY
    else:
        if not is_medical_query_ai(patient_text):
            doctor_response = "Please ask questions related to health or medical concerns only."
        else:
            last_medical_issue = patient_text
            doctor_response = analyze_text_only(patient_text)

    # Text to speech
    audio_output = text_to_speech_openai(doctor_response)

    if not audio_output or not os.path.exists(audio_output):
        audio_output = None

    return patient_text, doctor_response, audio_output

# ---------------- GRADIO UI ----------------
with gr.Blocks(
    title="AI Doctor Assistant",
    theme=gr.themes.Soft(),
    css=".container { max-width: 900px; margin: auto; }"
) as demo:

    gr.Markdown("""
    # 🩺 AI Doctor Assistant  
    Speak your health concern and upload a medical image if available.
    """)

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your problem"
        )

        image_input = gr.Image(
            type="filepath",
            label="Medical image (optional)"
        )

    submit = gr.Button("Consult Doctor", variant="primary")

    with gr.Row():
        patient_text = gr.Textbox(
            label="Patient Input",
            lines=3,
            interactive=False
        )

        doctor_text = gr.Textbox(
            label="Doctor Response",
            lines=4,
            interactive=False
        )

    doctor_audio = gr.Audio(
        label="Doctor Voice",
        type="filepath",
        autoplay=True
    )

    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[patient_text, doctor_text, doctor_audio]
    )

# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        show_error=True
    )
