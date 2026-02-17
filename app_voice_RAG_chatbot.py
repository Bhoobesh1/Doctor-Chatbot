import gradio as gr
import asyncio
import sys
import os
import requests
from openai import OpenAI

RAG_API_URL = "http://127.0.0.1:5000"

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

# ---------------- AI MEDICAL VALIDATION ----------------
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
            {"role": "user", "content": text}
        ],
        max_tokens=3
    )
    return response.choices[0].message.content.strip().upper() == "YES"

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

# ---------------- CORE VOICE + IMAGE LOGIC ----------------
def process_inputs(audio, image):
    global last_medical_issue

    if not audio:
        return "", "Please speak your health concern.", None

    patient_text = transcribe_with_openai(audio)
    text_lower = patient_text.lower()

    if any(x in text_lower for x in ["who are you", "introduce yourself", "your name"]):
        doctor_response = (
            "I am an AI doctor created to help you understand health concerns "
            "for learning and awareness purposes."
        )

    elif any(x in text_lower for x in ["thank you", "thanks"]):
        doctor_response = "You are welcome take care and feel free to ask anytime."

    elif image and os.path.exists(image):
        last_medical_issue = patient_text if patient_text.strip() else "Please analyze the medical image."

        if not is_medical_query_ai(patient_text):
            image_prompt = SYSTEM_PROMPT + " Analyze the medical image and explain any visible health related concerns."
        else:
            image_prompt = SYSTEM_PROMPT + " " + patient_text

        doctor_response = analyze_image_with_query(
            image_prompt,
            encode_image(image)
        )

    else:
        if not is_medical_query_ai(patient_text):
            doctor_response = "Please ask questions related to health or medical concerns only."
        else:
            last_medical_issue = patient_text
            doctor_response = analyze_text_only(patient_text)

    audio_output = text_to_speech_openai(doctor_response)
    if not audio_output or not os.path.exists(audio_output):
        audio_output = None

    return patient_text, doctor_response, audio_output

# ---------------- CHATBOT BACKEND ----------------
def chat_backend(message, history):
    if not message.strip():
        return history, ""

    if history is None:
        history = []

    try:
        response = requests.post(
            f"{RAG_API_URL}/ask",
            json={"question": message},
            timeout=30
        )

        if response.status_code == 200:
            answer = response.json().get("answer", "No response received.")
        else:
            answer = "Error from RAG backend."

    except Exception as e:
        answer = "Unable to connect to backend."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""


def open_chat():
    return gr.update(visible=True)

def close_chat():
    return gr.update(visible=False)

# ---------------- UI ----------------a
with gr.Blocks(title="AI Doctor Assistant") as demo:

    gr.Markdown("""
    # 🩺 AI Doctor Assistant  
    Speak your health concern and upload a medical image if available.
    """)

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your problem")
        image_input = gr.Image(type="filepath", label="Medical image (optional)")

    submit = gr.Button("Consult Doctor", variant="primary")

    with gr.Row():
        patient_text = gr.Textbox(label="Patient Input", lines=3, interactive=False)
        doctor_text = gr.Textbox(label="Doctor Response", lines=4, interactive=False)

    doctor_audio = gr.Audio(label="Doctor Voice", type="filepath", autoplay=True)

    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[patient_text, doctor_text, doctor_audio]
    )

    # -------- CHAT PANEL --------
    with gr.Column(visible=False) as chat_panel:
        gr.Markdown("### 💬 Chat with AI Doctor")
        chatbot = gr.Chatbot(height=350)
        chat_input = gr.Textbox(placeholder="Type your health question...")
        send_btn = gr.Button("Send")
        close_btn = gr.Button("Close Chat ❌")

    chat_icon = gr.Button("💬", elem_id="chatbot-fab")

    chat_icon.click(open_chat, outputs=chat_panel)
    close_btn.click(close_chat, outputs=chat_panel)

    send_btn.click(
        fn=chat_backend,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input]
    )

# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 900px; margin: auto; }
        #chatbot-fab {
            position: fixed;
            bottom: 25px;
            right: 25px;
            background-color: #4f46e5;
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 28px;
            cursor: pointer;
            border: none;
            z-index: 1000;
        }
        """
    )
