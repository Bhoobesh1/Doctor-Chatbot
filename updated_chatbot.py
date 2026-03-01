import gradio as gr
import asyncio
import sys
import os
import requests
from openai import OpenAI

# ---------------- CONFIG ----------------
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

# ---------------- RAG BACKEND CALL ----------------
def ask_rag_backend(question):
    try:
        response = requests.post(
            f"{RAG_API_URL}/ask",
            json={"question": question},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("answer", "No response received.")
        else:
            return "Error from RAG backend."
    except Exception:
        return "Unable to connect to backend."

# ---------------- CORE VOICE + IMAGE LOGIC ----------------
def process_inputs(audio, image):
    global last_medical_issue

    if not audio:
        return "", "Please speak your health concern.", None

    patient_text = transcribe_with_openai(audio)

    if not patient_text:
        return "", "Unable to understand audio.", None

    text_lower = patient_text.lower()

    if any(x in text_lower for x in ["who are you", "introduce yourself", "your name"]):
        doctor_response = (
            "I am an AI doctor created to help you understand health concerns "
            "for learning and awareness purposes."
        )
    elif any(x in text_lower for x in ["thank you", "thanks"]):
        doctor_response = "You are welcome. Take care and feel free to ask anytime."
    else:
        if image and os.path.exists(image):
            image_prompt = SYSTEM_PROMPT + " " + patient_text
            image_analysis = analyze_image_with_query(image_prompt, encode_image(image))
            combined_query = f"{patient_text}\nImage Findings: {image_analysis}"
            doctor_response = ask_rag_backend(combined_query)
        else:
            doctor_response = ask_rag_backend(patient_text)

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
    answer = ask_rag_backend(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


def open_chat():
    return gr.update(visible=True)

def close_chat():
    return gr.update(visible=False)


# ---------------- CUSTOM CSS ----------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:      #0b1e3d;
    --navy-mid:  #122650;
    --blue:      #1a5cff;
    --blue-lt:   #3d7bff;
    --teal:      #00c9b1;
    --teal-glow: rgba(0,201,177,0.18);
    --white:     #f4f8ff;
    --muted:     #8ba3c9;
    --card:      rgba(255,255,255,0.04);
    --border:    rgba(255,255,255,0.08);
    --radius:    14px;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--white) !important;
}

.gradio-container {
    max-width: 1140px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── Header ── */
#main-header {
    background: linear-gradient(135deg, #0b1e3d 0%, #122650 50%, #0d2a60 100%);
    border-bottom: 1px solid var(--border);
    padding: 36px 48px 28px;
    position: relative;
    overflow: hidden;
}
#main-header::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,201,177,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
#main-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.3rem !important;
    letter-spacing: -0.5px !important;
    margin: 0 !important;
    background: linear-gradient(90deg, #ffffff, var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#main-header p {
    font-size: 0.92rem;
    color: var(--muted);
    margin: 6px 0 0 !important;
    letter-spacing: 0.3px;
}

/* ── Disclaimer banner ── */
#disclaimer {
    background: rgba(0,201,177,0.07);
    border: 1px solid rgba(0,201,177,0.25);
    border-radius: var(--radius);
    padding: 10px 18px;
    margin: 18px 24px 0;
    font-size: 0.82rem;
    color: #7fd8ce;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Section labels ── */
.section-title {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--teal) !important;
    margin-bottom: 14px !important;
}

/* ── Cards / panels ── */
.panel-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    backdrop-filter: blur(6px);
}

/* ── All Gradio inputs & textareas ── */
input[type=text], textarea,
.gr-textbox textarea, .gr-textbox input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--white) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s;
}
input[type=text]:focus, textarea:focus {
    border-color: var(--teal) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,201,177,0.12) !important;
}

/* ── Labels ── */
label, .gr-form > label, .gr-block label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.4px !important;
    color: var(--muted) !important;
    margin-bottom: 6px !important;
}

/* ── Primary button ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, var(--blue), var(--blue-lt)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 13px 28px !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 18px rgba(26,92,255,0.35) !important;
    width: 100% !important;
}
.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(26,92,255,0.45) !important;
}
.gr-button-primary:active, button.primary:active {
    transform: translateY(0) !important;
}

/* ── Secondary / close button ── */
button.secondary {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
button.secondary:hover {
    background: rgba(255,255,255,0.1) !important;
    color: var(--white) !important;
}

/* ── Audio widget ── */
.gr-audio {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.03) !important;
    overflow: hidden !important;
}

/* ── Image upload ── */
.gr-image {
    border-radius: var(--radius) !important;
    border: 1px dashed rgba(0,201,177,0.3) !important;
    background: rgba(0,201,177,0.03) !important;
    transition: border-color 0.2s !important;
}
.gr-image:hover {
    border-color: var(--teal) !important;
}

/* ── Chatbot ── */
.gr-chatbot {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.message.user {
    background: linear-gradient(135deg, var(--blue), var(--blue-lt)) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #fff !important;
    font-size: 0.9rem !important;
    box-shadow: 0 2px 12px rgba(26,92,255,0.2) !important;
}
.message.bot {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: var(--white) !important;
    font-size: 0.9rem !important;
}

/* ── FAB chat button ── */
#chatbot-fab {
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    background: linear-gradient(135deg, var(--teal), #00a896) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 14px 22px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 6px 24px rgba(0,201,177,0.4) !important;
    cursor: pointer !important;
    z-index: 1000 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    width: auto !important;
}
#chatbot-fab:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(0,201,177,0.5) !important;
}

/* ── Chat panel ── */
#chat-panel {
    background: rgba(11,30,61,0.97) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(20px) !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 24px 0;
}

/* ── Status badge ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px var(--teal);
    margin-right: 6px;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
"""


# ---------------- UI ----------------
with gr.Blocks(title="AI Doctor Assistant") as demo:

    # ── Header ──
    gr.HTML("""
<div id="main-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <div>
            <h1 style="
                font-family: 'DM Serif Display', serif;
                font-size: 2.3rem;
                letter-spacing: -0.5px;
                margin: 0;
                background: linear-gradient(90deg, #00c9b1, #3d7bff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">AI Doctor Assistant</h1>
            <p style="font-size:0.92rem; color:#8ba3c9; margin: 6px 0 0; letter-spacing:0.3px;">
                <span class="status-dot"></span>
                AI-Powered Medical Guidance &nbsp;·&nbsp; Voice &amp; Image Enabled
            </p>
        </div>
    </div>
</div>
    <div id="disclaimer">
        ⚠ &nbsp;<strong>For educational purposes only.</strong>&nbsp; This tool does not replace professional medical advice, diagnosis, or treatment.
    </div>
    """)

    gr.HTML("<div style='height:24px'></div>")

    with gr.Row(equal_height=False):

        # ── LEFT — Patient Input ──
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<p class="section-title">Patient Input</p>')

            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Voice Recording"
            )

            gr.HTML("<div style='height:16px'></div>")

            image_input = gr.Image(
                type="filepath",
                label="Medical Image  (optional)"
            )

            gr.HTML("<div style='height:20px'></div>")

            submit = gr.Button(
                "⚕  Consult AI Doctor",
                variant="primary"
            )

            gr.HTML("""
            <p style="text-align:center; font-size:0.78rem; color:#4e6a92; margin-top:14px;">
                Speak clearly into your microphone.<br>
                Image upload is optional for visual analysis.
            </p>
            """)

        # ── RIGHT — AI Response ──
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<p class="section-title">AI Doctor Response</p>')

            patient_text = gr.Textbox(
                label="What You Said",
                lines=3,
                interactive=False,
                placeholder="Your transcribed speech will appear here…"
            )

            gr.HTML("<div style='height:14px'></div>")

            doctor_text = gr.Textbox(
                label="Medical Guidance",
                lines=6,
                interactive=False,
                placeholder="The AI doctor's response will appear here…"
            )

            gr.HTML("<div style='height:14px'></div>")

            doctor_audio = gr.Audio(
                label="Voice Response",
                type="filepath",
                autoplay=True
            )

    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[patient_text, doctor_text, doctor_audio]
    )

    # ── Live Chat Panel ──
    with gr.Column(visible=False, elem_id="chat-panel") as chat_panel:

        gr.HTML('<p class="section-title">💬 Live Chat with AI Doctor</p>')

        chatbot = gr.Chatbot(
            height=340,
            show_label=False,
        )

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Type your health question…",
                show_label=False,
                scale=5
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        gr.HTML("<div style='height:8px'></div>")
        close_btn = gr.Button("✕  Close Chat", variant="secondary")

    # FAB
    chat_icon = gr.Button("💬  Chat", elem_id="chatbot-fab")

    # Events
    chat_icon.click(open_chat, outputs=chat_panel)
    close_btn.click(close_chat, outputs=chat_panel)

    send_btn.click(
        fn=chat_backend,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input]
    )
    chat_input.submit(
        fn=chat_backend,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input]
    )


# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=["DM Sans", "sans-serif"],
        ),
    )