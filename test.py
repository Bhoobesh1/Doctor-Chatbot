import gradio as gr
import asyncio
import sys
import os
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
RAG_API_URL         = "http://127.0.0.1:5000"
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY")

# Windows asyncio fix
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import warnings
warnings.filterwarnings("ignore", message=".*connection.*", category=ResourceWarning)
client = OpenAI()

# ─────────────────────────────────────────────
#  YOUR EXISTING MODULES
# ─────────────────────────────────────────────
from brain_of_the_doctor  import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_openai
from voice_of_the_doctor  import text_to_speech_openai

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  REJECTION PHRASES  (out-of-scope detection)
# ─────────────────────────────────────────────
REJECTION_PHRASES = [
    "can only help with", "only help with", "cannot help with",
    "can't help with", "not able to help", "outside my scope",
    "not a medical question", "only medical", "only answer medical",
    "please ask a medical", "only assist with", "only provide medical",
    "not related to", "unrelated to", "this assistant provides only",
    "provides only medical", "not my area", "beyond my expertise",
    "i am not able to assist", "i'm not able to assist",
    "i cannot assist", "i can't assist",
]

def is_rejection(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in REJECTION_PHRASES)

# ─────────────────────────────────────────────
#  CASUAL / NON-MEDICAL DETECTION
# ─────────────────────────────────────────────
CASUAL_PHRASES = [
    "hi", "hello", "hey", "hii", "helo", "sup", "good morning",
    "good evening", "good night", "good afternoon", "how are you",
    "what's up", "whats up", "bye", "goodbye", "ok", "okay",
    "thanks", "thank you", "cool", "nice", "great", "sure",
    "who are you", "introduce yourself", "your name",
]

def is_casual(text: str) -> bool:
    t = text.strip().lower()
    return any(t == p or t.startswith(p + " ") for p in CASUAL_PHRASES) or len(t) <= 4

# ─────────────────────────────────────────────
#  RAG BACKEND
# ─────────────────────────────────────────────
def ask_rag(question: str) -> str:
    try:
        r = requests.post(
            f"{RAG_API_URL}/ask",
            json={"question": question},
            timeout=30
        )
        if r.status_code == 200:
            return r.json().get("answer", "No response received.")
        return "Error from RAG backend."
    except Exception:
        return "Unable to connect to backend."

# ─────────────────────────────────────────────
#  LLM-BASED CATEGORY DETECTION
# ─────────────────────────────────────────────
CATEGORY_SEARCH_TERMS = {
    "dermatology":      "skin clinic dermatologist",
    "ophthalmology":    "eye hospital ophthalmologist",
    "orthopedic":       "orthopedic hospital bone specialist",
    "cardiology":       "heart hospital cardiologist",
    "neurology":        "neurology hospital neurologist",
    "gastroenterology": "gastroenterology hospital stomach specialist",
    "pulmonology":      "lung hospital pulmonologist",
    "ent":              "ENT hospital ear nose throat specialist",
    "urology":          "urology hospital urologist",
    "gynecology":       "gynecology hospital gynecologist",
    "psychiatry":       "psychiatry hospital mental health",
    "dentistry":        "dental clinic dentist",
    "general":          "hospital",
}

CATEGORY_DETECTION_PROMPT = """You are a medical triage assistant.
Given the patient's query and/or the doctor's response, identify the most relevant medical specialty from this list:

dermatology, ophthalmology, orthopedic, cardiology, neurology, gastroenterology,
pulmonology, ent, urology, gynecology, psychiatry, dentistry, general

Rules:
- Reply with ONLY the single category word (lowercase), nothing else.
- If the topic covers multiple specialties, pick the most relevant one.
- If unclear or non-medical, reply with: general

Examples:
Patient: "my eye is red and itchy" → ophthalmology
Patient: "I have chest pain and shortness of breath" → cardiology
Patient: "skin rash on my arm" → dermatology
Patient: "stomach ache and nausea" → gastroenterology
Patient: "ringing in my ears" → ent
Patient: "feeling depressed and anxious" → psychiatry
"""

def detect_category_llm(patient_query: str, doctor_response: str = "") -> str:
    try:
        combined = f"Patient query: {patient_query}"
        if doctor_response:
            combined += f"\nDoctor response summary: {doctor_response[:300]}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CATEGORY_DETECTION_PROMPT},
                {"role": "user",   "content": combined}
            ],
            max_tokens=10,
            temperature=0,
        )
        category = response.choices[0].message.content.strip().lower()
        if category not in CATEGORY_SEARCH_TERMS:
            return "general"
        return category
    except Exception as e:
        print(f"[Category LLM error] {e}")
        return detect_category_keyword_fallback(patient_query)

def detect_category_keyword_fallback(query: str) -> str:
    keywords = {
        "dermatology":      ["skin","rash","itch","acne","pimple","eczema","psoriasis","hives"],
        "ophthalmology":    ["eye","eyes","vision","blur","blurry","red eye","watering","itchy eyes","reddish"],
        "orthopedic":       ["bone","fracture","joint","knee","back pain","ortho","spine","shoulder"],
        "cardiology":       ["heart","chest pain","cardio","palpitation","blood pressure"],
        "neurology":        ["headache","migraine","dizziness","seizure","numbness","nerve","brain"],
        "gastroenterology": ["stomach","nausea","vomit","diarrhea","constipation","abdomen","bowel","gut"],
        "pulmonology":      ["lung","cough","breathe","breathing","asthma","chest","respiratory"],
        "ent":              ["ear","nose","throat","sinus","hearing","tonsil","ringing"],
        "urology":          ["urine","kidney","bladder","urination","prostate"],
        "gynecology":       ["period","menstrual","ovary","uterus","pregnancy","vaginal"],
        "psychiatry":       ["anxiety","depression","stress","mental","mood","panic","sleep"],
        "dentistry":        ["tooth","teeth","gum","dental","jaw","mouth"],
    }
    q = query.lower()
    for category, kws in keywords.items():
        if any(k in q for k in kws):
            return category
    return "general"

# ─────────────────────────────────────────────
#  GOOGLE MAPS  — embed + search link
# ─────────────────────────────────────────────
MAP_PLACEHOLDER = """
<div style="
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(0,201,177,0.2);
    border-radius: 12px;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4e6a92;
    font-size: 0.88rem;
">
    📍 Enable location and consult the doctor to see nearby hospitals on the map
</div>
"""

def generate_google_map(lat: float, lon: float, search_term: str) -> str:
    if GOOGLE_MAPS_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY":
        maps_url = (
            f"https://maps.google.com/maps"
            f"?q={search_term}&ll={lat},{lon}&z=14&output=embed"
        )
    else:
        maps_url = (
            f"https://www.google.com/maps/embed/v1/search"
            f"?key={GOOGLE_MAPS_API_KEY}"
            f"&q={requests.utils.quote(search_term)}"
            f"&center={lat},{lon}"
            f"&zoom=14"
        )

    return f"""
<div style="position:relative; width:100%; border-radius:12px; overflow:hidden;">

  <div style="
      position:absolute; top:12px; left:50%; transform:translateX(-50%);
      z-index:10; background:#0b2044; border:1.5px solid #00c9b1;
      border-radius:999px; padding:6px 16px;
      color:#00c9b1; font-size:0.82rem; font-weight:600;
      white-space:nowrap; box-shadow:0 2px 10px rgba(0,0,0,0.4);
  ">
      ● Your Location: {round(lat,4)}, {round(lon,4)}
  </div>

  <iframe
      width="100%" height="450"
      style="border:0; border-radius:12px; display:block;"
      loading="lazy" allowfullscreen
      referrerpolicy="no-referrer-when-downgrade"
      src="{maps_url}">
  </iframe>

</div>
"""
def google_maps_link(lat: float, lon: float, search_term: str) -> str:
    return (
        f"https://www.google.com/maps/search/"
        f"{requests.utils.quote(search_term)}/@{lat},{lon},14z"
    )

def build_map(query_text: str, doctor_response: str, lat, lon) -> str:
    if not lat or not lon:
        return MAP_PLACEHOLDER
    try:
        category    = detect_category_llm(query_text, doctor_response)
        search_term = CATEGORY_SEARCH_TERMS.get(category, "hospital")
        print(f"[Map] Detected category: {category} → searching: {search_term}")
        return generate_google_map(float(lat), float(lon), search_term)
    except Exception as e:
        print(f"[Map error] {e}")
        return MAP_PLACEHOLDER

def maps_link_html(lat, lon, query_text: str, doctor_response: str = "") -> str:
    if not lat or not lon:
        return ""
    try:
        category    = detect_category_llm(query_text, doctor_response)
        search_term = CATEGORY_SEARCH_TERMS.get(category, "hospital")
        link        = google_maps_link(float(lat), float(lon), search_term)
        return f'''
    <br><br>
    <a href="{link}" target="_blank" style="
    color:#00c9b1;
    font-weight:600;
    text-decoration:none;">
    Open in Google Maps
    </a>
    '''
    except Exception:
        return ""
    
def format_response_html(text):
    text = text.replace("**", "")  # remove markdown

    lines = text.split("\n")
    formatted = ""
    in_list = False

    for line in lines:
        line = line.strip()

        if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
            if not in_list:
                formatted += "<ol>"
                in_list = True
            formatted += f"<li>{line[2:].strip()}</li>"
        else:
            if in_list:
                formatted += "</ol>"
                in_list = False
            if line:
                formatted += f"<p>{line}</p>"

    if in_list:
        formatted += "</ol>"

    return formatted
# ─────────────────────────────────────────────
#  CORE VOICE + IMAGE HANDLER
# ─────────────────────────────────────────────
def process_inputs(audio, image, lat, lon):
    patient_text_val = ""
    doctor_response  = ""

    if audio and os.path.exists(audio):
        patient_text_val = transcribe_with_openai(audio)
        if not patient_text_val:
            return "", "Unable to understand audio.", None, MAP_PLACEHOLDER
    elif not image:
        return "", "Please provide a voice recording or an image.", None, MAP_PLACEHOLDER

    if image and os.path.exists(image) and not patient_text_val:
        patient_text_val = "Please analyze this medical image and provide guidance."

    text_lower = patient_text_val.lower()

    if any(x in text_lower for x in ["who are you", "introduce yourself", "your name"]):
        doctor_response = (
            "I am an AI doctor created to help you understand health concerns "
            "for learning and awareness purposes."
        )
        audio_output = text_to_speech_openai(doctor_response)
        return patient_text_val, doctor_response, audio_output or None, MAP_PLACEHOLDER

    elif any(x in text_lower for x in ["thank you", "thanks"]):
        doctor_response = "You are welcome. Take care and feel free to ask anytime."
        audio_output = text_to_speech_openai(doctor_response)
        return patient_text_val, doctor_response, audio_output or None, MAP_PLACEHOLDER

    if image and os.path.exists(image):
        image_prompt      = f"{SYSTEM_PROMPT} {patient_text_val}"
        image_analysis    = analyze_image_with_query(image_prompt, encode_image(image))
        combined_query    = f"{patient_text_val}\nImage Findings: {image_analysis}"
        doctor_response   = ask_rag(combined_query)
        map_context_query = f"{patient_text_val} {image_analysis}"
    else:
        doctor_response   = ask_rag(patient_text_val)
        map_context_query = patient_text_val

    if is_rejection(doctor_response):
        map_html = MAP_PLACEHOLDER
    else:
        map_html         = build_map(map_context_query, doctor_response, lat, lon)
        doctor_response += maps_link_html(lat, lon, map_context_query, doctor_response)
        doctor_response = format_response_html(doctor_response)
    audio_output = text_to_speech_openai(doctor_response)
    if not audio_output or not os.path.exists(audio_output):
        audio_output = None

    return patient_text_val, doctor_response, audio_output, map_html

# ─────────────────────────────────────────────
#  CHAT BACKEND
# ─────────────────────────────────────────────
def chat_backend(message, history, lat, lon):
    if not message.strip():
        return history, "", gr.update()

    history = history or []
    answer  = ask_rag(message)

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": answer})

    # Don't update map for casual/greeting messages or rejections
    if is_casual(message) or is_rejection(answer):
        return history, "", gr.update()   # leave map unchanged

    link = maps_link_html(lat, lon, message, answer)
    if link:
        history.append({"role": "assistant", "content": link})
    map_html = build_map(message, answer, lat, lon)

    return history, "", map_html

# ─────────────────────────────────────────────
#  CHAT PANEL HELPERS
# ─────────────────────────────────────────────
def open_chat():
    return gr.update(visible=True)

def close_chat():
    return gr.update(visible=False)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
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

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--white) !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

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

.section-title {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--teal) !important;
    margin-bottom: 14px !important;
}

.panel-card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    backdrop-filter: blur(6px);
}

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

label, .gr-form > label, .gr-block label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.4px !important;
    color: var(--muted) !important;
    margin-bottom: 6px !important;
}

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

.gr-audio {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.03) !important;
    overflow: hidden !important;
}
.gr-image {
    border-radius: var(--radius) !important;
    border: 1px dashed rgba(0,201,177,0.3) !important;
    background: rgba(0,201,177,0.03) !important;
    transition: border-color 0.2s !important;
}
.gr-image:hover { border-color: var(--teal) !important; }

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

#chat-panel {
    background: rgba(11,30,61,0.97) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(20px) !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

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

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
"""

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
with gr.Blocks(title="AI Doctor Assistant") as demo:

    # ── Header ──
    gr.HTML("""
<div id="main-header">
    <div style="display:flex; align-items:center; gap:16px;">
        <div>
            <h1 style="
                font-family:'DM Serif Display',serif;
                font-size:2.3rem; letter-spacing:-0.5px; margin:0;
                background:linear-gradient(90deg,#00c9b1,#3d7bff);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                background-clip:text;">AI Doctor Assistant</h1>
            <p style="font-size:0.92rem;color:#8ba3c9;margin:6px 0 0;letter-spacing:0.3px;">
                <span class="status-dot"></span>
                AI-Powered Medical Guidance &nbsp;·&nbsp; Voice &amp; Image Enabled &nbsp;·&nbsp; Live Hospital Map
            </p>
        </div>
    </div>
</div>
<div id="disclaimer">
    ⚠ &nbsp;<strong>For educational purposes only.</strong>&nbsp;
    This tool does not replace professional medical advice, diagnosis, or treatment.
</div>
    """)

    gr.HTML("<div style='height:24px'></div>")

    # Hidden location state
    lat = gr.Textbox(visible=False)
    lon = gr.Textbox(visible=False)

    # Location button
    with gr.Row():
        with gr.Column():
            location_btn = gr.Button("📍 Get My Location", variant="secondary")
            gr.HTML("""<p style="font-size:0.78rem;color:#4e6a92;margin-top:6px;text-align:center;">
                Allow location access to find nearby hospitals</p>""")

    location_btn.click(
        fn=None, inputs=None, outputs=[lat, lon],
        js="""
        async () => {
            return await new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve([
                        pos.coords.latitude.toString(),
                        pos.coords.longitude.toString()
                    ]),
                    () => {
                        alert("Location permission denied. Map will not be available.");
                        resolve(["", ""]);
                    }
                );
            });
        }
        """
    )

    gr.HTML("<div style='height:12px'></div>")

    # ── Input / Response Row ──
    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<p class="section-title">Patient Input</p>')
            audio_input = gr.Audio(sources=["microphone"], type="filepath",
                                   label="Voice Recording  (optional if image provided)")
            gr.HTML("<div style='height:16px'></div>")
            image_input = gr.Image(type="filepath", label="Medical Image  (optional)")
            gr.HTML("<div style='height:20px'></div>")
            submit = gr.Button("⚕  Consult AI Doctor", variant="primary")
            gr.HTML("""<p style="text-align:center;font-size:0.78rem;color:#4e6a92;margin-top:14px;">
                Speak or upload an image — or both together.<br>
                Image analysis auto-detects the specialty for the map.<br>
                Enable location for nearby hospital suggestions.</p>""")

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<p class="section-title">AI Doctor Response</p>')
            patient_text = gr.Textbox(label="What You Said / Image Context", lines=3,
                                      interactive=False,
                                      placeholder="Your transcribed speech or image context will appear here…")
            gr.HTML("<div style='height:14px'></div>")
            doctor_text  = gr.HTML(label="Medical Guidance")
            gr.HTML("<div style='height:14px'></div>")
            doctor_audio = gr.Audio(label="Voice Response", type="filepath", autoplay=True)

    # ── Map Section ──
    gr.HTML("<div style='height:20px'></div>")
    gr.HTML("""
    <div style="font-size:0.72rem;font-weight:600;letter-spacing:1.5px;
                text-transform:uppercase;color:#00c9b1;margin-bottom:14px;padding:0 4px;">
        🗺 Nearby Recommended Hospitals
    </div>""")

    map_out = gr.HTML(value=MAP_PLACEHOLDER)

    # Wire main button
    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, lat, lon],
        outputs=[patient_text, doctor_text, doctor_audio, map_out]
    )

    # ── Chat Panel ──
    with gr.Column(visible=False, elem_id="chat-panel") as chat_panel:
        gr.HTML('<p class="section-title">💬 Live Chat with AI Doctor</p>')
        chatbot = gr.Chatbot(height=340, show_label=False)
        with gr.Row():
            chat_input = gr.Textbox(placeholder="Type your health question…",
                                    show_label=False, scale=5)
            send_btn   = gr.Button("Send", variant="primary", scale=1)
        gr.HTML("<div style='height:8px'></div>")
        close_btn = gr.Button("✕  Close Chat", variant="secondary")

    chat_icon = gr.Button("💬  Chat", elem_id="chatbot-fab")

    chat_icon.click(open_chat, outputs=chat_panel)
    close_btn.click(close_chat, outputs=chat_panel)

    send_btn.click(
        fn=chat_backend,
        inputs=[chat_input, chatbot, lat, lon],
        outputs=[chatbot, chat_input, map_out]
    )
    chat_input.submit(
        fn=chat_backend,
        inputs=[chat_input, chatbot, lat, lon],
        outputs=[chatbot, chat_input, map_out]
    )

# ─────────────────────────────────────────────
#  LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        debug=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=["DM Sans", "sans-serif"],
        ),
    )