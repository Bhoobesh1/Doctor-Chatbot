import gradio as gr
import asyncio
import sys
import requests
from openai import OpenAI

# ---------------- WINDOWS FIX ----------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------- OPENAI CLIENT ----------------
client = OpenAI()

# ---------------- IMPORT YOUR MODULES ----------------
from voice_of_the_patient import transcribe_with_openai
from voice_of_the_doctor import text_to_speech_openai

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are a highly experienced senior medical doctor.

Respond ONLY to medical or health-related questions.
If not medical, reply exactly:
"This assistant provides only medical guidance."

Do NOT give confirmed diagnoses.
Do NOT assume symptoms not mentioned.
Keep answers short, calm and clear.

If symptoms involve:
- Eye → Ophthalmologist
- Skin → Dermatologist
- Bone/joint → Orthopedic doctor
- Heart/chest → Cardiologist or emergency care
- General fever/cold → General physician

If severe symptoms like chest pain, breathing difficulty,
advise immediate emergency care.
"""

# ---------------- MEDICAL CHECK ----------------
def is_medical_query_ai(text):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a strict classifier.

If the user message contains:
- symptoms
- body pain
- bleeding
- blood
- health concerns
- diseases
- food related to health
- treatment questions

Reply ONLY: YES

If it is clearly non-medical (coding, travel, politics, etc.), reply ONLY: NO
"""
            },
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=3
    )

    return r.choices[0].message.content.strip().upper() == "YES"
# ---------------- DOCTOR RESPONSE ----------------
def analyze_text_only(query):
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.2,
        max_tokens=200
    )
    return r.choices[0].message.content.strip()

# ---------------- FETCH NEARBY HOSPITALS ----------------
def get_nearby_hospitals(lat, lon):
    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:5000,{lat},{lon});
      node["amenity"="clinic"](around:5000,{lat},{lon});
    );
    out body;
    """

    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            timeout=30
        )
        data = r.json()
    except:
        return []

    hospitals = []
    for e in data.get("elements", []):
        name = e.get("tags", {}).get("name")
        if name:
            hospitals.append({
                "name": name,
                "lat": e["lat"],
                "lon": e["lon"]
            })

    return hospitals[:50]

# ---------------- CATEGORY DETECTION ----------------
def detect_category(query):
    q = query.lower()

    if any(word in q for word in [
        "skin", "rash", "itch", "acne", "pimple", "eczema"
    ]):
        return "dermatology"

    elif any(word in q for word in [
        "eye", "eyes", "vision", "blur", "blurry",
        "red eye", "watering", "itchy eyes","reddish","itchy eyes"
    ]):
        return "ophthalmology"

    elif any(word in q for word in [
        "bone", "fracture", "joint", "knee", "back pain", "ortho"
    ]):
        return "orthopedic"

    elif any(word in q for word in [
        "heart", "chest pain", "cardio", "palpitation"
    ]):
        return "cardiology"

    else:
        return "general"

# ---------------- HOSPITAL FILTER ----------------
def classify_hospitals(hospitals, query_text):

    category = detect_category(query_text)
    filtered = []

    specialist_keywords = {
        "ophthalmology": ["eye", "vision", "ophthal", "retina"],
        "dermatology": ["skin", "derma", "cosmetic"],
        "orthopedic": ["ortho", "bone", "joint", "spine"],
        "cardiology": ["heart", "cardio", "cardiac"]
    }

    # Words to exclude in general case
    all_specialist_words = [
        "eye", "vision", "ophthal", "retina",
        "skin", "derma",
        "ortho", "bone", "joint", "spine",
        "heart", "cardio",
        "cancer", "oncology",
        "dental", "maternity", "neuro"
    ]

    general_keywords = [
        "general",
        "multi", "multispeciality", "multi-speciality",
        "government",
        "medical college",
        "clinic"
    ]

    for h in hospitals:
        name = h["name"].lower()

        # ---------------- SPECIALIST CASE ----------------
        if category in specialist_keywords:

            # If hospital name matches specialty keywords
            if any(word in name for word in specialist_keywords[category]):
                filtered.append(h)
                continue

            # Also include multi-speciality & government hospitals
            if any(word in name for word in ["multi", "multispeciality", "government", "general hospital"]):
                filtered.append(h)

        # ---------------- GENERAL CASE ----------------
        elif category == "general":

            # Skip specialist hospitals
            if any(word in name for word in all_specialist_words):
                continue

            # Allow only true general hospitals
            if any(word in name for word in general_keywords) or "hospital" in name:
                filtered.append(h)

    # ---------------- FINAL FALLBACK ----------------
    if not filtered:
        for h in hospitals:
            name = h["name"].lower()
            if "hospital" in name or "clinic" in name:
                filtered.append(h)

    return filtered
# ---------------- MAP GENERATOR ----------------
def generate_map(lat, lon, hospitals):

    markers = ""
    for h in hospitals:
        markers += f"""
        L.marker([{h['lat']},{h['lon']}])
        .addTo(map)
        .bindPopup("{h['name']}");
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
</head>
<body style="margin:0">
<div id="map" style="height:450px;"></div>
<script>
var map = L.map("map").setView([{lat},{lon}],14);

L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
    maxZoom: 19
}}).addTo(map);

L.marker([{lat},{lon}])
.addTo(map)
.bindPopup("You are here")
.openPopup();

{markers}
</script>
</body>
</html>
"""

    return f'<iframe srcdoc="{html.replace(chr(34),"&quot;")}" style="width:100%;height:470px;border:none;"></iframe>'
# ---------------- MAIN FUNCTION ----------------
def process_inputs(audio, text_input, lat, lon):
    try:
        if text_input and text_input.strip() != "":
            text = text_input.strip()
        elif audio:
            text = transcribe_with_openai(audio)
        else:
            return "", "Please speak or type your health concern", None, None

        if not is_medical_query_ai(text):
            return text, "Please ask health related questions only", None, None

        response = analyze_text_only(text)
        map_html = None

        if lat and lon:
            hospitals = get_nearby_hospitals(float(lat), float(lon))
            classified = classify_hospitals(hospitals, text)

            if classified:
                response += "\n\nRecommended hospitals:\n"
                response += "\n".join(h["name"] for h in classified)

            map_html = generate_map(float(lat), float(lon), classified)

        audio_out = text_to_speech_openai(response)

        return text, response, audio_out, map_html

    except Exception as e:
        print("ERROR:", e)
        return "", "System error occurred", None, None

# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.Markdown("## 🩺 AI Doctor with Live Location")

    lat = gr.Textbox(visible=False)
    lon = gr.Textbox(visible=False)

    location_btn = gr.Button("📍 Get My Location")

    location_btn.click(
        fn=None,
        inputs=None,
        outputs=[lat, lon],
        js="""
        async () => {
            return await new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        resolve([
                            position.coords.latitude.toString(),
                            position.coords.longitude.toString()
                        ]);
                    },
                    () => {
                        alert("Location permission denied");
                        resolve(["", ""]);
                    }
                );
            });
        }
        """
    )

    audio = gr.Audio(sources=["microphone"], type="filepath")
    text_input = gr.Textbox(
        label="Type Your Health Concern",
        placeholder="Describe your symptoms here...",
        lines=3
    )

    consult_btn = gr.Button("Consult Doctor")

    with gr.Row():
        txt_in = gr.Textbox(label="Patient Input", lines=6)
        txt_out = gr.Textbox(label="Doctor Response", lines=12)

    audio_out = gr.Audio(autoplay=True)
    map_out = gr.HTML()

    consult_btn.click(
        process_inputs,
        inputs=[audio, text_input, lat, lon],
        outputs=[txt_in, txt_out, audio_out, map_out]
    )

# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7861, debug=True)