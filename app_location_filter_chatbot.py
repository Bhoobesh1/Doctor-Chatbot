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
You are a senior medical doctor with decades of clinical experience.

Respond only to medical or health-related questions.
If the question is not related to health or medicine, reply:
"This assistant provides only medical guidance."

For medical questions:

• Explain likely causes calmly and clearly.
• Suggest simple home remedies when appropriate.
• Do not assume symptoms that were not mentioned.
• Do not provide a confirmed diagnosis.

IMPORTANT:
If symptoms relate to a specific organ system such as eyes, heart, chest, bones, skin, or general illness,
clearly recommend consulting the appropriate specialist or visiting a nearby hospital.

For example:
- Eye symptoms → recommend an ophthalmologist.
- Heart or chest symptoms → recommend a cardiologist or emergency care.

Keep the response short, natural, and reassuring.
Avoid medical jargon.
Do not exaggerate.
Start directly without greetings.
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
        model="gpt-4o-mini",
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

# ---------------- DETECT CATEGORY ----------------
def detect_category(query):
    q = query.lower()

    if any(word in q for word in ["skin", "rash", "itch", "acne"]):
        return "dermatology"

    elif any(word in q for word in ["eyes","eye","vision", "blur", "red eye"]):
        return "ophthalmology"

    elif any(word in q for word in ["bone", "fracture", "joint", "ortho"]):
        return "orthopedic"

    elif any(word in q for word in ["heart", "chest pain", "cardio"]):
        return "cardiology"

    else:
        return "general"

# ---------------- SMART CLASSIFICATION ----------------
def classify_hospitals(hospitals, query_text):

    category = detect_category(query_text)
    filtered = []

    # Specialist keywords to exclude in general cases
    specialist_words = [
        "eye", "cardio", "heart", "ortho", "bone",
        "skin", "derma", "maternity", "child",
        "cancer", "dental", "ayurveda", "homeo"
    ]

    for h in hospitals:
        name = h["name"].lower()

        # ---------------- GENERAL CASE ----------------
        if category == "general":

            # Skip specialist hospitals
            if any(word in name for word in specialist_words):
                continue

            # Allow general hospitals AND clinics
            if any(word in name for word in [
                "hospital",
                "clinic",
                "general",
                "multi",
                "government",
                "medical college"
            ]):
                filtered.append(h)

        # ---------------- SPECIALIST CASES ----------------
        else:

            # Always allow multi speciality hospitals
            if any(word in name for word in [
                "multi", "multispeciality", "multi-speciality",
                "general hospital", "government"
            ]):
                filtered.append(h)
                continue

            if category == "dermatology":
                if any(word in name for word in ["skin", "derma"]):
                    filtered.append(h)

            elif category == "ophthalmology":
                if any(word in name for word in ["eye", "vision", "ophthal"]):
                    filtered.append(h)

            elif category == "orthopedic":
                if any(word in name for word in ["ortho", "bone"]):
                    filtered.append(h)

            elif category == "cardiology":
                if any(word in name for word in ["heart", "cardio"]):
                    filtered.append(h)

    # Fallback: if nothing found, show general hospitals
    if not filtered:
        for h in hospitals:
            name = h["name"].lower()
            if "hospital" in name or "clinic" in name:
                filtered.append(h)

    return filtered
# ---------------- GENERATE MAP ----------------
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
def process_inputs(audio, lat, lon):
    try:
        if not audio:
            return "", "Please speak your health concern", None, None

        text = transcribe_with_openai(audio)

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
    consult_btn = gr.Button("Consult Doctor")

    with gr.Row():
        txt_in = gr.Textbox(
            label="Patient Input",
            lines=10,
            max_lines=10, 
            scale=1)

        txt_out = gr.Textbox(
            label="Doctor Response",
            lines=14,
            max_lines=20,
            scale=2
        )

    audio_out = gr.Audio(autoplay=True)
    map_out = gr.HTML()

    consult_btn.click(
        process_inputs,
        inputs=[audio, lat, lon],
        outputs=[txt_in, txt_out, audio_out, map_out]
    )

# ---------------- SAFE LAUNCH ----------------
if __name__ == "__main__":
    demo.queue()   
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        debug=True
    )