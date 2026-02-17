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
You are a highly experienced doctor.
Respond only for awareness purposes.
Suggest likely causes and simple home remedies.
Speak calmly and clearly.
Keep response short and reassuring.
Avoid medical jargon.
Start directly without greeting.
"""

# ---------------- MEDICAL CHECK ----------------
def is_medical_query_ai(text):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply YES or NO. Is this medical?"},
            {"role": "user", "content": text}
        ],
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
        max_tokens=200
    )
    return r.choices[0].message.content.strip()

# ---------------- OSM SEARCH ----------------
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
        r = requests.post("https://overpass-api.de/api/interpreter", data=query, timeout=30)
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

    return hospitals[:5]

# ---------------- MAP ----------------
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
<div id="map" style="height:400px;"></div>
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

    return f'<iframe srcdoc="{html.replace(chr(34),"&quot;")}" style="width:100%;height:420px;border:none;"></iframe>'

# ---------------- MAIN LOGIC ----------------
def process_inputs(audio, lat, lon):

    if not audio:
        return "", "Please speak your health concern", None, None

    text = transcribe_with_openai(audio)

    is_medical = is_medical_query_ai(text)

    if not is_medical:
        return text, "Please ask health related questions only", None, None

    response = analyze_text_only(text)

    map_html = None

    if lat and lon:
        hospitals = get_nearby_hospitals(float(lat), float(lon))

        if hospitals:
            response += "\n\nNearby hospitals include:\n"
            response += "\n".join(h["name"] for h in hospitals)

        map_html = generate_map(float(lat), float(lon), hospitals)

    audio_out = text_to_speech_openai(response)

    return text, response, audio_out, map_html

# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.Markdown("## 🩺 AI Doctor with Live Location")

    lat = gr.Textbox(visible=False)
    lon = gr.Textbox(visible=False)

    location_btn = gr.Button("📍 Get My Location")

    # ✅ Correct JS → Python Location Passing
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

    txt_in = gr.Textbox(label="Patient Input")
    txt_out = gr.Textbox(label="Doctor Response")
    audio_out = gr.Audio(autoplay=True)
    map_out = gr.HTML()

    consult_btn.click(
        process_inputs,
        inputs=[audio, lat, lon],
        outputs=[txt_in, txt_out, audio_out, map_out]
    )

demo.launch(server_port=7861)
