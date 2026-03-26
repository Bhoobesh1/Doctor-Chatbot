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

SPECIALTY_PROMPT = """
Given the following medical question or symptom, identify the most relevant hospital
specialty or department (e.g. dermatologist, cardiologist, orthopedic, general physician,
neurologist, ophthalmologist, ENT, dentist, gynecologist, psychiatrist, etc.).
Return ONLY the specialty keyword (one or two words, lowercase).
If unsure, return: general physician
"""

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
        return "Error from RAG backend."
    except Exception:
        return "Unable to connect to backend."


# ---------------- DETECT MEDICAL SPECIALTY ----------------
def detect_specialty(user_text):
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SPECIALTY_PROMPT},
                {"role": "user",   "content": user_text}
            ],
            max_tokens=20,
            temperature=0
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        return "general physician"


# ---------------- CORE VOICE + IMAGE LOGIC ----------------
def process_inputs(audio, image):
    if not audio:
        return "", "Please speak your health concern.", None, ""

    patient_text = transcribe_with_openai(audio)
    if not patient_text:
        return "", "Unable to understand audio.", None, ""

    text_lower = patient_text.lower()

    if any(x in text_lower for x in ["who are you", "introduce yourself", "your name"]):
        doctor_response = (
            "I am an AI doctor created to help you understand health concerns "
            "for learning and awareness purposes."
        )
        specialty = ""
    elif any(x in text_lower for x in ["thank you", "thanks"]):
        doctor_response = "You are welcome. Take care and feel free to ask anytime."
        specialty = ""
    else:
        if image and os.path.exists(image):
            image_prompt    = SYSTEM_PROMPT + " " + patient_text
            image_analysis  = analyze_image_with_query(image_prompt, encode_image(image))
            combined_query  = f"{patient_text}\nImage Findings: {image_analysis}"
            doctor_response = ask_rag_backend(combined_query)
        else:
            doctor_response = ask_rag_backend(patient_text)
        specialty = detect_specialty(patient_text)

    audio_output = text_to_speech_openai(doctor_response)
    if not audio_output or not os.path.exists(audio_output):
        audio_output = None

    return patient_text, doctor_response, audio_output, specialty


# ---------------- CHATBOT BACKEND ----------------
def chat_backend(message, history):
    if not message.strip():
        return history, "", ""
    if history is None:
        history = []
    answer    = ask_rag_backend(message)
    specialty = detect_specialty(message)
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, "", specialty

def open_chat():
    return gr.update(visible=True)

def close_chat():
    return gr.update(visible=False)


# ============================================================
# CUSTOM CSS
# ============================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:     #0b1e3d;
    --navy-mid: #122650;
    --blue:     #1a5cff;
    --blue-lt:  #3d7bff;
    --teal:     #00c9b1;
    --white:    #f4f8ff;
    --muted:    #8ba3c9;
    --card:     rgba(255,255,255,0.04);
    --border:   rgba(255,255,255,0.08);
    --radius:   14px;
}
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
#main-header {
    background: linear-gradient(135deg,#0b1e3d 0%,#122650 50%,#0d2a60 100%);
    border-bottom: 1px solid var(--border);
    padding: 36px 48px 28px;
    position: relative; overflow: hidden;
}
#main-header::before {
    content:""; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(0,201,177,0.12) 0%,transparent 70%);
    border-radius:50%; pointer-events:none;
}
#disclaimer {
    background:rgba(0,201,177,0.07);
    border:1px solid rgba(0,201,177,0.25);
    border-radius:var(--radius); padding:10px 18px;
    margin:18px 24px 0; font-size:0.82rem; color:#7fd8ce;
    display:flex; align-items:center; gap:8px;
}
.section-title {
    font-size:0.72rem !important; font-weight:600 !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important;
    color:var(--teal) !important; margin-bottom:14px !important;
}
.panel-card {
    background:var(--card) !important; border:1px solid var(--border) !important;
    border-radius:var(--radius) !important; padding:24px !important;
    backdrop-filter:blur(6px);
}
input[type=text], textarea,
.gr-textbox textarea, .gr-textbox input {
    background:rgba(255,255,255,0.05) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:10px !important; color:var(--white) !important;
    font-family:'DM Sans',sans-serif !important;
    font-size:0.93rem !important; padding:12px 14px !important;
}
label, .gr-form > label, .gr-block label {
    font-size:0.8rem !important; font-weight:500 !important;
    color:var(--muted) !important; margin-bottom:6px !important;
}
.gr-button-primary, button.primary {
    background:linear-gradient(135deg,var(--blue),var(--blue-lt)) !important;
    border:none !important; border-radius:10px !important;
    color:#fff !important; font-family:'DM Sans',sans-serif !important;
    font-size:0.9rem !important; font-weight:600 !important;
    padding:13px 28px !important; cursor:pointer !important;
    box-shadow:0 4px 18px rgba(26,92,255,0.35) !important;
    width:100% !important;
    transition:transform 0.15s,box-shadow 0.15s !important;
}
.gr-button-primary:hover, button.primary:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(26,92,255,0.45) !important;
}
button.secondary {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:10px !important; color:var(--muted) !important;
    font-family:'DM Sans',sans-serif !important;
    font-size:0.85rem !important; padding:10px 20px !important;
    cursor:pointer !important;
}
.gr-audio {
    border-radius:var(--radius) !important;
    border:1px solid var(--border) !important;
    background:rgba(255,255,255,0.03) !important; overflow:hidden !important;
}
.gr-image {
    border-radius:var(--radius) !important;
    border:1px dashed rgba(0,201,177,0.3) !important;
    background:rgba(0,201,177,0.03) !important;
}
.gr-chatbot {
    background:rgba(255,255,255,0.02) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--radius) !important;
}
.message.user {
    background:linear-gradient(135deg,var(--blue),var(--blue-lt)) !important;
    border-radius:18px 18px 4px 18px !important; color:#fff !important;
}
.message.bot {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid var(--border) !important;
    border-radius:18px 18px 18px 4px !important; color:var(--white) !important;
}
#chatbot-fab {
    position:fixed !important; bottom:28px !important; right:28px !important;
    background:linear-gradient(135deg,var(--teal),#00a896) !important;
    color:#fff !important; border:none !important; border-radius:50px !important;
    padding:14px 22px !important; font-size:0.88rem !important;
    font-weight:600 !important;
    box-shadow:0 6px 24px rgba(0,201,177,0.4) !important;
    cursor:pointer !important; z-index:1000 !important; width:auto !important;
    transition:transform 0.2s,box-shadow 0.2s !important;
}
#chatbot-fab:hover {
    transform:translateY(-3px) !important;
    box-shadow:0 10px 30px rgba(0,201,177,0.5) !important;
}
#chat-panel {
    background:rgba(11,30,61,0.97) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--radius) !important;
    padding:20px !important; margin-top:20px !important;
}
.status-dot {
    display:inline-block; width:8px; height:8px;
    border-radius:50%; background:var(--teal);
    box-shadow:0 0 8px var(--teal); margin-right:6px;
    animation:pulse-dot 2s infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:99px}
"""


# ============================================================
# MAP HTML  —  always visible in the page, never toggled
# ============================================================
MAP_HTML = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<div id="map-section" style="
  margin:28px 0 0 0;
  background:rgba(255,255,255,0.04);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:14px;
  padding:24px;
  font-family:'DM Sans',sans-serif;
">

  <!-- title -->
  <p style="font-size:0.72rem;font-weight:600;letter-spacing:1.5px;
            text-transform:uppercase;color:#00c9b1;margin:0 0 4px;">
    Nearby Hospitals
  </p>
  <p id="map-subtitle" style="font-size:0.84rem;color:#8ba3c9;margin:0 0 18px;">
    Ask a health question — hospitals near you will appear here.
  </p>

  <!-- location permission banner -->
  <div id="loc-banner" style="
    background:linear-gradient(135deg,rgba(0,201,177,0.08),rgba(26,92,255,0.08));
    border:1px solid rgba(0,201,177,0.28);
    border-radius:10px; padding:14px 18px;
    display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap; gap:12px;
    margin-bottom:18px;
  ">
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.6rem;">📍</span>
      <div>
        <p style="margin:0;font-size:0.9rem;font-weight:600;color:#f4f8ff;">
          Enable Location Access
        </p>
        <p style="margin:3px 0 0;font-size:0.77rem;color:#8ba3c9;">
          Used only in your browser — never sent to any server.
        </p>
      </div>
    </div>
    <button id="loc-btn"
      onclick="window.AI_DOC && window.AI_DOC.requestLocation()"
      style="
        background:linear-gradient(135deg,#00c9b1,#00a896);
        border:none; border-radius:8px; color:#fff;
        font-family:'DM Sans',sans-serif; font-size:0.84rem;
        font-weight:600; padding:10px 22px; cursor:pointer;
        box-shadow:0 4px 14px rgba(0,201,177,0.35);
        white-space:nowrap; transition:transform 0.15s;
      "
      onmouseover="this.style.transform='translateY(-2px)'"
      onmouseout="this.style.transform='translateY(0)'">
      🔓 Allow Location
    </button>
  </div>

  <!-- status badges -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
    <span style="background:rgba(0,201,177,0.12);color:#00c9b1;
                 border:1px solid rgba(0,201,177,0.28);border-radius:20px;
                 padding:4px 14px;font-size:0.74rem;font-weight:600;">
      📍 5 km radius
    </span>
    <span id="loc-status" style="background:rgba(255,255,255,0.06);color:#8ba3c9;
                 border:1px solid rgba(255,255,255,0.1);border-radius:20px;
                 padding:4px 14px;font-size:0.74rem;">
      ⏳ Location pending
    </span>
    <span id="spec-badge" style="display:none;
                 background:rgba(26,92,255,0.12);color:#3d7bff;
                 border:1px solid rgba(26,92,255,0.25);border-radius:20px;
                 padding:4px 14px;font-size:0.74rem;font-weight:600;">
    </span>
  </div>

  <!-- map + list -->
  <div style="display:flex;gap:16px;flex-wrap:wrap;">

    <div style="flex:1.6;min-width:280px;position:relative;">
      <div id="hospital-map" style="
        width:100%;height:420px;border-radius:10px;
        border:1px solid rgba(0,201,177,0.18);
        background:#0d1f3c;overflow:hidden;">
      </div>
      <!-- overlay shown before location granted -->
      <div id="map-overlay" style="
        position:absolute;inset:0;background:#0d1f3c;
        border-radius:10px;display:flex;flex-direction:column;
        align-items:center;justify-content:center;
        z-index:500;pointer-events:none;">
        <div style="font-size:3rem;margin-bottom:12px;">🗺️</div>
        <p id="overlay-msg" style="color:#8ba3c9;font-size:0.88rem;
           text-align:center;margin:0;line-height:1.6;">
          Allow location access above<br>to load the map.
        </p>
      </div>
    </div>

    <div id="hosp-list" style="
      flex:1;min-width:240px;max-height:420px;overflow-y:auto;
      display:flex;flex-direction:column;gap:10px;">
      <p style="color:#8ba3c9;font-size:0.85rem;text-align:center;margin-top:60px;">
        Hospitals will appear here after your query.
      </p>
    </div>
  </div>

  <!-- progress bar -->
  <div id="prog-wrap" style="display:none;margin-top:14px;">
    <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:5px;overflow:hidden;">
      <div id="prog-bar" style="height:100%;width:0%;
           background:linear-gradient(90deg,#00c9b1,#1a5cff);
           border-radius:99px;transition:width 0.4s ease;"></div>
    </div>
    <p id="prog-txt" style="font-size:0.78rem;color:#8ba3c9;
       margin:6px 0 0;text-align:center;"></p>
  </div>

</div><!-- /#map-section -->

<script>
/* Wrap everything so variables don't leak to global scope */
(function(){

  var _map     = null;
  var _markers = [];
  var _circle  = null;
  var _uMark   = null;
  var _lat     = null;
  var _lng     = null;
  var _locOK   = false;
  var _pending = null;

  function $id(id){ return document.getElementById(id); }

  function setProgress(show, pct, msg){
    $id('prog-wrap').style.display = show ? 'block' : 'none';
    $id('prog-bar').style.width    = (pct||0)+'%';
    if(msg) $id('prog-txt').textContent = msg;
  }

  function setListMsg(txt){
    $id('hosp-list').innerHTML =
      '<p style="color:#8ba3c9;font-size:0.85rem;text-align:center;'+
      'margin-top:60px;font-family:DM Sans,sans-serif;">'+txt+'</p>';
  }

  function setLocStatus(txt,color){
    var el=$id('loc-status');
    el.textContent=txt; el.style.color=color||'#8ba3c9';
  }

  /* ---------- Leaflet init ---------- */
  function initMap(lat,lng){
    /* hide overlay */
    var ov=$id('map-overlay'); if(ov) ov.style.display='none';

    if(!_map){
      _map=L.map('hospital-map',{zoomControl:true}).setView([lat,lng],14);
      L.tileLayer(
        'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        {attribution:'&copy; OpenStreetMap &copy; CARTO',subdomains:'abcd',maxZoom:19}
      ).addTo(_map);
    } else {
      _map.setView([lat,lng],14);
    }

    if(_uMark)  _map.removeLayer(_uMark);
    if(_circle) _map.removeLayer(_circle);

    _uMark=L.marker([lat,lng],{icon:L.divIcon({
      className:'',
      html:'<div style="width:18px;height:18px;background:#00c9b1;border:3px solid #fff;'+
           'border-radius:50%;box-shadow:0 0 0 6px rgba(0,201,177,0.22);"></div>',
      iconSize:[18,18],iconAnchor:[9,9]
    })}).addTo(_map)
      .bindPopup('<b style="font-family:DM Sans,sans-serif;color:#0b1e3d;">📍 You are here</b>');

    _circle=L.circle([lat,lng],{
      radius:5000,color:'#00c9b1',weight:1,
      opacity:0.5,fillColor:'#00c9b1',fillOpacity:0.04
    }).addTo(_map);

    /* Fix grey tiles when Leaflet renders inside a hidden/flex container */
    setTimeout(function(){ _map.invalidateSize(); },400);
  }

  /* ---------- numbered pin ---------- */
  function addPin(el,num){
    var icon=L.divIcon({
      className:'',
      html:'<div style="width:32px;height:32px;'+
           'background:linear-gradient(135deg,#1a5cff,#3d7bff);'+
           'border:2px solid #fff;border-radius:50%;'+
           'display:flex;align-items:center;justify-content:center;'+
           'color:#fff;font-size:13px;font-weight:700;'+
           'font-family:DM Sans,sans-serif;'+
           'box-shadow:0 2px 10px rgba(26,92,255,0.55);">'+num+'</div>',
      iconSize:[32,32],iconAnchor:[16,16]
    });
    var name =(el.tags&&el.tags.name)||'Hospital';
    var addr =(el.tags&&(el.tags['addr:street']||el.tags['addr:full']))||'';
    var phone=(el.tags&&(el.tags.phone||el.tags['contact:phone']))||'';
    var km   =(el._dist/1000).toFixed(1);
    var popup='<div style="font-family:DM Sans,sans-serif;min-width:190px;">'+
      '<b style="color:#0b1e3d;font-size:0.9rem;">'+name+'</b><br>'+
      (addr?'<span style="font-size:0.78rem;color:#555;">'+addr+'</span><br>':'')+
      '<span style="font-size:0.78rem;color:#1a5cff;">📍 '+km+' km away</span><br>'+
      (phone?'<span style="font-size:0.78rem;color:#555;">📞 '+phone+'</span><br>':'')+
      '<a href="https://www.openstreetmap.org/directions?to='+el._lat+','+el._lng+'"'+
      ' target="_blank" style="font-size:0.78rem;color:#00c9b1;text-decoration:none;">'+
      '🗺 Get directions</a></div>';
    var m=L.marker([el._lat,el._lng],{icon:icon}).addTo(_map).bindPopup(popup);
    _markers.push(m);
  }

  /* ---------- sidebar list ---------- */
  function renderList(arr){
    var list=$id('hosp-list'); list.innerHTML='';
    arr.forEach(function(el,i){
      var name =(el.tags&&el.tags.name)||('Hospital '+(i+1));
      var addr =(el.tags&&(el.tags['addr:street']||el.tags['addr:full']||el.tags['addr:city']))||'Address not listed';
      var phone=(el.tags&&(el.tags.phone||el.tags['contact:phone']))||'';
      var km   =(el._dist/1000).toFixed(1);
      var type =(el.tags&&(el.tags.amenity||el.tags.healthcare))||'hospital';
      var card=document.createElement('div');
      card.style.cssText='display:flex;gap:12px;background:rgba(255,255,255,0.04);'+
        'border:1px solid rgba(255,255,255,0.08);border-radius:10px;'+
        'padding:12px 14px;cursor:pointer;'+
        'transition:border-color 0.2s,background 0.2s;font-family:DM Sans,sans-serif;';
      card.onmouseenter=function(){card.style.borderColor='#00c9b1';card.style.background='rgba(0,201,177,0.06)';};
      card.onmouseleave=function(){card.style.borderColor='rgba(255,255,255,0.08)';card.style.background='rgba(255,255,255,0.04)';};
      card.innerHTML=
        '<div style="font-size:1.5rem;margin-top:2px;">🏥</div>'+
        '<div style="flex:1;min-width:0;">'+
          '<div style="font-size:0.88rem;font-weight:600;color:#f4f8ff;margin-bottom:3px;">'+(i+1)+'. '+name+'</div>'+
          '<div style="font-size:0.76rem;color:#8ba3c9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'+addr+'</div>'+
          '<div style="font-size:0.76rem;color:#8ba3c9;margin-top:2px;">📍 <b style="color:#00c9b1;">'+km+' km</b> away</div>'+
          (phone?'<div style="font-size:0.74rem;color:#8ba3c9;margin-top:2px;">📞 '+phone+'</div>':'')+
          '<div style="margin-top:6px;"><span style="background:rgba(26,92,255,0.15);color:#3d7bff;'+
          'border:1px solid rgba(26,92,255,0.25);border-radius:20px;'+
          'padding:2px 9px;font-size:0.7rem;font-weight:600;">'+type+'</span></div>'+
        '</div>';
      (function(captEl,captIdx){
        card.addEventListener('click',function(){
          if(_map&&_markers[captIdx]){_map.setView([captEl._lat,captEl._lng],16);_markers[captIdx].openPopup();}
        });
      })(el,i);
      list.appendChild(card);
    });
    var note=document.createElement('p');
    note.style.cssText='font-size:0.74rem;color:#4e6a92;text-align:center;margin:4px 0 0;';
    note.textContent='Click a card to zoom · Click pin for directions';
    list.appendChild(note);
  }

  /* ---------- Overpass search ---------- */
  function doSearch(lat,lng,specialty){
    setProgress(true,20,'Searching OpenStreetMap…');
    setListMsg('Searching for hospitals…');
    var q='[out:json][timeout:25];\n(\n'+
      'node["amenity"="hospital"](around:5000,'+lat+','+lng+');\n'+
      'way["amenity"="hospital"](around:5000,'+lat+','+lng+');\n'+
      'node["amenity"="clinic"](around:5000,'+lat+','+lng+');\n'+
      'way["amenity"="clinic"](around:5000,'+lat+','+lng+');\n'+
      'node["healthcare"="hospital"](around:5000,'+lat+','+lng+');\n'+
      'node["healthcare"="clinic"](around:5000,'+lat+','+lng+');\n'+
      ');\nout center 20;';
    setProgress(true,50,'Querying OpenStreetMap database…');
    fetch('https://overpass-api.de/api/interpreter',{method:'POST',body:'data='+encodeURIComponent(q)})
    .then(function(r){return r.json();})
    .then(function(data){
      setProgress(true,88,'Rendering results…');
      var els=data.elements||[];
      if(!els.length){setProgress(false);setListMsg('No hospitals found within 5 km.');return;}
      var uLL=L.latLng(lat,lng);
      var sorted=els.map(function(el){
        var eLat=el.lat||(el.center&&el.center.lat);
        var eLng=el.lon||(el.center&&el.center.lon);
        if(!eLat||!eLng)return null;
        return Object.assign({},el,{_lat:eLat,_lng:eLng,_dist:uLL.distanceTo(L.latLng(eLat,eLng))});
      }).filter(Boolean).sort(function(a,b){return a._dist-b._dist;}).slice(0,5);
      _markers.forEach(function(m){_map.removeLayer(m);}); _markers=[];
      renderList(sorted);
      sorted.forEach(function(el,i){addPin(el,i+1);});
      var coords=sorted.map(function(el){return[el._lat,el._lng];});
      coords.push([lat,lng]);
      _map.fitBounds(L.latLngBounds(coords),{padding:[40,40]});
      setProgress(false);
    }).catch(function(){
      setProgress(false);
      setListMsg('⚠ Could not reach OpenStreetMap. Check internet.');
    });
  }

  /* ---------- trigger search after location granted ---------- */
  function triggerSearch(specialty){
    $id('map-subtitle').textContent='Showing '+specialty+' hospitals near you';
    var badge=$id('spec-badge');
    badge.textContent='🩺 '+specialty; badge.style.display='inline-block';
    doSearch(_lat,_lng,specialty);
  }

  /* ---------- PUBLIC API ---------- */

  /* Called by the "Allow Location" button */
  window.AI_DOC_requestLocation = function(){
    var btn=$id('loc-btn');
    if(btn){btn.textContent='⏳ Requesting…';btn.disabled=true;}
    setLocStatus('⏳ Requesting…','#f0c040');
    if(!navigator.geolocation){
      setLocStatus('❌ Not supported','#ff6b6b');
      setListMsg('⚠ Geolocation not supported by your browser.');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      function(pos){
        _lat=pos.coords.latitude; _lng=pos.coords.longitude; _locOK=true;
        var b=$id('loc-banner'); if(b)b.style.display='none';
        setLocStatus('✅ Location active','#00c9b1');
        initMap(_lat,_lng);
        if(_pending){var spec=_pending;_pending=null;triggerSearch(spec);}
      },
      function(err){
        var msg=err.code===1?'❌ Permission denied':err.code===2?'❌ Position unavailable':'❌ Timed out';
        setLocStatus(msg,'#ff6b6b');
        setListMsg(msg+'. Click "Allow Location" and grant permission.');
        if(btn){btn.textContent='🔓 Allow Location';btn.disabled=false;}
      },
      {timeout:12000,maximumAge:60000,enableHighAccuracy:false}
    );
  };

  /* Called from Gradio JS bridge when specialty changes */
  window.AI_DOC_showForSpecialty = function(specialty){
    if(!specialty||!specialty.trim())return;
    if(!_locOK){
      _pending=specialty;
      $id('map-subtitle').textContent='Allow location to find '+specialty+' hospitals near you.';
      setListMsg('👆 Click "Allow Location" above, then hospitals will load automatically.');
      return;
    }
    triggerSearch(specialty);
  };

  /* Expose a single namespace object too (used by inline onclick) */
  window.AI_DOC = {
    requestLocation:  window.AI_DOC_requestLocation,
    showForSpecialty: window.AI_DOC_showForSpecialty
  };

})();
</script>
"""


# ============================================================
# JS BRIDGE  — polling loop so it works regardless of load order
# ============================================================
TRIGGER_MAP_JS = """
function(specialty) {
    if (!specialty || specialty.trim() === '') return specialty;
    var attempts = 0;
    var iv = setInterval(function(){
        attempts++;
        if (typeof window.AI_DOC_showForSpecialty === 'function') {
            clearInterval(iv);
            window.AI_DOC_showForSpecialty(specialty);
        }
        if (attempts > 50) clearInterval(iv);
    }, 100);
    return specialty;
}
"""


# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks(title="AI Doctor Assistant", css=CUSTOM_CSS) as demo:

    gr.HTML("""
<div id="main-header">
  <div style="display:flex;align-items:center;gap:16px;">
    <div>
      <h1 style="font-family:'DM Serif Display',serif;font-size:2.3rem;
                 letter-spacing:-0.5px;margin:0;
                 background:linear-gradient(90deg,#00c9b1,#3d7bff);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">
        AI Doctor Assistant
      </h1>
      <p style="font-size:0.92rem;color:#8ba3c9;margin:6px 0 0;letter-spacing:0.3px;">
        <span class="status-dot"></span>
        AI-Powered Medical Guidance &nbsp;·&nbsp; Voice &amp; Image Enabled
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

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<p class="section-title">Patient Input</p>')
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Recording")
            gr.HTML("<div style='height:16px'></div>")
            image_input = gr.Image(type="filepath", label="Medical Image  (optional)")
            gr.HTML("<div style='height:20px'></div>")
            submit = gr.Button("⚕  Consult AI Doctor", variant="primary")
            gr.HTML("""
            <p style="text-align:center;font-size:0.78rem;color:#4e6a92;margin-top:14px;">
              Speak clearly into your microphone.<br>Image upload is optional.
            </p>""")

        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<p class="section-title">AI Doctor Response</p>')
            patient_text = gr.Textbox(
                label="What You Said", lines=3, interactive=False,
                placeholder="Your transcribed speech will appear here…"
            )
            gr.HTML("<div style='height:14px'></div>")
            doctor_text = gr.Textbox(
                label="Medical Guidance", lines=6, interactive=False,
                placeholder="The AI doctor's response will appear here…"
            )
            gr.HTML("<div style='height:14px'></div>")
            doctor_audio = gr.Audio(label="Voice Response", type="filepath", autoplay=True)

    specialty_state = gr.Textbox(visible=False)

    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[patient_text, doctor_text, doctor_audio, specialty_state]
    )
    specialty_state.change(
        fn=None, inputs=[specialty_state],
        outputs=[specialty_state], js=TRIGGER_MAP_JS
    )

    # Map is always in the DOM — no visibility toggle
    gr.HTML(MAP_HTML)

    # Chat panel
    with gr.Column(visible=False, elem_id="chat-panel") as chat_panel:
        gr.HTML('<p class="section-title">💬 Live Chat with AI Doctor</p>')
        chatbot = gr.Chatbot(height=340, show_label=False)
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Type your health question…",
                show_label=False, scale=5
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        gr.HTML("<div style='height:8px'></div>")
        close_btn = gr.Button("✕  Close Chat", variant="secondary")

    chat_icon            = gr.Button("💬  Chat", elem_id="chatbot-fab")
    chat_specialty_state = gr.Textbox(visible=False)

    chat_icon.click(open_chat,  outputs=chat_panel)
    close_btn.click(close_chat, outputs=chat_panel)

    send_btn.click(
        fn=chat_backend,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input, chat_specialty_state]
    )
    chat_input.submit(
        fn=chat_backend,
        inputs=[chat_input, chatbot],
        outputs=[chatbot, chat_input, chat_specialty_state]
    )
    chat_specialty_state.change(
        fn=None, inputs=[chat_specialty_state],
        outputs=[chat_specialty_state], js=TRIGGER_MAP_JS
    )


# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=["DM Sans", "sans-serif"],
        ),
    )