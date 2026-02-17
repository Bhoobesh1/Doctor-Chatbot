from flask import Flask, request, jsonify
import PyPDF2
import faiss
import numpy as np
from openai import OpenAI

# ---------------- APP SETUP ----------------
app = Flask(__name__)
client = OpenAI()

# ---------------- GLOBAL STORAGE ----------------
chunks = []
index = None

# ---------------- CHAT MEMORY ----------------
conversation_memory = []
MAX_MEMORY = 6   # store last 6 turns only

# ---------------- SMALL TALK ----------------
def handle_small_talk(user_input: str):
    text = user_input.lower().strip()

    greetings = [
        "hi", "hello", "hey",
        "good morning", "good afternoon", "good evening"
    ]

    closing = [
        "bye", "thank you", "thanks",
        "ok thank you", "ok thanks", "that's all"
    ]

    for g in greetings:
        if text == g or text.startswith(g):
            return "Hi 👋 How can I help you?"

    for c in closing:
        if c in text:
            return "You're welcome 😊 Feel free to ask anytime."

    return None

# ---------------- CHUNKING ----------------
def make_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ---------------- EMBEDDINGS ----------------
def get_embeddings(chunks):
    vectors = []

    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        vectors.append(response.data[0].embedding)

    return np.array(vectors).astype("float32")

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dimension)
    idx.add(embeddings)
    return idx

# ---------------- PDF UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global chunks, index, conversation_memory

    conversation_memory = []  # reset memory on new PDF

    if "pdf" not in request.files:
        return jsonify({"message": "❌ PDF file missing"}), 400

    pdf = request.files["pdf"]

    if not pdf.filename.endswith(".pdf"):
        return jsonify({"message": "❌ Please upload a valid PDF file"}), 400

    reader = PyPDF2.PdfReader(pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        return jsonify({"message": "❌ No readable text found in PDF"}), 400

    chunks = make_chunks(text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    return jsonify({"message": "✅ PDF processed successfully!"})

# ---------------- ASK QUESTION ----------------
@app.route("/ask", methods=["POST"])
def ask():
    global chunks, index, conversation_memory

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "❌ Question cannot be empty."})

    # ---- Small talk handling ----
    small_talk = handle_small_talk(question)
    if small_talk:
        return jsonify({"answer": small_talk})

    if index is None:
        return jsonify({"answer": "❌ Please upload a PDF first."})

    # ---- Embed question ----
    q_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    q_embedding = np.array([q_embedding]).astype("float32")

    distances, indices = index.search(q_embedding, 3)

    if distances[0][0] > 1.4 and not conversation_memory:
        return jsonify({"answer": "❌ Please ask a valid question related to the document."})

    # ---- Build context ----
    context = "\n\n".join([chunks[i] for i in indices[0]])

    # ---- Build memory text ----
    memory_text = ""
    for m in conversation_memory:
        memory_text += f"User: {m['question']}\nAssistant: {m['answer']}\n\n"

    # ---- Prompt ----
    prompt = f"""
You are a helpful assistant.

Previous conversation:
{memory_text}

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    # ---- Save to memory ----
    conversation_memory.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_memory) > MAX_MEMORY:
        conversation_memory.pop(0)

    return jsonify({"answer": answer})

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
