from flask import Flask, request, jsonify
import PyPDF2
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------- APP SETUP ----------------
app = Flask(__name__)
client = OpenAI()

# ---------------- LOAD EMBEDDING MODEL ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- FILE PATHS ----------------
CHUNKS_FILE = "chunks.pkl"
INDEX_FILE = "faiss.index"

# ---------------- GLOBAL STORAGE ----------------
chunks = []
index = None

# ---------------- CHAT MEMORY ----------------
conversation_memory = []
MAX_MEMORY = 6

# ---------------- RETRIEVAL SETTINGS ----------------
TOP_K = 10                  
DISTANCE_THRESHOLD = 1.4    
CHUNK_SIZE = 450            
CHUNK_OVERLAP = 120       
# ---------------- SMALL TALK ----------------
def handle_small_talk(user_input: str):
    text = user_input.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    closing = ["bye", "thank you", "thanks", "ok thank you", "ok thanks", "that's all"]
    for g in greetings:
        if text == g or text.startswith(g + " ") or text.startswith(g + ","):
            return "Hi 👋 I'm your dermatology assistant. How can I help you with skin-related questions?"
    for c in closing:
        if c in text:
            return "You're welcome 😊 Feel free to ask about any skin concerns anytime!"
    return None

# ---------------- TOPIC RELEVANCE CHECK (GPT only, no keywords) ----------------
def is_skin_related(question: str) -> bool:
    try:
        check_response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a topic classifier. "
                        "Reply with only 'YES' if the question is related to skin, dermatology, "
                        "skincare, hair, scalp, nails, skin conditions, diseases, treatments, "
                        "cosmetics, or general health questions that may involve skin. "
                        "Be lenient — if unsure, reply YES. "
                        "Reply with only 'NO' if the question is clearly unrelated (e.g. math, coding, geography)."
                    )
                },
                {"role": "user", "content": question}
            ]
        )
        verdict = check_response.choices[0].message.content.strip().upper()
        return verdict != "NO"
    except Exception:
        return True 

# ---------------- CHUNKING ----------------
def make_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        result.append(text[start:end])
        start += chunk_size - overlap
    return result

# ---------------- BUILD FAISS ----------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dimension)
    idx.add(embeddings)
    return idx

# ---------------- LOAD SAVED DATA ----------------
def load_saved_data():
    global chunks, index
    if os.path.exists(CHUNKS_FILE) and os.path.exists(INDEX_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        index = faiss.read_index(INDEX_FILE)
        print("✅ Loaded saved embeddings and index")
    else:
        print("⚠ No saved embeddings found")

# Load at startup
load_saved_data()

# ---------------- BUILD MEMORY TEXT ----------------
def build_memory_text() -> str:
    if not conversation_memory:
        return ""
    lines = []
    for m in conversation_memory:
        lines.append(f"User: {m['question']}\nAssistant: {m['answer']}")
    return "\n\n".join(lines) + "\n\n"

# ---------------- PDF UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global chunks, index, conversation_memory
    conversation_memory = []  # Reset memory on new PDF upload

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

    # Create Chunks
    chunks = make_chunks(text)

    # Create Embeddings
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS
    index = build_faiss_index(embeddings)

    # Save to Disk
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_FILE)

    return jsonify({
        "message": f"✅ PDF processed successfully! {len(chunks)} chunks created and embeddings saved."
    })

# ---------------- ASK QUESTION ----------------
@app.route("/ask", methods=["POST"])
def ask():
    global chunks, index, conversation_memory

    data = request.get_json()
    if not data:
        return jsonify({"answer": "❌ Invalid JSON payload."})

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "❌ Question cannot be empty."})

    # --- Small talk check ---
    small_talk = handle_small_talk(question)
    if small_talk:
        return jsonify({"answer": small_talk})

    # --- Topic relevance check ---
    # Combine with last question for follow-up context awareness
    combined_question = question
    if conversation_memory:
        last_question = conversation_memory[-1]["question"]
        combined_question = last_question + " " + question

    if not is_skin_related(combined_question):
        return jsonify({
            "answer": (
                "🩺 I'm a dermatology assistant and can only help with skin, "
                "hair, nail, or skincare-related questions. "
                "Please ask something related to skin health or conditions!"
            )
        })

    # --- PDF check ---
    if index is None or not chunks:
        return jsonify({"answer": "❌ Please upload a PDF first so I can answer your question."})

    # --- Build contextual query for retrieval ---
    recent_memory = []
    for m in conversation_memory[-3:]:
        recent_memory.append(m["question"])
        recent_memory.append(m["answer"])

    retrieval_query = " ".join(recent_memory + [question])

    # --- Embed Question with Context ---
    q_embedding = embedding_model.encode([retrieval_query])
    q_embedding = np.array(q_embedding).astype("float32")

    # Retrieve TOP_K chunks, filter by distance threshold
    k = min(TOP_K, len(chunks))
    distances, indices = index.search(q_embedding, k)

    context_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks) and dist <= DISTANCE_THRESHOLD:
            context_chunks.append(chunks[idx])

    # Fallback: if all chunks were filtered, use top 3 anyway
    if not context_chunks:
        context_chunks = [chunks[i] for i in indices[0][:3] if i < len(chunks)]

    context = "\n\n".join(context_chunks)

    # --- Build Memory ---
    memory_text = build_memory_text()

    # --- Build Prompt (less strict) ---
    prompt = f"""You are a helpful dermatology assistant. Answer the user's question using the provided context from the uploaded PDF.

Guidelines:
- Prefer answers from the provided context.
- If the context partially covers the topic, answer based on what is available and mention any limitations.
- Only say "Information not available in knowledge base." if the context has absolutely NO relevant information.
- Be clear, concise, and helpful.
- If the question references a previous answer (e.g. "tell me more", "explain that"), use the conversation history to understand what was asked before.
- Do not make up clinical facts not supported by the context.

{f"Previous Conversation:{chr(10)}{memory_text}" if memory_text else ""}
Context from PDF:
{context}

Current Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful dermatology assistant. "
                        "Answer based on the provided context. "
                        "Be informative and helpful — only say information is unavailable "
                        "if the context truly has nothing relevant."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"answer": f"❌ Error generating answer: {str(e)}"})

    # --- Save to Memory ---
    conversation_memory.append({"question": question, "answer": answer})
    if len(conversation_memory) > MAX_MEMORY:
        conversation_memory.pop(0)

    return jsonify({"answer": answer})

# ---------------- CLEAR MEMORY ----------------
@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    global conversation_memory
    conversation_memory = []
    return jsonify({"message": "✅ Conversation memory cleared."})

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "pdf_loaded": index is not None,
        "chunks_count": len(chunks),
        "memory_length": len(conversation_memory)
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)