from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import threading
import time
import sqlite3
import uuid
from dotenv import load_dotenv
import mimetypes
import imghdr
from typing import TypedDict
import pdfplumber

from crewai.llm import LLM
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langgraph.graph import StateGraph, END

app = Flask(__name__)

# Load environment variables from a .env file if present
load_dotenv()

# LLM / Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini/gemini-2.5-flash")
BASE_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
os.environ["CREWAI_TELEMETRY"] = os.getenv("CREWAI_TELEMETRY", "False")

# App/server configuration
APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('APP_PORT', '5000'))
APP_DEBUG = os.getenv('APP_DEBUG', 'False').lower() in ['1', 'true', 'yes', 'on']
USE_WAITRESS = os.getenv('USE_WAITRESS', 'False').lower() in ['1', 'true', 'yes', 'on']

# Upload configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = set([
	'txt', 'md', 'pdf', 'csv', 'json', 'xml', 'yaml', 'yml',
	'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'
])
MAX_CONTENT_LENGTH_MB = int(os.getenv('MAX_UPLOAD_MB', '25'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

# Global variables for tracking crew execution
crew_status = {
    "is_running": False,
    "current_agent": None,
    "current_task": None,
    "progress": 0,
    "output": "",
    "error": None
}

# Database initialization
def init_db():
    conn = sqlite3.connect('crewai_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_history (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            output TEXT,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT (datetime('now', 'utc')),
            completed_at TIMESTAMP,
            error_message TEXT
        )
    ''')

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT (datetime('now', 'utc'))
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            task_id TEXT,
            created_at TIMESTAMP DEFAULT (datetime('now', 'utc')),
            FOREIGN KEY(thread_id) REFERENCES chat_threads(id)
        )
        """
    )

    # Lightweight migration: add thread_id to task_history (if missing)
    try:
        cursor.execute("PRAGMA table_info(task_history)")
        cols = [r[1] for r in cursor.fetchall()]
        if "thread_id" not in cols:
            cursor.execute("ALTER TABLE task_history ADD COLUMN thread_id TEXT")
    except Exception:
        # If migration fails, continue without blocking app startup
        pass
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize LLMs
try:    
    planner_llm = LLM(
        model=MODEL_NAME,
        api_key=GEMINI_API_KEY,
        temperature=BASE_TEMPERATURE,
        verbose=True,
    )
    writer_llm = LLM(
        model=MODEL_NAME,
        api_key=GEMINI_API_KEY,
        temperature=BASE_TEMPERATURE,
        verbose=True,
    )
    reviewer_llm = LLM(
        model=MODEL_NAME,
        api_key=GEMINI_API_KEY,
        temperature=BASE_TEMPERATURE,
        verbose=True,
    )
except Exception as e:
    print(f"Error initializing LLMs: {str(e)}")
    planner_llm = writer_llm = reviewer_llm = None


# Initialize ChromaDB for semantic history
try:
    chroma_client = chromadb.PersistentClient(
        path=os.getenv("CHROMA_DB_PATH", "chroma_db")
    )
    history_collection = chroma_client.get_or_create_collection(
        name="task_history",
        embedding_function=DefaultEmbeddingFunction(),
    )
    documents_collection = chroma_client.get_or_create_collection(
        name="uploaded_documents",
        embedding_function=DefaultEmbeddingFunction(),
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {str(e)}")
    history_collection = None
    documents_collection = None


class GraphState(TypedDict):
    prompt: str 
    attachments_note: str
    history_context: str
    chat_history: str
    plan: str
    draft: str
    final: str
    uploaded_context: str


def _ensure_thread(thread_id: str) -> None:
    conn = sqlite3.connect("crewai_history.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_threads (id) VALUES (?)
        ON CONFLICT(id) DO NOTHING
        """,
        (thread_id,),
    )
    conn.commit()
    conn.close()


def _add_chat_message(thread_id: str, role: str, content: str, task_id: str | None = None) -> None:
    conn = sqlite3.connect("crewai_history.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_messages (thread_id, role, content, task_id)
        VALUES (?, ?, ?, ?)
        """,
        (thread_id, role, content, task_id),
    )
    conn.commit()
    conn.close()


def _get_chat_history_text(thread_id: str, max_messages: int = 12) -> str:
    """Return recent messages formatted for prompt context."""
    conn = sqlite3.connect("crewai_history.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content
        FROM chat_messages
        WHERE thread_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (thread_id, max_messages),
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return ""
    rows.reverse()
    parts = []
    for role, content in rows:
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)


def add_task_to_vector_store(task_id, prompt, output, status):
    """Store completed tasks in ChromaDB for semantic history lookup."""
    if history_collection is None:
        return
    try:
        document = f"Prompt: {prompt}\n\nOutput:\n{output or ''}\n\nStatus: {status}"
        history_collection.add(
            ids=[task_id],
            documents=[document],
            metadatas=[{"status": status}],
        )
    except Exception as e:
        # Log but do not interrupt main flow
        print(f"Error saving task {task_id} to ChromaDB: {str(e)}")


def retrieve_similar_history(query: str, top_k: int = 3) -> str:
    """Retrieve semantically similar past tasks from ChromaDB."""
    if history_collection is None:
        return ""
    try:
        results = history_collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])
        if not docs or not docs[0]:
            return ""
        flattened = docs[0]
        return "\n\n---\n\n".join(flattened)
    except Exception as e:
        print(f"Error querying ChromaDB: {str(e)}")
        return ""


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    """Cap prompt context length to keep model latency predictable."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Simple char-based chunking with overlap for retrieval."""
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def _extract_text_from_file(file_path: str, max_chars: int = 50000) -> str:
    """Extract text only (no OCR) from supported upload types."""
    basename = os.path.basename(file_path)
    ext = os.path.splitext(basename)[1].lower().lstrip(".")
    try:
        # Plain text-like formats
        if ext in ["txt", "md", "yaml", "yml", "xml", "html"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return _truncate_for_prompt(f.read(), max_chars)

        # JSON: store the raw JSON as text (still grounded)
        if ext == "json":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            dumped = json.dumps(data, ensure_ascii=False)
            return _truncate_for_prompt(dumped, max_chars)

        # CSV: store header + rows as text (cap size)
        if ext == "csv":
            import csv

            rows: list[str] = []
            total_chars = 0
            with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= 2000:  # hard stop to keep ingestion bounded
                        break
                    line = ",".join(row)
                    if line:
                        rows.append(line)
                        total_chars += len(line)
                        if total_chars >= max_chars:
                            break
            return _truncate_for_prompt("\n".join(rows), max_chars)

        # PDF: extract textual content only (no OCR)
        if ext == "pdf":
            full_text_parts: list[str] = []
            total_chars = 0
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        continue
                    full_text_parts.append(page_text)
                    total_chars += len(page_text)
                    if total_chars >= max_chars:
                        break
            return _truncate_for_prompt("\n".join(full_text_parts), max_chars)

        # Images and other binary types: no extracted text (no OCR)
        return ""
    except Exception as e:
        print(f"Error extracting text from {basename}: {str(e)}")
        return ""


def delete_uploaded_documents(thread_id: str) -> None:
    """Delete previously ingested chunks for this thread."""
    if documents_collection is None:
        return
    try:
        documents_collection.delete(where={"thread_id": thread_id})
    except Exception as e:
        # If delete fails, ingestion will still proceed (best-effort).
        print(f"Error deleting uploaded docs for thread {thread_id}: {str(e)}")


def ingest_uploaded_documents(thread_id: str, file_paths: list[str]) -> int:
    """Ingest extracted text chunks from uploads into Chroma for RAG."""
    if documents_collection is None:
        return 0

    if not file_paths:
        return 0

    delete_uploaded_documents(thread_id)

    max_chunks_total = int(os.getenv("RAG_MAX_TOTAL_CHUNKS", "60"))
    max_chunks_per_file = int(os.getenv("RAG_MAX_CHUNKS_PER_FILE", "30"))

    docs_added = 0
    for file_path in file_paths:
        source = os.path.basename(file_path)
        text = _extract_text_from_file(file_path)
        if not text.strip():
            continue

        chunks = _chunk_text(text)
        if not chunks:
            continue

        # Cap chunks to keep ingestion and prompts bounded
        chunks = chunks[:max_chunks_per_file]
        for chunk_index, chunk in enumerate(chunks):
            if docs_added >= max_chunks_total:
                return docs_added
            doc_id = f"{thread_id}:{source}:{chunk_index}"
            documents_collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[{"thread_id": thread_id, "source": source, "chunk_index": chunk_index}],
            )
            docs_added += 1

    return docs_added


def retrieve_uploaded_context(query: str, thread_id: str, top_k: int = 4) -> str:
    """Retrieve relevant chunks from uploaded files only."""
    if documents_collection is None:
        return ""
    if not query.strip():
        return ""

    try:
        results = documents_collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"thread_id": thread_id},
        )
        docs = results.get("documents", [[]])
        if not docs or not docs[0]:
            return ""
        return "\n\n---\n\n".join(docs[0])
    except Exception as e:
        print(f"Error retrieving uploaded context: {str(e)}")
        return ""


def planner_node(state: GraphState) -> GraphState:
    """LangGraph planner node."""
    if planner_llm is None:
        raise RuntimeError("Planner LLM is not initialized")

    global crew_status
    crew_status["current_agent"] = "Planner"
    crew_status["current_task"] = "Breaking down your request into subtasks..."
    crew_status["progress"] = 20

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert planner for answering the user. "
                "Create a brief plan (2-4 short steps)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{state['prompt']}"
            ),
        },
    ]
    t0 = time.perf_counter()
    plan = planner_llm.call(messages)
    crew_status.setdefault("timings", {})
    crew_status["timings"]["planner_ms"] = (time.perf_counter() - t0) * 1000
    new_state: GraphState = dict(state)
    new_state["plan"] = str(plan)
    return new_state


def writer_node(state: GraphState) -> GraphState:
    """LangGraph writer node."""
    if writer_llm is None:
        raise RuntimeError("Writer LLM is not initialized")

    global crew_status
    crew_status["current_agent"] = "Writer"
    crew_status["current_task"] = "Generating detailed content..."
    crew_status["progress"] = 50

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded writer. Answer using ONLY the provided "
                "`uploaded_context` (extracted text from user uploads). "
                "If `uploaded_context` is empty, answer normally. "
                "Otherwise, if the answer is not present in the uploaded_context, "
                "respond exactly: \"I couldn't find that in the uploaded text.\""
            ),
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{state['prompt']}\n\n"
                f"Plan:\n{state.get('plan', '')}\n\n"
                f"uploaded_context:\n{_truncate_for_prompt(state.get('uploaded_context') or '', 6000)}"
            ),
        },
    ]
    t0 = time.perf_counter()
    draft = writer_llm.call(messages)
    crew_status.setdefault("timings", {})
    crew_status["timings"]["writer_ms"] = (time.perf_counter() - t0) * 1000
    new_state: GraphState = dict(state)
    new_state["draft"] = str(draft)
    return new_state


def reviewer_node(state: GraphState) -> GraphState:
    """LangGraph reviewer node."""
    if reviewer_llm is None:
        raise RuntimeError("Reviewer LLM is not initialized")

    global crew_status
    crew_status["current_agent"] = "Reviewer"
    crew_status["current_task"] = "Reviewing and polishing content..."
    crew_status["progress"] = 80

    draft_text = _truncate_for_prompt(state.get("draft", ""), 5000)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a finalizer. Polish the draft into the final answer. "
                "Do not add any new facts beyond what is already in the draft. "
                "Return ONLY the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{state['prompt']}\n\n"
                f"Draft answer:\n{draft_text}\n\n"
                "Rewrite this into the final user-facing answer."
            ),
        },
    ]
    crew_status.setdefault("timings", {})
    crew_status["timings"]["reviewer_retry"] = False
    t0 = time.perf_counter()
    final = reviewer_llm.call(messages)
    crew_status["timings"]["reviewer_first_ms"] = (time.perf_counter() - t0) * 1000
    final_text = str(final).strip()

    # Guard against reviewer-style meta output leaking to user.
    meta_phrases = (
        "the draft answer",
        "no changes are needed",
        "this draft",
        "the response is ready",
    )
    if any(p in final_text.lower() for p in meta_phrases):
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "Return only the final answer to the user request below. "
                    "Never critique the draft. Never mention drafting or reviewing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request:\n{state['prompt']}\n\n"
                    f"Draft content:\n{draft_text}"
                ),
            },
        ]
        crew_status["timings"]["reviewer_retry"] = True
        t1 = time.perf_counter()
        final = reviewer_llm.call(retry_messages)
        crew_status["timings"]["reviewer_retry_ms"] = (time.perf_counter() - t1) * 1000
    new_state: GraphState = dict(state)
    new_state["final"] = str(final)
    return new_state


def build_workflow():
    """Build the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", END)
    return workflow.compile()
    

workflow = build_workflow()


def save_task_to_history(task_id, prompt, status, output=None, error_message=None, thread_id: str | None = None):
    """Save task to history database and index completed tasks in ChromaDB."""
    conn = sqlite3.connect("crewai_history.db")
    cursor = conn.cursor()

    # Use UTC time to avoid timezone issues
    completed_at = datetime.utcnow().isoformat() if status in ["completed", "error"] else None

    cursor.execute(
        """
        INSERT INTO task_history (id, prompt, output, status, completed_at, error_message, thread_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            prompt = excluded.prompt,
            output = excluded.output,
            status = excluded.status,
            completed_at = excluded.completed_at,
            error_message = excluded.error_message,
            thread_id = excluded.thread_id
        """,
        (task_id, prompt, output, status, completed_at, error_message, thread_id),
    )

    conn.commit()
    conn.close()

    # Store only completed tasks in ChromaDB for semantic history
    if status == "completed":
        t0 = time.perf_counter()
        add_task_to_vector_store(task_id, prompt, output or "", status)
        crew_status.setdefault("timings", {})
        crew_status["timings"]["chroma_add_ms"] = crew_status["timings"].get("chroma_add_ms", 0) + (
            (time.perf_counter() - t0) * 1000
        )

def run_crew_async(user_prompt, task_id, attachments=None, thread_id: str | None = None):
    """Run the LangGraph workflow in a separate thread."""
    global crew_status
    
    try:
        crew_status["is_running"] = True
        crew_status["error"] = None
        crew_status["output"] = ""
        crew_status["progress"] = 0
        crew_status["timings"] = {}
        
        # RAG uses extracted text chunks; we don't inject attachment summaries into prompts.
        attachments_note = ""

        # Conversation thread memory (SQLite)
        if not thread_id:
            thread_id = str(uuid.uuid4())
        _ensure_thread(thread_id)
        chat_history = ""

        # Make uploaded files visible in the conversation thread UI.
        if attachments:
            file_names = [os.path.basename(p) for p in attachments if p]
            if file_names:
                _add_chat_message(
                    thread_id,
                    "user",
                    "Uploaded files:\n" + "\n".join(file_names),
                    task_id=task_id,
                )

        # Add the actual user question after the uploaded files list.
        _add_chat_message(thread_id, "user", user_prompt, task_id=task_id)

        # Save initial task to history as running (linked to thread)
        save_task_to_history(task_id, user_prompt, "running", thread_id=thread_id)

        # RAG: ingest uploaded docs and retrieve relevant chunks for grounding
        t0 = time.perf_counter()
        ingest_uploaded_documents(thread_id, attachments or [])
        crew_status["timings"]["rag_ingest_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        uploaded_context = retrieve_uploaded_context(user_prompt, thread_id)
        crew_status["timings"]["rag_retrieve_ms"] = (time.perf_counter() - t1) * 1000

        history_context = ""
        initial_state: GraphState = {
            "prompt": user_prompt,
            "attachments_note": attachments_note,
            "history_context": history_context,
            "chat_history": chat_history,
            "plan": "",
            "draft": "",
            "final": "",
            "uploaded_context": uploaded_context,
        }

        crew_status["current_task"] = "Executing LangGraph workflow..."
        crew_status["progress"] = 10

        # Run the LangGraph workflow
        final_state = workflow.invoke(initial_state)
        final_output = (
            final_state.get("final")
            or final_state.get("draft")
            or final_state.get("plan")
            or ""
        )
        
        crew_status["output"] = str(final_output)
        crew_status["progress"] = 100
        crew_status["current_agent"] = "Complete"
        crew_status["current_task"] = "Task completed successfully!"
        
        # Save completed task to history
        save_task_to_history(task_id, user_prompt, "completed", str(final_output), thread_id=thread_id)

        # Save assistant response to conversation thread
        _add_chat_message(thread_id, "assistant", str(final_output), task_id=task_id)
        
    except Exception as e:
        crew_status["error"] = str(e)
        crew_status["current_agent"] = "Error"
        crew_status["current_task"] = "An error occurred during execution"
        
        # Save failed task to history
        save_task_to_history(task_id, user_prompt, "error", error_message=str(e), thread_id=thread_id)
        
    finally:
        crew_status["is_running"] = False

@app.route('/')
def index():
    return render_template('index.html')

def _allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/start_crew', methods=['POST'])
def start_crew():
    try:
        attachments = []
        user_prompt = ''
        thread_id = None
        
        # Support both JSON and multipart form submissions
        if request.content_type and request.content_type.startswith('application/json'):
            data = request.get_json()
            user_prompt = (data or {}).get('prompt', '')
            thread_id = (data or {}).get("thread_id") or None
        else:
            user_prompt = (request.form.get('prompt') or '').strip()
            thread_id = (request.form.get("thread_id") or "").strip() or None
            # Handle file uploads
            if 'files' in request.files:
                files = request.files.getlist('files')
                for file in files:
                    if file and _allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        # Avoid overwriting existing files by adding a suffix if needed
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(save_path):
                            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{counter}{ext}")
                            counter += 1
                        file.save(save_path)
                        attachments.append(save_path)

        if not user_prompt.strip():
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        if crew_status["is_running"]:
            return jsonify({'error': 'Crew is already running. Please wait for completion.'}), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Start crew in background thread
        if not thread_id:
            thread_id = str(uuid.uuid4())

        thread = threading.Thread(
            target=run_crew_async, args=(user_prompt, task_id, attachments, thread_id)
        )
        thread.daemon = True
        thread.start()
        
        # If no thread_id was provided, create one for this conversation.
        if not thread_id:
            thread_id = str(uuid.uuid4())

        return jsonify(
            {
                "message": "Crew started successfully",
                "task_id": task_id,
                "thread_id": thread_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/threads/<thread_id>/messages")
def get_thread_messages(thread_id: str):
    """Get recent messages for a conversation thread."""
    try:
        conn = sqlite3.connect("crewai_history.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content, task_id, created_at
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY id ASC
            LIMIT 200
            """,
            (thread_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        messages = [
            {"role": r[0], "content": r[1], "task_id": r[2], "created_at": r[3]}
            for r in rows
        ]
        return jsonify({"thread_id": thread_id, "messages": messages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/crew_status')
def get_crew_status():
    """Get current crew execution status"""
    return jsonify(crew_status)

@app.route('/crew_output')
def get_crew_output():
    """Get final crew output"""
    return jsonify({
        'output': crew_status['output'],
        'is_complete': not crew_status['is_running'] and crew_status['output'] != '',
        'error': crew_status['error']
    })

@app.route('/reset_crew', methods=['POST'])
def reset_crew():
    """Reset crew status"""
    global crew_status
    crew_status = {
        "is_running": False,
        "current_agent": None,
        "current_task": None,
        "progress": 0,
        "output": "",
        "error": None
    }
    return jsonify({'message': 'Crew status reset successfully'})

@app.route('/history')
def get_history():
    """Get task history"""
    try:
        conn = sqlite3.connect('crewai_history.db')
        cursor = conn.cursor()
        
        # Get all tasks ordered by creation date (newest first)
        cursor.execute('''
            SELECT id, prompt, output, status, created_at, completed_at, error_message
            FROM task_history
            ORDER BY created_at DESC
            LIMIT 50
        ''')
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                'id': row[0],
                'prompt': row[1],
                'output': row[2],
                'status': row[3],
                'created_at': row[4],
                'completed_at': row[5],
                'error_message': row[6]
            })
        
        conn.close()
        return jsonify({'tasks': tasks})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/threads")
def list_threads():
    """List chat threads (one row per conversation)."""
    try:
        conn = sqlite3.connect("crewai_history.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT t.id, t.created_at,
                   COALESCE(m_last.content, '') AS last_message,
                   COALESCE(m_last.role, '') AS last_role
            FROM chat_threads t
            LEFT JOIN (
                SELECT cm.thread_id, cm.content, cm.role
                FROM chat_messages cm
                INNER JOIN (
                    SELECT thread_id, MAX(id) AS max_id
                    FROM chat_messages
                    GROUP BY thread_id
                ) mx
                ON mx.thread_id = cm.thread_id AND mx.max_id = cm.id
            ) m_last
            ON m_last.thread_id = t.id
            ORDER BY t.created_at DESC
            LIMIT 50
            """
        )
        rows = cursor.fetchall()
        conn.close()
        threads = [
            {
                "thread_id": r[0],
                "created_at": r[1],
                "last_message": r[2],
                "last_role": r[3],
            }
            for r in rows
        ]
        return jsonify({"threads": threads})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/threads/<thread_id>", methods=["DELETE"])
def delete_thread(thread_id: str):
    """Delete an entire conversation thread and its messages."""
    try:
        conn = sqlite3.connect("crewai_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
        conn.commit()
        conn.close()
        return jsonify({"message": "Thread deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history/<task_id>')
def get_task_details(task_id):
    """Get details of a specific task"""
    try:
        conn = sqlite3.connect('crewai_history.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, prompt, output, status, created_at, completed_at, error_message
            FROM task_history
            WHERE id = ?
        ''', (task_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                'id': row[0],
                'prompt': row[1],
                'output': row[2],
                'status': row[3],
                'created_at': row[4],
                'completed_at': row[5],
                'error_message': row[6]
            })
        else:
            return jsonify({'error': 'Task not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a specific task from history"""
    try:
        conn = sqlite3.connect('crewai_history.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM task_history WHERE id = ?', (task_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Task deleted successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clear all history (tasks + chats + vector store)."""
    try:
        conn = sqlite3.connect('crewai_history.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM chat_messages')
        cursor.execute('DELETE FROM chat_threads')
        cursor.execute('DELETE FROM task_history')
        conn.commit()
        conn.close()

        # Also clear ChromaDB semantic history (best-effort)
        try:
            # Chroma versions can reject delete(where={}); delete by ids instead.
            if history_collection is not None:
                res = history_collection.get()
                ids = res.get("ids", []) if isinstance(res, dict) else []
                if ids:
                    history_collection.delete(ids=ids)

            if documents_collection is not None:
                res = documents_collection.get()
                ids = res.get("ids", []) if isinstance(res, dict) else []
                if ids:
                    documents_collection.delete(ids=ids)
        except Exception as chroma_error:
            print(f"Error clearing ChromaDB: {str(chroma_error)}")
        
        return jsonify({'message': 'History cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    if USE_WAITRESS:
        try:
            from waitress import serve
        except Exception as import_error:
            raise RuntimeError("USE_WAITRESS is enabled but 'waitress' is not installed. Run: pip install waitress") from import_error
        serve(app, host=APP_HOST, port=APP_PORT)
    else:
        app.run(debug=APP_DEBUG, host=APP_HOST, port=APP_PORT)