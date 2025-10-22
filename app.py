from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from langchain.memory import ConversationBufferMemory
import threading
import time
import sqlite3
import uuid
from dotenv import load_dotenv
import mimetypes
import imghdr

app = Flask(__name__)

# Load environment variables from a .env file if present
load_dotenv()

# Perplexity API configuration
API_KEY = os.getenv('PERPLEXITY_API_KEY')
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
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize memory and LLM
memory = ConversationBufferMemory(return_messages=True)

try:
    planner_llm = LLM(
        provider="perplexity",
        model="perplexity/sonar",
        api_key=API_KEY,
        temperature=0.7,
        verbose=True
    )
    writer_llm = LLM(
        provider="perplexity",
        model="perplexity/sonar",
        api_key=API_KEY,
        temperature=0.3,
        verbose=True
    )
    reviewer_llm = LLM(
        provider="perplexity",
        model="perplexity/sonar",
        api_key=API_KEY,
        temperature=0.1,
        verbose=True
    )
except Exception as e:
    print(f"Error initializing LLMs: {str(e)}")
    planner_llm = writer_llm = reviewer_llm = None

# Initialize agents
planner = Agent(
    role="Planner",
    goal="Break down user prompts into structured subtasks.",
    backstory="A strategic planner that turns ideas into action plans.",
    memory=True,
    verbose=True,
    llm=planner_llm
)

writer = Agent(
    role="Writer",
    goal="Generate detailed and clear content from research.",
    backstory="A writer who transforms insights into high-quality outputs.",
    memory=True,
    verbose=True,
    llm=writer_llm
)

reviewer = Agent(
    role="Reviewer",
    goal="Ensure clarity, quality, and correctness of final output.",
    backstory="A careful reviewer that improves readability and accuracy.",
    memory=True,
    verbose=True,
    llm=reviewer_llm
)

def save_task_to_history(task_id, prompt, status, output=None, error_message=None):
    """Save task to history database"""
    conn = sqlite3.connect('crewai_history.db')
    cursor = conn.cursor()
    
    # Use UTC time to avoid timezone issues
    completed_at = datetime.utcnow().isoformat() if status in ['completed', 'error'] else None
    
    cursor.execute('''
        INSERT OR REPLACE INTO task_history 
        (id, prompt, output, status, completed_at, error_message)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (task_id, prompt, output, status, completed_at, error_message))
    
    conn.commit()
    conn.close()

def run_crew_async(user_prompt, task_id, attachments=None):
    """Run the crew in a separate thread"""
    global crew_status
    
    try:
        crew_status["is_running"] = True
        crew_status["error"] = None
        crew_status["output"] = ""
        crew_status["progress"] = 0
        
        # Save initial task to history
        save_task_to_history(task_id, user_prompt, "running")
        
        # Task 1: Planner
        crew_status["current_agent"] = "Planner"
        crew_status["current_task"] = "Breaking down your request into subtasks..."
        crew_status["progress"] = 20
        
        attachments_note = ""
        if attachments:
            def summarize_attachment(file_path: str) -> str:
                try:
                    max_chars = 2000
                    basename = os.path.basename(file_path)
                    ext = os.path.splitext(basename)[1].lower().lstrip('.')
                    size_bytes = os.path.getsize(file_path)
                    # Text-like files
                    if ext in ["txt", "md", "yaml", "yml", "xml"]:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(max_chars)
                        snippet = content.strip()
                        return f"{basename} ({ext}, {size_bytes} bytes) snippet:\n" + snippet
                    # JSON
                    if ext == "json":
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            keys = list(data.keys())[:20]
                            return f"{basename} (json, {size_bytes} bytes) object with keys: {keys}"
                        if isinstance(data, list):
                            length = len(data)
                            first = data[0] if length > 0 else None
                            first_snippet = str(first)[:400] if first is not None else 'empty'
                            return f"{basename} (json array, {size_bytes} bytes) length={length}; first item preview: {first_snippet}"
                        return f"{basename} (json, {size_bytes} bytes)"
                    # CSV
                    if ext == "csv":
                        import csv
                        rows_preview = []
                        header = None
                        max_rows = 5
                        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                            reader = csv.reader(f)
                            try:
                                header = next(reader)
                            except StopIteration:
                                header = []
                            for i, row in enumerate(reader):
                                if i >= max_rows:
                                    break
                                rows_preview.append(row)
                        return f"{basename} (csv, {size_bytes} bytes) columns={len(header)} header={header} sample_rows={rows_preview}"
                    # Images
                    if ext in ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"]:
                        try:
                            from PIL import Image  # type: ignore
                            with Image.open(file_path) as img:
                                w, h = img.size
                                mode = img.mode
                                fmt = img.format
                            return f"{basename} (image {fmt}, {w}x{h}, mode={mode}, {size_bytes} bytes)"
                        except Exception:
                            kind = imghdr.what(file_path)
                            return f"{basename} (image {kind or 'unknown'}, {size_bytes} bytes)"
                    # PDFs or others
                    if ext == "pdf":
                        return f"{basename} (pdf, {size_bytes} bytes)"
                    # Fallback
                    mime, _ = mimetypes.guess_type(file_path)
                    return f"{basename} ({mime or 'unknown'}, {size_bytes} bytes)"
                except Exception as summarize_error:
                    return f"{os.path.basename(file_path)} (failed to summarize: {summarize_error})"

            summaries = [summarize_attachment(p) for p in attachments]
            combined = "\n\n".join(summaries)
            # Cap the note length to keep prompts manageable
            max_note_len = 4000
            if len(combined) > max_note_len:
                combined = combined[:max_note_len] + "\n... (truncated)"
            attachments_note = f"\n\nAttachments provided with summaries:\n{combined}"
        
        task1 = Task(
            description=f"Break down: '{user_prompt}' into subtasks. If attachments are provided, plan how to analyze and explain them clearly.{attachments_note}",
            expected_output="A structured list of subtasks.",
            agent=planner
        )
        
        # Task 2: Writer
        crew_status["current_agent"] = "Writer"
        crew_status["current_task"] = "Generating detailed content..."
        crew_status["progress"] = 50
        
        task2 = Task(
            description=f"Write full response to: '{user_prompt}' using inputs above. If files/images were provided, interpret them using the summaries and explain them clearly with any relevant stats or observations.{attachments_note}",
            expected_output="A detailed and complete answer.",
            agent=writer
        )
        
        # Task 3: Reviewer
        crew_status["current_agent"] = "Reviewer"
        crew_status["current_task"] = "Reviewing and polishing content..."
        crew_status["progress"] = 80
        
        task3 = Task(
            description=f"Review written content for clarity and accuracy. Ensure any attached files/images are clearly explained and any claims are consistent with the provided summaries.{attachments_note}",
            expected_output="Polished content with improved clarity and grammar.",
            agent=reviewer
        )
        
        # Create and run crew
        crew = Crew(
            agents=[planner, writer, reviewer],
            tasks=[task1, task2, task3],
            verbose=True
        )
        
        crew_status["current_task"] = "Executing multi-agent workflow..."
        crew_status["progress"] = 90
        
        final_output = crew.kickoff()
        
        crew_status["output"] = str(final_output)
        crew_status["progress"] = 100
        crew_status["current_agent"] = "Complete"
        crew_status["current_task"] = "Task completed successfully!"
        
        # Save completed task to history
        save_task_to_history(task_id, user_prompt, "completed", str(final_output))
        
    except Exception as e:
        crew_status["error"] = str(e)
        crew_status["current_agent"] = "Error"
        crew_status["current_task"] = "An error occurred during execution"
        
        # Save failed task to history
        save_task_to_history(task_id, user_prompt, "error", error_message=str(e))
        
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
        
        # Support both JSON and multipart form submissions
        if request.content_type and request.content_type.startswith('application/json'):
            data = request.get_json()
            user_prompt = (data or {}).get('prompt', '')
        else:
            user_prompt = (request.form.get('prompt') or '').strip()
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
        thread = threading.Thread(target=run_crew_async, args=(user_prompt, task_id, attachments))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Crew started successfully',
            'task_id': task_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    """Clear all task history"""
    try:
        conn = sqlite3.connect('crewai_history.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM task_history')
        conn.commit()
        conn.close()
        
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