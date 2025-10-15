# nano_personal_ai_v6.py
# Personalized Nano-AI v6.0
# Run: pip install fastapi uvicorn python-multipart
# Start: uvicorn nano_personal_ai_v6:app --host 0.0.0.0 --port 8000

import os
import json
import random
import asyncio
import threading
from collections import defaultdict
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

# ---------- CONFIG ----------
MEMORY_DIR = "user_brains"
GLOBAL_FILE = "global_brain.json"   # optional global core data
API_KEY_NAME = "X-API-KEY"
API_KEY = os.environ.get("NANO_API_KEY", "supersecretkey")  # change for production
SAVE_INTERVAL = 30  # seconds autosave interval
MAX_TONE_LEN = 30
# ----------------------------

app = FastAPI(title="Nano Personal AI v6.0")

# ensure dirs
os.makedirs(MEMORY_DIR, exist_ok=True)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
lock = threading.Lock()  # protect disk writes/reads

# ---------- Utility: per-user brain structure ----------
def default_brain():
    # words: adjacency weights; letters optional
    return {
        "context": {},      # key-value (name, likes, tone default)
        "words": {},        # { "hello": {"there": 3, "friend":1}, ... }
        "letters": {},      # { "v": {"i":1, ...}, ... } optional
        "meta": {"tone": "neutral"}  # tone: neutral/friendly/formal/funny
    }

def user_path(user_id: str) -> str:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(MEMORY_DIR, f"{safe}.json")

def load_brain(user_id: str) -> Dict[str, Any]:
    path = user_path(user_id)
    if os.path.exists(path):
        with lock:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    else:
        return default_brain()

def save_brain(user_id: str, brain: Dict[str, Any]) -> None:
    path = user_path(user_id)
    tmp = path + ".tmp"
    with lock:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(brain, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

def list_user_ids():
    files = os.listdir(MEMORY_DIR)
    ids = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    return ids

# ---------- Low-level learning functions ----------
def update_word_connections(brain: Dict[str, Any], text: str, weight_inc: float = 1.0):
    words = [w for w in text.strip().split() if w]
    for i in range(len(words)-1):
        a, b = words[i].lower(), words[i+1].lower()
        brain["words"].setdefault(a, {})
        brain["words"][a][b] = brain["words"][a].get(b, 0.0) + weight_inc
        # clamp or normalize optionally

def update_letter_connections(brain: Dict[str, Any], text: str, weight_inc: float = 0.1):
    # optional: keep small letter-level data
    for word in text.strip().split():
        letters = list(word)
        for i in range(len(letters)-1):
            a, b = letters[i], letters[i+1]
            brain["letters"].setdefault(a, {})
            brain["letters"][a][b] = brain["letters"][a].get(b, 0.0) + weight_inc

def learn_context_from_text(brain: Dict[str, Any], text: str):
    tl = text.lower()
    words = tl.split()
    # simple patterns
    if "my name is" in tl:
        name = tl.split("my name is")[-1].strip().split()[0]
        if name: brain["context"]["name"] = name
    elif "i am" in tl and "am" in words:
        try:
            i = words.index("am")
            if i+1 < len(words): brain["context"]["you"] = words[i+1]
        except ValueError:
            pass
    # likes
    if len(words) > 2 and words[0] == "i" and words[1] in ["like", "love"]:
        brain["context"]["likes"] = words[2]

# ---------- High-level behavior ----------
def predict_next_word(brain: Dict[str, Any], word: str) -> str:
    word = word.lower()
    if word in brain["words"] and brain["words"][word]:
        choices = list(brain["words"][word].keys())
        weights = list(brain["words"][word].values())
        return random.choices(choices, weights=weights, k=1)[0]
    # fallback to letter-based expansion
    if word and word[-1] in brain["letters"]:
        next_letter = max(brain["letters"][word[-1]], key=brain["letters"][word[-1]].get)
        return word + next_letter
    return "?"

def build_sentence_from(brain: Dict[str, Any], start_word: str, max_len: int = 8) -> str:
    sentence = [start_word]
    current = start_word
    for _ in range(max_len-1):
        nxt = predict_next_word(brain, current)
        if not nxt or nxt in sentence: break
        sentence.append(nxt)
        current = nxt
    return " ".join(sentence)

# tone templates
def apply_tone(user_brain: Dict[str, Any], base_text: str) -> str:
    tone = user_brain.get("meta", {}).get("tone", "neutral")
    name = user_brain.get("context", {}).get("name") or user_brain.get("context", {}).get("you") or ""
    if tone == "friendly":
        if name: return f"Hey {name}! {base_text} 😊"
        return base_text + " 😊"
    if tone == "funny":
        return base_text + " 😂"
    if tone == "formal":
        if name: return f"Hello {name}. {base_text}"
        return f"{base_text}"
    # neutral or default
    return base_text

# ---------- Modes & Teacher control ----------
async def require_api_key(header_key: str = Depends(api_key_header)):
    if header_key == API_KEY:
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")

def teacher_update(brain: Dict[str, Any], command: str) -> str:
    # simple command grammar: key=value  OR set_tone=funny
    cmd = command.strip()
    if "=" in cmd:
        k, v = cmd.split("=", 1)
        k, v = k.strip(), v.strip()
        # allow writing to context keys or meta.tone
        if k == "tone":
            brain["meta"]["tone"] = v[:MAX_TONE_LEN]
            return f"tone set to {v}"
        else:
            brain["context"][k] = v
            return f"context {k} set to {v}"
    return "invalid command"

# ---------- API Endpoints ----------

# Serve a simple static UI if static directory exists
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return "<html><body><h3>Nano Personal AI v6.0 — no static UI found. Use /chat endpoint.</h3></body></html>"

@app.post("/chat")
async def chat_endpoint(payload: Dict = Body(...)):
    """
    User chat:
    payload: { "user_id": "vishnu", "message": "My name is Vishnu", "mode": "user" }
    mode: "user" (default) => learning-only (no destructive edits)
          "teacher" (only allowed via /teach endpoint with API key) => not used here
    """
    user_id = str(payload.get("user_id", "guest")).strip() or "guest"
    message = str(payload.get("message", "")).strip()
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    brain = load_brain(user_id)

    # Normal user learning: learn patterns and context, but DO NOT auto-overwrite critical keys that teacher controls
    learn_context_from_text(brain, message)
    update_word_connections(brain, message, weight_inc=1.0)
    update_letter_connections(brain, message, weight_inc=0.05)

    # Optional small auto-adjust: decay rarely used edges (keeps brain fresh)
    # (light-weight: reduce all weights slightly)
    for a, targets in brain["words"].items():
        for b in list(targets.keys()):
            brain["words"][a][b] = max(0.0, brain["words"][a][b] * 0.997)

    # generate reply: prefer contextual answers
    # context replies
    tl = message.lower()
    if "who am i" in tl or "who i am" in tl or "what is my name" in tl:
        name = brain.get("context", {}).get("you") or brain.get("context", {}).get("name")
        if name:
            base = f"Your name is {name}."
            resp = apply_tone(brain, base)
            save_brain(user_id, brain)
            return {"reply": resp}
        else:
            base = "I don't know your name yet. You can say: 'My name is ...'"
            return {"reply": apply_tone(brain, base)}

    # Otherwise try transmitter/predictor
    words = message.split()
    start = words[-1] if words else ""
    predicted = build_sentence_from(brain, start) if start else "Hello"
    base_reply = predicted if predicted and predicted != start else random.choice([
        "Tell me more.", "Interesting.", "I see.", "Okay."
    ])
    response = apply_tone(brain, base_reply)

    # save learning
    save_brain(user_id, brain)
    return {"reply": response}

@app.post("/teach", dependencies=[Depends(require_api_key)])
async def teach_endpoint(payload: Dict = Body(...)):
    """
    Teacher endpoint (authorized by API key).
    payload: { "user_id":"vishnu", "cmd":"name=Anwar" }
    This endpoint can modify the stored brain directly.
    """
    user_id = str(payload.get("user_id", "global")).strip() or "global"
    cmd = str(payload.get("cmd", "")).strip()
    if not cmd:
        return JSONResponse({"error": "cmd required"}, status_code=400)
    brain = load_brain(user_id)
    res = teacher_update(brain, cmd)
    save_brain(user_id, brain)
    return {"status": "ok", "result": res}

@app.post("/set_tone", dependencies=[Depends(require_api_key)])
async def set_tone_endpoint(payload: Dict = Body(...)):
    """
    Set user tone securely: payload { "user_id":"vishnu", "tone":"friendly" }
    """
    user_id = str(payload.get("user_id", "guest")).strip() or "guest"
    tone = str(payload.get("tone", "neutral")).strip()[:MAX_TONE_LEN]
    brain = load_brain(user_id)
    brain["meta"]["tone"] = tone
    save_brain(user_id, brain)
    return {"status": "ok", "tone": tone}

@app.get("/memory")
async def memory_endpoint(user_id: str = "guest", api_key: str = Depends(api_key_header)):
    # if no valid API key, still allow user to fetch own memory (no key) — simplified:
    # if api_key matches API_KEY, can fetch any user's memory
    if api_key == API_KEY:
        # admin fetch
        brain = load_brain(user_id)
        return brain
    # else only allow public limited info for the requested user file
    brain = load_brain(user_id)
    safe = {"context": brain.get("context", {}), "meta": brain.get("meta", {})}
    return safe

@app.get("/list_users", dependencies=[Depends(require_api_key)])
async def list_users_endpoint():
    return {"users": list_user_ids()}

@app.post("/save", dependencies=[Depends(require_api_key)])
async def save_all_endpoint():
    # flush nothing special because saves are done per operation, but we can touch global file
    # placeholder to satisfy API
    return {"status": "saved"}

# ---------- Autosave loop (global small housekeeping) ----------
def autosave_background():
    # runs in separate thread to periodically ensure files are synced (no-op because we save on each chat)
    while True:
        try:
            # could implement global trimming/compaction here
            asyncio.run(asyncio.sleep(SAVE_INTERVAL))
        except Exception:
            pass

# start autosave thread daemon
t = threading.Thread(target=autosave_background, daemon=True)
t.start()

# ---------- on startup, ensure existing brains loaded (no-op) ----------
@app.on_event("startup")
async def startup_event():
    # create dir if missing
    os.makedirs(MEMORY_DIR, exist_ok=True)
    # ensure global file exists
    if not os.path.exists(GLOBAL_FILE):
        with open(GLOBAL_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

# ---------- graceful shutdown ----------
@app.on_event("shutdown")
def shutdown_event():
    # nothing special (brains saved during operations)
    pass
