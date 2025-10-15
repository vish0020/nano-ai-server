from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json, os, random
from collections import defaultdict

app = FastAPI()

MEMORY_FILE = "nano_ai_memory.json"

# Load or create simple memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        connections = json.load(f)
else:
    connections = defaultdict(dict)

def update_connections(text):
    words = text.lower().split()
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if b in connections[a]:
            connections[a][b] += 1
        else:
            connections[a][b] = 1
    with open(MEMORY_FILE, "w") as f:
        json.dump(connections, f)

def generate_reply(input_text):
    words = input_text.lower().split()
    if not words:
        return "Hmm?"
    last = words[-1]
    if last in connections and connections[last]:
        possible = list(connections[last].keys())
        return random.choice(possible)
    else:
        return random.choice(["ok", "hmm", "tell me more", "interesting"])

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    msg = data.get("message", "")
    update_connections(msg)
    reply = generate_reply(msg)
    return JSONResponse({"reply": reply})
