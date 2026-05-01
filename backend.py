#!/usr/bin/env python3
"""
NOVA Chatbot Backend
Bridges the frontend with Ollama (text) and Stable Diffusion (images)
"""

import json
import base64
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import os

OLLAMA_URL = "http://localhost:11434"
SD_URL     = "http://localhost:7860"
PORT       = 8081   # changed from 8080 (Jenkins uses that)

# Directory where index.html lives (same folder as this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────────

def json_request(url: str, data: dict = None, timeout: int = 120) -> dict:
    body = json.dumps(data).encode() if data else None
    req  = urllib.request.Request(
        url,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def stream_ollama(model: str, messages: list, temperature: float = 0.7):
    """Generator: yields text tokens from Ollama streaming API."""
    payload = {
        "model"  : model,
        "messages": messages,
        "stream" : True,
        "options": {"temperature": temperature},
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                done  = chunk.get("done", False)
                yield token, done
                if done:
                    break
            except json.JSONDecodeError:
                continue


# ── Request Handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    # ── CORS preflight ──
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── GET ──
    def do_GET(self):
        path = urlparse(self.path).path

        # ── Serve index.html ──
        if path == "/" or path == "/index.html":
            self._serve_file("index.html", "text/html")
            return

        if path == "/health":
            self._json(200, {"status": "ok", "port": PORT})

        elif path == "/api/models":
            try:
                data = json_request(f"{OLLAMA_URL}/api/tags", timeout=5)
                models = [m["name"] for m in data.get("models", [])]
                self._json(200, {"models": models})
            except Exception as e:
                self._json(503, {"error": str(e), "models": []})

        elif path == "/api/ollama-status":
            try:
                json_request(f"{OLLAMA_URL}/api/tags", timeout=3)
                self._json(200, {"running": True})
            except:
                self._json(200, {"running": False})

        elif path == "/api/sd-status":
            try:
                json_request(f"{SD_URL}/sdapi/v1/sd-models", timeout=3)
                self._json(200, {"running": True})
            except:
                self._json(200, {"running": False})

        else:
            self._json(404, {"error": "Not found"})

    # ── POST ──
    def do_POST(self):
        path    = urlparse(self.path).path
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "Invalid JSON"})
            return

        if path == "/api/chat":
            self._handle_chat(payload)

        elif path == "/api/generate-image":
            self._handle_image(payload)

        elif path == "/api/chat-stream":
            self._handle_chat_stream(payload)

        else:
            self._json(404, {"error": "Not found"})

    # ── Chat (non-streaming) ──
    def _handle_chat(self, payload):
        model    = payload.get("model", "openchat:7b")
        messages = payload.get("messages", [])
        temp     = float(payload.get("temperature", 0.7))

        if not messages:
            self._json(400, {"error": "messages required"})
            return

        try:
            result = json_request(f"{OLLAMA_URL}/api/chat", {
                "model"   : model,
                "messages": messages,
                "stream"  : False,
                "options" : {"temperature": temp},
            })
            reply = result.get("message", {}).get("content", "")
            self._json(200, {"reply": reply, "model": model})

        except urllib.error.URLError:
            self._json(503, {"error": "Ollama not running. Start with: ollama serve"})
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── Chat (streaming via SSE) ──
    def _handle_chat_stream(self, payload):
        model    = payload.get("model", "openchat:7b")
        messages = payload.get("messages", [])
        temp     = float(payload.get("temperature", 0.7))

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for token, done in stream_ollama(model, messages, temp):
                data = json.dumps({"token": token, "done": done})
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
                if done:
                    break
        except BrokenPipeError:
            pass
        except urllib.error.URLError:
            err = json.dumps({"error": "Ollama not running"})
            self.wfile.write(f"data: {err}\n\n".encode())
        except Exception as e:
            err = json.dumps({"error": str(e)})
            self.wfile.write(f"data: {err}\n\n".encode())

    # ── Image Generation ──
    def _handle_image(self, payload):
        prompt   = payload.get("prompt", "")
        style    = payload.get("style", "photorealistic")
        steps    = int(payload.get("steps", 20))
        width    = int(payload.get("width", 512))
        height   = int(payload.get("height", 512))

        if not prompt:
            self._json(400, {"error": "prompt required"})
            return

        full_prompt = f"{prompt}, {style} style, high quality, detailed, 4k"
        neg_prompt  = "blurry, bad quality, distorted, ugly, watermark"

        # Try Stable Diffusion
        try:
            result = json_request(f"{SD_URL}/sdapi/v1/txt2img", {
                "prompt"         : full_prompt,
                "negative_prompt": neg_prompt,
                "steps"          : steps,
                "width"          : width,
                "height"         : height,
                "cfg_scale"      : 7,
                "sampler_name"   : "DPM++ 2M Karras",
            }, timeout=120)

            images = result.get("images", [])
            if images:
                self._json(200, {
                    "image"    : images[0],   # base64 PNG
                    "source"   : "stable-diffusion",
                    "prompt"   : full_prompt,
                })
                return

        except Exception as sd_err:
            print(f"[SD] Not available: {sd_err}")

        # Fallback: ask Ollama to describe the image
        try:
            result = json_request(f"{OLLAMA_URL}/api/generate", {
                "model" : payload.get("model", "openchat:7b"),
                "prompt": f"Vividly describe this scene as a {style} artwork: {prompt}. Be detailed and creative.",
                "stream": False,
            })
            description = result.get("response", "")
            self._json(200, {
                "image"      : None,
                "source"     : "text-description",
                "description": description,
                "prompt"     : full_prompt,
            })

        except urllib.error.URLError:
            self._json(503, {"error": "Neither Stable Diffusion nor Ollama is running"})
        except Exception as e:
            self._json(500, {"error": str(e)})

    # ── Helpers ──
    def _serve_file(self, filename: str, content_type: str):
        filepath = os.path.join(BASE_DIR, filename)
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._json(404, {"error": f"{filename} not found. Make sure it's in the same folder as backend.py"})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════╗
║        NOVA Backend Server           ║
╠══════════════════════════════════════╣
║  Server  → http://localhost:{PORT}      ║
║  Ollama  → {OLLAMA_URL}     ║
║  SD API  → {SD_URL}         ║
╚══════════════════════════════════════╝

Open in browser → http://localhost:{PORT}

Endpoints:
  GET  /                   — serves index.html
  GET  /health             — health check
  GET  /api/models         — list Ollama models
  GET  /api/ollama-status  — check Ollama
  GET  /api/sd-status      — check Stable Diffusion
  POST /api/chat           — text generation
  POST /api/chat-stream    — streaming text (SSE)
  POST /api/generate-image — image generation

Starting server...
    """)

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
