# ADDA — AI Chatbot

A sophisticated local AI chatbot with text and image generation, powered by **Ollama** and **Stable Diffusion**.
The LLM is running in the local system.
## System Config:
MacBook Air
Chip : Apple M1
Memory : 8GB
macOS : 26.1

---

## Quick Start

### 1. Start Ollama
```bash
ollama serve
ollama run openchat:7b   # or any model you prefer
```

### 2. Start the Backend
```bash
python3 backend.py
# Runs at http://localhost:8080
```

### 3. Open the Frontend
```bash
# Just open index.html in your browser
open index.html         # Mac
start index.html        # Windows
xdg-open index.html     # Linux
```

---

## Features

- 💬 **Streaming chat** — real-time token-by-token responses
- 🎨 **Image generation** — via Stable Diffusion (AUTOMATIC1111)
- 🔄 **Multi-model** — switch between any Ollama model
- 📝 **Chat history** — saved locally in browser
- 📤 **Export** — download conversations as text
- ⚙️ **Settings** — adjust temperature and max tokens
- 🌓 **Dark theme** — sleek cyberpunk aesthetic

---

## Image Generation Setup (Optional)

Image generation requires **AUTOMATIC1111 Stable Diffusion WebUI**:

```bash
# Clone and install
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui

# Start with API enabled
python launch.py --api --listen
# Runs at http://localhost:7860
```

Without Stable Diffusion, the chatbot falls back to **text descriptions** of images using Ollama.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| GET | `/api/models` | List available Ollama models |
| GET | `/api/ollama-status` | Check if Ollama is running |
| GET | `/api/sd-status` | Check if Stable Diffusion is running |
| POST | `/api/chat` | Text generation (non-streaming) |
| POST | `/api/chat-stream` | Text generation (SSE streaming) |
| POST | `/api/generate-image` | Image generation |

### Example API calls

**Chat:**
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat:7b",
    "messages": [{"role":"user","content":"What is an epoch?"}],
    "temperature": 0.7
  }'
```

**Generate Image:**
```bash
curl -X POST http://localhost:8080/api/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic city at sunset",
    "style": "photorealistic",
    "width": 512,
    "height": 512
  }'
```

---

## Supported Models (via Ollama)

```bash
ollama pull openchat:7b      # recommended
ollama pull llama3.1:7b
ollama pull mistral:7b
ollama pull gemma2:7b
ollama pull phi3:3.8b
```

---

## Requirements

- Python 3.8+ (no pip installs — uses stdlib only)
- Ollama installed and running
- Modern browser (Chrome, Firefox, Edge)
- *(Optional)* AUTOMATIC1111 for real image generation
