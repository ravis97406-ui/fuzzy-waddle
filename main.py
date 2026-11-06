
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from groq import Groq

app = FastAPI(title="ClipForge AI")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
    <title>ClipForge AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    body {
        background:#10131c;
        color:white;
        font-family:Arial;
        text-align:center;
        padding:50px 15px;
    }
    .box {
        background:#202638;
        padding:30px;
        border-radius:15px;
        max-width:500px;
        margin:auto;
    }
    button {
        background:#7657ff;
        color:white;
        padding:14px 25px;
        border:0;
        border-radius:8px;
        margin-top:20px;
    }
    </style>
    </head>
    <body>
    <div class="box">
    <h1>🎬 ClipForge AI</h1>
    <p>Turn long videos into engaging short clips.</p>
    <form action="/upload" method="post"
    enctype="multipart/form-data">
    <input type="file" name="video"
    accept="video/mp4,video/webm" required>
    <br>
    <button type="submit">Upload & Transcribe</button>
    </form>
    </div>
    </body>
    </html>
    """

@app.get("/healthz")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):

    if not video.filename.lower().endswith((".mp4", ".webm")):
        return {"error": "Please upload an MP4 or WebM video."}

    if not os.getenv("GROQ_API_KEY"):
        return {"error": "Groq API key is not configured."}

    contents = await video.read(25 * 1024 * 1024 + 1)

    if len(contents) > 25 * 1024 * 1024:
        return {"error": "Video exceeds the 25 MB free-tier limit."}

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        result = await run_in_threadpool(
            lambda: client.audio.transcriptions.create(
                file=(video.filename, contents),
                model="whisper-large-v3-turbo",
                response_format="verbose_json"
            )
        )

        return {
            "message": "Transcription successful!",
            "filename": video.filename,
            "transcript": result.text
        }

    except Exception as e:
        return {"error": str(e)}

