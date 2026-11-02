from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import os
import shutil

app = FastAPI(title="ClipForge AI")

UPLOAD_DIR = "/tmp/clipforge_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>ClipForge AI</title>
        <style>
            body {
                background: #10131c;
                color: white;
                font-family: Arial;
                text-align: center;
                padding: 50px 15px;
            }
            .box {
                background: #202638;
                padding: 35px;
                border-radius: 15px;
                max-width: 500px;
                margin: auto;
            }
            button {
                background: #7657ff;
                color: white;
                padding: 14px 25px;
                border: 0;
                border-radius: 8px;
                margin-top: 20px;
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
                       accept="video/*" required>
                <br>
                <button type="submit">Upload Video</button>
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
    if not video.content_type or not video.content_type.startswith("video/"):
        return {"error": "Please upload a video file"}

    path = os.path.join(UPLOAD_DIR, video.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return {
        "message": "Video uploaded successfully!",
        "filename": video.filename
    }
