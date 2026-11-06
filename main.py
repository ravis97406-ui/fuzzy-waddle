
import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from groq import Groq

app = FastAPI(title="ClipForge AI")

MAX_SIZE = 25 * 1024 * 1024

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
    <button type="submit">Find Viral Clips</button>
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
        return {"error": "Upload an MP4 or WebM video."}

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {"error": "Groq API key is missing."}

    contents = await video.read(MAX_SIZE + 1)

    if len(contents) > MAX_SIZE:
        return {"error": "Video exceeds the 25 MB limit."}

    client = Groq(api_key=api_key)

    try:
        # STEP 1: Transcribe video audio with timestamps
        transcription = await run_in_threadpool(
            lambda: client.audio.transcriptions.create(
                file=(video.filename, contents),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        )

        transcript = transcription.text

        segments = []

for s in (transcription.segments or []):
    if isinstance(s, dict):
        start = s.get("start")
        end = s.get("end")
        text = s.get("text", "")
    else:
        start = getattr(s, "start", None)
        end = getattr(s, "end", None)
        text = getattr(s, "text", "")

    if start is not None and end is not None:
        segments.append({
            "start": float(start),
            "end": float(end),
            "text": str(text)
        })

        if not segments:
            return {
                "message": "Transcription successful, but no timestamped segments were returned.",
                "transcript": transcript,
                "highlights": []
            }

        # STEP 2: Ask AI to identify engaging moments
        prompt = f"""
You are an expert viral short-form video editor.

Analyse this timestamped transcript.

Find up to 5 interesting standalone clips.

Prioritize:
- Strong hooks
- Surprising statements
- Emotional moments
- Funny moments
- Dramatic moments
- Useful or informative content

Rules:
1. Each clip must be 15 to 60 seconds.
2. Use only timestamps from the supplied transcript.
3. Start and end times must be within the video's transcript duration.
4. Avoid overlapping clips.
5. Never invent dialogue.
6. Rank the clips from most engaging to least engaging.
7. If there is insufficient content, return an empty clips array.

Return JSON only, in this format:

{{
  "clips": [
    {{
      "start": 10.0,
      "end": 35.0,
      "title": "Interesting moment",
      "reason": "Why viewers may watch",
      "score": 9
    }}
  ]
}}

TIMESTAMPED TRANSCRIPT:
{json.dumps(segments, ensure_ascii=False)}
"""

        response = await run_in_threadpool(
            lambda: client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "You identify engaging video highlights. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
        )

        # STEP 3: Parse and validate AI results
        result = json.loads(
            response.choices[0].message.content
        )

        highlights = []

        for clip in result.get("clips", [])[:5]:
            start = float(clip["start"])
            end = float(clip["end"])

            if (
                0 <= start < end
                and 15 <= end - start <= 60
                and end <= max(s["end"] for s in segments) + 1
            ):
                highlights.append({
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 2),
                    "title": str(clip["title"])[:120],
                    "reason": str(clip["reason"])[:300],
                    "score": max(1, min(10, int(clip["score"])))
                })

        highlights.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "message": "AI analysis completed!",
            "filename": video.filename,
            "transcript": transcript,
            "highlights": highlights
        }

    except Exception as e:
        return {
            "error": "Video analysis failed.",
            "details": str(e)[:500]
        }


