from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from pypdf import PdfReader
import re
import requests
import tempfile
import os

app = FastAPI(
    title="HR Interviewer API",
    description="Upload résumé PDF → get brutal HR feedback in pure Markdown",
    version="1.1"
)

OLLAMA_URL = "https://ollama-gemma3-270m-gpu-746057898178.europe-west1.run.app/api/chat"

def extract_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 14000:
        text = text[:14000] + "\n\n...[truncated for model context]"
    return text

def interview_candidate(resume: str) -> str:
    payload = {
        "model": "gemma3:270m",
        "messages": [
            {
                "role": "system",
                "content": "You are a senior technical recruiter with 15 years of experience. "
                           "Be direct, professional, and brutally honest. "
                           "Analyze the résumé deeply and respond ONLY in clean, well-structured Markdown with these exact sections:\n"
                           "- Overall Impression\n"
                           "- Strengths\n"
                           "- Red Flags & Concerns\n"
                           "- Follow-up Questions (3–6)\n"
                           "- Final Recommendation (Strong Hire / Hire / Lean Pass / No Hire)\n\n"
                           "Never use HTML. Never add extra commentary outside the sections."
            },
            {
                "role": "user",
                "content": f"Full résumé text:\n\n{resume}"
            }
        ],
        "stream": False,
        "options": {"temperature": 0.7}
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"# Error\n\nFailed to reach interviewer.\n\n`{str(e)}`"

@app.post("/interview", response_class=PlainTextResponse)
async def interview(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="PDF only")

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw = extract_text(tmp_path)
        if len(raw.strip()) < 100:
            return "# No text found\n\nThis PDF appears to be empty or image-only."

        cleaned = clean_text(raw)
        markdown_response = interview_candidate(cleaned)
        return PlainTextResponse(markdown_response, media_type="text/markdown")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/")
async def root():
    return {
        "message": "HR Interviewer API ready",
        "endpoint": "POST /interview",
        "input": "multipart/form-data with 'file' = your_resume.pdf",
        "output": "text/markdown"
    }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")