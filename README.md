# Multimodal Intelligent Learning Assistant — Reference Implementation

This is a working end-to-end backend (FastAPI) + a minimal mobile-first
web dashboard, implementing every module from the proposal:

- Mobile-number + OTP login
- PDF / PPTX / DOCX / image / handwritten-note ingestion
- Explainable RAG question answering (text + voice)
- Knowledge-gap detection + learning dependency graph
- Adaptive quiz generation
- Personalised study scheduling + reminders
- Student web dashboard

Treat this as a **strong starting scaffold** for your MSc implementation —
it is functional but you should extend/tune each module and cite this
architecture as your own designed-and-built work once you've adapted it.

---

## 1. Project structure

```
msc-project/
├── app/
│   ├── main.py              # FastAPI app + all API routes
│   ├── auth.py               # mobile-number OTP login (JWT)
│   ├── database.py           # SQLAlchemy engine/session
│   ├── models.py             # ORM models (users, docs, chunks, concepts...)
│   ├── schemas.py             # Pydantic request/response models
│   ├── ingestion.py           # PDF/PPTX/DOCX/image/handwriting extraction
│   ├── rag.py                 # embeddings + FAISS + explainable answer generation
│   ├── graph_module.py        # dependency graph + knowledge-gap detection
│   ├── quiz.py                 # adaptive quiz generation
│   ├── scheduler.py            # personalised study schedule generation
│   └── notifications.py        # reminder scheduler (APScheduler)
├── frontend/
│   └── index.html              # plain HTML/JS mobile-first dashboard
├── requirements.txt
├── .env.example
└── README.md
```

## 2. System-level dependencies (install BEFORE pip install)

Some Python libraries wrap system binaries. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils ffmpeg
```

- `tesseract-ocr` → required by `pytesseract` (printed-text OCR)
- `poppler-utils` → required by `pdf2image`
- `ffmpeg` → required by `openai-whisper` (voice questions)

On macOS: `brew install tesseract poppler ffmpeg`
On Windows: install Tesseract and Poppler binaries manually and add them to PATH.

## 3. Python environment setup

```bash
cd msc-project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `torch`, `transformers` (for handwriting/TrOCR) and `openai-whisper`
> (voice) are large downloads. If you don't need handwriting or voice yet,
> comment those lines out of `requirements.txt` for a much faster initial setup.

## 4. Configure environment variables

```bash
cp .env.example .env
```

Fill in:
- `JWT_SECRET` — any long random string
- `ANTHROPIC_API_KEY` — get one from https://console.anthropic.com (used for
  grounded answer generation and quiz-question generation). **Without this,
  the system still runs**: Q&A falls back to showing the best-matching
  excerpt directly, and quiz generation falls back to a simple template —
  useful for testing the pipeline before you have API credits.
- `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` — get a
  free trial account at https://www.twilio.com for real SMS OTP delivery.
  **Without this, OTPs are printed to the console** instead of texted —
  fine for local development/demo.

## 5. Run the backend

```bash
uvicorn app.main:app --reload
```

- Interactive API docs: http://localhost:8000/docs
- The SQLite database file (`msc_learning_assistant.db`) is created automatically.

## 6. Run the frontend

The dashboard is a single static HTML file with no build step:

```bash
cd frontend
python3 -m http.server 5500
```

Open http://localhost:5500 in your browser (or on your phone, on the same
network, using your machine's LAN IP instead of localhost).

## 7. Typical end-to-end flow to demo for your assessment

1. **Login** — enter a mobile number, click "Send OTP". In dev mode the
   OTP is printed in the terminal running `uvicorn`. Enter it in the app.
2. **Upload** — upload a PDF/PPTX/DOCX or a photo of handwritten notes.
   Watch `status` move from `queued → processing → indexed`
   (`GET /documents`).
3. **Ask** — type a question. The answer will cite the exact document and
   page/slide it came from.
4. **Seed concepts** (one-time, via `/docs` Swagger UI, since the UI doesn't
   expose concept authoring yet):
   - `POST /concepts` to create a few concepts for your subject
   - `POST /concepts/{concept_id}/prerequisite/{prerequisite_id}` to link
     prerequisite relationships
5. **Quiz** — generate an adaptive quiz; answering questions updates
   per-concept mastery and prerequisite estimates automatically.
6. **Schedule** — set free-time + exam dates via `POST /schedule/generate`;
   view upcoming sessions in the Schedule tab. Reminders fire automatically
   ~15 minutes before each session (printed to console in dev mode).

## 8. What you should extend for a strong MSc submission

- Add automated tests (`pytest`) for ingestion, RAG retrieval accuracy, and
  mastery updates — this directly feeds your Evaluation chapter.
- Add a proper concept-authoring UI instead of using Swagger for `/concepts`.
- Add pagination/streaming for the LLM answer for a nicer UX.
- Swap the FAISS `IndexFlatIP` for `IndexIVFFlat` if you test with a large
  document set (faster search, still simple to reason about).
- Add unit-tested evaluation scripts that compute the metrics defined in
  your proposal's Evaluation Plan (retrieval accuracy, citation
  correctness, hallucination rate, schedule adherence, SUS score).
- Consider containerising with Docker + docker-compose (Postgres instead
  of SQLite, a proper `.env` secrets setup) if your write-up discusses
  deployment/production-readiness.

## 9. Academic integrity note

This scaffold is a generic reference architecture. To submit it as your
own MSc work: read every file, understand it, modify it to reflect your
own design decisions (module boundaries, model choices, schema fields),
and write your dissertation's implementation chapter describing what you
actually built and why — not this README. Keep your git commit history
from early, incremental commits as evidence of independent development.
