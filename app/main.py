"""
Main FastAPI application.

Run with:  uvicorn app.main:app --reload
Docs at:   http://localhost:8000/docs
"""
import os
import shutil
import json
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()   # must run before any os.getenv() calls in this app, including in other modules

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import (
    User, Document, Chunk, Concept, ConceptPrerequisite,
    FreeTimeSlot, StudySession, QuizQuestion,
)
from .schemas import (
    QuestionRequest, AnswerResponse, SourceRef, ScheduleRequest, QuizAnswerIn,
)
from . import ingestion, rag, graph_module, quiz as quiz_module, scheduler as scheduler_module
from .auth import router as auth_router, get_current_user
from .notifications import start_reminder_scheduler

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multimodal Adaptive Learning Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup_event():
    start_reminder_scheduler()


# ---------------------------------------------------------------------------
# 1. Document ingestion (PDF / PPTX / DOCX / image / handwritten notes)
# ---------------------------------------------------------------------------
def _detect_file_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return "pdf"
    if ext == "pptx":
        return "pptx"
    if ext == "docx":
        return "docx"
    if ext in ("jpg", "jpeg", "png"):
        return "image"
    raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")


def _process_document(document_id: int, path: str, file_type: str, is_handwritten: bool):
    """Runs in the background: extract -> chunk -> embed -> index."""
    from .database import SessionLocal
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "processing"
        db.commit()

        pages = ingestion.extract_document(path, file_type, is_handwritten)
        new_chunks = []
        for text, label in pages:
            for piece in ingestion.chunk_text(text):
                chunk = Chunk(document_id=document_id, text=piece, page_or_slide=label)
                db.add(chunk)
                new_chunks.append(chunk)
        db.commit()
        for c in new_chunks:
            db.refresh(c)

        rag.add_chunks_to_index(db, doc.owner_id, new_chunks)

        doc.status = "indexed"
        db.commit()
    except Exception as e:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
        print(f"Ingestion failed for document {document_id}: {e}")
    finally:
        db.close()


@app.post("/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
    is_handwritten: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    file_type = _detect_file_type(file.filename)
    save_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = Document(
        owner_id=current_user.id,
        filename=file.filename,
        file_type=file_type,
        subject=subject,
        status="queued",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(_process_document, doc.id, save_path, file_type, is_handwritten)
    return {"document_id": doc.id, "status": doc.status}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    return [{"id": d.id, "filename": d.filename, "status": d.status, "subject": d.subject} for d in docs]


# ---------------------------------------------------------------------------
# 2. Explainable RAG question answering (text + voice)
# ---------------------------------------------------------------------------
@app.post("/qa/ask", response_model=AnswerResponse)
def ask_question(
    payload: QuestionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chunks = rag.retrieve(db, current_user.id, payload.question)
    answer_text, grounded = rag.generate_grounded_answer(payload.question, chunks, db)

    sources = []
    for c in chunks:
        doc = db.query(Document).filter(Document.id == c.document_id).first()
        sources.append(SourceRef(
            document_name=doc.filename if doc else "Unknown",
            page_or_slide=c.page_or_slide,
            excerpt=c.text[:300],
        ))
    return AnswerResponse(answer=answer_text, sources=sources, grounded=grounded)


@app.post("/qa/ask-voice", response_model=AnswerResponse)
def ask_voice_question(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Transcribes a spoken question with Whisper, then answers via the same RAG path."""
    import whisper
    tmp_path = os.path.join(UPLOAD_DIR, f"voice_{current_user.id}_{audio.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    question_text = result["text"]

    chunks = rag.retrieve(db, current_user.id, question_text)
    answer_text, grounded = rag.generate_grounded_answer(question_text, chunks, db)
    sources = [
        SourceRef(
            document_name=(db.query(Document).filter(Document.id == c.document_id).first().filename),
            page_or_slide=c.page_or_slide,
            excerpt=c.text[:300],
        )
        for c in chunks
    ]
    return AnswerResponse(answer=f"(Heard: \"{question_text}\")\n\n{answer_text}", sources=sources, grounded=grounded)


# ---------------------------------------------------------------------------
# 3. Knowledge-gap detection / dependency graph
# ---------------------------------------------------------------------------
@app.get("/concepts/weak")
def weak_concepts(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return graph_module.get_weak_concepts(db, current_user.id)


@app.post("/concepts")
def create_concept(name: str, subject: Optional[str] = None, db: Session = Depends(get_db)):
    concept = Concept(name=name, subject=subject)
    db.add(concept)
    db.commit()
    db.refresh(concept)
    return {"id": concept.id, "name": concept.name}


@app.post("/concepts/{concept_id}/prerequisite/{prerequisite_id}")
def add_prerequisite(concept_id: int, prerequisite_id: int, db: Session = Depends(get_db)):
    edge = ConceptPrerequisite(concept_id=concept_id, prerequisite_id=prerequisite_id)
    db.add(edge)
    db.commit()
    return {"message": "prerequisite linked"}


# ---------------------------------------------------------------------------
# 4. Adaptive quiz generation + submission
# ---------------------------------------------------------------------------
@app.post("/quiz/generate")
def generate_quiz(
    num_questions: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    questions = quiz_module.generate_adaptive_quiz(db, current_user.id, num_questions)
    return [
        {
            "id": q.id,
            "question": q.question_text,
            "options": json.loads(q.options) if q.options else [],
        }
        for q in questions
    ]


@app.post("/quiz/answer")
def answer_quiz(
    payload: QuizAnswerIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    question = db.query(QuizQuestion).filter(QuizQuestion.id == payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    is_correct = payload.selected_answer.strip().lower() == question.correct_answer.strip().lower()
    graph_module.record_quiz_attempt(db, current_user.id, question.id, is_correct)
    return {"correct": is_correct, "correct_answer": question.correct_answer}


# ---------------------------------------------------------------------------
# 5. Personalised scheduling + reminders
# ---------------------------------------------------------------------------
@app.post("/schedule/free-time")
def set_free_time(
    slots: List[dict],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.query(FreeTimeSlot).filter(FreeTimeSlot.user_id == current_user.id).delete()
    for s in slots:
        db.add(FreeTimeSlot(
            user_id=current_user.id,
            day_of_week=s["day_of_week"],
            start_hour=s["start_hour"],
            end_hour=s["end_hour"],
        ))
    db.commit()
    return {"message": "free time saved"}


@app.post("/schedule/generate")
def generate_schedule(
    payload: ScheduleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Persist free-time slots first, then build the schedule from them.
    db.query(FreeTimeSlot).filter(FreeTimeSlot.user_id == current_user.id).delete()
    for slot in payload.free_time:
        db.add(FreeTimeSlot(
            user_id=current_user.id,
            day_of_week=slot.day_of_week,
            start_hour=slot.start_hour,
            end_hour=slot.end_hour,
        ))
    db.commit()

    sessions = scheduler_module.generate_schedule(db, current_user.id, payload.exam_dates)
    return [
        {
            "concept_id": s.concept_id,
            "start": s.scheduled_start.isoformat(),
            "end": s.scheduled_end.isoformat(),
        }
        for s in sessions
    ]


@app.get("/schedule/upcoming")
def upcoming_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sessions = (
        db.query(StudySession)
        .filter(StudySession.user_id == current_user.id, StudySession.completed == False)  # noqa: E712
        .order_by(StudySession.scheduled_start)
        .all()
    )
    return [
        {
            "id": s.id,
            "concept_id": s.concept_id,
            "start": s.scheduled_start.isoformat(),
            "end": s.scheduled_end.isoformat(),
        }
        for s in sessions
    ]


# ---------------------------------------------------------------------------
# 6. Dashboard summary endpoint (single call to populate the frontend)
# ---------------------------------------------------------------------------
@app.get("/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return {
        "documents": list_documents(db, current_user),
        "weak_concepts": graph_module.get_weak_concepts(db, current_user.id),
        "upcoming_sessions": upcoming_sessions(db, current_user),
    }
