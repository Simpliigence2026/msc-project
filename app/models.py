"""
ORM models for every entity in the system:
users, documents, chunks (RAG index metadata), quiz questions/attempts,
concepts + prerequisite edges (dependency graph), study sessions, reminders.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
)
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    mobile_number = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    institution = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="owner")
    attempts = relationship("QuizAttempt", back_populates="user")
    sessions = relationship("StudySession", back_populates="user")
    free_time_slots = relationship("FreeTimeSlot", back_populates="user")


class OTPCode(Base):
    """Short-lived one-time-passwords for mobile login."""
    __tablename__ = "otp_codes"
    id = Column(Integer, primary_key=True)
    mobile_number = Column(String, index=True, nullable=False)
    code_hash = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    consumed = Column(Boolean, default=False)


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)   # pdf | pptx | docx | image
    subject = Column(String, nullable=True)
    status = Column(String, default="queued")     # queued|processing|indexed|failed
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    """One retrievable, source-referenced piece of a document (RAG unit)."""
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text, nullable=False)
    page_or_slide = Column(String, nullable=True)   # e.g. "Page 4" / "Slide 7"
    vector_id = Column(Integer, nullable=True)       # position in the FAISS index

    document = relationship("Document", back_populates="chunks")


class Concept(Base):
    """A node in the learning dependency graph."""
    __tablename__ = "concepts"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    subject = Column(String, nullable=True)


class ConceptPrerequisite(Base):
    """Directed edge: concept_id depends on prerequisite_id."""
    __tablename__ = "concept_prerequisites"
    id = Column(Integer, primary_key=True)
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    prerequisite_id = Column(Integer, ForeignKey("concepts.id"))


class Mastery(Base):
    """Per-user, per-concept mastery estimate (0.0 - 1.0)."""
    __tablename__ = "mastery"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    score = Column(Float, default=0.5)
    updated_at = Column(DateTime, default=datetime.utcnow)


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    question_text = Column(Text, nullable=False)
    correct_answer = Column(Text, nullable=False)
    options = Column(Text, nullable=True)   # JSON-encoded list for MCQ
    difficulty = Column(Float, default=0.5)


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question_id = Column(Integer, ForeignKey("quiz_questions.id"))
    is_correct = Column(Boolean, nullable=False)
    answered_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="attempts")


class FreeTimeSlot(Base):
    """Recurring weekly free-time window used by the scheduler."""
    __tablename__ = "free_time_slots"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    day_of_week = Column(Integer)   # 0=Monday ... 6=Sunday
    start_hour = Column(Float)      # e.g. 17.5 = 17:30
    end_hour = Column(Float)

    user = relationship("User", back_populates="free_time_slots")


class StudySession(Base):
    __tablename__ = "study_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    scheduled_start = Column(DateTime, nullable=False)
    scheduled_end = Column(DateTime, nullable=False)
    reminder_sent = Column(Boolean, default=False)
    completed = Column(Boolean, default=False)

    user = relationship("User", back_populates="sessions")
