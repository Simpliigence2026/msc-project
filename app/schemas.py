from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class RequestOTP(BaseModel):
    mobile_number: str


class VerifyOTP(BaseModel):
    mobile_number: str
    code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QuestionRequest(BaseModel):
    question: str
    subject: Optional[str] = None


class SourceRef(BaseModel):
    document_name: str
    page_or_slide: Optional[str]
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    grounded: bool


class FreeTimeSlotIn(BaseModel):
    day_of_week: int
    start_hour: float
    end_hour: float


class ScheduleRequest(BaseModel):
    exam_dates: dict  # {"Subject": "YYYY-MM-DD"}
    free_time: List[FreeTimeSlotIn]


class QuizAnswerIn(BaseModel):
    question_id: int
    selected_answer: str
