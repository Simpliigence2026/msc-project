"""
Personalised study scheduling (FR15-FR17).

Takes the student's weekly free-time windows + exam dates + current mastery
(from graph_module) and produces StudySession rows, prioritising:
  1. Concepts with the lowest mastery
  2. Subjects with the nearest exam date
Sessions are 45 minutes by default and packed into free-time slots for the
next `horizon_days`.
"""
from datetime import datetime, timedelta, date
from typing import List, Dict
from sqlalchemy.orm import Session

from .models import FreeTimeSlot, StudySession, Concept
from .graph_module import get_weak_concepts

SESSION_LENGTH_MINUTES = 45


def _days_to_exam(exam_dates: Dict[str, str], subject: str) -> int:
    if subject not in exam_dates:
        return 999
    exam_day = datetime.strptime(exam_dates[subject], "%Y-%m-%d").date()
    return max((exam_day - date.today()).days, 0)


def generate_schedule(
    db: Session,
    user_id: int,
    exam_dates: Dict[str, str],
    horizon_days: int = 14,
) -> List[StudySession]:
    weak_concepts = get_weak_concepts(db, user_id)

    # Rank concepts: lower mastery + closer exam = higher priority
    def priority(w):
        concept = db.query(Concept).filter(Concept.id == w["concept_id"]).first()
        subject = concept.subject if concept else ""
        urgency = 1 / (1 + _days_to_exam(exam_dates, subject))
        weakness = 1 - w["mastery"]
        return -(urgency * 0.6 + weakness * 0.4)  # negative -> ascending sort = best first

    weak_concepts.sort(key=priority)

    slots = db.query(FreeTimeSlot).filter(FreeTimeSlot.user_id == user_id).all()
    if not slots:
        return []

    sessions = []
    concept_queue = list(weak_concepts)
    today = date.today()

    for day_offset in range(horizon_days):
        current_day = today + timedelta(days=day_offset)
        weekday = current_day.weekday()
        todays_slots = [s for s in slots if s.day_of_week == weekday]

        for slot in todays_slots:
            if not concept_queue:
                break
            slot_start_minutes = int(slot.start_hour * 60)
            slot_end_minutes = int(slot.end_hour * 60)
            cursor = slot_start_minutes

            while cursor + SESSION_LENGTH_MINUTES <= slot_end_minutes and concept_queue:
                target = concept_queue.pop(0)
                start_dt = datetime.combine(current_day, datetime.min.time()) + timedelta(minutes=cursor)
                end_dt = start_dt + timedelta(minutes=SESSION_LENGTH_MINUTES)

                session = StudySession(
                    user_id=user_id,
                    concept_id=target["concept_id"],
                    scheduled_start=start_dt,
                    scheduled_end=end_dt,
                )
                db.add(session)
                sessions.append(session)
                cursor += SESSION_LENGTH_MINUTES

        if not concept_queue:
            break

    db.commit()
    return sessions
