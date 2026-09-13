"""
Study/exam reminders (FR17).

Uses APScheduler to poll for study sessions starting soon and "sends" a
reminder (SMS via Twilio if configured, otherwise console/log — swap in
a push-notification provider such as Firebase Cloud Messaging as needed).
"""
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from .database import SessionLocal
from .models import StudySession, User

REMINDER_LEAD_MINUTES = 15
scheduler = BackgroundScheduler()


def _send_reminder(user: User, session: StudySession):
    message = f"Reminder: your study session starts at {session.scheduled_start.strftime('%H:%M')}."
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if sid and token and from_number:
        from twilio.rest import Client
        client = Client(sid, token)
        client.messages.create(body=message, from_=from_number, to=user.mobile_number)
    else:
        print(f"[DEV MODE reminder] -> {user.mobile_number}: {message}")


def check_and_send_reminders():
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        window_end = now + timedelta(minutes=REMINDER_LEAD_MINUTES)
        due_sessions = (
            db.query(StudySession)
            .filter(
                StudySession.reminder_sent == False,  # noqa: E712
                StudySession.scheduled_start >= now,
                StudySession.scheduled_start <= window_end,
            )
            .all()
        )
        for session in due_sessions:
            user = db.query(User).filter(User.id == session.user_id).first()
            if user:
                _send_reminder(user, session)
            session.reminder_sent = True
        db.commit()
    finally:
        db.close()


def start_reminder_scheduler():
    """Call once at app startup."""
    scheduler.add_job(check_and_send_reminders, "interval", minutes=1, id="reminder_check", replace_existing=True)
    scheduler.start()
