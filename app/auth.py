"""
Mobile-number + OTP authentication.

Flow:
 1. POST /auth/request-otp  {mobile_number}  -> sends a 6-digit OTP via SMS (Twilio)
 2. POST /auth/verify-otp   {mobile_number, code} -> returns a JWT access token
 3. Protected routes use `get_current_user` as a dependency.
"""
import os
import re
import random
import string
from datetime import datetime, timedelta

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from .database import get_db
from .models import User, OTPCode
from .schemas import RequestOTP, VerifyOTP, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/verify-otp")


def _hash_otp(code: str) -> str:
    return bcrypt.hashpw(code.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_otp_hash(code: str, code_hash: str) -> bool:
    return bcrypt.checkpw(code.encode("utf-8"), code_hash.encode("utf-8"))

# Optional leading "+" then 7-15 digits (E.164-ish). No letters/symbols allowed.
MOBILE_NUMBER_PATTERN = re.compile(r"^\+?[0-9]{7,15}$")


def _validate_mobile_number(mobile_number: str) -> str:
    mobile_number = mobile_number.strip()
    if not MOBILE_NUMBER_PATTERN.match(mobile_number):
        raise HTTPException(
            status_code=400,
            detail="Invalid mobile number. Use digits only, optionally starting with '+' and a country code.",
        )
    return mobile_number


def _generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def _send_sms(mobile_number: str, message: str):
    """
    Sends the OTP via Twilio. Falls back to printing to console if
    TWILIO_* env vars are not configured (useful for local dev/demo).
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not (sid and token and from_number):
        print(f"[DEV MODE - no SMS provider configured] OTP for {mobile_number}: {message}")
        return

    from twilio.rest import Client
    client = Client(sid, token)
    client.messages.create(body=message, from_=from_number, to=mobile_number)


@router.post("/request-otp")
def request_otp(payload: RequestOTP, db: Session = Depends(get_db)):
    mobile_number = _validate_mobile_number(payload.mobile_number)
    code = _generate_otp()
    otp = OTPCode(
        mobile_number=mobile_number,
        code_hash=_hash_otp(code),
        expires_at=datetime.utcnow() + timedelta(minutes=5),
    )
    db.add(otp)
    db.commit()

    _send_sms(mobile_number, f"Your Learning Assistant login code is {code}. Expires in 5 minutes.")
    return {"message": "OTP sent"}


@router.post("/verify-otp", response_model=TokenResponse)
def verify_otp(payload: VerifyOTP, db: Session = Depends(get_db)):
    mobile_number = _validate_mobile_number(payload.mobile_number)
    otp = (
        db.query(OTPCode)
        .filter(OTPCode.mobile_number == mobile_number, OTPCode.consumed == False)  # noqa: E712
        .order_by(OTPCode.id.desc())
        .first()
    )
    if not otp or otp.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="OTP expired or not found. Request a new one.")
    if not _verify_otp_hash(payload.code, otp.code_hash):
        raise HTTPException(status_code=400, detail="Incorrect OTP.")

    otp.consumed = True

    user = db.query(User).filter(User.mobile_number == mobile_number).first()
    if not user:
        user = User(mobile_number=mobile_number)
        db.add(user)
    db.commit()
    db.refresh(user)

    access_token = jwt.encode(
        {"sub": str(user.id), "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return TokenResponse(access_token=access_token)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user
