import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Load env vars
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not all([DATABASE_URL, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("❌ Thiếu các biến môi trường cần thiết trong file .env")

# Database & Supabase Clients
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Auth Dependency
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    try:
        user_response = supabase.auth.get_user(token)
        user_id = user_response.user.id
        return str(user_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc đã hết hạn.",
        )
