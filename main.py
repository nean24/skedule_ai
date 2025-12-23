from dotenv import load_dotenv
import os
import io
import base64
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy import text
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# --- 1. IMPORT LANGCHAIN & CONFIG ---
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_tool_calling_agent

from langchain.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory

from utils.thoi_gian_tu_nhien import parse_natural_time
from app_dependencies import get_current_user_id, engine, supabase

# Bật lại nếu bạn đã có file payment_service.py
# from payment_service import router as payment_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm_brain = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY, 
    temperature=0.7
)

# --- 2. XỬ LÝ ÂM THANH (TTS & STT) ---
def clean_text_for_speech(text: str) -> str:
    return text.replace('*', '').replace('#', '').replace('-', ' ').replace('_', '')

def text_to_base64_audio(text: str) -> str:
    try:
        if not text: return ""
        short_text = clean_text_for_speech(text)[:200]
        tts = gTTS(short_text, lang='vi')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return base64.b64encode(audio_fp.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Lỗi TTS: {e}")
        return ""

async def audio_to_text(audio_file: UploadFile) -> str:
    try:
        r = sr.Recognizer()
        audio_bytes = await audio_file.read()
        audio_fp = io.BytesIO(audio_bytes)
        sound = AudioSegment.from_file(audio_fp)
        wav_fp = io.BytesIO()
        sound.export(wav_fp, format="wav")
        wav_fp.seek(0)
        with sr.AudioFile(wav_fp) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data, language="vi-VN")
    except Exception as e:
        logger.error(f"Lỗi STT: {e}")
        return ""

# --- 3. CÁC CÔNG CỤ (TOOLS) ---

@tool
def lay_ten_nguoi_dung(user_id: str) -> str:
    """Lấy tên người dùng từ bảng profiles."""
    with engine.connect() as conn:
        res = conn.execute(text("SELECT name FROM profiles WHERE id = :uid"), {"uid": user_id}).fetchone()
        return f"Tên người dùng là {res.name}." if res else "Không rõ tên."

@tool
def tao_su_kien_toan_dien(tieu_de: str, loai_su_kien: str, user_id: str, mo_ta: Optional[str] = None,
                          bat_dau: Optional[str] = None, ket_thuc: Optional[str] = None,
                          uu_tien: str = 'medium') -> str:
    """Tạo sự kiện, task hoặc lịch trình. Tự động xử lý thời gian tự nhiên."""
    try:
        with engine.connect() as conn:
            with conn.begin():
                start_dt, end_dt = parse_natural_time(bat_dau, datetime.now()) if bat_dau else (None, None)
                if ket_thuc:
                    _, end_dt = parse_natural_time(ket_thuc, start_dt or datetime.now())

                event_id = conn.execute(text("""
                    INSERT INTO events (user_id, title, description, type, start_time, end_time)
                    VALUES (:uid, :title, :desc, :type, :start, :end) RETURNING id
                """), {
                    "uid": user_id, "title": tieu_de, "desc": mo_ta,
                    "type": loai_su_kien, "start": start_dt, "end": end_dt
                }).scalar()

                if loai_su_kien in ['task', 'deadline']:
                    conn.execute(text("""
                        INSERT INTO tasks (user_id, event_id, title, description, deadline, priority, status)
                        VALUES (:uid, :eid, :title, :desc, :dl, :pri, 'todo')
                    """), {
                        "uid": user_id, "eid": event_id, "title": tieu_de,
                        "desc": mo_ta, "dl": end_dt or start_dt, "pri": uu_tien
                    })
                return f"✅ Đã tạo {loai_su_kien}: '{tieu_de}'."
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

@tool
def cap_nhat_su_kien(tieu_de_cu: str, thoi_gian_moi: str, user_id: str) -> str:
    """Dùng khi user muốn dời lịch hoặc đổi giờ."""
    try:
        with engine.connect() as conn:
            with conn.begin():
                event = conn.execute(text("SELECT id FROM events WHERE user_id = :uid AND title ILIKE :t LIMIT 1"),
                                     {"uid": user_id, "t": f"%{tieu_de_cu}%"}).fetchone()
                if not event: return "⚠️ Không tìm thấy sự kiện."
                new_start, new_end = parse_natural_time(thoi_gian_moi, datetime.now())
                conn.execute(text("UPDATE events SET start_time=:s, end_time=:e WHERE id=:id"),
                             {"s": new_start, "e": new_end, "id": event.id})
                return f"✅ Đã dời lịch sang {new_start}."
    except Exception as e:
        return f"Lỗi: {e}"

@tool
def thong_ke_tong_quan(user_id: str) -> str:
    """Báo cáo tổng quan số lượng công việc, ghi chú và lịch trình."""
    try:
        with engine.connect() as conn:
            task_stats = conn.execute(text("""
                SELECT 
                    COUNT(*) FILTER (WHERE status = 'todo') as todo,
                    COUNT(*) FILTER (WHERE status = 'doing') as doing,
                    COUNT(*) FILTER (WHERE status = 'done') as done
                FROM tasks WHERE user_id = :uid
            """), {"uid": user_id}).fetchone()
            
            note_count = conn.execute(text("SELECT COUNT(*) FROM notes WHERE user_id = :uid"), {"uid": user_id}).scalar()
            
            event_count = conn.execute(text("""
                SELECT COUNT(*) FROM events WHERE user_id = :uid 
                AND start_time >= CURRENT_DATE AND start_time < CURRENT_DATE + INTERVAL '7 days'
            """), {"uid": user_id}).scalar()

            return (f"📊 BÁO CÁO:\n- Công việc: {task_stats.todo} chờ, {task_stats.doing} làm, {task_stats.done} xong.\n"
                    f"- Ghi chú: {note_count}\n- Lịch 7 ngày tới: {event_count} sự kiện.")
    except Exception as e:
        return f"Lỗi thống kê: {e}"

@tool
def liet_ke_danh_sach(user_id: str, loai: str = 'all', gioi_han: int = 5) -> str:
    """Liệt kê danh sách ghi chú hoặc sự kiện theo yêu cầu."""
    try:
        with engine.connect() as conn:
            if loai in ['ghi chú', 'note']:
                rows = conn.execute(text("SELECT content FROM notes WHERE user_id = :uid ORDER BY created_at DESC LIMIT :l"),
                                    {"uid": user_id, "l": gioi_han}).fetchall()
                return "📝 GHI CHÚ:\n" + "\n".join([f"- {r.content[:50]}..." for r in rows]) if rows else "📭 Trống."
            else:
                rows = conn.execute(text("SELECT title, start_time FROM events WHERE user_id = :uid ORDER BY start_time ASC LIMIT :l"),
                                    {"uid": user_id, "l": gioi_han}).fetchall()
                return "📋 SỰ KIỆN:\n" + "\n".join([f"- {r.title} ({r.start_time})" for r in rows]) if rows else "📭 Trống."
    except Exception as e:
        return f"Lỗi: {e}"

@tool
def xem_chi_tiet_su_kien(user_id: str, tu_khoa: str) -> str:
    """Tìm kiếm chi tiết nội dung sự kiện hoặc ghi chú."""
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT title, description FROM events WHERE user_id = :uid AND title ILIKE :k LIMIT 1"),
                               {"uid": user_id, "k": f"%{tu_khoa}%"}).fetchone()
            if res: return f"🔎 {res.title}: {res.description}"
            return "⚠️ Không tìm thấy."
    except Exception as e:
        return f"Lỗi: {e}"

@tool
def tao_ghi_chu_thong_minh(noi_dung: str, user_id: str, context_title: Optional[str] = None) -> str:
    """Lưu ghi chú mới vào hệ thống."""
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO notes (user_id, content) VALUES (:uid, :c)"),
                         {"uid": user_id, "c": noi_dung})
            return "✅ Đã lưu ghi chú."
    except Exception as e:
        return f"Lỗi: {e}"

@tool
def xoa_su_kien_toan_tap(tieu_de: str, user_id: str) -> str:
    """Xóa hoàn toàn một sự kiện theo tiêu đề."""
    try:
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM events WHERE user_id = :uid AND title ILIKE :t"),
                         {"uid": user_id, "t": f"%{tieu_de}%"})
            return f"🗑️ Đã xóa '{tieu_de}'."
    except Exception as e:
        return f"Lỗi: {e}"

# --- 4. LẮP RÁP AGENT ---
tools = [
    lay_ten_nguoi_dung, tao_su_kien_toan_dien, cap_nhat_su_kien,
    thong_ke_tong_quan, liet_ke_danh_sach, xem_chi_tiet_su_kien,
    tao_ghi_chu_thong_minh, xoa_su_kien_toan_tap
]

system_prompt = f"""
Bạn là Skedule AI Agent. Hôm nay là {date.today().strftime('%d/%m/%Y')}
QUY TẮC:
1. Luôn dùng `lay_ten_nguoi_dung` khi bắt đầu chào hỏi.
2. Trả lời ngắn gọn, tiếng Việt tự nhiên.
3. Tự suy luận loại công việc để dùng tool phù hợp.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "USER_ID: {user_id}\nPROMPT: {input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_executor = AgentExecutor(
    agent=create_tool_calling_agent(llm_brain, tools, prompt_template),
    tools=tools, 
    verbose=True
)

store = {}
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store: store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor, get_history, input_messages_key="input", history_messages_key="chat_history"
)

# --- 5. API ENDPOINTS ---
app = FastAPI(title="Skedule AI Agent v1.5")

@app.post("/chat")
async def chat(prompt: Optional[str] = Form(None), 
               audio_file: Optional[UploadFile] = File(None), 
               user_id: str = Depends(get_current_user_id)):
    
    user_prompt = await audio_to_text(audio_file) if audio_file else prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Thiếu nội dung.")

    result = agent_with_history.invoke(
        {"input": user_prompt, "user_id": user_id}, 
        config={"configurable": {"session_id": f"user_{user_id}"}}
    )
    
    ai_text = result.get("output", "Lỗi phản hồi.")
    return {
        "user_prompt": user_prompt if audio_file else None, 
        "text_response": ai_text, 
        "audio_base64": text_to_base64_audio(ai_text)
    }