# File: utils/thoi_gian_tu_nhien.py

import re
from datetime import datetime, timedelta

# --- CÁC HÀM PHỤ TRỢ ---


def add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)


def add_years(dt: datetime, years: int) -> datetime:
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        return dt.replace(month=2, day=28, year=dt.year + years)


def end_of_month(dt: datetime) -> datetime:
    next_month = add_months(dt, 1).replace(day=1)
    return next_month - timedelta(days=1)

# --- HÀM CHÍNH ĐỂ EXPORT ---


def parse_natural_time(expression: str, base_date: datetime) -> tuple[datetime, datetime]:
    """
    Chuyển cụm thời gian tiếng Việt thành (ngày_bắt_đầu, ngày_kết_thúc).
    Hàm này đã được sửa để xử lý cả ngày giờ cụ thể.
    """
    try:
        # Ưu tiên 1: Thử phân tích chuỗi ngày giờ đầy đủ (ví dụ: '2025-10-20 09:00:00')
        new_start = datetime.fromisoformat(expression)
        return new_start, new_start + timedelta(hours=1)
    except (ValueError, TypeError):
        # Nếu không phải chuỗi đầy đủ, tiếp tục xử lý ngôn ngữ tự nhiên
        pass

    expr = expression.lower().strip()
    new_start = base_date

    match = re.search(r"(\d+)\s*(ngày|tuần|tháng|năm)\s*(sau|tới|trước)", expr)
    if match:
        amount, unit, direction = int(
            match.group(1)), match.group(2), match.group(3)
        multiplier = 1 if direction in ["sau", "tới"] else -1

        if unit == "ngày":
            new_start += timedelta(days=amount * multiplier)
        elif unit == "tuần":
            new_start += timedelta(weeks=amount * multiplier)
        elif unit == "tháng":
            new_start = add_months(new_start, amount * multiplier)
        elif unit == "năm":
            new_start = add_years(new_start, amount * multiplier)

    # Thêm các logic khác nếu cần, ví dụ: "ngày mai", "tuần sau"
    elif "mai" in expr or "ngày sau" in expr:
        new_start += timedelta(days=1)

    return new_start, new_start + timedelta(hours=1)
