from pathlib import Path
import os
import logging
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "")

LOG_FILE = Path(__file__).resolve().parent / "trading.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

SYMBOLS_FILE = Path("symbols.txt")

if SYMBOLS_FILE.exists():
    with open(SYMBOLS_FILE, "r", encoding="utf-8") as f:
        SYMBOLS = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

TIMEFRAMES = ["5m", "15m", "1h", "4h"]
UPDATE_SECONDS = 120

# آستانه‌های موتور سیگنال در phase7.py تنظیم می‌شوند:
#   MIN_SCORE_THRESHOLD = 25   حداقل نمره ۰-۱۰۰ برای صدور سیگنال
#   MIN_SCORE_MARGIN    = 15   حداقل فاصله نمره صعودی و نزولی
#   MIN_CONFLUENCE_COUNT = 2   حداقل تعداد تایمفریم همراستا
# برای سیگنال بیشتر آستانه را کاهش و برای سیگنال باکیفیت‌تر افزایش دهید.

# تعداد کندل‌های ذخیره‌شده برای هر تایمفریم
MAX_CANDLES_PER_TF = {
    "5m": 1500,
    "15m": 1000,
    "1h": 500,
    "4h": 200,
}

# زمان چک کردن برای وجود کندل جدید (دقیقه)
# ۰ یعنی هر بار چک کن (مثل ۵ دقیقه)
CHECK_INTERVAL_MINUTES = {
    "5m": 0,      # هر بار (هیچ تغییری)
    "15m": 7,     # هر ۷ دقیقه
    "1h": 20,     # هر ۲۰ دقیقه
    "4h": 60,     # هر ۱ ساعت
}

# True: سیگنال‌ها فقط از کندل‌های confirmed/closed ساخته می‌شوند (آخرین کندل forming رد می‌شود).
# False: رفتار فعلی — آخرین کندل موجود در دیتافریم در نظر گرفته می‌شود.
USE_CLOSED_CANDLES_ONLY = True

# سرمایه پیشفرض برای محاسبه حجم معامله (بر حسب واحد ارز پایه، مثلاً USDT)
CAPITAL = 1000.0
