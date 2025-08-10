import os
from dotenv import load_dotenv
from pathlib import Path

# Tải biến môi trường từ file .env
load_dotenv()

# Đường dẫn gốc của dự án
ROOT_DIR = Path(__file__).parent.parent

# --- Cấu hình API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# --- Cấu hình đường dẫn ---
DATA_DIR = ROOT_DIR / "data"
DB_DIR = DATA_DIR / "chroma_db"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.json"
EVAL_RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

# --- Cấu hình Scraper ---
SCRAPER_URL = "https://thoibaotaichinhvietnam.vn/chung-khoan"

# --- Cấu hình mô hình ---
# Sử dụng một dictionary để dễ dàng quản lý và lựa chọn các mô hình
EMBEDDING_MODELS = {
    "vietnamese": "bkai-foundation-models/vietnamese-bi-encoder",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
}

GENERATION_MODEL_PRO = "gemini-1.5-pro-latest"
GENERATION_MODEL_FLASH = "gemini-1.5-flash-latest"

# --- Cấu hình ChromaDB ---
CHROMA_COLLECTION_NAME = "financial_news"
