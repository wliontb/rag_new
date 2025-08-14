import os
from dotenv import load_dotenv
from pathlib import Path
import torch

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
SCRAPER_URL = "https://thoibaotaichinhvietnam.vn/chung-khoan&s_cond=&BRSR="

# --- Cấu hình mô hình ---
# Sử dụng một dictionary để dễ dàng quản lý và lựa chọn các mô hình
EMBEDDING_MODELS = {
    "vietnamese": "bkai-foundation-models/vietnamese-bi-encoder",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
}

GENERATION_MODEL_PRO = "gemini-2.5-pro"
GENERATION_MODEL_FLASH = "gemini-2.5-flash"

# --- Cấu hình ChromaDB ---
CHROMA_COLLECTION_NAME = "financial_news"

# GPU Configuration
GPU_ENABLED = torch.cuda.is_available()
GPU_MEMORY_FRACTION = 0.7  # Sử dụng 70% GPU memory
EMBEDDING_BATCH_SIZE = 32 if GPU_ENABLED else 8
MIXED_PRECISION = True

# Log GPU status
if GPU_ENABLED:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
else:
    print("GPU not available, using CPU")
