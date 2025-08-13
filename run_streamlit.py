#!/usr/bin/env python3
"""
Script để chạy ứng dụng Streamlit RAG System.

Sử dụng:
    python run_streamlit.py [port]

Ví dụ:
    python run_streamlit.py 8501
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    # Lấy đường dẫn đến thư mục UI
    ui_dir = Path(__file__).parent / "ui"
    main_app = ui_dir / "main.py"
    
    # Port mặc định
    port = "8501"
    
    # Nếu có argument port
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    # Kiểm tra file main.py có tồn tại không
    if not main_app.exists():
        print(f"❌ Không tìm thấy file: {main_app}")
        print("Vui lòng đảm bảo bạn đang chạy script từ thư mục gốc của dự án.")
        sys.exit(1)
    
    print(f"🚀 Đang khởi chạy RAG System trên port {port}...")
    print(f"📂 File ứng dụng: {main_app}")
    print(f"🌐 URL: http://localhost:{port}")
    print("=" * 50)
    
    # Chạy Streamlit
    try:
        cmd = [
            "streamlit", "run", str(main_app),
            "--server.port", port,
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ]
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Đã dừng ứng dụng.")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Không tìm thấy Streamlit. Vui lòng cài đặt:")
        print("pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
