import streamlit as st
import sys
import os

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import settings

st.set_page_config(
    page_title="RAG System - Tài chính VN", 
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Navigation --- #
st.sidebar.title("🏦 RAG System")
st.sidebar.write("Hệ thống Hỏi-Đáp Tài chính Việt Nam")

page = st.sidebar.selectbox(
    "Chọn trang:",
    ["🏠 Trang chủ", "💬 Hỏi đáp", "📄 Xem Documents", "📊 Dashboard", "🗄️ Quản lý Collections"]
)

# --- Sidebar System Info --- #
st.sidebar.divider()
st.sidebar.subheader("ℹ️ Thông tin hệ thống")

# Hiển thị thông tin các embedding models
st.sidebar.write("**Embedding Models:**")
for name, model in settings.EMBEDDING_MODELS.items():
    st.sidebar.write(f"• {name}: `{model.split('/')[-1]}`")

# GPU status
gpu_status = "🟢 Available" if settings.GPU_ENABLED else "🔴 Not Available"
st.sidebar.write(f"**GPU:** {gpu_status}")

# Database info
st.sidebar.write(f"**Database:** ChromaDB")
st.sidebar.write(f"**Base Collection:** `{settings.CHROMA_COLLECTION_NAME}`")

# --- Main Content --- #
if page == "🏠 Trang chủ":
    st.title("🏦 Hệ thống RAG - Tin tức Tài chính Việt Nam")
    
    st.write("""
    Chào mừng bạn đến với hệ thống Retrieval-Augmented Generation (RAG) chuyên về tin tức tài chính Việt Nam.
    Hệ thống sử dụng nhiều embedding models để cung cấp trải nghiệm tìm kiếm thông tin tốt nhất.
    """)
    
    # Feature overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("💬 Hỏi đáp thông minh")
        st.write("""
        - Trả lời câu hỏi dựa trên dữ liệu tin tức
        - Hỗ trợ 2 embedding models
        - Tìm kiếm theo ngày tháng
        - GPU acceleration
        """)
    
    with col2:
        st.subheader("📄 Xem Documents")
        st.write("""
        - Duyệt tất cả documents
        - Tìm kiếm và lọc nâng cao
        - Xem chi tiết từng document
        - Export dữ liệu CSV
        """)
    
    with col3:
        st.subheader("📊 Dashboard đánh giá")
        st.write("""
        - So sánh hiệu suất models
        - Trực quan hóa kết quả
        - Metrics đa dạng
        - Export báo cáo
        """)
    
    with col4:
        st.subheader("🗄️ Quản lý dữ liệu")
        st.write("""
        - Xem thông tin collections
        - So sánh embedding models
        - Test queries
        - Quản lý database
        """)
    
    # Quick stats
    st.divider()
    st.subheader("📈 Thống kê nhanh")
    
    try:
        from rag_core.db_manager import MultiEmbeddingDBManager
        
        multi_db = MultiEmbeddingDBManager()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vietnamese_db = multi_db.get_db_manager('vietnamese')
            if vietnamese_db:
                vietnamese_stats = vietnamese_db.get_collection_stats()
                st.metric("Vietnamese Collection", vietnamese_stats.get('count', 0), "documents")
        
        with col2:
            nomic_db = multi_db.get_db_manager('nomic')
            if nomic_db:
                nomic_stats = nomic_db.get_collection_stats()
                st.metric("Nomic Collection", nomic_stats.get('count', 0), "documents")
        
        with col3:
            total_docs = vietnamese_stats.get('count', 0) + nomic_stats.get('count', 0)
            st.metric("Tổng Documents", total_docs, "trong 2 collections")
        
        with col4:
            # Hiển thị trạng thái GPU
            gpu_status = "🟢 Available" if settings.GPU_ENABLED else "🔴 CPU Only"
            st.metric("GPU Status", gpu_status, "")
            
    except Exception as e:
        st.warning("Không thể tải thông tin database. Vui lòng kiểm tra kết nối.")
    
    # Getting started
    st.divider()
    st.subheader("🚀 Bắt đầu")
    
    st.write("""
    **Để sử dụng hệ thống:**
    
    1. **Hỏi đáp:** Chọn trang "💬 Hỏi đáp" để bắt đầu đặt câu hỏi
    2. **Chọn Model:** Sử dụng dropdown để chọn embedding model phù hợp
    3. **Xem Documents:** Sử dụng "📄 Xem Documents" để duyệt và tìm kiếm tài liệu
    4. **Dashboard:** Xem "📊 Dashboard" để so sánh hiệu suất các models
    5. **Quản lý:** Sử dụng "🗄️ Quản lý Collections" để xem thông tin chi tiết
    """)

elif page == "💬 Hỏi đáp":
    # Chạy app hỏi đáp
    try:
        # Clear existing page config từ app.py
        if 'page_config_set' in st.session_state:
            del st.session_state.page_config_set
        
        # Import và chạy function
        from ui.app import run_chat_interface
        run_chat_interface()
    except Exception as e:
        st.error(f"Lỗi khi tải trang hỏi đáp: {e}")
        st.write("Vui lòng thử lại hoặc liên hệ support.")

elif page == "📄 Xem Documents":
    # Chạy document viewer
    try:
        from ui.document_viewer import run_document_viewer
        run_document_viewer()
    except Exception as e:
        st.error(f"Lỗi khi tải document viewer: {e}")
        st.write("Vui lòng thử lại hoặc liên hệ support.")

elif page == "📊 Dashboard":
    # Chạy dashboard
    try:
        dashboard_path = os.path.join(project_root, "ui", "dashboard.py")
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_code = f.read()
        exec(dashboard_code)
    except Exception as e:
        st.error(f"Lỗi khi tải dashboard: {e}")

elif page == "🗄️ Quản lý Collections":
    # Chạy collection manager
    try:
        collection_path = os.path.join(project_root, "ui", "collection_manager.py")
        with open(collection_path, 'r', encoding='utf-8') as f:
            collection_code = f.read()
        exec(collection_code)
    except Exception as e:
        st.error(f"Lỗi khi tải collection manager: {e}")
