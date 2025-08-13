import streamlit as st
import sys
import os
import pandas as pd

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rag_core.db_manager import MultiEmbeddingDBManager
from config import settings

def run_collection_manager():
    """Chạy giao diện quản lý collections."""
    
    st.title("🗄️ Quản lý Collections Database")
    st.write("Trang quản lý và so sánh các collections sử dụng các embedding models khác nhau.")

    # --- Khởi tạo Multi DB Manager --- #
    @st.cache_resource
    def get_multi_db_manager():
        return MultiEmbeddingDBManager(
            embedding_models=settings.EMBEDDING_MODELS,
            use_gpu=settings.GPU_ENABLED
        )

    multi_db = get_multi_db_manager()

    # --- Sidebar điều khiển --- #
    st.sidebar.title("⚙️ Điều khiển")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

    # --- Hiển thị thông tin tất cả collections --- #
    st.subheader("📊 Thông tin tất cả Collections")

    collections_info = []
    for model_name, db_manager in multi_db.db_managers.items():
        stats = db_manager.get_collection_stats()
        collections_info.append({
            "Model Name": model_name,
            "Collection Name": stats["collection_name"],
            "Document Count": stats["count"],
            "Embedding Model": stats["embedding_model"],
            "GPU Enabled": stats["gpu_enabled"],
            "Status": "✅ Active" if stats["count"] > 0 else "⚠️ Empty"
        })

    # Tạo DataFrame và hiển thị
    df_collections = pd.DataFrame(collections_info)
    st.dataframe(df_collections, use_container_width=True)

    # --- So sánh Collections --- #
    st.subheader("🔍 So sánh Collections")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Collection 1: Vietnamese Embedding**")
        vietnamese_db = multi_db.get_db_manager('vietnamese')
        if vietnamese_db:
            vietnamese_stats = vietnamese_db.get_collection_stats()
            st.json(vietnamese_stats)

    with col2:
        st.write("**Collection 2: Nomic Embedding**")
        nomic_db = multi_db.get_db_manager('nomic')
        if nomic_db:
            nomic_stats = nomic_db.get_collection_stats()
            st.json(nomic_stats)

    # --- Test Query trên tất cả Collections --- #
    st.subheader("🧪 Test Query trên tất cả Collections")

    test_query = st.text_input(
        "Nhập câu hỏi để test:",
        placeholder="VD: Tình hình kinh tế Việt Nam"
    )

    k_results = st.slider("Số lượng kết quả:", min_value=1, max_value=10, value=3)

    if st.button("🚀 Chạy Test Query") and test_query:
        with st.spinner("Đang truy vấn tất cả collections..."):
            try:
                results = multi_db.query_all_models(test_query, k=k_results)
                
                for model_name, docs in results.items():
                    st.write(f"### Kết quả từ {model_name.upper()} ({len(docs)} documents)")
                    
                    if docs:
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"Document {i}: {doc.metadata.get('title', 'No title')[:100]}..."):
                                st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                st.write(f"**Content:** {doc.page_content[:300]}...")
                                st.json(doc.metadata)
                    else:
                        st.warning(f"Không tìm thấy kết quả nào cho {model_name}")
                        
                    st.divider()
                    
            except Exception as e:
                st.error(f"Lỗi khi test query: {e}")

    # --- Quản lý Collections --- #
    st.subheader("🛠️ Quản lý Collections")

    st.warning("⚠️ **Cảnh báo:** Các thao tác bên dưới sẽ thay đổi dữ liệu trong database.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Xóa tất cả Collections", type="secondary"):
            if st.checkbox("Tôi hiểu rủi ro và muốn xóa tất cả"):
                try:
                    multi_db.delete_all_collections()
                    st.success("Đã xóa tất cả collections!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa: {e}")

    with col2:
        selected_model = st.selectbox(
            "Chọn model để xóa:",
            options=list(settings.EMBEDDING_MODELS.keys())
        )
        
        if st.button(f"🗑️ Xóa Collection {selected_model}"):
            if st.checkbox(f"Xóa collection {selected_model}"):
                try:
                    db_manager = multi_db.get_db_manager(selected_model)
                    if db_manager:
                        db_manager.delete_collection()
                        st.success(f"Đã xóa collection {selected_model}!")
                        st.cache_resource.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa: {e}")

    with col3:
        st.write("**Rebuild Collections**")
        if st.button("🔄 Rebuild từ dữ liệu"):
            st.info("Chức năng này cần được implement trong data_ingestion pipeline")

    # --- Footer --- #
    st.divider()
    st.caption("💡 **Tip:** Sử dụng trang này để kiểm tra và so sánh hiệu suất của các embedding models khác nhau.")

# Chạy interface nếu file được gọi trực tiếp
if __name__ == "__main__":
    run_collection_manager()
