import streamlit as st
import sys
import os

# Thêm đường dẫn gốc của dự án vào sys.path
# để có thể import các module từ các thư mục khác
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rag_core.agent import QueryAgent
from config import settings

# Chỉ set page config nếu chưa được set
if 'page_config_set' not in st.session_state:
    st.set_page_config(page_title="Hệ thống RAG - Tin tức Tài chính VN", layout="wide")
    st.session_state.page_config_set = True

def run_chat_interface():
    """Chạy giao diện chat chính."""

    
    st.title("Hỏi đáp về Tin tức Tài chính Việt Nam")
    st.write("""
    Chào mừng bạn đến với hệ thống Hỏi-Đáp thông minh, chuyên sâu về lĩnh vực tài chính Việt Nam. 
    Đặt câu hỏi vào ô bên dưới và nhận câu trả lời được tổng hợp từ các nguồn tin tức uy tín.
    """)

    # --- Chọn Embedding Model/Collection --- #
    st.sidebar.title("⚙️ Cấu hình Hệ thống")

    # Dropdown chọn embedding model
    embedding_options = {
        "vietnamese": "Vietnamese BiEncoder (Tiếng Việt tối ưu)",
        "nomic": "Nomic Embed v1.5 (Đa ngôn ngữ)"
    }

    selected_embedding = st.sidebar.selectbox(
        "Chọn mô hình Embedding:",
        options=list(embedding_options.keys()),
        format_func=lambda x: embedding_options[x],
        key="embedding_selection"
    )

    # Hiển thị thông tin về collection được chọn
    st.sidebar.info(f"""
    **Collection hiện tại:** `{settings.CHROMA_COLLECTION_NAME}_{selected_embedding}`

    **Mô hình:** `{settings.EMBEDDING_MODELS[selected_embedding]}`
    """)

    # --- Khởi tạo Agent --- #
    # Reset agent nếu embedding model thay đổi
    if 'current_embedding' not in st.session_state:
        st.session_state.current_embedding = selected_embedding

    if st.session_state.current_embedding != selected_embedding:
        st.session_state.current_embedding = selected_embedding
        if 'agent' in st.session_state:
            del st.session_state.agent
        st.rerun()

    # Khởi tạo agent với embedding được chọn
    if 'agent' not in st.session_state:
        with st.spinner(f"Đang khởi tạo hệ thống với {embedding_options[selected_embedding]}..."):
            try:
                st.session_state.agent = QueryAgent(embedding_method=selected_embedding)
                st.success(f"Hệ thống đã sẵn sàng với {embedding_options[selected_embedding]}!")
            except Exception as e:
                st.error(f"Lỗi khởi tạo hệ thống: {e}")
                st.stop()

    # --- Giao diện chính --- #
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input(
            "**Nhập câu hỏi của bạn:**", 
            placeholder="Ví dụ: Tình hình xuất khẩu thủy sản của Việt Nam trong quý 2 năm nay?"
        )

    with col2:
        st.write("")  # Spacer
        search_button = st.button("🔍 Tìm kiếm câu trả lời", use_container_width=True)

    # Hiển thị thống kê collection
    if st.sidebar.button("📊 Xem thống kê Collection"):
        try:
            collection_stats = st.session_state.agent.db_manager.get_collection_stats()
            st.sidebar.success(f"**Số lượng documents:** {collection_stats['count']}")
            if collection_stats.get('sample_metadata'):
                st.sidebar.json(collection_stats['sample_metadata'])
        except Exception as e:
            st.sidebar.error(f"Không thể lấy thống kê: {e}")

    if search_button:
        if not question:
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang tìm kiếm và tổng hợp câu trả lời..."):
                try:
                    answer, retrieved_docs = st.session_state.agent.answer(question)

                    if not retrieved_docs:
                        st.warning("Không tìm thấy thông tin liên quan đến câu hỏi của bạn.")
                    else:
                        st.subheader("Câu trả lời tổng hợp:")
                        st.markdown(answer)

                        st.subheader("Các nguồn thông tin được sử dụng:")
                        for i, doc in enumerate(retrieved_docs):
                            with st.expander(f"Nguồn {i+1}: {doc.metadata.get('title', 'Không có tiêu đề')}"):
                                st.write(f"**Nguồn:** [{doc.metadata.get('source')}]({doc.metadata.get('source')})")
                                st.write("**Nội dung trích dẫn:**")
                                st.caption(doc.page_content)
                                
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
                    # Debug information
                    with st.expander("Chi tiết lỗi (Debug)"):
                        st.code(str(e))
                        import traceback
                        st.code(traceback.format_exc())

# Chạy interface nếu file được gọi trực tiếp
if __name__ == "__main__":
    run_chat_interface()
