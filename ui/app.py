import streamlit as st
import sys
import os

# Thêm đường dẫn gốc của dự án vào sys.path
# để có thể import các module từ các thư mục khác
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rag_core.agent import QueryAgent
from config import settings

st.set_page_config(page_title="Hệ thống RAG - Tin tức Tài chính VN", layout="wide")

st.title("Hỏi đáp về Tin tức Tài chính Việt Nam")
st.write("""
Chào mừng bạn đến với hệ thống Hỏi-Đáp thông minh, chuyên sâu về lĩnh vực tài chính Việt Nam. 
Đặt câu hỏi vào ô bên dưới và nhận câu trả lời được tổng hợp từ các nguồn tin tức uy tín.
""")

# --- Khởi tạo Agent --- #
# Sử dụng session state để chỉ khởi tạo Agent một lần
if 'agent' not in st.session_state:
    with st.spinner("Đang khởi tạo hệ thống, vui lòng chờ..."):
        try:
            # Mặc định sử dụng mô hình embedding "vietnamese"
            st.session_state.agent = QueryAgent(embedding_method='vietnamese')
            st.success("Hệ thống đã sẵn sàng!")
        except Exception as e:
            st.error(f"Lỗi khởi tạo hệ thống: {e}")
            st.stop()

# --- Giao diện chính --- #
question = st.text_input(
    "**Nhập câu hỏi của bạn:**", 
    placeholder="Ví dụ: Tình hình xuất khẩu thủy sản của Việt Nam trong quý 2 năm nay?"
)

if st.button("Tìm kiếm câu trả lời"):
    if not question:
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("Đang tìm kiếm và tổng hợp câu trả lời..."):
            try:
                answer, retrieved_docs = st.session_state.agent.answer(question)

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
