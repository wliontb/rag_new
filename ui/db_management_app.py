import streamlit as st
import pandas as pd
import logging
import os
import sys

# Đảm bảo có thể import các module từ thư mục gốc của dự án
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_core.db_manager import ChromaDBManager

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Quản lý Dữ liệu ChromaDB", layout="wide")

@st.cache_resource
def get_db_manager():
    """Khởi tạo và cache lại đối tượng ChromaDBManager."""
    return ChromaDBManager()

def load_data(db_manager):
    """Tải dữ liệu từ ChromaDB và chuyển thành DataFrame."""
    with st.spinner("Đang tải dữ liệu từ ChromaDB..."):
        data = db_manager.view_all_documents()
        if data and data['ids']:
            df = pd.DataFrame({
                'ID': data['ids'],
                'Nguồn': [meta.get('source', '') for meta in data['metadatas']],
                'Tiêu đề': [meta.get('title', '') for meta in data['metadatas']],
                'Nội dung': data['documents']
            })
            return df
    return pd.DataFrame()

def main():
    st.title("Bảng điều khiển Quản lý Dữ liệu ChromaDB")
    st.write("Xem và quản lý các tài liệu được lưu trữ trong cơ sở dữ liệu vector ChromaDB.")

    db_manager = get_db_manager()

    # --- Các nút hành động ---
    col1, col2, col3 = st.columns([1,1, 8])
    with col1:
        if st.button("Làm mới Dữ liệu", key="refresh"):
            st.cache_data.clear() # Xóa cache để tải lại dữ liệu mới
            st.rerun()

    with col2:
        if st.button("Xóa Toàn bộ Dữ liệu", key="delete"):
            st.session_state.show_confirmation = True

    # --- Hộp thoại xác nhận xóa ---
    if st.session_state.get('show_confirmation', False):
        st.warning("**Bạn có chắc chắn muốn xóa toàn bộ dữ liệu không? Hành động này không thể hoàn tác.**")
        c1, c2 = st.columns(2)
        if c1.button("Xác nhận Xóa", key="confirm_delete"):
            with st.spinner("Đang xóa collection..."):
                db_manager.delete_collection()
            st.session_state.show_confirmation = False
            st.success("Đã xóa thành công toàn bộ dữ liệu!")
            st.rerun()
        if c2.button("Hủy", key="cancel_delete"):
            st.session_state.show_confirmation = False
            st.rerun()

    # --- Hiển thị dữ liệu ---
    st.header("Danh sách Tài liệu trong ChromaDB")
    df = load_data(db_manager)

    if not df.empty:
        st.dataframe(df, height=600, use_container_width=True)
        st.info(f"Tìm thấy tổng cộng {len(df)} tài liệu.")
    else:
        st.info("Không có tài liệu nào trong cơ sở dữ liệu.")

if __name__ == "__main__":
    main()
