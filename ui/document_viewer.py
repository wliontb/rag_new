import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rag_core.db_manager import MultiEmbeddingDBManager
from config import settings

def run_document_viewer():
    """Chạy giao diện xem documents."""
    
    st.title("📄 Document Viewer")
    st.write("Xem và tìm kiếm tất cả documents trong các collections")

    # --- Sidebar điều khiển --- #
    st.sidebar.title("🔍 Bộ lọc & Tìm kiếm")

    # Chọn collection
    embedding_options = {
        "vietnamese": "Vietnamese BiEncoder",
        "nomic": "Nomic Embed v1.5"
    }

    selected_collection = st.sidebar.selectbox(
        "Chọn Collection:",
        options=list(embedding_options.keys()),
        format_func=lambda x: embedding_options[x],
        key="collection_selection"
    )

    # Khởi tạo DB Manager
    @st.cache_resource
    def get_multi_db_manager():
        return MultiEmbeddingDBManager(
            embedding_models=settings.EMBEDDING_MODELS,
            use_gpu=settings.GPU_ENABLED
        )

    multi_db = get_multi_db_manager()
    db_manager = multi_db.get_db_manager(selected_collection)

    if not db_manager:
        st.error(f"Không thể kết nối với collection: {selected_collection}")
        return

    # Lấy thông tin collection
    collection_stats = db_manager.get_collection_stats()
    
    # Hiển thị thống kê
    st.sidebar.info(f"""
    **Collection:** `{collection_stats['collection_name']}`
    **Tổng documents:** {collection_stats['count']:,}
    **Model:** `{collection_stats['embedding_model'].split('/')[-1]}`
    """)

    # Tìm kiếm text
    search_text = st.sidebar.text_input(
        "🔍 Tìm kiếm trong nội dung:",
        placeholder="Nhập từ khóa..."
    )

    # Lọc theo ngày
    st.sidebar.subheader("📅 Lọc theo ngày")
    date_filter = st.sidebar.checkbox("Bật lọc ngày")
    
    start_date = None
    end_date = None
    if date_filter:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Từ ngày:", value=None)
        with col2:
            end_date = st.date_input("Đến ngày:", value=None)

    # Số lượng documents hiển thị
    limit = st.sidebar.slider("Số documents hiển thị:", 10, 200, 50)

    # Refresh button
    if st.sidebar.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    # --- Main Content --- #

    # Lấy tất cả documents
    @st.cache_data
    def get_all_documents(collection_name, limit_docs):
        try:
            collection = db_manager.get_collection()
            if collection:
                # Lấy documents với metadata
                results = collection.get(
                    limit=limit_docs,
                    include=["metadatas", "documents", "ids"]
                )
                return results
            return None
        except Exception as e:
            st.error(f"Lỗi khi lấy documents: {e}")
            return None

    # Load documents
    with st.spinner("Đang tải documents..."):
        documents_data = get_all_documents(collection_stats['collection_name'], limit)

    if not documents_data or not documents_data.get('documents'):
        st.warning("Không có documents nào trong collection này.")
        return

    # Chuyển đổi thành DataFrame để dễ xử lý
    docs_list = []
    for i, (doc_id, document, metadata) in enumerate(zip(
        documents_data['ids'],
        documents_data['documents'], 
        documents_data['metadatas']
    )):
        docs_list.append({
            'id': doc_id,
            'content': document,
            'title': metadata.get('title', 'No title'),
            'source': metadata.get('source', 'Unknown'),
            'date': metadata.get('date', 'Unknown'),
            'metadata': metadata
        })

    df = pd.DataFrame(docs_list)

    # Áp dụng filters
    filtered_df = df.copy()

    # Text search filter
    if search_text:
        mask = (
            filtered_df['content'].str.contains(search_text, case=False, na=False) |
            filtered_df['title'].str.contains(search_text, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    # Date filter
    if date_filter and start_date and end_date:
        # Convert date strings to datetime for comparison
        def parse_date(date_str):
            try:
                if isinstance(date_str, str):
                    return datetime.strptime(date_str.split('T')[0], '%Y-%m-%d').date()
                return None
            except:
                return None
        
        filtered_df['parsed_date'] = filtered_df['date'].apply(parse_date)
        mask = (
            (filtered_df['parsed_date'] >= start_date) & 
            (filtered_df['parsed_date'] <= end_date)
        )
        filtered_df = filtered_df[mask]

    # Hiển thị kết quả
    st.subheader(f"📊 Kết quả: {len(filtered_df):,} / {len(df):,} documents")

    if len(filtered_df) == 0:
        st.warning("Không tìm thấy documents nào phù hợp với bộ lọc.")
        return

    # Hiển thị options
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        view_mode = st.selectbox(
            "Chế độ hiển thị:",
            ["📋 Danh sách", "📄 Chi tiết", "📊 Bảng"]
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sắp xếp theo:",
            ["date", "title", "id"],
            format_func=lambda x: {"date": "Ngày", "title": "Tiêu đề", "id": "ID"}[x]
        )
    
    with col3:
        sort_order = st.selectbox("Thứ tự:", ["desc", "asc"], format_func=lambda x: "↓" if x == "desc" else "↑")

    # Sắp xếp
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=(sort_order == "asc"))

    # Hiển thị theo mode
    if view_mode == "📊 Bảng":
        # Table view
        display_df = filtered_df[['title', 'source', 'date']].copy()
        display_df['content_preview'] = filtered_df['content'].str[:100] + "..."
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "title": "Tiêu đề",
                "source": st.column_config.LinkColumn("Nguồn"),
                "date": "Ngày",
                "content_preview": "Nội dung (preview)"
            }
        )

    elif view_mode == "📋 Danh sách":
        # List view
        for idx, row in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"📅 {row['date']} | 🔗 {row['source']}")
                    st.write(row['content'][:200] + "..." if len(row['content']) > 200 else row['content'])
                
                with col2:
                    if st.button("👁️ Xem", key=f"view_{idx}"):
                        st.session_state[f"show_detail_{idx}"] = True
                
                # Show detail if clicked
                if st.session_state.get(f"show_detail_{idx}", False):
                    with st.expander("📄 Nội dung đầy đủ", expanded=True):
                        st.write("**Nội dung:**")
                        st.text_area("", value=row['content'], height=200, disabled=True, key=f"content_{idx}")
                        st.write("**Metadata:**")
                        st.json(row['metadata'])
                        
                        if st.button("❌ Đóng", key=f"close_{idx}"):
                            st.session_state[f"show_detail_{idx}"] = False
                            st.rerun()
                
                st.divider()

    else:  # Chi tiết view
        # Detailed view
        selected_idx = st.selectbox(
            "Chọn document để xem chi tiết:",
            range(len(filtered_df)),
            format_func=lambda x: f"{filtered_df.iloc[x]['title'][:50]}..." if len(filtered_df.iloc[x]['title']) > 50 else filtered_df.iloc[x]['title']
        )
        
        if selected_idx is not None:
            row = filtered_df.iloc[selected_idx]
            
            st.subheader(f"📄 {row['title']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ID", row['id'])
            with col2:
                st.metric("Ngày", row['date'])
            with col3:
                st.metric("Độ dài", f"{len(row['content'])} ký tự")
            
            st.write("**🔗 Nguồn:**", row['source'])
            
            st.write("**📝 Nội dung:**")
            st.text_area("", value=row['content'], height=300, disabled=True)
            
            st.write("**🏷️ Metadata:**")
            st.json(row['metadata'])

    # Export functionality
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export CSV"):
            csv_data = filtered_df[['id', 'title', 'source', 'date', 'content']].to_csv(index=False)
            st.download_button(
                label="⬇️ Tải xuống CSV",
                data=csv_data,
                file_name=f"documents_{selected_collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Thống kê nhanh"):
            st.write("### 📈 Thống kê Documents")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng documents", len(filtered_df))
            with col2:
                avg_length = filtered_df['content'].str.len().mean()
                st.metric("Độ dài TB", f"{avg_length:.0f} ký tự")
            with col3:
                unique_sources = filtered_df['source'].nunique()
                st.metric("Nguồn khác nhau", unique_sources)
            
            # Chart: Documents per day
            if 'parsed_date' in filtered_df.columns:
                date_counts = filtered_df.groupby('parsed_date').size().reset_index(name='count')
                if len(date_counts) > 1:
                    st.line_chart(date_counts.set_index('parsed_date'))

# Chạy interface nếu file được gọi trực tiếp
if __name__ == "__main__":
    run_document_viewer()
