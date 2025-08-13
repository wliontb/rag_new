# 🏦 Hệ thống RAG - Tin tức Tài chính Việt Nam

## Tính năng mới: Multi-Collection Support

Hệ thống hiện đã được cập nhật để hỗ trợ nhiều embedding models và collections:

### 🆕 Các tính năng chính:

1. **Lựa chọn Embedding Model linh hoạt**
   - Vietnamese BiEncoder (tối ưu cho tiếng Việt)
   - Nomic Embed v1.5 (đa ngôn ngữ)

2. **Giao diện Streamlit cải tiến**
   - Dropdown chọn embedding model
   - Thông tin collection real-time
   - Thống kê documents

3. **Multi-page Navigation**
   - 🏠 Trang chủ với thống kê tổng quan
   - 💬 Hỏi đáp với lựa chọn model
   - � Xem Documents với tìm kiếm nâng cao
   - �📊 Dashboard đánh giá hiệu suất
   - 🗄️ Quản lý collections

4. **Document Viewer** (mới)
   - Duyệt tất cả documents trong collection
   - Tìm kiếm text trong nội dung
   - Lọc theo ngày tháng
   - 3 chế độ hiển thị: Danh sách, Chi tiết, Bảng
   - Export dữ liệu CSV
   - Thống kê nhanh về documents

5. **Collection Management**
   - Xem thống kê tất cả collections
   - So sánh hiệu suất các models
   - Test query trên nhiều collections
   - Quản lý dữ liệu database

## 🚀 Cách sử dụng

### 1. Chạy Data Ingestion (nếu chưa có dữ liệu)
```bash
python -m data_ingestion.main
```

### 2. Chạy ứng dụng Streamlit
```bash
# Cách 1: Sử dụng script tiện ích
python run_streamlit.py

# Cách 2: Chạy trực tiếp
streamlit run ui/main.py

# Cách 3: Chỉ chạy trang hỏi đáp
streamlit run ui/app.py
```

### 3. Sử dụng hệ thống

1. **Truy cập giao diện:** Mở trình duyệt tại `http://localhost:8501`

2. **Chọn Embedding Model:**
   - Vào trang "💬 Hỏi đáp"
   - Sử dụng dropdown ở sidebar để chọn model
   - Vietnamese: tối ưu cho câu hỏi tiếng Việt
   - Nomic: tốt cho câu hỏi đa ngôn ngữ

3. **Duyệt Documents:**
   - Vào trang "📄 Xem Documents"
   - Chọn collection muốn xem
   - Sử dụng bộ lọc tìm kiếm:
     - Tìm kiếm text trong nội dung
     - Lọc theo ngày tháng
     - Chọn số lượng documents hiển thị
   - Xem documents ở 3 chế độ: Danh sách, Chi tiết, Bảng
   - Export dữ liệu CSV nếu cần

4. **Đặt câu hỏi:**
   - Nhập câu hỏi về tài chính Việt Nam
   - Hệ thống sẽ tìm kiếm trong collection được chọn
   - Xem câu trả lời và các nguồn tham khảo

5. **Quản lý Collections:**
   - Vào trang "🗄️ Quản lý Collections"
   - Xem thống kê và so sánh collections
   - Test query trên nhiều collections cùng lúc

## 📊 Thông tin Collections

Hệ thống tạo các collections sau:
- `financial_news_vietnamese`: Sử dụng Vietnamese BiEncoder
- `financial_news_nomic`: Sử dụng Nomic Embed v1.5

Mỗi collection chứa cùng dữ liệu nhưng với vector embeddings khác nhau.

## 🔧 Cấu hình

### Thay đổi Embedding Models
Chỉnh sửa trong `config/settings.py`:
```python
EMBEDDING_MODELS = {
    "vietnamese": "bkai-foundation-models/vietnamese-bi-encoder",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    # Thêm models khác nếu cần
}
```

### GPU Support
- Tự động detect GPU và sử dụng nếu có
- Hiển thị trạng thái GPU trong sidebar
- Tối ưu batch processing cho GPU

## 🛠️ Troubleshooting

### Lỗi Collection không tồn tại
```bash
# Chạy lại data ingestion để tạo collections
python -m data_ingestion.main
```

### Lỗi Memory khi sử dụng GPU
```python
# Giảm batch size trong settings.py
EMBEDDING_BATCH_SIZE = 16  # thay vì 32
```

### Lỗi Import Module
```bash
# Đảm bảo đang ở thư mục gốc của dự án
cd /path/to/rag_new
python run_streamlit.py
```

## 📁 Cấu trúc File UI mới

```
ui/
├── main.py              # Trang điều hướng chính  
├── app.py               # Giao diện hỏi đáp
├── document_viewer.py   # Trang xem và duyệt documents
├── dashboard.py         # Dashboard đánh giá
├── collection_manager.py # Quản lý collections
└── db_management_app.py # (legacy)
```

## 🎯 Workflow khuyến nghị

1. **Setup lần đầu:**
   - Chạy data ingestion để tạo cả 2 collections
   - Kiểm tra collections qua trang quản lý

2. **Sử dụng hàng ngày:**
   - Chọn model phù hợp với loại câu hỏi
   - Vietnamese model cho câu hỏi tiếng Việt thuần túy
   - Nomic model cho câu hỏi phức tạp hoặc đa ngôn ngữ
   - Sử dụng Document Viewer để duyệt và tìm hiểu dữ liệu

3. **Đánh giá hiệu suất:**
   - Sử dụng trang test query để so sánh
   - Chọn model tốt nhất cho use case cụ thể

## 💡 Tips

- **Hiệu suất:** Vietnamese model thường tốt hơn cho câu hỏi tiếng Việt
- **Linh hoạt:** Nomic model tốt cho câu hỏi có thuật ngữ nước ngoài  
- **Duyệt dữ liệu:** Sử dụng Document Viewer để hiểu rõ hơn về dataset
- **Tìm kiếm:** Combine text search và date filter để tìm documents chính xác
- **Export:** Download CSV để phân tích offline
- **Memory:** Sử dụng GPU nếu có để tăng tốc độ
- **Debugging:** Xem logs trong terminal khi chạy ứng dụng
