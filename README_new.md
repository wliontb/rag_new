# Blueprint: Hệ thống RAG cho Tin tức Tài chính Việt Nam

## 1. Tổng quan và Tầm nhìn

Tài liệu này mô tả kiến trúc và cách triển khai một hệ thống **Retrieval-Augmented Generation (RAG)** hoàn chỉnh, chuyên biệt cho lĩnh vực tin tức tài chính Việt Nam. Mục tiêu là xây dựng một hệ thống không chỉ trả lời câu hỏi chính xác dựa trên dữ liệu mới nhất mà còn phải có cấu trúc rõ ràng, dễ bảo trì, mở rộng và có hiệu năng được đo lường, tối ưu hóa một cách bài bản.

Dự án này sẽ đóng vai trò là một khuôn mẫu (blueprint) chi tiết, từ việc thu thập dữ liệu, đánh giá mô hình, cho đến việc triển khai giao diện tương tác.

### Các tính năng cốt lõi

*   **Thu thập dữ liệu tự động**: Cào dữ liệu từ các trang tin tức tài chính (`thoibaotaichinhvietnam.vn`).
*   **Tạo sinh dữ liệu đánh giá (Synthetic Ground Truth)**: Tự động tạo các cặp Hỏi-Đáp (QA) từ nội dung bài báo bằng LLM, làm cơ sở cho việc đánh giá.
*   **Lưu trữ và truy vấn Vector**: Chuyển hóa văn bản thành vector embeddings và lưu trữ trong **ChromaDB**.
*   **Kiến trúc RAG thông minh**:
    *   **Phân tích thời gian trong truy vấn**: Tự động nhận diện và trích xuất các mốc thời gian (ví dụ: "tuần trước", "quý 2 năm nay") để lọc kết quả tìm kiếm.
    *   **Pipeline linh hoạt**: Hỗ trợ và so sánh nhiều mô hình embedding khác nhau (ví dụ: `vietnamese-bi-encoder`, `nomic-embed-text-v1.5`).
*   **Quy trình đánh giá toàn diện**: Sử dụng framework **Ragas** để đo lường hiệu suất của pipeline RAG với các chỉ số như `faithfulness`, `answer_relevancy`, `context_precision`, và `context_recall`.
*   **Giao diện tương tác**: Cung cấp UI trực quan bằng **Streamlit** để người dùng cuối có thể dễ dàng đặt câu hỏi và nhận câu trả lời.

---

## 2. Kiến trúc hệ thống được đề xuất

Để đảm bảo tính module hóa và dễ quản lý, chúng ta nên cấu trúc lại dự án theo các thành phần logic riêng biệt.

```
+-----------------------+      +-----------------------+      +---------------------+
|   Data Ingestion      |----->|       RAG Core        |----->|    API & UI Layer   |
| (Scraper, QA Gen)     |      | (VectorDB, QueryAgent)|      |     (Streamlit)     |
+-----------------------+      +-----------------------+      +---------------------+
           |                              |                              |
           |                              |                              |
           v                              v                              v
+-----------------------+      +-----------------------+      +---------------------+
|   Raw Data Storage    |      |   Vector Database     |      |   User Interaction  |
|      (JSON/CSV)       |      |      (ChromaDB)       |      |                     |
+-----------------------+      +-----------------------+      +---------------------+
           |                              ^
           |                              |
           +------------------------------+
           |
           v
+-----------------------+
|   Evaluation Engine   |
|        (Ragas)        |
+-----------------------+
```

### Cấu trúc thư mục đề xuất

Cấu trúc này phân tách rõ ràng các nhiệm vụ, giúp việc phát triển và gỡ lỗi trở nên dễ dàng hơn.

```
/
├── config/
│   ├── __init__.py
│   └── settings.py             # Cấu hình tập trung: API keys, paths, model names
├── data/
│   ├── qa_dataset.json         # Ground truth data cho việc đánh giá
│   └── chroma_db/              # Cơ sở dữ liệu vector ChromaDB
├── data_ingestion/
│   ├── __init__.py
│   ├── main.py                 # Script chính để chạy toàn bộ pipeline thu thập dữ liệu
│   ├── scrapper.py             # Logic cào dữ liệu
│   └── question_generator.py   # Logic tạo sinh QA dataset
├── rag_core/
│   ├── __init__.py
│   ├── agent.py                # Chứa QueryAgent, xử lý logic RAG chính
│   ├── db_manager.py           # Quản lý các thao tác với ChromaDB
│   └── prompt_templates.py     # Lưu trữ các mẫu prompt cho LLM
├── evaluation/
│   ├── __init__.py
│   ├── main_evaluation.py      # Script chính để chạy đánh giá bằng Ragas
│   └── results/                # Thư mục chứa các file CSV kết quả
├── ui/
│   ├── __init__.py
│   ├── app.py                  # Giao diện Streamlit chính
│   └── dashboard.py            # Dashboard trực quan hóa kết quả đánh giá
├── .env.example                # File mẫu cho biến môi trường
├── requirements.txt            # Các thư viện Python cần thiết
└── README.md                   # Tài liệu dự án
```

---

## 3. Hướng dẫn cài đặt và vận hành

### 3.1. Cài đặt môi trường

1.  **Clone repository:**
    ```bash
    git clone <your-repository-url>
    cd <project-folder>
    ```

2.  **Cài đặt các thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình biến môi trường:**
    Tạo một file `.env` từ file `.env.example` và điền `GOOGLE_API_KEY` của bạn.
    ```
    # .env
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    File `config/settings.py` sẽ tự động tải biến môi trường này.

### 3.2. Quy trình vận hành

#### Bước 1: Thu thập dữ liệu và xây dựng VectorDB

Chạy script `main.py` trong module `data_ingestion` để bắt đầu:
```bash
python -m data_ingestion.main
```
Quy trình này sẽ:
1.  **Cào dữ liệu**: Tải các bài báo từ nguồn đã định.
2.  **Tạo QA Dataset**: Dùng LLM (Gemini) để tạo các cặp QA và lưu vào `data/qa_dataset.json`.
3.  **Indexing**: Mã hóa nội dung bài báo thành vector và lưu vào ChromaDB.

#### Bước 2: Đánh giá hiệu suất hệ thống

Để đo lường chất lượng của các pipeline RAG khác nhau:
```bash
python -m evaluation.main_evaluation
```
Script này sẽ:
1.  Tải bộ `qa_dataset.json`.
2.  Lặp qua từng phương pháp embedding đã cấu hình (`vietnamese`, `nomic`, `baseline`).
3.  Gửi câu hỏi đến `QueryAgent` để nhận câu trả lời và ngữ cảnh.
4.  Sử dụng Ragas để tính toán các chỉ số và lưu kết quả vào thư mục `evaluation/results/`.

#### Bước 3: Khởi chạy ứng dụng Web

Sau khi có dữ liệu, khởi chạy giao diện người dùng:
```bash
streamlit run ui/app.py
```
Truy cập `http://localhost:8501` để tương tác với hệ thống.

---

## 4. Phân tích các thành phần chính

### 4.1. Data Ingestion (`data_ingestion`)

*   **`scrapper.py`**: Sử dụng `requests` và `BeautifulSoup` để thu thập dữ liệu. Cần có cơ chế xử lý lỗi và `sleep` để tránh làm quá tải server nguồn.
*   **`question_generator.py`**:
    *   Sử dụng `ChatGoogleGenerativeAI` với prompt được thiết kế để yêu cầu LLM đóng vai trò chuyên gia, tạo ra các cặp QA chất lượng cao.
    *   Yêu cầu output ở định dạng JSON và có logic `clean_json_response` để đảm bảo tính toàn vẹn của dữ liệu.

### 4.2. RAG Core (`rag_core`)

*   **`agent.py` (`QueryAgent`)**:
    *   Là trung tâm xử lý logic. Nhận một câu hỏi và một phương thức embedding (`method`).
    *   **Date Extraction**: Sử dụng một prompt LLM chuyên biệt để trích xuất thông tin thời gian từ câu hỏi, chuyển đổi thành định dạng `(start_date, end_date)`.
    *   **Vector Search**: Truy vấn ChromaDB với bộ lọc (filter) ngày tháng để tăng độ chính xác.
    *   **Answer Generation**: Xây dựng prompt cuối cùng với ngữ cảnh được truy xuất và câu hỏi của người dùng để LLM tổng hợp câu trả lời.
*   **`db_manager.py`**: Cung cấp các hàm trừu tượng để làm việc với ChromaDB (thêm, xóa, truy vấn dữ liệu), giúp tách biệt logic RAG khỏi việc triển khai cơ sở dữ liệu cụ thể.

### 4.3. Chiến lược đánh giá (`evaluation`)

Đây là một thành phần quan trọng để đảm bảo và cải thiện chất lượng của hệ thống.

*   **Mục tiêu**: So sánh hiệu quả của các mô hình embedding khác nhau để tìm ra lựa chọn tối ưu cho bộ dữ liệu và tác vụ cụ thể.
*   **Các phương pháp so sánh**:
    *   **Baseline (Zero-shot)**: LLM trả lời không cần ngữ cảnh. Dùng để đo lường kiến thức nền của LLM.
    *   **Vietnamese BiEncoder**: Sử dụng `bkai-foundation-models/vietnamese-bi-encoder`, một model chuyên cho tiếng Việt.
    *   **Nomic Embed**: Sử dụng `nomic-ai/nomic-embed-text-v1.5`, một model đa ngôn ngữ hiệu suất cao.
*   **Các chỉ số chính (Metrics)**:
    *   **`faithfulness`**: Mức độ câu trả lời bám sát vào ngữ cảnh được cung cấp.
    *   **`answer_relevancy`**: Mức độ câu trả lời liên quan đến câu hỏi.
    *   **`context_precision`**: Tỷ lệ các văn bản thực sự liên quan trong số các văn bản được truy xuất.
    *   **`context_recall`**: Tỷ lệ các văn bản liên quan được truy xuất thành công so với tổng số văn bản liên quan.
    *   **ROUGE Score**: Đo lường sự trùng lặp n-gram giữa câu trả lời của máy và câu trả lời chuẩn (ground truth).

---

## 5. Hướng phát triển và cải tiến

*   **Mở rộng nguồn dữ liệu**: Tích hợp thêm các scraper cho các trang tin tức khác (CafeF, Vietstock,...).
*   **Tối ưu hóa RAG**:
    *   **Chunking Strategy**: Thử nghiệm các chiến lược chia nhỏ văn bản (ví dụ: `RecursiveCharacterTextSplitter` với các kích thước và overlap khác nhau).
    *   **Re-ranking**: Áp dụng một mô hình re-ranker (ví dụ: Cohere Rerank) sau bước retrieval để cải thiện chất lượng của các văn bản ngữ cảnh trước khi đưa vào LLM.
*   **Thử nghiệm mô hình**: Tích hợp và đánh giá các mô hình Embedding và LLM mới (ví dụ: các model từ VinAI, Llama-3).
*   **Cải thiện UI/UX**: Thêm tính năng lịch sử chat, cho phép người dùng đánh giá câu trả lời (thumbs up/down) để thu thập feedback.
*   **Containerization**: Đóng gói ứng dụng bằng Docker để đơn giản hóa việc triển khai trên mọi môi trường.
*   **CI/CD**: Thiết lập pipeline CI/CD (ví dụ: GitHub Actions) để tự động chạy đánh giá mỗi khi có sự thay đổi trong codebase, đảm bảo chất lượng không bị suy giảm.
