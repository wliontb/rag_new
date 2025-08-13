import logging
import datetime
import json
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor
import asyncio

from config import settings
from rag_core.db_manager import ChromaDBManager
from rag_core.prompt_templates import RAG_PROMPT, DATE_EXTRACTION_PROMPT

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueryAgent:
    def __init__(self, embedding_method: str = 'vietnamese'):
        if embedding_method not in settings.EMBEDDING_MODELS:
            raise ValueError(f"Phương thức embedding '{embedding_method}' không hợp lệ.")

        # LLM mạnh mẽ cho việc sinh câu trả lời chính
        self.main_llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_PRO,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1
        )

        # LLM nhanh và rẻ cho các tác vụ phụ trợ
        self.fast_llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_FLASH,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0
        )
        
        # GPU-optimized embedding cho vector search
        self.gpu_enabled = torch.cuda.is_available()
        if self.gpu_enabled:
            logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        
        # Sử dụng collection suffix để chọn đúng collection
        embedding_model_name = settings.EMBEDDING_MODELS[embedding_method]
        self.db_manager = ChromaDBManager(
            embedding_function_name=embedding_model_name,
            use_gpu=self.gpu_enabled,
            collection_suffix=embedding_method  # Thêm collection_suffix để tương thích với multi-collection
        )
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self) -> Runnable:
        """Tạo chuỗi xử lý RAG với LLM chính."""
        return RAG_PROMPT | self.main_llm | StrOutputParser()

    def _extract_dates(self, question: str):
        """Sử dụng LLM nhanh để trích xuất ngày tháng."""
        try:
            date_extraction_chain = DATE_EXTRACTION_PROMPT | self.fast_llm | StrOutputParser()
            today = datetime.date.today().strftime("%Y-%m-%d")
            
            response = date_extraction_chain.invoke({"question": question, "today": today})
            
            json_str = response[response.find('{'):response.rfind('}') + 1]
            dates = json.loads(json_str)
            
            start_date = dates.get('start_date')
            end_date = dates.get('end_date')
            
            logging.info(f"Trích xuất ngày tháng: Start: {start_date}, End: {end_date}")
            return start_date, end_date
        except Exception as e:
            logging.error(f"Không thể trích xuất ngày tháng từ câu hỏi: '{question}'. Lỗi: {e}")
            return None, None

    def answer(self, question: str):
        """
        Thực hiện toàn bộ quy trình RAG với GPU optimization.
        """
        logging.info(f"Nhận câu hỏi: {question}")

        # Bước 1: Trích xuất ngày tháng
        start_date, end_date = self._extract_dates(question)

        # Bước 2: Vector search (không dùng date filter trong query)
        logging.info("Bắt đầu truy vấn ngữ cảnh từ VectorDB...")
        
        if self.gpu_enabled:
            torch.cuda.empty_cache()
            
        # Query nhiều documents hơn nếu có date filter
        k_query = 20 if (start_date or end_date) else 10
        retrieved_docs = self.db_manager.query(
            query_text=question, 
            k=k_query
        )

        # Post-process date filtering nếu cần
        if (start_date or end_date) and retrieved_docs:
            filtered_docs = []
            for doc in retrieved_docs:
                doc_date = doc.metadata.get('date')
                if doc_date:
                    try:
                        if isinstance(doc_date, str):
                            doc_date = doc_date.split('T')[0]
                        
                        if start_date and doc_date < start_date:
                            continue
                        if end_date and doc_date > end_date:
                            continue
                    except:
                        # Nếu không parse được date, vẫn include
                        pass
                
                filtered_docs.append(doc)
                if len(filtered_docs) >= 10:  # Giới hạn 10 docs
                    break
            
            retrieved_docs = filtered_docs

        if not retrieved_docs:
            logging.warning("Không tìm thấy tài liệu nào liên quan.")
            return "Tôi không tìm thấy thông tin phù hợp trong dữ liệu để trả lời câu hỏi này.", []

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        logging.info(f"Đã truy xuất được {len(retrieved_docs)} tài liệu.")

        # Bước 3: Answer generation
        logging.info("Bắt đầu sinh câu trả lời...")
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        # GPU cleanup
        if self.gpu_enabled:
            torch.cuda.empty_cache()

        return response, retrieved_docs

