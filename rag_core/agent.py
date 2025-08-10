import logging
import datetime
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler

from config import settings
from rag_core.db_manager import ChromaDBManager
from rag_core.prompt_templates import RAG_PROMPT, DATE_EXTRACTION_PROMPT

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueryAgent:
    def __init__(self, embedding_method: str = 'vietnamese', callback_handler: BaseCallbackHandler = None):
        if embedding_method not in settings.EMBEDDING_MODELS:
            raise ValueError(f"Phương thức embedding '{embedding_method}' không hợp lệ.")

        # LLM mạnh mẽ cho việc sinh câu trả lời chính
        self.main_llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_PRO,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            callbacks=[callback_handler] if callback_handler else None
        )

        # LLM nhanh và rẻ cho các tác vụ phụ trợ
        self.fast_llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_FLASH,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0,
            callbacks=[callback_handler] if callback_handler else None
        )
        
        embedding_model_name = settings.EMBEDDING_MODELS[embedding_method]
        self.db_manager = ChromaDBManager(embedding_function_name=embedding_model_name)
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
        Thực hiện toàn bộ quy trình RAG để trả lời câu hỏi.
        """
        logging.info(f"Nhận câu hỏi: {question}")

        # (Tùy chọn) Bước 1: Trích xuất ngày tháng bằng LLM nhanh
        # start_date, end_date = self._extract_dates(question)

        # Bước 2: Truy vấn VectorDB để lấy ngữ cảnh
        logging.info("Bắt đầu truy vấn ngữ cảnh từ VectorDB...")
        retrieved_docs = self.db_manager.query(question, k=5)

        if not retrieved_docs:
            logging.warning("Không tìm thấy tài liệu nào liên quan.")
            return "Tôi không tìm thấy thông tin phù hợp trong dữ liệu để trả lời câu hỏi này.", []

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        logging.info(f"Đã truy xuất được {len(retrieved_docs)} tài liệu.")

        # Bước 3: Sinh câu trả lời dựa trên ngữ cảnh (sử dụng LLM chính)
        logging.info("Bắt đầu sinh câu trả lời...")
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        return response, retrieved_docs

