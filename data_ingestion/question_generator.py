import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_TEMPLATE = """
BẠN LÀ MỘT CHUYÊN GIA PHÂN TÍCH TÀI CHÍNH. Dựa vào nội dung bài báo dưới đây, hãy tạo ra một cặp câu hỏi và câu trả lời (QA) thật chất lượng.

**Yêu cầu:**
1.  **Câu hỏi:** Phải sâu sắc, đúng trọng tâm, không hỏi những thông tin quá chung chung. Câu hỏi phải có thể được trả lời hoàn toàn dựa vào nội dung được cung cấp.
2.  **Câu trả lời:** Phải chính xác, súc tích, và trích xuất trực tiếp từ những thông tin quan trọng nhất trong bài báo.
3.  **Định dạng:** Trả về kết quả dưới dạng một JSON object duy nhất có hai key là "question" và "answer".

**Nội dung bài báo:**
---
{context}
---

**JSON Output:**
"""

def clean_json_response(response_text):
    """Làm sạch và parse JSON từ text trả về của LLM."""
    try:
        # Cố gắng tìm JSON object trong text
        json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        logging.error(f"Không thể parse JSON từ response: {response_text} - Lỗi: {e}")
        return None

def generate_qa_from_article(article_content: str):
    """Sử dụng LLM để tạo QA từ nội dung bài báo."""
    if not article_content or not isinstance(article_content, str) or len(article_content) < 100:
        logging.warning("Nội dung bài báo quá ngắn hoặc không hợp lệ, bỏ qua.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_PRO,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context"],
        )

        chain = prompt | llm

        response = chain.invoke({"context": article_content})
        qa_pair = clean_json_response(response.content)

        if qa_pair and 'question' in qa_pair and 'answer' in qa_pair:
            logging.info(f"Đã tạo QA thành công: Q: {qa_pair['question']}")
            return qa_pair
        else:
            logging.error("QA pair không hợp lệ hoặc thiếu key.")
            return None

    except Exception as e:
        logging.error(f"Lỗi trong quá trình gọi API của LLM: {e}")
        return None
