import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from config import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_TEMPLATE = """
Extract key propositions from the following text. A proposition is a single, atomic statement.

**Text:**
{text}

**Instructions:**
- Identify all the core propositions in the text.
- Each proposition should be a standalone, complete sentence.
- Present the propositions in a JSON list.

**JSON Output:**
{{ "propositions": ["proposition 1", "proposition 2", ...] }}
"""

class Propositionizer:
    def __init__(self, callback_handler: BaseCallbackHandler = None):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_FLASH,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0,
            callbacks=[callback_handler] if callback_handler else None
        )
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["text"],
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def get_propositions(self, text: str) -> List[str]:
        """Sử dụng LLM để trích xuất các mệnh đề từ văn bản."""
        try:
            response = self.chain.invoke({"text": text})
            return response.get("propositions", [])
        except Exception as e:
            logging.error(f"Lỗi khi trích xuất mệnh đề: {e}")
            return []

def chunk_articles_into_propositions(articles: List[Dict[str, str]], callback_handler: BaseCallbackHandler = None) -> List[Dict[str, str]]:
    """Chia các bài báo thành các mệnh đề và trả về một danh sách các tài liệu mới."""
    propositionizer = Propositionizer(callback_handler=callback_handler)
    proposition_documents = []

    for article in articles:
        propositions = propositionizer.get_propositions(article['content'])
        for prop in propositions:
            proposition_documents.append({
                'content': prop,
                'url': article['url'],
                'title': article['title']
            })
    
    return proposition_documents
