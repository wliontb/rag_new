import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import settings
from data_ingestion.common import Data

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_TEMPLATE = """
Hãy phân tách phần "Nội dung" thành các mệnh đề rõ ràng và đơn giản, đảm bảo rằng chúng có thể hiểu được ngay cả khi tách khỏi ngữ cảnh.
Tách câu ghép thành các câu đơn. Giữ nguyên cách diễn đạt ban đầu trong đầu vào bất cứ khi nào có thể.
Với bất kỳ thực thể nào có tên riêng đi kèm với thông tin mô tả bổ sung, hãy tách thông tin này thành một mệnh đề riêng biệt.
Phi ngữ cảnh hóa các mệnh đề bằng cách thêm các từ bổ nghĩa cần thiết vào danh từ hoặc toàn bộ câu, và thay thế các đại từ (ví dụ: "nó", "anh ấy", "cô ấy", "họ", "điều này", "điều đó") bằng tên đầy đủ của thực thể mà chúng đề cập đến.
Dựa trên thông tin được cung cấp, hãy tóm tắt thành một mảng JSON.

**Nội dung:**
{text}

**Hướng dẫn:**
- Xác định tất cả các mệnh đề chính trong văn bản.
- Mỗi mệnh đề nên là một câu hoàn chỉnh, độc lập.
- Trình bày các mệnh đề trong một danh sách JSON.

**Đầu ra JSON:**
{{ "propositions": ["mệnh đề 1", "mệnh đề 2", ...] }}
"""

class Propositionizer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_FLASH,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0
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

class OptimizedPropositionizer:
    def __init__(self, max_workers: int = 4):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_FLASH,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.0
        )
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["text"],
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.max_workers = max_workers
        
    async def get_propositions_async(self, text: str, max_retries: int = 3) -> List[str]:
        """Xử lý bất đồng bộ với retry mechanism."""
        for attempt in range(max_retries):
            try:
                # Sử dụng ThreadPoolExecutor để không block event loop
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(
                        executor, 
                        lambda: self.chain.invoke({"text": text})
                    )
                return response.get("propositions", [])
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"All attempts failed for text chunk")
                    return []
                    
    async def process_articles_batch(self, articles: List[Data]) -> List[Dict[str, str]]:
        """Xử lý batch articles với concurrency control."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_article(article):
            async with semaphore:
                propositions = await self.get_propositions_async(article.content)
                return [(prop, article) for prop in propositions]
        
        # Xử lý tất cả articles song song
        tasks = [process_single_article(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten kết quả
        proposition_documents = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error processing article: {result}")
                continue
            for prop, article in result:
                proposition_documents.append({
                    'content': prop,
                    'url': article.url,
                    'title': article.title
                })
        
        return proposition_documents

def chunk_articles_into_propositions(articles: List[Data]) -> List[Dict[str, str]]:
    """Chia các bài báo thành các mệnh đề và trả về một danh sách các tài liệu mới."""
    propositionizer = Propositionizer()
    proposition_documents = []

    for article in articles:
        propositions = propositionizer.get_propositions(article.content)
        for prop in propositions:
            proposition_documents.append({
                'content': prop,
                'url': article.url,
                'title': article.title
            })
    
    return proposition_documents

async def chunk_articles_into_propositions_optimized(
    articles: List[Data], 
    batch_size: int = 10,
    max_workers: int = 4
) -> List[Dict[str, str]]:
    """Phiên bản tối ưu với batch processing và async."""
    propositionizer = OptimizedPropositionizer(max_workers)
    all_proposition_documents = []
    
    # Chia articles thành các batch nhỏ
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1}")
        
        batch_results = await propositionizer.process_articles_batch(batch)
        all_proposition_documents.extend(batch_results)
        
        # Rate limiting để tránh hit API limits
        if i + batch_size < len(articles):
            await asyncio.sleep(1)
    
    return all_proposition_documents
