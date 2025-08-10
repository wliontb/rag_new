import chromadb
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict

from config import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaDBManager:
    def __init__(self, embedding_function_name: str = settings.EMBEDDING_MODELS['vietnamese']):
        """Khởi tạo DB Manager với một embedding function cụ thể."""
        self.client = chromadb.PersistentClient(path=str(settings.DB_DIR))
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_function_name,
            model_kwargs={'trust_remote_code': True}
        )
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        
        # Khởi tạo Langchain Chroma wrapper
        self.langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def get_collection(self):
        """Lấy collection từ ChromaDB."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception as e:
            logging.warning(f"Collection '{self.collection_name}' chưa tồn tại. Sẽ tạo mới khi thêm dữ liệu. Lỗi: {e}")
            return None

    def add_documents(self, articles: List[Dict[str, str]]):
        """Chuyển đổi bài báo thành Documents và thêm vào ChromaDB."""
        if not articles:
            logging.warning("Không có bài báo nào để thêm vào DB.")
            return

        documents = []
        for article in articles:
            doc = Document(
                page_content=article['content'],
                metadata={
                    'source': article['url'],
                    'title': article['title'],
                    # Thêm các metadata khác nếu cần, ví dụ: ngày xuất bản
                }
            )
            documents.append(doc)

        logging.info(f"Bắt đầu thêm {len(documents)} documents vào collection '{self.collection_name}'...")
        self.langchain_chroma.add_documents(documents)
        logging.info("Thêm documents thành công.")

    def query(self, query_text: str, k: int = 5, filters: Dict = None):
        """Truy vấn các documents tương tự từ ChromaDB."""
        try:
            results = self.langchain_chroma.similarity_search(
                query=query_text,
                k=k,
                filter=filters
            )
            return results
        except Exception as e:
            logging.error(f"Lỗi khi truy vấn ChromaDB: {e}")
            return []

    def view_all_documents(self):
        """Xem tất cả các tài liệu trong collection."""
        collection = self.get_collection()
        if collection:
            # The get() method without any arguments retrieves all items.
            return collection.get(include=["metadatas", "documents"])
        return None

    def delete_collection(self):
        """Xóa collection hiện tại."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logging.info(f"Collection '{self.collection_name}' đã được xóa thành công.")
            # Re-initialize the Langchain Chroma wrapper to create a new collection
            self.langchain_chroma = Chroma.from_documents(
                documents=[],
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                persist_directory=str(settings.DB_DIR),
            )
        except Exception as e:
            logging.error(f"Lỗi khi xóa collection: {e}")