import chromadb
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict

import torch

from config import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaDBManager:
    def __init__(self, embedding_function_name: str = settings.EMBEDDING_MODELS['vietnamese'], use_gpu: bool = False):
        """Khởi tạo DB Manager với một embedding function cụ thể."""
        self.client = chromadb.PersistentClient(path=str(settings.DB_DIR))

        # GPU optimization cho embedding
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if embedding_function_name:
            if self.use_gpu:
                # GPU-optimized embedding
                model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
                encode_kwargs = {
                    'batch_size': 64,  # Larger batch for GPU
                    'device': 'cuda',
                    'normalize_embeddings': True,
                    'convert_to_tensor': True
                }
                
                self.embedding_function = HuggingFaceEmbeddings(
                    model_name=embedding_function_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                logging.info(f"Using GPU-accelerated embeddings: {embedding_function_name}")
            else:
                # CPU fallback
                self.embedding_function = HuggingFaceEmbeddings(
                    model_name=embedding_function_name
                )
        else:
            self.embedding_function = None

        logging.info(f"Đã khởi tạo embedding model trên: {'GPU' if self.use_gpu else 'CPU'}")

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
        """Chuyển đổi bài báo thành Documents và thêm vào ChromaDB với GPU optimization."""
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

        logging.info(f"Đang thêm {len(documents)} documents vào collection '{self.collection_name}'...")
        # Batch processing cho GPU
        batch_size = 16 if torch.cuda.is_available() else 8
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                # Sử dụng GPU memory optimization
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        self.langchain_chroma.add_documents(batch)
                else:
                    self.langchain_chroma.add_documents(batch)
                    
                logging.info(f"Đã xử lý batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Clear GPU cache sau mỗi batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.error(f"Lỗi khi thêm batch {i//batch_size + 1}: {e}")
                continue

        logging.info(f"Đã thêm thành công {len(documents)} documents vào ChromaDB.")

    def query(self, query_text: str, k: int = 5, start_date=None, end_date=None):
        """Truy vấn các documents tương tự từ ChromaDB."""
        try:
            # Đơn giản hóa: bỏ date filtering trong database query
            results = self.langchain_chroma.similarity_search(
                query=query_text,
                k=k
            )
            
            logging.info(f"Đã truy vấn được {len(results)} documents từ ChromaDB")
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

    def query_optimized(self, query_text: str, k: int = 5, start_date=None, end_date=None, use_gpu=False):
        """GPU-optimized query với date filtering."""
        
        # Date filtering
        where_clause = {}
        if start_date and end_date:
            where_clause = {
                "$and": [
                    {"date": {"$gte": start_date}},
                    {"date": {"$lte": end_date}}
                ]
            }
        
        if use_gpu and self.use_gpu:
            # GPU memory management
            torch.cuda.empty_cache()
            
            # Enable mixed precision cho embedding
            with torch.cuda.amp.autocast():
                results = self.langchain_chroma.similarity_search(
                    query_text, 
                    k=k,
                    where=where_clause if where_clause else None
                )
        else:
            results = self.langchain_chroma.similarity_search(
                query_text, 
                k=k,
                where=where_clause if where_clause else None
            )
        
        return results