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
    def __init__(self, embedding_function_name: str = settings.EMBEDDING_MODELS['vietnamese'], use_gpu: bool = False, collection_suffix: str = ""):
        """Khởi tạo DB Manager với một embedding function cụ thể."""
        self.client = chromadb.PersistentClient(path=str(settings.DB_DIR))
        self.embedding_model_name = embedding_function_name

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
                    model_name=embedding_function_name,
                    model_kwargs={'trust_remote_code': True}
                )
        else:
            self.embedding_function = None

        logging.info(f"Đã khởi tạo embedding model trên: {'GPU' if self.use_gpu else 'CPU'}")

        # Tạo collection name dựa trên embedding model
        if collection_suffix:
            self.collection_name = f"{settings.CHROMA_COLLECTION_NAME}_{collection_suffix}"
        else:
            # Tạo suffix từ tên model để đảm bảo unique
            model_suffix = embedding_function_name.split('/')[-1].replace('-', '_')
            self.collection_name = f"{settings.CHROMA_COLLECTION_NAME}_{model_suffix}"
        
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

    def get_collection_stats(self):
        """Lấy thống kê về collection hiện tại."""
        try:
            collection = self.get_collection()
            if collection:
                # Lấy tổng số documents
                count = collection.count()
                
                # Lấy một sample document để xem metadata
                if count > 0:
                    sample = collection.get(limit=1, include=["metadatas", "documents"])
                    sample_metadata = sample["metadatas"][0] if sample["metadatas"] else {}
                    sample_content = sample["documents"][0][:200] + "..." if sample["documents"] else ""
                else:
                    sample_metadata = {}
                    sample_content = ""
                
                return {
                    "collection_name": self.collection_name,
                    "count": count,
                    "embedding_model": self.embedding_model_name,
                    "sample_metadata": sample_metadata,
                    "sample_content": sample_content,
                    "gpu_enabled": self.use_gpu
                }
            else:
                return {
                    "collection_name": self.collection_name,
                    "count": 0,
                    "embedding_model": self.embedding_model_name,
                    "sample_metadata": {},
                    "sample_content": "",
                    "gpu_enabled": self.use_gpu,
                    "error": "Collection chưa tồn tại"
                }
        except Exception as e:
            logging.error(f"Lỗi khi lấy thống kê collection: {e}")
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "count": 0
            }

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


class MultiEmbeddingDBManager:
    """Manager để quản lý nhiều embedding models và collections."""
    
    def __init__(self, embedding_models: Dict[str, str] = None, use_gpu: bool = False):
        """
        Khởi tạo manager với nhiều embedding models.
        
        Args:
            embedding_models: Dict với key là tên model (ví dụ: 'vietnamese', 'nomic') 
                             và value là model path
            use_gpu: Có sử dụng GPU hay không
        """
        if embedding_models is None:
            embedding_models = settings.EMBEDDING_MODELS
        
        self.embedding_models = embedding_models
        self.use_gpu = use_gpu
        self.db_managers = {}
        
        # Khởi tạo DB manager cho mỗi embedding model
        for model_name, model_path in embedding_models.items():
            logging.info(f"Khởi tạo DB manager cho model: {model_name} ({model_path})")
            self.db_managers[model_name] = ChromaDBManager(
                embedding_function_name=model_path,
                use_gpu=use_gpu,
                collection_suffix=model_name
            )
    
    def add_documents_to_all(self, articles: List[Dict[str, str]]):
        """Thêm documents vào tất cả các collections."""
        for model_name, db_manager in self.db_managers.items():
            logging.info(f"Đang thêm {len(articles)} documents vào collection cho model: {model_name}")
            db_manager.add_documents(articles)
            logging.info(f"Hoàn tất thêm documents cho model: {model_name}")
    
    def get_db_manager(self, model_name: str) -> ChromaDBManager:
        """Lấy DB manager cho một model cụ thể."""
        return self.db_managers.get(model_name)
    
    def query_all_models(self, query_text: str, k: int = 5, start_date=None, end_date=None):
        """Truy vấn trên tất cả các models và trả về kết quả từ mỗi model."""
        results = {}
        for model_name, db_manager in self.db_managers.items():
            logging.info(f"Đang truy vấn model: {model_name}")
            results[model_name] = db_manager.query(query_text, k, start_date, end_date)
        return results
    
    def delete_all_collections(self):
        """Xóa tất cả collections."""
        for model_name, db_manager in self.db_managers.items():
            logging.info(f"Đang xóa collection cho model: {model_name}")
            db_manager.delete_collection()
    
    def list_collections(self):
        """Liệt kê tất cả collections."""
        collections = {}
        for model_name, db_manager in self.db_managers.items():
            collections[model_name] = db_manager.collection_name
        return collections