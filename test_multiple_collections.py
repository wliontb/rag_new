#!/usr/bin/env python3
"""
Script để test việc truy vấn từ nhiều collections với các embedding models khác nhau.
"""

from rag_core.db_manager import MultiEmbeddingDBManager
from config import settings
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_multiple_collections():
    """Test việc truy vấn từ nhiều collections."""
    
    # Khởi tạo MultiEmbeddingDBManager
    multi_db_manager = MultiEmbeddingDBManager(
        embedding_models=settings.EMBEDDING_MODELS,
        use_gpu=settings.GPU_ENABLED
    )
    
    # Hiển thị thông tin các collections
    collections = multi_db_manager.list_collections()
    logging.info("Các collections hiện có:")
    for model_name, collection_name in collections.items():
        logging.info(f"  - Model '{model_name}': Collection '{collection_name}'")
    
    # Test query
    test_query = "cổ phiếu ngân hàng"
    logging.info(f"\n=== Test truy vấn: '{test_query}' ===")
    
    # Truy vấn từ từng model riêng lẻ
    for model_name in settings.EMBEDDING_MODELS.keys():
        logging.info(f"\n--- Kết quả từ model: {model_name} ---")
        db_manager = multi_db_manager.get_db_manager(model_name)
        
        if db_manager:
            results = db_manager.query(test_query, k=3)
            logging.info(f"Tìm thấy {len(results)} kết quả:")
            
            for i, doc in enumerate(results, 1):
                logging.info(f"  {i}. Score: {doc.metadata.get('score', 'N/A')}")
                logging.info(f"     Title: {doc.metadata.get('title', 'N/A')}")
                logging.info(f"     Content: {doc.page_content[:200]}...")
                logging.info(f"     Source: {doc.metadata.get('source', 'N/A')}")
                logging.info("")
        else:
            logging.error(f"Không tìm thấy DB manager cho model: {model_name}")
    
    # Test truy vấn từ tất cả models cùng lúc
    logging.info("\n=== Test truy vấn từ tất cả models ===")
    all_results = multi_db_manager.query_all_models(test_query, k=2)
    
    for model_name, results in all_results.items():
        logging.info(f"\nModel '{model_name}': {len(results)} kết quả")
        for i, doc in enumerate(results, 1):
            logging.info(f"  {i}. {doc.metadata.get('title', 'N/A')}")

if __name__ == "__main__":
    test_multiple_collections()
