import json
import logging
from tqdm import tqdm
import asyncio
from config import settings
from data_ingestion.scrapper import fetch_article_urls, scrape_article_content
from data_ingestion.question_generator import generate_qa_from_article
from data_ingestion.proposition_chunker import chunk_articles_into_propositions, chunk_articles_into_propositions_optimized
from rag_core.db_manager import ChromaDBManager, MultiEmbeddingDBManager
from data_ingestion.common import Data

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_data_ingestion_pipeline_async():
    """Phiên bản async của pipeline."""
    logging.info("Bắt đầu pipeline thu thập dữ liệu...")

    # --- Bước 1: Cào dữ liệu --- #
    logging.info(f"Bắt đầu cào URL từ: {settings.SCRAPER_URL}")
    article_urls = fetch_article_urls(settings.SCRAPER_URL, max_pages=3) # Giới hạn 1 trang để test
    logging.info(f"Đã tìm thấy tổng cộng {len(article_urls)} URL duy nhất.")

    scraped_articles: list[Data] = []
    for url in tqdm(article_urls, desc="Đang cào nội dung các bài báo"):
        content = scrape_article_content(url)
        if content:
            scraped_articles.append(content)
    logging.info(f"Đã cào thành công nội dung của {len(scraped_articles)} bài báo.")

    if not scraped_articles:
        logging.warning("Không có bài báo nào được cào. Dừng pipeline.")
        return

    # --- Bước 2: Tạo QA Dataset --- #

    # Check nếu qa_dataset.json đã có dữ liệu thì bỏ qua bước này mà lấy dữ liệu đang có để chunking luôn
    try:
        with open(settings.QA_DATASET_PATH, 'r', encoding='utf-8') as f:
            qa_dataset = json.load(f)
            logging.info(f"Đã tìm thấy {len(qa_dataset)} cặp QA trong file: {settings.QA_DATASET_PATH}")
    except FileNotFoundError:
        logging.warning(f"Không tìm thấy file: {settings.QA_DATASET_PATH}. Bắt đầu tạo mới...")
        qa_dataset = []
        logging.info("Bắt đầu tạo QA dataset từ các bài báo đã cào...")
        for article in tqdm(scraped_articles, desc="Đang tạo QA"):
            qa_pair = generate_qa_from_article(article.content)
            if qa_pair:
                qa_dataset.append({
                    "id": article.id,
                    "date": article.date,
                    "title": article.title,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "contexts": [article.content],
                    "ground_truth": qa_pair["answer"]
                })

        with open(settings.QA_DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=4)
        logging.info(f"Đã lưu {len(qa_dataset)} cặp QA vào file: {settings.QA_DATASET_PATH}")

    # --- Bước 3: Optimized Proposition Chunking --- #
    logging.info("Bắt đầu chia nhỏ các bài báo thành các mệnh đề (optimized)...")
    
    proposition_documents = await chunk_articles_into_propositions_optimized(
        scraped_articles,
        batch_size=10,  # Xử lý 10 articles song song
        max_workers=4   # Giới hạn concurrent requests
    )
    
    logging.info(f"Đã tạo {len(proposition_documents)} mệnh đề.")

    # --- Bước 4: Indexing dữ liệu vào ChromaDB với nhiều embedding models --- #
    logging.info("Bắt đầu indexing dữ liệu vào ChromaDB với nhiều embedding models...")
    
    # Khởi tạo MultiEmbeddingDBManager để lưu vào cả 2 collections
    multi_db_manager = MultiEmbeddingDBManager(
        embedding_models=settings.EMBEDDING_MODELS,
        use_gpu=settings.GPU_ENABLED
    )
    
    # Hiển thị thông tin các collections sẽ được tạo
    collections = multi_db_manager.list_collections()
    logging.info(f"Sẽ lưu dữ liệu vào các collections: {collections}")
    
    # Thêm documents vào tất cả các collections
    multi_db_manager.add_documents_to_all(proposition_documents)
    
    logging.info("Hoàn tất indexing dữ liệu vào tất cả embedding models.")
    logging.info(f"Dữ liệu đã được lưu vào {len(settings.EMBEDDING_MODELS)} collections khác nhau:")
    for model_name, collection_name in collections.items():
        logging.info(f"  - Model '{model_name}': Collection '{collection_name}'")

    logging.info("Pipeline thu thập dữ liệu đã hoàn tất.")

def run_data_ingestion_pipeline():
    """Wrapper để chạy async pipeline."""
    asyncio.run(run_data_ingestion_pipeline_async())

if __name__ == "__main__":
    run_data_ingestion_pipeline()
