import json
import logging
from tqdm import tqdm

from config import settings
from data_ingestion.scrapper import fetch_article_urls, scrape_article_content
from data_ingestion.question_generator import generate_qa_from_article
from data_ingestion.proposition_chunker import chunk_articles_into_propositions
from rag_core.db_manager import ChromaDBManager
from evaluation.cost_tracker import CostCallbackHandler

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_data_ingestion_pipeline():
    """Chạy toàn bộ pipeline thu thập và xử lý dữ liệu."""
    logging.info("Bắt đầu pipeline thu thập dữ liệu...")

    # Khởi tạo callback để theo dõi chi phí
    cost_callback = CostCallbackHandler()

    # --- Bước 1: Cào dữ liệu --- #
    logging.info(f"Bắt đầu cào URL từ: {settings.SCRAPER_URL}")
    article_urls = fetch_article_urls(settings.SCRAPER_URL, max_pages=2) # Giới hạn 1 trang để test
    logging.info(f"Đã tìm thấy tổng cộng {len(article_urls)} URL duy nhất.")

    scraped_articles = []
    for url in tqdm(article_urls, desc="Đang cào nội dung các bài báo"):
        content = scrape_article_content(url)
        if content:
            scraped_articles.append(content)
    logging.info(f"Đã cào thành công nội dung của {len(scraped_articles)} bài báo.")

    if not scraped_articles:
        logging.warning("Không có bài báo nào được cào. Dừng pipeline.")
        return

    # --- Bước 2: Tạo QA Dataset --- #
    # (Tạm thời bỏ qua theo dõi chi phí ở bước này để đơn giản)
    logging.info("Bắt đầu tạo QA dataset từ các bài báo đã cào...")
    qa_dataset = []
    for article in tqdm(scraped_articles, desc="Đang tạo QA"):
        qa_pair = generate_qa_from_article(article['content'])
        if qa_pair:
            qa_dataset.append({
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "contexts": [article['content']],
                "ground_truth": qa_pair["answer"]
            })

    with open(settings.QA_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=4)
    logging.info(f"Đã lưu {len(qa_dataset)} cặp QA vào file: {settings.QA_DATASET_PATH}")

    # --- Bước 3: Proposition Chunking và Indexing --- #
    logging.info("Bắt đầu chia nhỏ các bài báo thành các mệnh đề...")
    cost_callback.reset() # Reset chi phí trước khi bắt đầu chunking
    proposition_documents = chunk_articles_into_propositions(scraped_articles, callback_handler=cost_callback)
    logging.info(f"Đã tạo {len(proposition_documents)} mệnh đề.")
    chunking_costs = cost_callback.get_costs()
    logging.info(f"Chi phí cho bước Proposition Chunking: {chunking_costs}")

    logging.info("Bắt đầu indexing dữ liệu vào ChromaDB...")
    db_manager = ChromaDBManager()
    db_manager.add_documents(proposition_documents)
    logging.info("Hoàn tất indexing dữ liệu.")

    logging.info("Pipeline thu thập dữ liệu đã hoàn tất.")

if __name__ == "__main__":
    run_data_ingestion_pipeline()
