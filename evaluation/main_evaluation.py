import logging
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)
from rouge_score import rouge_scorer
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from rag_core.agent import QueryAgent
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rouge_score(predictions, references):
    """Tính toán điểm ROUGE-1."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores.append(scores['rouge1'].fmeasure)
    return rouge_scores

async def run_evaluation_optimized():
    """Phiên bản tối ưu với batch processing và GPU."""
    
    
    logging.info("Bắt đầu quy trình đánh giá tối ưu...")

    # LLM vẫn dùng API
    gemini_llm_for_ragas = ChatGoogleGenerativeAI(
        model=settings.GENERATION_MODEL_PRO,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.1
    )
    ragas_gemini_llm = LangchainLLMWrapper(langchain_llm=gemini_llm_for_ragas)

    # Embedding dùng GPU local để tăng tốc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {'device': device, 'trust_remote_code': True}
    
    # Tối ưu batch size cho GPU
    encode_kwargs = {
        'batch_size': 64 if device == 'cuda' else 8,
        'device': device,
        'normalize_embeddings': True,
    }
    
    # Sử dụng embedding model local với GPU
    gpu_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Model nhẹ và nhanh
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    ragas_gpu_embeddings = LangchainEmbeddingsWrapper(embeddings=gpu_embeddings)
    
    logging.info(f"Ragas embedding running on: {device}")

    try:
        qa_dataset = Dataset.from_json(str(settings.QA_DATASET_PATH))
        # Chuyển đổi Dataset thành dictionary of lists
        qa_dict = qa_dataset.to_dict()
        logging.info(f"Đã tải thành công {len(qa_dataset)} mẫu từ bộ dữ liệu đánh giá.")
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file dataset tại: {settings.QA_DATASET_PATH}")
        return

    embedding_methods = list(settings.EMBEDDING_MODELS.keys()) + ['baseline']

    for method in embedding_methods:
        logging.info(f"--- Đang đánh giá phương pháp: {method.upper()} ---")

        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": qa_dict['ground_truth']  # Sử dụng qa_dict
        }

        if method == 'baseline':
            logging.info("Bỏ qua baseline trong phiên bản này.")
            continue

        agent = QueryAgent(embedding_method=method)

        # Tạo list of dictionaries từ qa_dict
        qa_list = []
        for i in range(len(qa_dict['question'])):
            qa_list.append({
                'question': qa_dict['question'][i],
                'answer': qa_dict['answer'][i] if 'answer' in qa_dict else None,
                'contexts': qa_dict['contexts'][i] if 'contexts' in qa_dict else None,
                'ground_truth': qa_dict['ground_truth'][i]
            })

        # Batch processing với ThreadPoolExecutor
        async def process_qa_batch(qa_batch):
            results = []
            for qa_pair in qa_batch:
                question = qa_pair['question']
                response, retrieved_docs = agent.answer(question)
                results.append((question, response, retrieved_docs))
            return results

        # Chia dataset thành batches
        batch_size = 10 if torch.cuda.is_available() else 5
        qa_batches = [qa_list[i:i + batch_size] for i in range(0, len(qa_list), batch_size)]
        
        for i, batch in enumerate(qa_batches):
            logging.info(f"Processing batch {i+1}/{len(qa_batches)}")
            
            # Xử lý batch
            batch_results = await process_qa_batch(batch)
            
            for question, response, retrieved_docs in batch_results:
                eval_data["question"].append(question)
                eval_data["answer"].append(response)
                eval_data["contexts"].append([doc.page_content for doc in retrieved_docs])
            
            # GPU memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        eval_dataset = Dataset.from_dict(eval_data)

        metrics_to_evaluate = [
            answer_relevancy,
            context_recall,
            faithfulness,
            answer_similarity,
            answer_correctness,
        ]

        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_gemini_llm,
            embeddings=ragas_gpu_embeddings  # Sử dụng GPU embeddings
        )

        # Merge kết quả metrics với dữ liệu gốc
        df_result_metrics = result.to_pandas()
        df_eval_data = pd.DataFrame(eval_data)
        df_result = pd.concat([df_eval_data, df_result_metrics], axis=1)

        # Tính ROUGE-1
        logging.info("Calculating ROUGE-1 scores...")
        df_result['rouge_score_1'] = calculate_rouge_score(
            df_result['answer'], df_result['ground_truth']
        )

        # Chỉ lấy các cột quan trọng và rename
        column_map = {
            "answer_relevancy": "Answer Relevancy (Mức độ liên quan của câu trả lời)",
            "context_recall": "Context Recall (Mức độ đầy đủ của ngữ cảnh)",
            "faithfulness": "Faithfulness (Mức độ trung thực của câu trả lời)",
            "answer_similarity": "Semantic Similarity (Mức độ tương đồng ngữ nghĩa)",
            "rouge_score_1": "ROUGE-1 Score (Điểm ROUGE-1)",
            "answer_correctness": "Answer Accuracy (Mức độ chính xác của câu trả lời)"
        }

        ordered_columns = [col for col in column_map.keys() if col in df_result.columns]
        df_result_ordered = df_result[ordered_columns].rename(columns=column_map)

        logging.info(f"Kết quả đánh giá cho phương pháp '{method}':\n{df_result_ordered}")

        output_path = settings.EVAL_RESULTS_DIR / f"evaluation_results_{method}.csv"
        df_result_ordered.to_csv(output_path, index=False)
        logging.info(f"Đã lưu kết quả đánh giá vào file: {output_path}")

    logging.info("Quy trình đánh giá đã hoàn tất.")

def run_evaluation():
    """Wrapper để chạy async evaluation."""
    asyncio.run(run_evaluation_optimized())

if __name__ == "__main__":
    run_evaluation()