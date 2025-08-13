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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from rag_core.agent import QueryAgent
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineAgent:
    """Agent cho phương pháp baseline (zero-shot) - chỉ sử dụng LLM mà không retrieve."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_PRO,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Prompt template cho zero-shot
        self.baseline_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một AI assistant thông minh. Hãy trả lời câu hỏi của người dùng một cách chính xác và chi tiết nhất có thể dựa trên kiến thức của bạn.

Lưu ý:
- Nếu bạn không chắc chắn về thông tin, hãy thành thật nói rằng bạn không biết
- Đưa ra câu trả lời có cấu trúc và dễ hiểu
- Sử dụng tiếng Việt để trả lời"""),
            ("human", "{question}")
        ])
        
        self.chain = self.baseline_prompt | self.llm | StrOutputParser()
        logging.info("BaselineAgent đã được khởi tạo (Zero-shot)")
    
    def answer(self, question: str):
        """Trả lời câu hỏi bằng phương pháp zero-shot."""
        response = self.chain.invoke({"question": question})
        # Trả về empty list cho contexts để tương thích với format đánh giá
        return response, []

def calculate_rouge_score(predictions, references):
    """Tính toán điểm ROUGE-1."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores.append(scores['rouge1'].fmeasure)
    return rouge_scores

def create_comparison_table(evaluation_results):
    """Tạo bảng so sánh kết quả đánh giá của các phương pháp."""
    comparison_data = []
    
    for method, results in evaluation_results.items():
        avg_scores = {
            'Phương pháp': method.upper(),
            'Answer Relevancy': f"{results['answer_relevancy'].mean():.4f}",
            'Context Recall': f"{results['context_recall'].mean():.4f}" if 'context_recall' in results else "N/A",
            'Faithfulness': f"{results['faithfulness'].mean():.4f}" if 'faithfulness' in results else "N/A", 
            'Answer Similarity': f"{results['answer_similarity'].mean():.4f}" if 'answer_similarity' in results else "N/A",
            'Answer Correctness': f"{results['answer_correctness'].mean():.4f}",
            'ROUGE-1': f"{results['rouge_score_1'].mean():.4f}"
        }
        comparison_data.append(avg_scores)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

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
        model_name="BAAI/bge-m3",  # Model nhẹ và nhanh
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

    # Thêm baseline vào danh sách phương pháp
    embedding_methods = list(settings.EMBEDDING_MODELS.keys()) + ['baseline']
    evaluation_results = {}  # Dictionary để lưu kết quả của tất cả phương pháp

    for method in embedding_methods:
        logging.info(f"--- Đang đánh giá phương pháp: {method.upper()} ---")

        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": qa_dict['ground_truth']  # Sử dụng qa_dict
        }

        # Khởi tạo agent phù hợp
        if method == 'baseline':
            agent = BaselineAgent()
            logging.info("Sử dụng BaselineAgent (Zero-shot)")
        else:
            agent = QueryAgent(embedding_method=method)
            logging.info(f"Sử dụng QueryAgent với embedding: {method}")

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
                # Xử lý contexts cho baseline (empty list)
                if method == 'baseline':
                    eval_data["contexts"].append(["No context used (zero-shot)"])
                else:
                    eval_data["contexts"].append([doc.page_content for doc in retrieved_docs])
            
            # GPU memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        eval_dataset = Dataset.from_dict(eval_data)

        # Chọn metrics phù hợp cho từng phương pháp
        if method == 'baseline':
            # Baseline không có context, nên bỏ qua context-related metrics
            metrics_to_evaluate = [
                answer_relevancy,
                answer_similarity,
                answer_correctness,
            ]
        else:
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

        # Lưu kết quả để so sánh sau này
        evaluation_results[method] = df_result

        # Chỉ lấy các cột quan trọng và rename
        if method == 'baseline':
            column_map = {
                "answer_relevancy": "Answer Relevancy (Mức độ liên quan của câu trả lời)",
                "answer_similarity": "Semantic Similarity (Mức độ tương đồng ngữ nghĩa)",
                "rouge_score_1": "ROUGE-1 Score (Điểm ROUGE-1)",
                "answer_correctness": "Answer Accuracy (Mức độ chính xác của câu trả lời)"
            }
        else:
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

    # Tạo bảng so sánh sau khi đánh giá xong tất cả phương pháp
    logging.info("\n" + "="*80)
    logging.info("TẠNG BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ")
    logging.info("="*80)
    
    comparison_table = create_comparison_table(evaluation_results)
    print("\n📊 BẢNG SO SÁNH ĐIỂM TRUNG BÌNH CÁC PHƯƠNG PHÁP:")
    print("="*100)
    print(comparison_table.to_string(index=False))
    print("="*100)
    
    # Lưu bảng so sánh
    comparison_path = settings.EVAL_RESULTS_DIR / "comparison_results.csv"
    comparison_table.to_csv(comparison_path, index=False)
    logging.info(f"Đã lưu bảng so sánh vào file: {comparison_path}")

    logging.info("Quy trình đánh giá đã hoàn tất.")

def run_evaluation():
    """Wrapper để chạy async evaluation."""
    asyncio.run(run_evaluation_optimized())

if __name__ == "__main__":
    run_evaluation()