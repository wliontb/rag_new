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

from config import settings
from rag_core.agent import QueryAgent
from evaluation.cost_tracker import CostCallbackHandler

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

def run_evaluation():
    logging.info("Bắt đầu quy trình đánh giá...")

    cost_callback = CostCallbackHandler()

    gemini_llm_for_ragas = ChatGoogleGenerativeAI(
        model=settings.GENERATION_MODEL_PRO,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.1,
        callbacks=[cost_callback]
    )
    ragas_gemini_llm = LangchainLLMWrapper(langchain_llm=gemini_llm_for_ragas)

    vertex_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    ragas_vertex_embeddings = LangchainEmbeddingsWrapper(embeddings=vertex_embeddings)

    try:
        qa_dataset = Dataset.from_json(str(settings.QA_DATASET_PATH))
        logging.info(f"Đã tải thành công {len(qa_dataset)} mẫu từ bộ dữ liệu đánh giá.")
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file dataset tại: {settings.QA_DATASET_PATH}")
        return

    embedding_methods = list(settings.EMBEDDING_MODELS.keys()) + ['baseline']
    total_costs_summary = []

    for method in embedding_methods:
        logging.info(f"--- Đang đánh giá phương pháp: {method.upper()} ---")

        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": qa_dataset['ground_truth']
        }

        if method == 'baseline':
            logging.info("Bỏ qua baseline trong phiên bản này.")
            continue

        cost_callback.reset()
        agent = QueryAgent(embedding_method=method, callback_handler=cost_callback)

        for i, qa_pair in enumerate(qa_dataset):
            question = qa_pair['question']
            response, retrieved_docs = agent.answer(question)

            eval_data["question"].append(question)
            eval_data["answer"].append(response)
            eval_data["contexts"].append([doc.page_content for doc in retrieved_docs])
            logging.info(f"[Sample {i+1}/{len(qa_dataset)}] Q: {question} -> A: {response[:100]}...")

        query_time_costs = cost_callback.get_costs()
        logging.info(f"Chi phí Query Time cho phương pháp '{method}': {query_time_costs}")

        eval_dataset = Dataset.from_dict(eval_data)

        metrics_to_evaluate = [
            answer_relevancy,
            context_recall,
            faithfulness,
            answer_similarity,
            answer_correctness,
        ]

        cost_callback.reset()
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_gemini_llm,
            embeddings=ragas_vertex_embeddings,
            callbacks=[cost_callback]
        )
        ragas_eval_costs = cost_callback.get_costs()
        logging.info(f"Chi phí Đánh giá Ragas cho phương pháp '{method}': {ragas_eval_costs}")

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

        total_costs_summary.append({
            "method": method,
            "query_time_cost_usd": query_time_costs['total_cost_usd'],
            "ragas_eval_cost_usd": ragas_eval_costs['total_cost_usd']
        })

    logging.info("--- TỔNG KẾT CHI PHÍ SUY LUẬN ---")
    df_costs = pd.DataFrame(total_costs_summary)
    logging.info(f"\n{df_costs.to_string(index=False)}")
    logging.info("Quy trình đánh giá đã hoàn tất.")

if __name__ == "__main__":
    run_evaluation()