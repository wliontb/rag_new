#!/usr/bin/env python3
"""
Script để so sánh kết quả đánh giá RAGAS giữa các embedding models khác nhau.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_evaluation_results():
    """So sánh kết quả đánh giá giữa các embedding models."""
    
    results_dir = Path("evaluation/results")
    
    # Đọc kết quả từ các file CSV
    models = {}
    
    for file_path in results_dir.glob("evaluation_results_*.csv"):
        model_name = file_path.stem.replace("evaluation_results_", "")
        try:
            df = pd.read_csv(file_path)
            models[model_name] = df
            logging.info(f"Đã tải kết quả cho model: {model_name}")
        except Exception as e:
            logging.error(f"Lỗi khi đọc file {file_path}: {e}")
    
    if len(models) < 2:
        logging.error("Cần ít nhất 2 models để so sánh")
        return
    
    # Tính toán metrics trung bình cho mỗi model
    summary_stats = {}
    
    for model_name, df in models.items():
        stats = {}
        
        # Tính mean cho các metric chính (bỏ qua NaN)
        for col in df.columns:
            if 'Score' in col or 'Accuracy' in col or 'Relevancy' in col or 'Recall' in col or 'Similarity' in col or 'Faithfulness' in col:
                stats[col] = df[col].mean()
        
        summary_stats[model_name] = stats
        
        logging.info(f"\n=== Kết quả trung bình cho {model_name.upper()} ===")
        for metric, value in stats.items():
            logging.info(f"{metric}: {value:.4f}")
    
    # Tạo DataFrame để so sánh
    comparison_df = pd.DataFrame(summary_stats).T
    
    # Tính toán sự khác biệt
    if len(models) == 2:
        model_names = list(models.keys())
        model1, model2 = model_names[0], model_names[1]
        
        logging.info(f"\n=== So sánh {model1.upper()} vs {model2.upper()} ===")
        
        for metric in comparison_df.columns:
            diff = comparison_df.loc[model1, metric] - comparison_df.loc[model2, metric]
            better_model = model1 if diff > 0 else model2
            logging.info(f"{metric}:")
            logging.info(f"  {model1}: {comparison_df.loc[model1, metric]:.4f}")
            logging.info(f"  {model2}: {comparison_df.loc[model2, metric]:.4f}")
            logging.info(f"  Khác biệt: {abs(diff):.4f} (Tốt hơn: {better_model})")
            logging.info("")
    
    # Lưu kết quả so sánh
    comparison_output_path = results_dir / "model_comparison_summary.csv"
    comparison_df.to_csv(comparison_output_path)
    logging.info(f"Đã lưu kết quả so sánh vào: {comparison_output_path}")
    
    logging.info("=== TỔNG KẾT SO SÁNH ===")
    logging.info(f"Đã so sánh {len(models)} models")
    logging.info("Xem file model_comparison_summary.csv để biết chi tiết")
    
    return comparison_df

def analyze_per_question_performance():
    """Phân tích performance theo từng câu hỏi."""
    
    results_dir = Path("evaluation/results")
    
    # Đọc kết quả từ các file CSV
    models = {}
    
    for file_path in results_dir.glob("evaluation_results_*.csv"):
        model_name = file_path.stem.replace("evaluation_results_", "")
        try:
            df = pd.read_csv(file_path)
            models[model_name] = df
        except Exception as e:
            logging.error(f"Lỗi khi đọc file {file_path}: {e}")
            continue
    
    if len(models) < 2:
        return
    
    # So sánh performance từng câu hỏi
    model_names = list(models.keys())
    
    # Tạo DataFrame kết hợp
    combined_analysis = pd.DataFrame()
    
    for i, (model_name, df) in enumerate(models.items()):
        temp_df = df.copy()
        temp_df['model'] = model_name
        temp_df['question_id'] = range(len(temp_df))
        combined_analysis = pd.concat([combined_analysis, temp_df], ignore_index=True)
    
    # Lưu kết quả phân tích chi tiết
    detailed_output_path = results_dir / "detailed_question_analysis.csv"
    combined_analysis.to_csv(detailed_output_path, index=False)
    logging.info(f"Đã lưu phân tích chi tiết vào: {detailed_output_path}")
    
    return combined_analysis

if __name__ == "__main__":
    logging.info("=== BẮT ĐẦU SO SÁNH KẾT QUẢ ĐÁNH GIÁ ===")
    
    # So sánh tổng quan
    comparison_summary = compare_evaluation_results()
    
    # Phân tích chi tiết theo câu hỏi
    detailed_analysis = analyze_per_question_performance()
    
    logging.info("=== HOÀN TẤT SO SÁNH ===")
