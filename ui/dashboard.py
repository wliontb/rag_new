import streamlit as st
import pandas as pd
import glob
import os

# Thêm đường dẫn gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import settings

st.set_page_config(page_title="Dashboard Đánh giá RAG", layout="wide")

st.title("📊 Dashboard Trực quan hóa Kết quả Đánh giá")
st.write("Nơi đây hiển thị và so sánh hiệu suất của các pipeline RAG khác nhau.")

# --- Tải dữ liệu kết quả --- #
results_path = str(settings.EVAL_RESULTS_DIR)
csv_files = glob.glob(os.path.join(results_path, "*.csv"))

if not csv_files:
    st.warning("Không tìm thấy file kết quả nào trong thư mục `evaluation/results`.")
    st.info("Vui lòng chạy quy trình đánh giá trước bằng lệnh: `python -m evaluation.main_evaluation`")
else:
    all_results = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Lấy tên phương pháp từ tên file
            method_name = os.path.basename(f).replace("evaluation_results_", "").replace(".csv", "")
            df['method'] = method_name
            all_results.append(df)
        except Exception as e:
            st.error(f"Lỗi khi đọc file {f}: {e}")

    if all_results:
        # Ghép tất cả kết quả lại
        full_df = pd.concat(all_results, ignore_index=True)

        st.subheader("So sánh Hiệu suất Trung bình")
        
        # Tính toán các chỉ số trung bình
        avg_scores = full_df.groupby('method').mean(numeric_only=True)
        st.dataframe(avg_scores.style.format("{:.3f}").background_gradient(cmap='viridis'))

        st.subheader("Phân phối của các Chỉ số")
        
        # Chọn chỉ số để xem
        metric_to_plot = st.selectbox("Chọn một chỉ số để trực quan hóa:", avg_scores.columns)

        if metric_to_plot:
            st.bar_chart(full_df, x='method', y=metric_to_plot)

        st.subheader("Chi tiết Kết quả")
        st.dataframe(full_df)
