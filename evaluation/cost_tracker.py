import logging
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bảng giá (ví dụ cho Gemini 1.5 Flash, đơn vị: USD cho 1 triệu token)
# Cần cập nhật nếu bạn dùng model khác
MODEL_PRICES = {
    "gemini-1.5-flash-latest": {
        "input": 0.35 / 1_000_000,
        "output": 0.70 / 1_000_000,
    },
    # Thêm các model khác nếu cần
    "gemini-pro": {
        "input": 0.5 / 1_000_000,
        "output": 1.5 / 1_000_000,
    }
}

class CostCallbackHandler(BaseCallbackHandler):
    """Callback handler để theo dõi và tính toán chi phí API của LLM."""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """Reset lại các bộ đếm."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.successful_requests = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Chạy khi LLM bắt đầu."""
        pass # Không cần làm gì ở đây

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Chạy khi LLM kết thúc, thu thập thông tin token và chi phí."""
        try:
            if response.llm_output and 'token_usage' in response.llm_output:
                token_usage = response.llm_output['token_usage']
                model_name = response.llm_output.get('model_name', 'gemini-1.5-flash-latest')

                # Lấy thông tin token
                prompt_tokens = token_usage.get('prompt_token_count', 0)
                completion_tokens = token_usage.get('completion_token_count', 0)
                total_tokens = token_usage.get('total_token_count', 0)

                # Cập nhật bộ đếm
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                self.successful_requests += 1

                # Tính toán chi phí
                if model_name in MODEL_PRICES:
                    price_info = MODEL_PRICES[model_name]
                    cost = (prompt_tokens * price_info['input']) + (completion_tokens * price_info['output'])
                    self.total_cost += cost
                else:
                    logging.warning(f"Không tìm thấy thông tin giá cho model: {model_name}")

        except Exception as e:
            logging.error(f"Lỗi trong CostCallbackHandler.on_llm_end: {e}")

    def get_costs(self) -> Dict[str, Any]:
        """Lấy thông tin chi phí đã tích lũy."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "successful_requests": self.successful_requests,
            "total_cost_usd": f"{self.total_cost:.6f}"
        }
