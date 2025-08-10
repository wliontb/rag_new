from langchain_core.prompts import PromptTemplate

# Prompt để trích xuất ngày tháng từ câu hỏi của người dùng
DATE_EXTRACTION_TEMPLATE = """
BẠN LÀ MỘT TRỢ LÝ GIỎI VỀ VIỆC XỬ LÝ NGÔN NGỮ TỰ NHIÊN.
Trích xuất ngày bắt đầu (start_date) và ngày kết thúc (end_date) từ câu hỏi của người dùng. 
Hãy chuyển đổi các mốc thời gian tương đối (ví dụ: "tuần trước", "tháng này", "năm ngoái", "quý 2 năm 2023") thành định dạng YYYY-MM-DD.

Hôm nay là ngày: {today}

Câu hỏi: "{question}"

Nếu không tìm thấy thông tin ngày tháng, hãy trả về một JSON object với hai key "start_date" và "end_date" có giá trị là null.
Chỉ trả về JSON object, không thêm bất kỳ giải thích nào.

JSON Output:
"""

DATE_EXTRACTION_PROMPT = PromptTemplate(
    template=DATE_EXTRACTION_TEMPLATE,
    input_variables=["question", "today"],
)


# Prompt chính để sinh câu trả lời dựa trên ngữ cảnh
RAG_TEMPLATE = """
BẠN LÀ MỘT TRỢ LÝ AI CHUYÊN VỀ TIN TỨC TÀI CHÍNH VIỆT NAM.
Hãy sử dụng những thông tin trong phần "Ngữ cảnh" dưới đây để trả lời câu hỏi của người dùng một cách chính xác và đầy đủ.

**Yêu cầu:**
1.  Chỉ dựa vào thông tin được cung cấp trong "Ngữ cảnh". Không sử dụng kiến thức bên ngoài.
2.  Nếu "Ngữ cảnh" không chứa thông tin để trả lời câu hỏi, hãy nói rằng: "Tôi không tìm thấy thông tin phù hợp trong dữ liệu để trả lời câu hỏi này."
3.  Trình bày câu trả lời một cách rõ ràng, chuyên nghiệp và súc tích.

**Ngữ cảnh:**
---
{context}
---

**Câu hỏi:** {question}

**Câu trả lời:**
"""

RAG_PROMPT = PromptTemplate(
    template=RAG_TEMPLATE,
    input_variables=["context", "question"],
)
