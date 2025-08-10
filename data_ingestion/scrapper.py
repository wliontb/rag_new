import requests
from bs4 import BeautifulSoup
import time
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_article_urls(base_url, max_pages=1):
    """Lấy danh sách URL của các bài báo từ trang chính."""
    urls = []
    for page in range(1, max_pages + 1):
        try:
            # Thêm logic để xử lý phân trang nếu cần
            # Ví dụ: page_url = f"{base_url}/page/{page}"
            page_url = base_url # Giả sử chỉ lấy từ trang đầu
            response = requests.get(page_url, timeout=10)
            response.raise_for_status() # Ném lỗi nếu request không thành công

            soup = BeautifulSoup(response.content, 'html.parser')

            # --- Logic để tìm link bài báo --- 
            # Cần tùy chỉnh selector cho trang web cụ thể
            # Ví dụ:
            article_links = soup.select("div.article-bdt-20 h3.article-title a") 
            if not article_links:
                logging.warning(f"Không tìm thấy link bài báo nào trên trang: {page_url}")

            for link in article_links:
                href = link.get('href')
                if href and href.startswith('/'):
                    full_url = f"{base_url.rstrip('/')}{href}"
                    urls.append(full_url)
                elif href:
                    urls.append(href)

            logging.info(f"Đã tìm thấy {len(article_links)} URLs trên trang {page}")
            time.sleep(1) # Tạm dừng để tránh làm quá tải server

        except requests.exceptions.RequestException as e:
            logging.error(f"Lỗi khi truy cập {page_url}: {e}")
            break # Dừng nếu có lỗi mạng
    print(urls)
    return list(set(urls)) # Trả về danh sách duy nhất

def scrape_article_content(url):
    """Cào nội dung chi tiết từ một URL bài báo."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Logic để trích xuất nội dung --- 
        # Cần tùy chỉnh các selector cho trang web cụ thể
        title_element = soup.select_one("h1.post-title")
        title = title_element.text.strip() if title_element else "Không có tiêu đề"

        # Lấy tất cả các thẻ p trong phần thân bài báo
        content_elements = soup.select("div.post-content p")
        content = "\n".join([p.text.strip() for p in content_elements])

        if not content:
            logging.warning(f"Không tìm thấy nội dung cho URL: {url}")
            return None

        return {
            "url": url,
            "title": title,
            "content": content
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi cào nội dung từ {url}: {e}")
        return None
