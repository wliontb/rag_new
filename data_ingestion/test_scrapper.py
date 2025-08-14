import requests
from bs4 import BeautifulSoup
import time
import logging
# from data_ingestion.common import Data

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

def get_article_links(url):
    print(f"Đang truy cập URL: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Không thể truy cập trang chủ:", response.status_code)
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    article_links = []

    listRes = soup.find("div", class_="col-left col-left-new col-left-subfolder")
    if not listRes:
        print("Không tìm thấy danh sách bài viết.")
        return []

    res = listRes.find("div", class_="width_common list-news-subfolder has-border-right")
    all_hrefs = res.find_all("article", class_="item-news item-news-common thumb-left") if res else []
    print(f"Tổng số bài viết tìm thấy: {len(all_hrefs)}")

    for href in all_hrefs:
        h2 = href.find("h2", class_="title-news")
        link = h2.find("a")
        if link and link.has_attr("href"):
            article_links.append(link["href"])

    print("Đã tìm thấy các liên kết bài viết:", article_links)
    return article_links

def fetch_article_urls(base_url='https://vnexpress.net/kinh-doanh/chung-khoan-p', max_pages=3):
    """Lấy danh sách URL của các bài báo từ trang chính."""
    urls = []
    # Số lượng trang muốn cào (cứ mỗi 15 là 1 trang mới)
    for offset in range(1, 4, 1):  # Ví dụ: 4 trang đầu tiên
        curr_url = base_url + str(offset)
        print(f"Đang duyệt trang: {curr_url}")
        urls.extend(get_article_links(curr_url))

    return urls

def scrape_article_content(url):
    """Cào nội dung chi tiết từ một URL bài báo."""
    try:
        # Lấy ID bài viết từ link
        article_id = url.split("-")[-1].replace(".html", "")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        print("SOUP", soup)

        # --- Logic để trích xuất nội dung --- 
        title_tag = soup.find("h1", class_="title-detail")
        title = title_tag.text.strip() if title_tag else "Không tìm thấy tiêu đề"

        date = soup.find("span", class_="date")
        date_text = date.text.strip() if date else "Không tìm thấy ngày"

        # Lấy tất cả các thẻ p trong phần thân bài báo
        content_div = soup.find("article", class_="fck_detail")
        if not content_div:
            print(f"Không tìm thấy nội dung bài viết: {url}")
            return None

        paragraphs = content_div.find_all("p")
        paragraph_texts = [p.text.strip() for p in paragraphs if p.text.strip()]
        full_content = "\n".join(paragraph_texts)

        if not full_content:
            logging.warning(f"Không tìm thấy nội dung cho URL: {url}")
            return None

        return {
            "id": article_id,
            "url": url,
            "title": title,
            "date": date_text,
            "content": full_content
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi cào nội dung từ {url}: {e}")
        return None

list_url = fetch_article_urls()
print(scrape_article_content(list_url[0]))  # Cào nội dung của bài viết đầu tiên
print(f"Đã tìm thấy {len(list_url)} bài viết.")