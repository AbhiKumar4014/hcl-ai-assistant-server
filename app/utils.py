import concurrent.futures
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_url(url: str) -> dict:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
                          "(KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise error if status code != 200

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = (soup.title.string or "").strip() if soup.title else ""

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc.get("content", "").strip() if meta_desc else ""

        # Replace anchor tags with "text (url)"
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            full_url = urljoin(url, a["href"])
            a.replace_with(f"{text} ({full_url})" if text else f"{full_url}")

        # Replace img tags with Markdown-style ![alt](url)
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                full_url = urljoin(url, src)
                alt = img.get("alt", "").strip()
                img.replace_with(f"![{alt}]({full_url})" if alt else full_url)

        # Remove non-visible/script tags
        for tag in soup(["style", "noscript", "script"]):
            tag.decompose()

        # Extract visible text
        page_text = soup.get_text(separator=' ', strip=True)

        return {
            "source_page_url": url,
            "title": title,
            "description": description,
            "page_text": page_text
        }

    except Exception as e:
        logging.error(f"Error extracting text from URL: {url} - {e}")
        return None


def extract_all_text_parallel(hcl_urls: dict, max_workers=10):
    context = {}
    tasks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for site_name, url_list in hcl_urls.items():
            if not url_list:
                continue
            future = executor.submit(extract_text_from_url, url_list[0])  # Assuming only first URL is needed
            tasks.append((site_name, future))

        for site_name, future in tasks:
            result = future.result()
            if site_name not in context:
                context[site_name] = result

    return context



def fetch_attachment_links(api_url: str, base_url: str) -> list[dict]:
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if "itemlist" not in data or not isinstance(data["itemlist"], list):
            raise ValueError("Invalid response format")

        result = [
            {
                "title": item.get("title", ""),
                "url": f"{base_url}{item.get('attachment', '')}"
            }
            for item in data["itemlist"]
        ]

        return result

    except Exception as e:
        logging.error("Error: in except", e)
        return []

def extract_all_doc_urls() -> dict:
    BASE_URL = "http://www.hcl-software.com"
    response = {}
    with open('document_urls.json', 'r') as file:
        documents_urls = json.load(file)
    if documents_urls:
        for endpoint, url in documents_urls.items():
            logging.info("Extracting")
            data = fetch_attachment_links(url, BASE_URL)
            if data:
                response[endpoint] = data
    return response


def is_youtube_video_valid(url: str) -> tuple[bool, str]:
    oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
    # print(f"Checking YouTube video: {oembed_url}")
    try:
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            thumbnail_url = data.get("thumbnail_url", "")
            return True, thumbnail_url
        else:
            return False, ""
    except requests.RequestException as e:
        logging.error(f"Error checking YouTube video '{url}': {e}")
        return False, ""

import random

