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
