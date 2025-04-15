import concurrent.futures
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from bs4 import BeautifulSoup
import requests

def extract_text_from_url(url: str) -> dict:
    # logging.info(f"Extracting text from URL: {url}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else ""

        # Extract meta description
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        description = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""

        # Remove unnecessary tags
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Extract visible text
        text = soup.get_text(separator=' ', strip=True)

        # Extract image URLs with alt text
        img_tags = soup.find_all("img")
        image_map = {}
        unnamed_count = 1

        for img in img_tags:
            src = img.get("src")
            if src:
                full_url = requests.compat.urljoin(url, src)
                alt = img.get("alt", "").strip()

                if alt:
                    image_map[alt] = full_url
                else:
                    image_map[f"image_{unnamed_count}"] = full_url
                    unnamed_count += 1

        # logging.info(f"Successfully extracted text from URL: {url}")
        return {
            "source_page_url": url,
            "title": title,
            "description": description,
            "page_text": text,
            "image_urls": image_map
        }

    except Exception as e:
        logging.error(f"Error extracting text from URL: {url} - {str(e)}")
        return {
            "error": str(e)
        }


def extract_all_text_parallel(hcl_urls: dict, max_workers=700):
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
