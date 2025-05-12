import scrapetube
from rapidfuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer, util
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin, urlparse, parse_qs
import json
import re
import random
from langchain.schema import Document

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
    try:
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            thumbnail_url = data.get("thumbnail_url", "")
            return True, thumbnail_url
        else:
            logging.warning("Invalid YouTube video URL: %s", url)
            return False, ""
    except requests.RequestException as e:
        logging.error(f"Error checking YouTube video '{url}': {e}")
        return False, ""

def get_images(urls: list[str], count: int = 6) -> list[str]:
    collected_images = []

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            overview_div = soup.find("div", id="overview")
            if not overview_div:
                logging.warning(f"No <div id='overview'> found for {url}")
                continue

            found = False
            image_count = 0

            for tag in soup.find_all(True):
                if tag == overview_div:
                    found = True

                if found and tag.name == "img" and tag.get("src"):
                    img_url = urljoin(url, tag["src"])

                    # Check image size before accepting
                    try:
                        img_resp = requests.get(img_url, timeout=5)
                        img = Image.open(BytesIO(img_resp.content))
                        width, height = img.size
                        aspect_ratio = width / height
                        if width > 195 and height > 97:
                            if aspect_ratio == 1.5:
                                collected_images.append(img_url)
                                image_count += 1
                                if image_count == count:
                                    break

                    except Exception as img_error:
                        logging.warning(f"Skipping image {img_url}: {img_error}")
                        continue
        except Exception as e:
            # logging.error(f"Error processing URL {url}: {e}")
            continue

    # If fewer than `count` images are collected, fetch additional images from images.json
    if len(collected_images) < count:
        try:
            with open('images.json', 'r') as file:
                images_data = json.load(file)
                all_images = images_data

                # Calculate how many more images are needed
                remaining_count = count - len(collected_images)

                # Select random images to fill the gap
                additional_images = random.sample(all_images, min(remaining_count, len(all_images)))
                collected_images.extend(additional_images)
        except Exception as json_error:
            logging.error(f"Error reading images.json: {json_error}")

    # Ensure only `count` images are returned
    return collected_images[:count]

def extract_channel_videos(channel_id: str="UC07b9GB8a-4c6-T6pd2bbBQ", limit: int = None):
    videos = scrapetube.get_channel(channel_id=channel_id, limit=limit)
    video_list = []

    for video in videos:
        video_id = video.get('videoId')
        title = video.get('title', {}).get('runs', [{}])[0].get('text')
        description = video.get('descriptionSnippet', {}).get('runs', [{}])[0].get('text') if 'descriptionSnippet' in video else None
        published_time = video.get('publishedTimeText', {}).get('simpleText')
        view_count = video.get('viewCountText', {}).get('simpleText')
        thumbnail_url = video.get('thumbnail', {}).get('thumbnails', [{}])[-1].get('url') if 'thumbnail' in video else None
        video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None

        video_data = {
            'videoId': video_id,
            'title': title,
            'description': description,
            'publishedTime': published_time,
            'viewCount': view_count,
            'thumbnailUrl': thumbnail_url,
            'videoUrl': video_url
        }
        video_list.append(video_data)

    return {
        'videos': video_list
    }

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_doc_embeddings(docs, embeddings):
    documents = []
    texts = []

    for key in docs:
        for doc in docs[key]:
            title = doc.get('title', '')
            url = doc.get('url', '')
            combined_text = f"{title}. {url}"

            # Create Document object
            document = Document(page_content=combined_text, metadata={"title": title, "url": url})
            documents.append(document)
            texts.append(combined_text)

    # Generate embeddings
    embeddings_list = embeddings.embed_documents(texts)

    # Attach embeddings to documents
    for i, doc in enumerate(documents):
        doc.metadata['embedding'] = embeddings_list[i]

    return documents

def create_video_embeddings(items, embeddings):
    documents = []
    texts = []

    for item in items:
        title = item.get('title', '')
        description = item.get('description', '')
        video_url = item.get('videoUrl', '')
        thumbnail_url = item.get('thumbnailUrl', '')
        combined_text = f"{title}. {description}. {video_url} {thumbnail_url}"

        # Create Document object
        document = Document(page_content=combined_text, metadata={"title": title, "description": description, "videoUrl": video_url, "thumbnailUrl": thumbnail_url})
        documents.append(document)
        texts.append(combined_text)

    # Generate embeddings
    embeddings_list = embeddings.embed_documents(texts)

    # Attach embeddings to documents
    for i, doc in enumerate(documents):
        doc.metadata['embedding'] = embeddings_list[i]

    return documents

def search_videos(query_embedding, video_vectorstore, user_query_text, top_k=3):
    """
    Perform a similarity search in the vectorstore, then rerank results
    by title (75%) and description (25%) matching the raw query text.
    """
    # Step 1: retrieve a superset of candidates
    hits = video_vectorstore.similarity_search_by_vector(
        query_embedding,
        k=top_k,
        fetch_k=10
    )

    # Prepare for reranking
    scored = []
    for doc in hits:
        metadata = doc.metadata
        title = metadata.get('title', '')
        desc  = metadata.get('description', '')

        # Compute fuzzy-match scores normalized to [0,1]
        title_score = fuzz.token_sort_ratio(user_query_text, title) / 100.0
        desc_score  = fuzz.token_sort_ratio(user_query_text, desc)  / 100.0

        # Mix with 75% weight on title and 25% on description
        mixed_score = 0.75 * title_score + 0.25 * desc_score

        scored.append((mixed_score, doc))

    # Step 2: sort by mixed score and select top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:top_k]]

    # Build result dicts
    results = []
    for doc in top_docs:
        metadata = doc.metadata
        video_info = {
            'title': metadata.get('title'),
            'description': metadata.get('description'),
            'thumbnail_url': metadata.get('thumbnailUrl'),
            'video_url': metadata.get('videoUrl'),
            "manual_search": True,
        }
        results.append(video_info)

    return results

def search_documents(query_embedding, doc_vectorstore, top_k=3):
    # Perform similarity search using the vectorstore
    hits = doc_vectorstore.similarity_search_by_vector(query_embedding, k=top_k, fetch_k=20)

    results = []
    for doc in hits:
        metadata = doc.metadata
        doc_info = {
            'title': metadata.get('title'),
            'description': metadata.get('description'),
            'document_url': metadata.get('url'),
            # 'score': round(doc.score, 4)  # Uncomment if 'score' attribute is available
        }
        results.append(doc_info)
    return results

ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt']


def validate_document_url(url: str) -> bool:
    url_pattern = re.compile(
        r'^(https?://)?(www\.)?([\w-]+)\.[a-z]{2,6}(/\S*)?$'
    )
    if not re.match(url_pattern, url):
        return False

    # Parse the URL to extract the path and query parameters
    parsed_url = urlparse(url)
    path = parsed_url.path
    query_params = parse_qs(parsed_url.query)

    # Check file extension in path or query parameters
    extensions = [ext for ext in ALLOWED_EXTENSIONS if path.lower().endswith(f'.{ext}')]

    if not extensions:
        for value in query_params.values():
            for item in value:
                if any(item.lower().endswith(f'.{ext}') for ext in ALLOWED_EXTENSIONS):
                    extensions.append(item)

    if not extensions:
        return False

    # Define allowed document content types
    allowed_content_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword', 'text/plain']

    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get('Content-Type')
        return response.status_code == 200 and content_type in allowed_content_types

    except requests.RequestException:
        return False
