# from diffusers import StableDiffusionPipeline
# from PIL import Image
# import torch
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import logging
from langchain.embeddings.base import Embeddings
import numpy as np

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
        
        video_map = {}
        seen_videos = set()

        youtube_anchors = soup.find_all("a", href=True)

        for a_tag in youtube_anchors:
            href = a_tag["href"]
            if "youtube.com/watch" in href or "youtu.be/" in href:
                video_url = requests.compat.urljoin(url, href)
                if video_url in seen_videos:
                    continue
                seen_videos.add(video_url)

                img_tag = a_tag.find("img")

                if img_tag:
                    img_src = img_tag.get("src")
                    img_alt = img_tag.get("alt", "").strip()
                    if img_src:
                        full_img_url = requests.compat.urljoin(url, img_src)
                        key = img_alt if img_alt else f"video_{len(video_map)+1}"
                        video_map[key] = {
                            "video_url": video_url,
                            "thumbnail_url": full_img_url
                        }
                else:
                    key = f"video_{len(video_map)+1}"
                    video_map[key] = {
                        "video_url": video_url,
                        "thumbnail_url": None
                    }

        return {
            "source_page_url": url,
            "title": title,
            "description": description,
            "page_text": text,
            "image_urls": image_map,
            "video_urls": video_map
        }

    except Exception as e:
        logging.error(f"Error extracting text from URL: {url} - {str(e)}")
        return {
            "error": str(e)
        }


def extract_all_text_parallel(hcl_urls: dict, max_workers=100):
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


def extract_relevant_media(
    query: str, source_docs: list, embeddings: Embeddings, top_k=3
):
    image_data = []
    video_data = []

    for doc in source_docs:
        # Extract images
        image_dict = doc.metadata.get("image_urls", {})
        for alt, url in image_dict.items():
            if alt and url:
                image_data.append(
                    {"alt": alt, "url": url, "doc_metadata": doc.metadata}
                )

        # Extract videos
        video_dict = doc.metadata.get("video_urls", {})
        for alt, info in video_dict.items():
            video_url = info.get("video_url")
            if alt and video_url:
                video_data.append(
                    {
                        "alt": alt,
                        "url": video_url,
                        "thumbnail": info.get("thumbnail_url"),
                        "doc_metadata": doc.metadata,
                    }
                )

    # Process image similarities
    top_images = []
    if image_data:
        alt_texts = [item["alt"] for item in image_data]
        image_vectors = embeddings.embed_documents(alt_texts)
        query_vector = embeddings.embed_query(query)
        similarities = np.dot(image_vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_images = [image_data[i] for i in top_indices]

    # Process video similarities (if more than 3, else return all)
    top_videos = []
    if len(video_data) > top_k:
        alt_texts = [item["alt"] for item in video_data]
        video_vectors = embeddings.embed_documents(alt_texts)
        query_vector = embeddings.embed_query(query)
        similarities = np.dot(video_vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_videos = [video_data[i] for i in top_indices]
    else:
        top_videos = video_data

    return {"images": top_images, "videos": top_videos}


# def generate_image_from_text(prompt_text: str, output_path: str = "generated_image.png"):
#     # Load the Stable Diffusion model (only once)
#     model_id = "runwayml/stable-diffusion-v1-5"
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id, torch_dtype=torch.float16
#     )
#     pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

#     # Generate image
#     image = pipe(prompt_text).images[0]
#     image.save(output_path)
#     return output_path
