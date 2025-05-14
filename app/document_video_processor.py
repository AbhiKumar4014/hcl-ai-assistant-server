from flask import jsonify
from concurrent.futures import ThreadPoolExecutor
from utils import is_youtube_video_valid, validate_document_url, search_videos, search_documents

# Constants for maximum count
DEFAULT_MAX_DOCUMENTS = 3
DEFAULT_MAX_VIDEOS = 3

def append_valid_documents_and_videos(parsed_response, query_embeddings, query, docs_vectorstore, videos_vectorstore, max_documents=DEFAULT_MAX_DOCUMENTS, max_videos=DEFAULT_MAX_VIDEOS):
    """
    Appends valid documents and videos to the parsed response concurrently.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        if "videos" in parsed_response:
            futures["videos"] = executor.submit(_process_and_filter_videos, parsed_response["videos"], max_videos)

        if "documents" in parsed_response:
            futures["documents"] = executor.submit(_process_and_filter_documents, parsed_response["documents"], max_documents)

        for key, future in futures.items():
            parsed_response[key] = future.result()

    # Add manually extracted videos and documents concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_videos = executor.submit(_add_manually_extracted_videos, parsed_response, query_embeddings, query, videos_vectorstore, max_videos)
        future_docs = executor.submit(_add_manually_extracted_documents, parsed_response, query_embeddings, docs_vectorstore, max_documents)

        parsed_response = future_videos.result()
        parsed_response = future_docs.result()

    return parsed_response


def _process_and_filter_videos(videos, max_videos):
    valid_videos = []
    for video in videos:
        url = video.get("video_url", "")
        is_valid, thumbnail_url = is_youtube_video_valid(url)
        if is_valid:
            video["thumbnail_url"] = thumbnail_url
            valid_videos.append(video)
    return valid_videos[:max_videos]


def _process_and_filter_documents(documents, max_documents):
    valid_docs = []
    for doc in documents:
        url = doc.get("document_url", "")
        is_valid = validate_document_url(url)
        if is_valid:
            valid_docs.append(doc)
    return valid_docs[:max_documents]


def _add_manually_extracted_videos(parsed_response, query_embeddings, query, videos_vectorstore, max_videos):
    if len(parsed_response.get("videos", [])) < max_videos:
        existing_urls = {video['video_url'] for video in parsed_response.get("videos", [])}
        relevant_videos = search_videos(query_embeddings, videos_vectorstore, query)
        unique_videos = [video for video in relevant_videos if video['video_url'] not in existing_urls]
        parsed_response["videos"].extend(unique_videos[:max_videos - len(parsed_response["videos"])])
    return parsed_response


def _add_manually_extracted_documents(parsed_response, query_embeddings, docs_vectorstore, max_documents):
    if len(parsed_response.get("documents", [])) < max_documents:
        existing_urls = {doc['document_url'] for doc in parsed_response.get("documents", [])}
        relevant_docs = search_documents(query_embeddings, docs_vectorstore)
        unique_docs = [doc for doc in relevant_docs if doc['document_url'] not in existing_urls]
        parsed_response["documents"].extend(unique_docs[:max_documents - len(parsed_response["documents"])])
    return parsed_response
