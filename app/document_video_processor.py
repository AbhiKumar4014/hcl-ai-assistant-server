from flask import jsonify
from utils import is_youtube_video_valid, validate_document_url, search_videos, search_documents

# Constants for maximum count
DEFAULT_MAX_DOCUMENTS = 3
DEFAULT_MAX_VIDEOS = 3

def append_valid_documents_and_videos(parsed_response, query_embeddings, query, docs_vectorstore, videos_vectorstore, max_documents=DEFAULT_MAX_DOCUMENTS, max_videos=DEFAULT_MAX_VIDEOS):
    """
    Appends valid documents and videos to the parsed response, ensuring the maximum counts are respected.

    Args:
    - parsed_response (dict): The response data containing 'videos' and 'documents' to be processed.
    - max_documents (int): The maximum number of documents to include in the response.
    - max_videos (int): The maximum number of videos to include in the response.

    Returns:
    - Response (JSON): The modified response with valid videos and documents, adhering to the maximum counts.
    """

    # Process and filter videos
    if "videos" in parsed_response:
        parsed_response["videos"] = _process_and_filter_videos(parsed_response["videos"], max_videos)

    # Process and filter documents
    if "documents" in parsed_response:
        parsed_response["documents"] = _process_and_filter_documents(parsed_response["documents"], max_documents)

    # Add manually extracted videos if necessary
    parsed_response = _add_manually_extracted_videos(parsed_response, query_embeddings, query, videos_vectorstore, max_videos)

    # Add manually extracted documents if necessary
    parsed_response = _add_manually_extracted_documents(parsed_response, query_embeddings, docs_vectorstore, max_documents)

    return parsed_response


def _process_and_filter_videos(videos, max_videos):
    """
    Filters valid videos and limits the number of videos.

    Args:
    - videos (list): List of video dictionaries.
    - max_videos (int): Maximum count of videos to return.

    Returns:
    - List of valid and limited videos.
    """
    valid_videos = []
    for video in videos:
        url = video.get("video_url", "")
        is_valid, thumbnail_url = is_youtube_video_valid(url)
        if is_valid:
            video["thumbnail_url"] = thumbnail_url
            valid_videos.append(video)
    return valid_videos[:max_videos]


def _process_and_filter_documents(documents, max_documents):
    """
    Filters valid documents and limits the number of documents.

    Args:
    - documents (list): List of document dictionaries.
    - max_documents (int): Maximum count of documents to return.

    Returns:
    - List of valid and limited documents.
    """
    valid_docs = []
    for doc in documents:
        url = doc.get("document_url", "")
        is_valid = validate_document_url(url)
        if is_valid:
            valid_docs.append(doc)
    return valid_docs[:max_documents]


def _add_manually_extracted_videos(parsed_response, query_embeddings, query, videos_vectorstore, max_videos):
    """
    Adds manually extracted videos if the count is insufficient.

    Args:
    - parsed_response (dict): The response data containing 'videos' to be processed.
    - max_videos (int): The maximum number of videos to include in the response.

    Returns:
    - dict: The updated parsed response with additional videos if necessary.
    """
    if len(parsed_response.get("videos", [])) < max_videos:
        existing_urls = {video['video_url'] for video in parsed_response.get("videos", [])}
        relevant_videos = search_videos(query_embeddings, videos_vectorstore, query)
        unique_videos = [video for video in relevant_videos if video['video_url'] not in existing_urls]
        parsed_response["videos"].extend(unique_videos[:max_videos - len(parsed_response["videos"])])
    return parsed_response


def _add_manually_extracted_documents(parsed_response, query_embeddings, docs_vectorstore, max_documents):
    """
    Adds manually extracted documents if the count is insufficient.

    Args:
    - parsed_response (dict): The response data containing 'documents' to be processed.
    - max_documents (int): The maximum number of documents to include in the response.

    Returns:
    - dict: The updated parsed response with additional documents if necessary.
    """
    if len(parsed_response.get("documents", [])) < max_documents:
        existing_urls = {doc['document_url'] for doc in parsed_response.get("documents", [])}
        relevant_docs = search_documents(query_embeddings, docs_vectorstore)
        unique_docs = [doc for doc in relevant_docs if doc['document_url'] not in existing_urls]
        parsed_response["documents"].extend(unique_docs[:max_documents - len(parsed_response["documents"])])
    return parsed_response
