from pathlib import Path
import json
import concurrent.futures
import time
from dotenv import load_dotenv
import os
import logging

from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from utils import *  # Assuming the necessary utils are imported from your utils module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=""" You are a helpful and intelligent assistant that answers questions using information from the HCLSoftware website.
    You speak in a friendly, conversational way and include only relevant HCLSoftware URLs in a structured format.
    You are a highly knowledgeable and professional assistant that provides comprehensive, accurate, and well-formatted responses based on information from the HCLSoftware website.

    Response Guidelines:
    - Maintain a friendly, informative, and conversational tone.
    - Use structured formatting, including headings, bullet points, and numbered lists where appropriate.
    - Provide detailed, well-researched, and descriptive responses with a minimum of 500 words, thoroughly addressing the query.
    - Include only relevant HCLSoftware URLs using the specified JSON format, prioritizing official product, solution, and service pages.

    Reference Inclusion:
    - Include up-to-date product, solution, or service links, prioritizing official HCLSoftware pages.
    - If the query is blog-related, prioritize blog links and clearly state the "Source" URL.
    - Ensure transparency and traceability by accurately identifying content origins.
    - Avoid stating that something is "not related to HCLSoftware" unless it clearly isn’t.
    - Limit references to the most relevant links, ensuring they are unique and contextually appropriate.
    - Include at least one contact link in every response. If the query specifies a particular product, include the relevant product-specific contact page.

    Response Structure:
    - Deliver responses in markdown format, without code block markers.
    - Avoid including URLs in the "answer" field; only use the "reference_url" array.
    - Each reference JSON must include a brief, informative description summarizing its relevance.
    - Ensure the contact or support link is included and contextually relevant.

    Video Content:
    - Include only verified, contextually relevant YouTube video links found in the source context.
    - If no relevant video is available, do not generate or include a video link.

    Formatting and Structure:
    - Maintain structured formatting in responses, using headings, bullet points, and lists.
    - Ensure each response is clear, well-organized, and easy to navigate.
    - Do not include any HTML tags, code, or HTML formatting in the "answer" field under any circumstances.
    - Responses must not hardcode any specific count for links or references; ensure relevance and contextual accuracy.

    History Handling:
    - Retain conversation context only when explicitly referenced by the user, ensuring each response remains focused and relevant to the immediate query.
    - Avoid using previous conversation context unless the user clearly refers to it.

    Content Limitations:
    - Do not provide opinions, speculative content, or comparisons with third-party platforms.
    - Maintain professionalism, accuracy, and relevance in all responses.

    - The assistant should retain conversation history for context but not let it influence responses unless the user explicitly refers to it in their query. History should be considered only when directly mentioned or requested by the user.
    The final output must be strictly well-formatted and valid JSON, without any extra commentary, or code block markers.

    Context: {context}
    Question: {question}

    You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers:

    {{"answer": "Your markdown answer",
    "is_valid_query": True or False,
    "references": [{{"title": "Title (max 20 characters)",
                    "reference_url": "",
                    "description": "Short summary (max 80 characters)"}}],
    "videos": [{{"title": "Short video title (max 20 characters)",
                    "video_url": "Exact YouTube URL",
                    "description": "Brief summary of the video"}}],
    "documents": [{{"title": "Short document title",
                    "document_url": "Exact document link",
                    "description": "Brief summary of document"}}],
    "enhanced_user_query": "Your enhanced query"}}
    """
)

# Define the chunking function for document embedding
def chunk_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

# Define the function for embedding documents in batches
def embed_documents_in_batches(documents, embeddings, persist_dir, batch_size=100, delay=45):
    logging.info("Embedding documents in batches to respect rate limits.")
    all_vectors = None

    for idx, batch in enumerate(chunk_documents(documents, batch_size)):
        logging.info(f"Processing batch {idx + 1}: {len(batch)} documents")
        try:
            batch_store = FAISS.from_documents(batch, embedding=embeddings)
            if all_vectors is None:
                all_vectors = batch_store
            else:
                all_vectors.merge_from(batch_store)
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Error in batch {idx + 1}: {e}")

    if all_vectors:
        all_vectors.save_local(persist_dir)
        logging.info("FAISS vectorstore saved locally.")
    return all_vectors

# Helper function to create URL documents
def create_url_document(entry_id, content_data):
    if content_data is None or not isinstance(content_data, dict):
        logging.warning(f"Skipping URL entry {entry_id} with unknown source")
        return None

    title = content_data.get("title", "unknown")
    description = content_data.get("description", "unknown")
    source_url = content_data.get("source_page_url", "unknown")
    page_text = content_data.get("page_text", "unknown")
    page_content = f"[Source: {source_url}]\n\n{page_text}"

    return Document(
        page_content=page_content,
        metadata={
            "source": source_url,
            "title": title,
            "description": description,
            "entry_id": entry_id,
        },
    )

# Helper function to create document content
def create_document_document(entry_id, content_data):
    if content_data is None or not isinstance(content_data, dict):
        logging.warning(f"Skipping Document entry {entry_id} with unknown source")
        return None

    title = content_data.get("title", "unknown")
    description = content_data.get("description", "unknown")
    document_url = content_data.get("document_url", "unknown")
    page_text = content_data.get("content", "unknown")
    page_content = f"[Document URL: {document_url}]\n\n{page_text}"

    return Document(
        page_content=page_content,
        metadata={
            "source": document_url,
            "title": title,
            "description": description,
            "entry_id": entry_id,
        },
    )

# Helper function to create video content
def create_video_document(entry_id, content_data):
    if content_data is None or not isinstance(content_data, dict):
        logging.warning(f"Skipping Video entry {entry_id} with unknown source")
        return None

    title = content_data.get("title", "unknown")
    description = content_data.get("description", "unknown")
    video_url = content_data.get("videoUrl", "unknown")
    video_description = content_data.get("description", "unknown")
    page_content = f"[Video URL: {video_url}]\n\n{video_description}"

    return Document(
        page_content=page_content,
        metadata={
            "source": video_url,
            "title": title,
            "description": description,
            "entry_id": entry_id,
        },
    )

# Function to load model data from various sources
def load_model_data(source_data=None, source_type: str = "faiss", persist_dir: str = "./faiss_index"):
    retriever = docs_vectorstore = videos_vectorstore = None
    logging.info(f"Loading model data from source type: {source_type}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

    if source_type == "faiss":
        logging.info(f"Loading FAISS vectorstore from: {persist_dir}")
        try:
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            docs_vectorstore = FAISS.load_local("./faiss_index/docs", embeddings, allow_dangerous_deserialization=True)
            videos_vectorstore = FAISS.load_local("./faiss_index/videos", embeddings, allow_dangerous_deserialization=True)

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10,
                    "fetch_k": 50,
                    "lambda_mult": 0.5,
                    "score_threshold": 0.7,
                },
            )
            logging.info("FAISS vectorstore loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading FAISS vectorstore: {e}")
    else:
        if source_type == "json_file":
            if not source_data:
                raise ValueError("source_data must be a file path when source_type is 'json_file'")
            logging.info(f"Loading data from JSON file: {source_data}")
            context = json.loads(Path(source_data).read_text(encoding="utf-8"))
        elif source_type == "json_object":
            if not isinstance(source_data, dict):
                raise ValueError("source_data must be a dictionary when source_type is 'json_object'")
            logging.info("Loading data from JSON object")
            context = source_data
        else:
            raise ValueError("Invalid source_type. Must be 'faiss', 'json_file', or 'json_object'.")

        logging.info(f"Creating documents from context with {len(context)} entries")
        all_documents = []

        # Process content (URL type)
        for i, content_data in enumerate(context.get("content", [])):
            document = create_url_document(i, content_data)
            if document:
                all_documents.append(document)

        # Process documents (Document type)
        for i, content_data in enumerate(context.get("documents", [])):
            document = create_document_document(i, content_data)
            if document:
                all_documents.append(document)

        # Process videos (Video type)
        for i, content_data in enumerate(context.get("videos", [])):
            document = create_video_document(i, content_data)
            if document:
                all_documents.append(document)

        logging.info("Creating FAISS vectorstore from documents with batching")
        vectorstore = embed_documents_in_batches(all_documents, embeddings, persist_dir)
        docs_vectorstore = embed_documents_in_batches(create_doc_embeddings(context["documents"], embeddings), embeddings, "./faiss_index/docs")
        videos_vectorstore = embed_documents_in_batches(create_video_embeddings(context["videos"], embeddings), embeddings, "./faiss_index/videos")

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 50,
                "lambda_mult": 0.5,
                "score_threshold": 0.7,
            },
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.2, google_api_key=google_api_key
    )

    if retriever is not None:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )
    else:
        logging.error("Retriever is not initialized.")
        qa_chain = None

    logging.info("QA chain loaded successfully.")
    return qa_chain, docs_vectorstore, videos_vectorstore, embeddings