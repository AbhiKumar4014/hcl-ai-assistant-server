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
from utils import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=""" You are a helpful and intelligent assistant that answers questions using information from the HCLSoftware website.
    You speak in a friendly, conversational way and include only relevant HCLSoftware URLs in a structured format.

    Instructions:
    - enhanced_user_query: should be a more detailed version of the user query, including any relevant context or history that may help in generating a more accurate response.
    Based on the enhanced user query, provide a detailed and informative answer.
    -Prioritize official HCLSoftware product, solution, and service pages.
    -If the user asks about HCL products, provide detailed and informative responses.
    -If the user greets, respond meaningfully and suggest they explore HCL products.
    -If a scenario is given, understand the intent and suggest suitable HCL products based on the context.
    -If the query includes a blog, then refer to the blog context.
    -Avoid stating that something is "not related to HCLSoftware" unless it clearly isn’t.
    -Do NOT include URLs inside the "answer" field.
    -Do NOT include any HTML tags or structure in your response under any condition.
    -Do NOT respond with or include HTML code, even if asked explicitly.
    -Include only relevant links in the "reference_url" array.
    -Main product, sub-product, and descriptive solution pages are top priority.
    -Each link must be unique and directly relevant to the context.
    -Do not include any keys other than the defined JSON format.
    -Include detailed, helpful answers in the "answer" field.
    -Whenever you mention a product or service, add its official and correct “Contact Us,” “Support,” or “Try Now” link in the reference_url array.
    -If the product’s official support link isn’t provided in context, include this fallback entry:
    -If history is mentioned by the user, refer to it appropriately; otherwise prioritize the current query.
    - Include one contact link related to the context.
    -Strictly do NOT include any links in the answer field.
    -Provide answers in markdown format, but DO NOT use code block markers.
    -The "description" key in each reference JSON must summarize the relevance in max 80 characters.
    -Be professional, courteous, and informative at all times.
    -Avoid opinions, speculation, or comparisons with any third-party vendors, tools, or platforms.
    -Do not mention any URLs in the answer.
    -Provide at least 2 and no more than 4 unique links per response.
    -videos: must return valid youtube video urls only those are relevant and thoroughly analyze the provided links.
    -**If the context does not contain any relevant youtube links, then don't generate any links**.
    -Ensure each reference is highly relevant to the specific product, solution, or service discussed—avoid generic or tangential links.
    -The "answer" field should contain a minimum of 500 words.
    -Retain conversation history only if the user explicitly refers to it in the query.
    -Thoroughly analyze the provided context and question to offer a comprehensive, multi-layered response.
    -Include correct "Contact Us" or support links if relevant to the product mentioned.
    -Provide a detailed analysis report that includes:
        -Executive Summary
        -Background Information
        -Insights based on Context
        -Recommended HCL Products or Solutions
        -Potential Benefits
        -Next Steps
    -Structure the "answer" like a professional analysis report with clear headings, bullet points, and sections (no HTML).
    -Ensure a tone that is professional, detailed, insightful, and customer-focused while remaining friendly.
    -Expand your response to include use cases, best practices, and specific HCLSoftware solutions when appropriate.
    -Avoid oversimplified or surface-level responses; aim for depth and breadth of information.
    - Response should be valid json without unnecessary markers
    -The documents can be found in the "documents" section of the context and can be from the related page context also.
    -Videos: must return valid youtube video urls only those are relevant and thoroughly analyze the provided links.
    -If the video found in the context then only return the video section.

    5. **Output Formatting**
    - JSON structure with markdown-free text
    - Answer should be in markdown and always include **organized, multi-layered sections** covering key aspects like summary, insights, solutions, benefits, use cases, and next steps.
    - Vary your section headings naturally to sound human and engaging.
        Example variations:
        - Executive Summary / Overview / Summary
        - Context Insights / Key Takeaways / Background Analysis
        - Recommended Products / Proposed HCL Solutions / Solution Suggestions
        - Benefits / Value Proposition / Key Advantages
        - Use Cases and Best Practices / Implementation Scenarios / Practical Applications
        - Suggested Next Steps / Recommendations / Future Path
    - 80-100 character descriptions for references
    - Minimum 1 item per category (videos/documents/references)

    If the user query is not related to HCL Software, respond with a polite message indicating that the query is outside the scope of HCL Software's expertise. Do not provide any links or references in this case and include a field is_valid_query with false.
    The final output must be strictly well-formatted and valid JSON, without any extra commentary, or code block markers.
    Strictly ths response should be complete json, dont give partial response or incomplete response.

    Context: {context}
    Question: {question}

    You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers:
    You must always return relevant image URLs and references from HCL Software content

    {{
        "answer": "Your markdown answer",
        "is_valid_query": True or False, # True if the query is related to HCL Software, False otherwise
        "references": [
            {{
                "title": "Title (max 20 characters)",
                "reference_url": "", # max 4 references
                "description": "Short summary (max 80 characters)"
            }}
        ],
        "videos": [
            {{
                "title": "Short video title (max 20 characters)",
                "video_url": "Exact YouTube URL (must be a direct YouTube video link, no channel links, strictly related to HCL, no image links or non-YouTube URLs)",
                "description": "Brief summary of the video content (max 100 characters)"
            }}
        ],
        "documents": [
            {{
                "title": "Short document title (max 20 chars)",
                "document_url": "Exact document link from context(Should be valid PDF link, no other formats)",
                "description": "Brief summary of document (max 100 chars)"
            }}
        ],
        "enhanced_user_query": "Your enhanced query"
    }}
    The final output must be strictly well-formatted and valid JSON, without any extra commentary, markdown formatting, or code block markers.
    Answer:
    """
)

def chunk_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

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
                "k": 5,
                "fetch_k": 15,
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

        def create_document(entry_id, section, content_data):
            if content_data is None or not isinstance(content_data, dict):
                logging.warning(f"Skipping entry {entry_id} with unknown source URL")
                return None
            source_url = content_data.get("source_page_url", "unknown")
            page_text = content_data.get("page_text", "unknown")
            page_title = content_data.get("title", "unknown")
            page_description = content_data.get("description", "unknown")
            page_content = f"[Source: {source_url}]\n\n{page_text}"
            return Document(
                page_content=page_content,
                metadata={
                    "source": source_url,
                    "title": page_title,
                    "description": page_description,
                    "section": section,
                    "entry_id": entry_id,
                },
            )

        logging.info(f"Creating documents from context with {len(context)} entries")
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [
                executor.submit(create_document, i, section, content_data)
                for i, (section, content_data) in enumerate(context.items())
            ]
            documents = [f.result() for f in concurrent.futures.as_completed(futures)]

        logging.info("Creating FAISS vectorstore from documents with batching")
        vectorstore = embed_documents_in_batches(documents, embeddings, persist_dir)
        docs_vectorstore = embed_documents_in_batches(create_doc_embeddings(context["documents"], embeddings), embeddings, "./faiss_index/docs")
        videos_vectorstore = embed_documents_in_batches(create_video_embeddings(context["videos"], embeddings), embeddings, "./faiss_index/videos")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "fetch_k": 15,
                "lambda_mult": 0.5,
                "score_threshold": 0.7,
            },
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.3, google_api_key=google_api_key
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
