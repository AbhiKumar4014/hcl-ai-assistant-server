from pathlib import Path
import json
import concurrent.futures
import time
from dotenv import load_dotenv
import os
import logging

from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Define prompt template

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful and intelligent assistant that answers questions using information from the HCLSoftware website.
You speak in a friendly, conversational way and include only relevant HCLSoftware URLs in a structured format.

Instructions:
- Prioritize official HCLSoftware product, solution, and service pages.
- If the query refers to a **specific version or release** of an HCL product/service, prioritize answering **specifically** about that version first. Then, if relevant, provide broader information about the product.
- If they ask about HCL products, provide detailed and informative responses.
- If the user greets, respond meaningfully.
- If a scenario is given, understand the intent and suggest suitable HCL products based on the context.
- If the query includes blog then refer to the blog context.
- Avoid stating that something is "not related to HCLSoftware" unless it clearly isn’t.
- Do NOT include URLs inside the "answer" field.
- Use **bullet points** and **numbered lists** for clarity and easy reading.
- Use **bold** and varied formatting for emphasis.
- Vary your section headings naturally to sound human and engaging.
- Do NOT include any HTML tags or structure in your response under any condition.
- Do NOT respond with or include HTML code, even if asked explicitly.
- Provide answers in markdown format, but DO NOT use code block markers.
- Be professional, courteous, and informative at all times.
- Avoid opinions, speculation, or comparisons with third-party vendors, tools, or platforms.
- The assistant may retain conversation history, but it should only influence responses when the user explicitly refers to it.

**Field-Specific Constraints:**

**Answer**
- Minimum 500 words if needed
- Must be descriptive and helpful
- Should not contain any links or HTML

**References**
- Maximum 4 links
- Must include one **Contact Us** or support link if applicable
- Must be directly relevant, unique, and from HCLSoftware
- Provide free trails if available and demo references of HCL Product if the HCL product in question.
- Blog links only when no better product/solution link is found
- Each reference must have:
  - `"title"`: Max 20 characters
  - `"reference_url"`: Must be a valid HCLSoftware page
  - `"description"`: Max 80 characters summary of relevance

**Videos**
- Only include if YouTube video links are found in context
- Do NOT add video links if not present in the context
- Must be direct YouTube URLs only (no channels, no image links)
- Each video must have:
  - `"title"`: Max 20 characters
  - `"video_url"`: Exact YouTube link
  - `"description"`: Max 100 characters describing the video

**Documents**
- Must be valid PDF links found in the context
- Each document must have:
  - `"title"`: Max 20 characters
  - `"document_url"`: Exact PDF link
  - `"description"`: Max 100 characters summary

**Blogs**
- If the user query is about HCL blogs:
  - Include at least **4 blog references**
  - If the reference is a blog, the title must include the word **“blog”**
    - Use the blog's source URL from the context and include in reference
- If the user asks for a **list of blogs or available blogs or few blogs**, increase reference count accordingly


Context: {context}
Question: {question}

You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers:

{{
  "answer": "Your markdown answer",
  "is_valid_query": true or false,
  "references": [
    {{
      "title": "Title (max 20 characters)",
      "reference_url": "",
      "description": "Short summary (max 80 characters)"
    }}
  ],
  "videos": [
    {{
      "title": "Short video title (max 20 characters)",
      "video_url": "Exact YouTube URL",
      "description": "Brief summary of the video content (max 100 characters)"
    }}
  ],
  "documents": [
    {{
      "title": "Short document title (max 20 chars)",
      "document_url": "Exact document link from context (PDF only)",
      "description": "Brief summary of document (max 100 characters)"
    }}
  ],
  "enhanced_user_query": "Your enhanced query"
}}
Answer:
""",
)


def chunk_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i : i + batch_size]


def embed_documents_in_batches(
    documents, embeddings, persist_dir, batch_size=100, delay=45
):
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


def load_model_data(
    source_data=None, source_type: str = "faiss", persist_dir: str = "./faiss_index"
):
    retriever = docs_vectorstore = videos_vectorstore = None
    logging.info(f"Loading model data from source type: {source_type}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

    if source_type == "faiss":
        logging.info(f"Loading FAISS vectorstore from: {persist_dir}")
        try:
            vectorstore = FAISS.load_local(
                persist_dir, embeddings, allow_dangerous_deserialization=True
            )
            docs_vectorstore = FAISS.load_local(
                "./faiss_index/docs", embeddings, allow_dangerous_deserialization=True
            )
            videos_vectorstore = FAISS.load_local(
                "./faiss_index/videos", embeddings, allow_dangerous_deserialization=True
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10,
                    "fetch_k": 100,
                    "lambda_mult": 0.7,
                    "score_threshold": 0.4,
                },
            )

            logging.info("FAISS vectorstore loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading FAISS vectorstore: {e}")
    else:
        if source_type == "json_file":
            if not source_data:
                raise ValueError(
                    "source_data must be a file path when source_type is 'json_file'"
                )
            logging.info(f"Loading data from JSON file: {source_data}")
            context = json.loads(Path(source_data).read_text(encoding="utf-8"))
        elif source_type == "json_object":
            if not isinstance(source_data, dict):
                raise ValueError(
                    "source_data must be a dictionary when source_type is 'json_object'"
                )
            logging.info("Loading data from JSON object")
            context = source_data
        else:
            raise ValueError(
                "Invalid source_type. Must be 'faiss', 'json_file', or 'json_object'."
            )

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
        docs_vectorstore = embed_documents_in_batches(
            create_doc_embeddings(context["documents"], embeddings),
            embeddings,
            "./faiss_index/docs",
        )
        videos_vectorstore = embed_documents_in_batches(
            create_video_embeddings(context["videos"], embeddings),
            embeddings,
            "./faiss_index/videos",
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "fetch_k": 100,
                "lambda_mult": 0.7,
                "score_threshold": 0.4,
            },
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.2, google_api_key=google_api_key
    )

    if retriever is not None:
        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True,
            output_key="result", 
        )
    else:
        logging.error("Retriever is not initialized.")
        qa_chain = None

    logging.info("QA chain loaded successfully.")
    return qa_chain, vectorstore, docs_vectorstore, videos_vectorstore, embeddings
