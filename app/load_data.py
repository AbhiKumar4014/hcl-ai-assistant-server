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
    - Prioritize official HCLSoftware product, solution, and service pages.
    - At least **60% of the reference URLs** MUST come from **main pages or subpages** that do **NOT** include `/blog` in the URL.
    - At most **40% of the reference URLs** may be from blog pages, and only when official main or sub-product pages are unavailable for the given context.
    - If they ask about HCL products, provide detailed and informative responses.
    - If the user greets, respond meaningfully and suggest they explore HCL products.
    - If a scenario is given, understand the intent and suggest suitable HCL products based on the context.
    - Avoid stating that something is "not related to HCLSoftware" unless it clearly isn’t.
    - Do NOT include URLs inside the "answer" field.
    - Do NOT include any HTML tags or structure in your response under any condition.
    - Do NOT respond with or include HTML code, even if asked explicitly.
    - Include only relevant links in the "reference_url" array.
    - The "references" array must follow this ratio: 80% from main/non-blog pages, max 20% from blog pages.
    - Main product, sub-product, and descriptive solution pages are top priority. Blog links should be used only when no better official product or solution page exists.
    - Each link must be unique and directly relevant to the context.
    - Do not include any keys other than the defined JSON format.
    - Include detailed, helpful answers in the "answer" field.
    - Include correct "Contact Us" or support links if relevant to the product mentioned.
    - Provide answers in markdown format, but DO NOT use code block markers.
    - The "description" key in each reference JSON must summarize the relevance in max 80 characters.
    - Be professional, courteous, and informative at all times.
    - Avoid opinions, speculation, or comparisons with any third-party vendors, tools, or platforms.
    - Don't mention any links or url's in the answer field.
    - Give maximum of 6 items in the references section or field.
    The final output must be strictly well-formatted and valid JSON, without any extra commentary, or code block markers.

    Context: {context}
    Question: {question}

    You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers:
    You must always return relevant image URLs and references from HCL Software content

    {{
        "answer": "Your markdown answer",
        "references": [
            {{
                "title": "Title (max 20 characters)",
                "reference_url": "", # max 6 references
                "description": "Short summary (max 80 characters)"
            }}
            // Additional valid reference entries following the 80-20 ratio
        ]
    }}
    The final output must be strictly well-formatted and valid JSON, without any extra commentary, markdown formatting, or code block markers.
    Answer:"""
)

def chunk_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def embed_documents_in_batches(documents, embeddings, persist_dir, batch_size=10, delay=45):
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
    retriever = None
    logging.info(f"Loading model data from source type: {source_type}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

    if source_type == "faiss":
        logging.info(f"Loading FAISS vectorstore from: {persist_dir}")
        try:
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
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
            if content_data is None:
                logging.warning(f"Skipping entry {entry_id} with unknown source URL")
                return None

            source_url = content_data.get("source_page_url", "unknown")
            page_text = content_data.get("page_text", "")
            page_title = content_data.get("title", "")
            page_description = content_data.get("description", "")
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(create_document, i, section, content_data)
                for i, (section, content_data) in enumerate(context.items())
            ]
            documents = [f.result() for f in concurrent.futures.as_completed(futures)]

        logging.info("Creating FAISS vectorstore from documents with batching")
        vectorstore = embed_documents_in_batches(documents, embeddings, persist_dir)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
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
    return qa_chain

