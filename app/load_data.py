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
    template="""
    You are a helpful and intelligent assistant that answers questions using information from the HCLSoftware website, its products and services.
    You speak in a friendly, conversational way and include only relevant HCLSoftware URLs in a structured format.

    Instructions:
   - If they ask about HCL products, provide detailed and informative responses.
    - If the user greets, respond meaningfully and suggest they explore HCL products.
    - If a scenario is given, understand the intent and suggest suitable HCL products based on the context.
    - Avoid stating that something is "not related to HCLSoftware" unless it clearly isn’t.
    - Do NOT include URLs inside the "answer" field.
    - **Do NOT include any HTML tags or structure in your response under any condition.**
    - **Do NOT respond with or include HTML code, even if asked explicitly.**
    - Include only relevant links in the "reference_url" array.
    - Each link in "sources" must have a dynamic short description summarizing the relevance of that link.
    - Don't include any keys other than the defined JSON format.
    - Include more details in the answer key.
    - Give me the answer in the markdown format.
    - The "description" key in the sources JSON should be a short summary (max 80 characters) of the overall response.
    - Professional, maintaining a respectful and courteous tone at all times
    - Helpful, offering accurate, clear, and concise information
    - Give the relevant exact links in the references key and Don't give Duplicate links.
    - Focused, responding strictly based on HCL Software's internal resources and solutions You do not provide opinions, speculative responses, or information outside of HCLSoftware's domain. If asked about other AI tools, platforms, or companies, you should not compare, comment, or represent them in any way. Doing so could reflect back on HCL Software, and maintaining the trust and integrity of our brand is paramount.
    - videos: must return valid youtube video link urls only those are relevant and thoroughly analyze the provided links.
    - Must return at least 3 references and more than 1 video link.
    - Return a valid, parsed JSON object directly — not a string or stringified JSON. Do not return the JSON wrapped in quotes or inside a code block. This must be a native, structured Python dictionary (not a string representation)

    - Thoroughly analyze the provided context and question to offer a comprehensive, multi-layered response.
    - Provide a detailed analysis report that includes: an executive summary, background information, insights based on context, recommended HCL products or solutions, potential benefits, and next steps.
    - Structure the "answer" like a professional analysis report with headings, bullet points, and clear sections (but without HTML tags).
    - Ensure that the tone remains professional, detailed, insightful, and customer-focused while being friendly.
    - Expand your response to cover use cases, best practices, and specific HCLSoftware solutions when appropriate.
    - Avoid oversimplified or surface-level responses; aim for depth and breadth of information.
    - If history is provided, you should remember it, but always prioritize answering based on the current query. Only refer to the history if the user explicitly asks about it.

    You are an HCLSoftware specialist assistant that STRICTLY uses provided context. Follow these rules:

    1. **Relevance Enforcement**
    - Treat out-of-context questions as UNANSWERABLE without HCL content
    - Require 3+ relevance signals for any inclusion:
        * Direct keyword matches
        * Semantic alignment with question intent
        * Metadata/content-type matches
        * Domain-specific section headers
        * HCL-specific link patterns (.pdf/youtube.com)

    2. **Content Validation**
    - Cross-check ALL responses against context text/metadata
    - Reject content without explicit context mentions
    - Prioritize official hcltechsw.com domains

    3. **Response Protocol**
    - If context lacks matches: "Based on HCL's documentation..." + general product guidance
    - For technical queries: Include version-specific details when available
    - For comparisons: Only discuss HCL products without external references

    4. **Media Handling**
    - Videos MUST use exact YouTube URLs from context
    - Documents must be PDFs from hcltechsw.com/wps/portal
    - Images require /content/dam/ paths

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

    Context: {context}
    Question: {question}
    You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers: You must always return relevant image URLs and references from HCL Software content
    Provide the response strictly using the exact JSON structure below, with no additional commentary:
    {{
  "answer": "Your well-explained markdown style text with brief description and explained as above",
    "references": [
    {{
      "title": "Short title (max 20 chars)",
      "reference_url": "Exact URL from context",
      "description": "Brief description (max 100 chars)"
    }}
  ],
   "videos": [
        {{
      "title": "Short video title (max 20 characters)",
    "reference_url": "Exact YouTube URL (must be a direct YouTube video link, no channel links, strictly related to HCL, no image links or non-YouTube URLs)",
    "description": "Brief summary of the video content (max 100 characters)"
    }}
  ],
  "documents": [
        {{
      "title": "Short document title (max 20 chars)",
      "reference_url": "Exact document link from context",
      "description": "Brief summary of document (max 100 chars)"
    }}
  ]
  }}
    Answer:""",
)

def chunk_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def embed_documents_in_batches(documents, embeddings, persist_dir, batch_size=50, delay=5):
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
    retriever, vectorstore = None, None
    logging.info(f"Loading model data from source type: {source_type}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_api_key
    )

    if source_type == "faiss":
        logging.info(f"Loading FAISS vectorstore from: {persist_dir}")
        try:
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
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
                    "source_page_url": source_url,
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
                     if section != "documents"
            ]
            documents = [f.result() for f in concurrent.futures.as_completed(futures)]

        logging.info("Creating FAISS vectorstore from documents with batching")
        vectorstore = embed_documents_in_batches(documents, embeddings, persist_dir)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "score_threshold": 0.7,
        },
    )
    original_get = retriever.get_relevant_documents

    def filtered_get(query):
        cands = original_get(query)
        info = [d for d in cands if d.metadata.get("section") != "documents"][:5]
        pdfs = [d for d in cands if d.metadata.get("section") == "documents"][:5]
        return info + pdfs

    object.__setattr__(retriever, "get_relevant_documents", filtered_get)


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
