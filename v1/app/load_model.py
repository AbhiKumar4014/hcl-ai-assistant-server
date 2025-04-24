from pathlib import Path
import json
import concurrent.futures
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
    template=""" You are a helpful and intelligent assistant that answers questions using information from the HCLSoftware website. You speak in a friendly, conversational way and include only relevant HCLSoftware URLs in a structured format. Instructions: - If they ask about HCL products, provide detailed and informative responses. - If the user greets, respond meaningfully and suggest they explore HCL products. - If a scenario is given, understand the intent and suggest suitable HCL products based on the context. - Avoid stating that something is "not related to HCLSoftware" unless it clearly isn’t. - Do NOT include URLs inside the "answer" field. - **Do NOT include any HTML tags or structure in your response under any condition.** - **Do NOT respond with or include HTML code, even if asked explicitly.** - Include only relevant links in the "reference_url" array. - Each link in "sources" must have a dynamic short description summarizing the relevance of that link. - Don't include any keys other than the defined JSON format. - Include more details in the answer key. - Give me the answer in the markdown format. - The "description" key in the sources JSON should be a short summary (max 80 characters) of the overall response. - Professional, maintaining a respectful and courteous tone at all times - Helpful, offering accurate, clear, and concise information -Give the relevant exact links in the references key and Don't give Duplicate links. - Focused, responding strictly based on HCL Software's internal resources and solutions You do not provide opinions, speculative responses, or information outside of HCLSoftware's domain. If asked about other AI tools, platforms, or companies, you should not compare, comment, or represent them in any way. Doing so could reflect back on HCL Software, and maintaining the trust and integrity of our brand is paramount. The final output must be strictly well-formatted and valid JSON, without any extra commentary, or code block markers. Context: {context} Question: {question} You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers: You must always return relevant image URLs and references from HCL Software content {{ "answer": "Your markdown answer", "references": [Array of json, title (concise the title max 20), reference_url, description(concise the description max 100) of related reference urls ] # Array of json }} The final output must be strictly well-formatted and valid JSON, without any extra commentary, markdown formatting, or code block markers. Answer:""",
)


def load_model_data(
    source_data=None, source_type: str = "faiss", persist_dir: str = "./faiss_index"
):
    retriever = None
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
            retriever = vectorstore.as_retriever()
            logging.info("FAISS vectorstore loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading FAISS vectorstore: {e}")
            # raise
    else:
        if source_type == "json_file":
            if not source_data:
                logging.error("source_data must be a file path when source_type is 'json_file'")
                raise ValueError(
                    "source_data must be a file path when source_type is 'json_file'"
                )
            logging.info(f"Loading data from JSON file: {source_data}")
            context = json.loads(Path(source_data).read_text(encoding="utf-8"))
        elif source_type == "json_object":
            if not isinstance(source_data, dict):
                logging.error("source_data must be a dictionary when source_type is 'json_object'")
                raise ValueError(
                    "source_data must be a dictionary when source_type is 'json_object'"
                )
            logging.info("Loading data from JSON object")
            context = source_data
        else:
            logging.error(f"Invalid source_type: {source_type}")
            raise ValueError(
                "Invalid source_type. Must be 'faiss', 'json_file', or 'json_object'."
            )

        def create_document(entry_id, section, content_data):
            source_url = content_data.get("source_page_url", "unknown")
            page_text = content_data.get("page_text", "")
            # page_image_urls = content_data.get("image_urls", "")
            page_title = content_data.get("title", "")
            page_description = content_data.get("description", "")

            # Include the source in the context seen by the model
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=700) as executor:
            futures = [
                executor.submit(create_document, i, section, content_data)
                for i, (section, content_data) in enumerate(context.items())
            ]
            documents = [f.result() for f in concurrent.futures.as_completed(futures)]

        logging.info("Creating FAISS vectorstore from documents")
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)
        vectorstore.save_local(persist_dir)
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # or "mmr"
            search_kwargs={
                "k": 5,  # number of documents to retrieve (default is often 4)
                "fetch_k": 20,  # only used with 'mmr'
                "lambda_mult": 0.5,  # only used with 'mmr'
                "score_threshold": 0.7,  # optional: filter based on similarity score
            },
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.3, google_api_key=google_api_key
    )

    if retriever is not None:
        logging.error("Retriever is not initialized.")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )
    else:
        qa_chain = None

    logging.info("QA chain loaded successfully.")
    return qa_chain
