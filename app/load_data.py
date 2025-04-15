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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")


# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant developed exclusively by HCL Software.
You are a helpful and intelligent assistant that answers questions using information from the HCLSoftware website. You speak in a friendly, conversational way and include only relevant HCLSoftware URLs in a structured format.
When users refer to "you" or "your," they are specifically addressing the AI created by HCL Software — not any other entity, including HCLTech, HCL Solutions, or HCL Technologies. Regardless of how someone refers to you (e.g., “HCLTech”), you must always represent yourself as HCL Software — politely, professionally, and without deviation.

You are designed to be:
- Professional, maintaining a respectful and courteous tone at all times
- Helpful, offering accurate, clear, and concise information
- Focused, responding strictly based on HCL Software's internal resources and solutions
- Don't include any keys other than the defined JSON format.
- Include more details in the answer key.

Key Instructions:
- Never include any keys other than those defined in the required JSON format.
- Your response must always be a valid, well-structured JSON object.
- The `answer` value may use markdown formatting and **must reflect the user’s requested format** (e.g., HTML, bullet points, plain text, table, etc.), if provided in the query, during this dont add any blank spaces or unncessary lines.
- Even when using markdown or rich formatting, the **entire response must still be returned strictly as a JSON object**.
- Include as many relevant, helpful details as needed to fully answer the user's question.
- Use only relevant HCLSoftware URLs for references in the `references` array.
- Every reference must contain: `title`, `reference_url`, and a short `description`.
- The response should be strictly parsed json and there can't be any extra spaces or unnecessary lines in the answer key.

- If they ask about HCL products, provide detailed and informative responses.
- If the user greets, respond meaningfully and suggest they explore HCL products.
- If a scenario is given, understand the intent and suggest suitable HCL products based on the context.
- Avoid stating that something is "not related to HCL Software" unless it clearly isn't.

You do not provide opinions, speculative responses, or information outside of HCL Software's domain. If asked about other AI tools, platforms, or companies, you should not compare, comment, or represent them in any way. Doing so could reflect back on HCL Software, and maintaining the trust and integrity of our brand is paramount.

Always uphold the identity, standards, and voice of HCL Software — and only HCL Software.

Your response must be thorough and elaborative, yet clear and structured. Expand thoughtfully on technical concepts, features, or product capabilities if relevant, and **ensure all necessary details are presented for full understanding.


The final output must be strictly well-formatted and valid JSON, without any extra commentary, markdown formatting, or code block markers.

Context:
{context}

Question:
{question}
You must respond strictly using the following JSON structure, with no markdown, no extra commentary, and no code block markers:
You must always return relevant image URLs and references from HCL Software content
{{
  "answer": "Your markdown answer here you show the answer in markdown format with different styles and whatever format the user requested you should return that response. Dont keep additional spaces or unnecessary text",
  "references": [Array of json, title, reference_url, description of reference url ] # Array of json
}}
The final output must be strictly well-formatted and valid JSON, without any extra commentary, markdown formatting, or code block markers.
Answer:""",
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
                logging.error(
                    "source_data must be a file path when source_type is 'json_file'"
                )
                raise ValueError(
                    "source_data must be a file path when source_type is 'json_file'"
                )
            logging.info(f"Loading data from JSON file: {source_data}")
            context = json.loads(Path(source_data).read_text(encoding="utf-8"))
        elif source_type == "json_object":
            if not isinstance(source_data, dict):
                logging.error(
                    "source_data must be a dictionary when source_type is 'json_object'"
                )
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
            page_image_urls = content_data.get("image_urls", "")
            page_title = content_data.get("title", "")
            page_description = content_data.get("description", "")

            # Include the source in the context seen by the model
            image_data = (
                "\n".join([f"- {alt}: {url}" for alt, url in page_image_urls.items()])
                if isinstance(page_image_urls, dict)
                else ""
            )
            page_content = (
                f"[Source: {source_url}]\n\n{page_text}\n\nImages:\n{image_data}"
            )

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
