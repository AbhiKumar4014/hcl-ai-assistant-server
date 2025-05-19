from concurrent.futures import ThreadPoolExecutor
import re
import base64
import logging

from flask import Flask, json, request, jsonify
from flask_cors import CORS
from langchain_core.messages import AIMessage
import json

from load_page_urls import *
from load_data import *
from utils import *
from document_video_processor import append_valid_documents_and_videos

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
CORS(app)  # Enable CORS

# For backup
# qa_chain, docs_vectorstore, videos_vectorstore, embeddings = load_model_data(source_type="json_file", source_data="hcl_sites_data.json")

qa_chain, vectorstore, docs_vectorstore, videos_vectorstore, embeddings = (
    load_model_data()
)


@app.route("/query/health-check", methods=["GET"])
def healthcheck():
    health_status = {
        "qa_chain_loaded": qa_chain is not None,
        "vectorstore_loaded": vectorstore is not None,
        "docs_vectorstore_loaded": docs_vectorstore is not None,
        "videos_vectorstore_loaded": videos_vectorstore is not None,
        "embeddings_loaded": embeddings is not None,
    }
    # Determine overall model readiness
    all_loaded = all(list(health_status.values()))

    message = (
        "Server is healthy and model loaded successfully"
        if all_loaded
        else "Server healthy, but some components are not loaded"
    )
    return jsonify({"message": message, "status": health_status}), (
        200 if all_loaded else 206
    )


@app.route("/query/load", methods=["POST"])
def load():
    global qa_chain
    data = request.get_json()
    if not data or "sitemap_urls" not in data:
        logging.warning("Missing 'sitemap_url' in request body")
        return jsonify({"error": "Missing 'sitemap_url' in request body"}), 400

    sitemap_data = data["sitemap_urls"]
    if isinstance(sitemap_data, str):
        sitemap_urls = [sitemap_data]
    elif isinstance(sitemap_data, list) and all(
        isinstance(url, str) for url in sitemap_data
    ):
        sitemap_urls = sitemap_data
    else:
        return (
            jsonify({"error": "'sitemap_url' must be a string or list of strings"}),
            400,
        )

    logging.info("Load requested")

    hcl_urls = load_hcl_sitemap(sitemap_urls)  # Load HCL sitemap URLs
    if not hcl_urls:
        logging.error("Failed to load HCL sitemap URLs")
        return jsonify(
            {"error": "Failed to load HCL sitemap URLs or no URLs found"}
        ), 400

    context = extract_all_text_parallel(hcl_urls)  # Extract text from HCL URLs

    context["documents"] = extract_all_doc_urls()  # Extract document URLs

    context["videos"] = extract_channel_videos()["videos"]  # Extract video URLs

    # Save the context to a JSON file for backup
    with open("hcl_sites_data.json", "w") as file:
        json.dump(context, file)

    # Load the model with the context
    qa_chain = load_model_data(context, source_type="json_object")

    if qa_chain is not None:
        logging.info("Model loaded successfully")
        return jsonify({"message": "Model loaded successfully"})
    else:
        logging.error("Failed to load model")
        return jsonify({"error": "Failed to load model"}), 500


@app.route("/query/ask", methods=["POST"])
def ask():
    global qa_chain, vectorstore, docs_vectorstore, videos_vectorstore, embeddings

    # Get data from JSON body
    data = request.get_json()
    query = data.get("query") if data else None
    history = data.get("history", "") if data else ""

    # Validate query
    if not query:
        logging.warning("Missing 'query' parameter")
        return jsonify({"error": "Missing 'query' parameter"}), 400

    # Check QA chain
    if qa_chain is None:
        logging.error("QA chain is not initialized")
        return (
            jsonify(
                {
                    "error": "Oops! Something went wrong. The QA chain is not initialized. Please try again later or call /load first."
                }
            ),
            500,
        )

    # Storing the query with date in file
    log_query_in_background(query)
        
    # Combine history with the current query if provided
    full_query = (
        f"{query}\n\n[Previous three conversation:\n{history}]" if history else query
    )
    try:
        # Step 1 & 2: Run both embeddings and vectorstore searches in parallel
        if not history:
            # No history: single search with broader scope
            unique_docs = vectorstore.max_marginal_relevance_search_by_vector(
                embeddings.embed_query(query), k=20, fetch_k=20
            )
        else:
            # With history: run both searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_query_only = executor.submit(
                    lambda: vectorstore.max_marginal_relevance_search_by_vector(
                        embeddings.embed_query(query), k=8, fetch_k=20
                    )
                )

                combined_query = f"{query} {history}"
                future_query_plus_history = executor.submit(
                    lambda: vectorstore.max_marginal_relevance_search_by_vector(
                        embeddings.embed_query(combined_query), k=10, fetch_k=20
                    )
                )

                docs_query_only = future_query_only.result()
                docs_query_plus_history = future_query_plus_history.result()

            # Merge and deduplicate by page content
            combined_docs = docs_query_only + docs_query_plus_history
            seen_hashes = set()
            unique_docs = []
            for doc in combined_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)

        # Step 4: Pass question and context into qa_chain
        result = qa_chain.invoke({"question": full_query, "context": unique_docs})
        
        # Normalize response
        if isinstance(result, AIMessage):
            raw_response = result.content.strip()

        elif isinstance(result, dict) and "result" in result:
            inner = result["result"]
            if isinstance(inner, AIMessage):
                raw_response = inner.content.strip()
            elif isinstance(inner, str):
                raw_response = inner.strip()
            else:
                raise TypeError(f"Unsupported type for result['result']: {type(inner)}")

        elif isinstance(result, str):
            raw_response = result.strip()

        else:
            raise TypeError(f"Unexpected result type: {type(result)}")

            logging.error(f"Unexpected response format from model: {result}")
            return (
                jsonify(
                    {
                        "error": "Unexpected response format from model",
                        "details": str(result),
                    }
                ),
                500,
            )

        # Remove only starting ```json and ending ``` (if present)
        if raw_response.startswith("```json"):
            raw_response = raw_response[len("```json") :].lstrip()
        if raw_response.endswith("```"):
            raw_response = raw_response[: -len("```")].rstrip()

        # Parse the cleaned JSON
        parsed = json.loads(raw_response)
        query_embeddings = embeddings.embed_query(parsed["enhanced_user_query"])
        if parsed.get("is_valid_query"):
            parsed = append_valid_documents_and_videos(
                parsed, query_embeddings, query, docs_vectorstore, videos_vectorstore
            )
        del parsed["is_valid_query"]
        return jsonify(parsed)

    except json.JSONDecodeError as json_err:
        logging.warning(
            f"Primary JSON decode failed: {json_err}. \n Attempting fallback extraction"
        )

        answer = ""
        references = []
        videos = []
        documents = []
        enhanced_user_query = ""
        is_valid_query = False
        final_response = {}

        try:
            is_valid_query_match = re.search(
                r'"is_valid_query"\s*:\s*(True|False)', raw_response, re.DOTALL
            )
            if is_valid_query_match:
                is_valid_query = is_valid_query_match.group(1).strip().lower() == "true"
            enhanced_user_query_match = re.search(
                r'"enhanced_user_query"\s*:\s*"(.*?)"(,|\n|\r)', raw_response, re.DOTALL
            )

            if enhanced_user_query_match:
                raw_string = enhanced_user_query_match.group(1)
                # Properly unescape using JSON loader
                enhanced_user_query = json.loads(f'"{raw_string}"')

                # Extract the "answer" field using regex
            answer_match = re.search(
                r'"answer"\s*:\s*"(.*?)"(,|\n|\r)', raw_response, re.DOTALL
            )
            if answer_match:
                try:
                    # Attempt to parse the answer as JSON
                    answer = json.loads(answer_match.group(1))
                except json.JSONDecodeError:
                    # If parsing fails, treat it as a string
                    answer = (
                        answer_match.group(1)
                        .replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace("\\r", "\r")
                    )

            # Extract "references" array
            references_match = re.search(
                r'"references"\s*:\s*(\[[\s\S]*?\])', raw_response
            )
            if references_match:
                try:
                    references = json.loads(references_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Could not parse `references` field")

            # Extract "videos" array
            videos_match = re.search(r'"videos"\s*:\s*(\[[\s\S]*?\])', raw_response)
            if videos_match:
                try:
                    videos = json.loads(videos_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Could not parse `videos` field")

            # Extract "documents" array
            documents_match = re.search(
                r'"documents"\s*:\s*(\[[\s\S]*?\])', raw_response
            )
            if documents_match:
                try:
                    documents = json.loads(documents_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Could not parse `documents` field")

            final_response = {
                "answer": answer,
                "references": references,
                "videos": videos,
                "documents": documents,
                "enhanced_user_query": enhanced_user_query,
            }
            if is_valid_query:
                query_embeddings = embeddings.embed_query(enhanced_user_query)
                final_response = append_valid_documents_and_videos(
                    final_response,
                    query_embeddings,
                    query,
                    docs_vectorstore,
                    videos_vectorstore,
                )

            return jsonify(final_response), 200

        except Exception as extract_err:
            logging.error(f"Manual fallback also failed: {extract_err}")
            return (
                jsonify(
                    {
                        "error": "JSON parsing failed and fallback also failed.",
                        "details": str(json_err),
                        "raw": raw_response,
                    }
                ),
                206,
            )

    except Exception as e:
        logging.exception("Unexpected error during ask")
        return (
            jsonify(
                {
                    "error": "Oops! Something went wrong while processing your request.",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/query/call-chatbot-api", methods=["POST"])
def call_chatbot_api():
    data = request.get_json()
    additional_parameter = data.get("additional_parameter")

    if not additional_parameter:
        return jsonify({"error": "Missing 'additional_parameter' in request body"}), 400

    USERNAME = "user@hcl-software.com"
    DISPLAYNAME = "user@hcl-software.com"
    ACCESS_TOKEN = "xxxxxxxxxxxxxsssss"

    API_USERNAME = "80d590a9-dd10-48b5-8437-0ed66ffefeea"
    API_PASSWORD = "d64889b2-ad10-46ea-9230-5ede72da76ca"

    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode()

    payload = {
        "username": USERNAME,
        "displayname": DISPLAYNAME,
        "access_token": ACCESS_TOKEN,
        "additional_parameter": additional_parameter,
    }

    headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            "https://hclswaichatbot.eu.bigfixaex.ai/external/api/token",
            headers=headers,
            json=payload,
        )
        return jsonify({"status_code": response.status_code, "response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
