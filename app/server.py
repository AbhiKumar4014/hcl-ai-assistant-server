import os
from flask import Flask, json, request, jsonify
from flask_cors import CORS
import logging

from load_page_urls import *
from load_data import *
from utils import *
import re
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

# qa_chain = None
# qa_chain = load_model_data(source_type="json_file", source_data="hcl_sites_data.json")
qa_chain = load_model_data()

@app.route("/query/health-check", methods=["GET"])
def healthcheck():
    logging.info("Health check requested")
    if qa_chain is not None:
        return jsonify({"message": "Server is healthy and model loaded successfully"})
    return jsonify({"message": "Server is healthy!"})

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
    elif isinstance(sitemap_data, list) and all(isinstance(url, str) for url in sitemap_data):
        sitemap_urls = sitemap_data
    else:
        return jsonify({"error": "'sitemap_url' must be a string or list of strings"}), 400

    logging.info("Load requested")
    hcl_urls = load_hcl_sitemap(sitemap_urls)
    context = extract_all_text_parallel(hcl_urls)
    doc_urls = extract_all_doc_urls()

    # Initialize the documents list
    context["documents"] = []

    # Populate with name, variant, url, and source
    for endpoint, docs_list in doc_urls.items():
        for meta in docs_list:
            raw = meta.get("title") or meta.get("name")
            url = meta.get("url")
            if not raw or not url:
                continue

            # split camelCase / PascalCase into words:
            split = re.sub(r"(?<!^)(?=[A-Z])", " ", raw).strip()
            # also make sure the version with spaces inserted if missing is covered:
            spaced = re.sub(r"(\d+)", r" \1", split).strip()

            context["documents"].append(
                {
                    "name": raw,  # e.g. "AppScan360" or "AppScan 360"
                    "variant": spaced,  # e.g. "App Scan 360"
                    "url": url,
                    "source": endpoint,
                }
            )

    with open("hcl_sites_data.json", "w") as file:
        json.dump(context, file)

    qa_chain = load_model_data(context, source_type="json_object")
    if qa_chain is not None:
        print(qa_chain)
        logging.info("Model loaded successfully")
        return jsonify({"message": "Model loaded successfully"})
    else:
        logging.error("Failed to load model")
        return jsonify({"error": "Failed to load model"}), 500

@app.route('/query/ask', methods=['POST'])
def ask():
    global qa_chain

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
        return jsonify({
            "error": "Oops! Something went wrong. The QA chain is not initialized. Please try again later or call /load first."
        }), 500

    # Combine history with the current query if provided
    full_query = f"(Attached Last three conversation{history})\n{query}" if history else query
    
    def clean_json_input(text):
        try:
            cleaned = text.replace('\u2003', ' ').replace('\n', '')
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON format: " + str(e))

    try:
        result = qa_chain.invoke(full_query)
        source_docs = result.get("source_documents", [])

        with open("source_documents.json", "w") as file:
            logging.info("Saving source documents to source_documents.json")
            json.dump([str(doc) for doc in source_docs], file, indent=4)

        # Normalize response
        if isinstance(result, dict) and "result" in result:
            raw_response = result["result"].strip()
        elif isinstance(result, str):
            raw_response = result.strip()
        else:
            logging.error(f"Unexpected response format from model: {result}")
            return jsonify({
                "error": "Unexpected response format from model",
                "details": str(result)
            }), 500

        # Remove only starting ```json and ending ``` (if present)
        if raw_response.startswith("```json"):
            raw_response = raw_response[len("```json"):].lstrip()
        if raw_response.endswith("```"):
            raw_response = raw_response[: -len("```")].rstrip()

        # Parse the cleaned JSON
        parsed = json.loads(raw_response)
        if "videos" in parsed:
            # logging.info(f"""{parsed["videos"]} found in parsed response""")
            valid_videos = []
            for video in parsed["videos"]:
                url = video.get("video_url", "")
                is_valid, thumbnail_url = is_youtube_video_valid(url)
                if is_valid:
                    video["thumbnail_url"] = thumbnail_url
                    valid_videos.append(video)
            parsed["videos"] = valid_videos  

        if "documents" in parsed:
            parsed["documents"] = parsed["documents"][:3]

        source_page_urls = [doc.metadata.get("source_page_url", "") for doc in source_docs]
        if source_page_urls:
            images = get_images(source_page_urls, 6)
            parsed["images"] = images


        return jsonify(parsed)

    except json.JSONDecodeError as json_err:
        logging.warning(f"Primary JSON decode failed: {json_err}. \n Attempting fallback extraction")
    
        answer = ""
        references = []
        videos = []
        documents = []
    
        try:
            # Extract the "answer" field using regex
            answer_match = re.search(r'"answer"\s*:\s*"(.*?)"(,|\n|\r)', raw_response, re.DOTALL)
            if answer_match:
                answer = json.loads(f'"{answer_match.group(1)}"')  # Safe unescaping
    
            # Extract "references" array
            references_match = re.search(r'"references"\s*:\s*(\[[\s\S]*?\])', raw_response)
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
            documents_match = re.search(r'"documents"\s*:\s*(\[[\s\S]*?\])', raw_response)
            if documents_match:
                try:
                    documents = json.loads(documents_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Could not parse `documents` field")
    
            return jsonify({
                "error": "Partial JSON parsing fallback triggered.",
                "details": str(json_err),
                "answer": answer,
                "references": references,
                "videos": videos,
                "documents": documents,
                "raw": raw_response
            }), 200
    
        except Exception as extract_err:
            logging.error(f"Manual fallback also failed: {extract_err}")
            return jsonify({
                "error": "JSON parsing failed and fallback also failed.",
                "details": str(json_err),
                "raw": raw_response
            }), 206

    except Exception as e:
        logging.exception("Unexpected error during ask")
        return jsonify({
            "error": "Oops! Something went wrong while processing your request.",
            "details": str(e)
        }), 500

@app.route('/query/call-chatbot-api', methods=['POST'])
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
    "additional_parameter": additional_parameter
    }

    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://hclswaichatbot.eu.bigfixaex.ai/external/api/token", headers=headers, json=payload)
        return jsonify({
            "status_code": response.status_code,
            "response": response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
