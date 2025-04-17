import os
from flask import Flask, json, request, jsonify
from flask_cors import CORS
import logging

from load_page_urls import *
from load_data import *
import json
from utils import *
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

qa_chain = None
# qa_chain = load_model_data(source_type="json_file", source_data="hcl_sites_data.json")
qa_chain = load_model_data()

@app.route("/health-check", methods=["GET"])
def healthcheck():
    logging.info("Health check requested")
    if qa_chain is not None:
        return jsonify({"message": "Server is healthy and model loaded successfully"})
    return jsonify({"message": "Server is healthy!"})

@app.route("/load", methods=["GET"])
def load():
    global qa_chain
    logging.info("Load requested")
    # Extract the urls from the sitemap
    hcl_urls = load_hcl_sitemap()

    # Scrap the web page contents parallely
    context = extract_all_text_parallel(hcl_urls)

    # Extract the document URLs from the JSON file
    doc_urls = extract_all_doc_urls()
    for endpoint, urls in doc_urls.items():
        if endpoint in context:
            context[endpoint][endpoint] = urls

    # Backup the context to a JSON file
    with open("hcl_sites_data.json", "w") as file:
        json.dump(context, file)

    # Load the model data
    qa_chain = load_model_data(context, source_type="json_object")
    if qa_chain is not None:
        print(qa_chain)
        logging.info("Model loaded successfully")
        return jsonify({"message": "Model loaded successfully"})
    else:
        logging.error("Failed to load model")
        return jsonify({"error": "Failed to load model"}), 500

@app.route('/ask', methods=['POST'])
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
    full_query = f"(Attached Last three converastion{history})\n{query}" if history else query

    try:
        result = qa_chain.invoke({"query": full_query})
        # logging.info(f"test type {type(result["result"])}")

        # Normalize response
        if isinstance(result, dict) and "result" in result:
            raw_response = result["result"].strip()
        elif isinstance(result, str):
            raw_response = result.strip()
        else:
            logging.error(f"Unexpected response format from model: {result}.")
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
        # logging.info("Successfully processed query and returning result")
        return jsonify(parsed)

    except json.JSONDecodeError as json_err:
            logging.warning(f"Primary JSON decode failed: {json_err}. \n Attempting fallback extraction")

            answer = ""
            references = []

            try:
                # Extract the "answer" field using regex
                answer_match = re.search(r'"answer"\s*:\s*"(.*?)"(,|\n|\r)', raw_response, re.DOTALL)
                answer = answer_match
                if answer_match:
                    answer = answer_match.group(1).encode().decode('unicode_escape')

                references_match = re.search(r'"references"\s*:\s*({.*?})\s*(,|\n|$)', raw_response, re.DOTALL)
                logging.info(references_match)
                if references_match:
                    references_json = references_match.group(1)
                    try:
                        references = json.loads(references_json)
                    except json.JSONDecodeError:
                        logging.warning("Could not parse `references` field")

                return jsonify({
                    "error": "Partial JSON parsing fallback triggered.",
                    "details": str(json_err),
                    "answer": answer,
                    "reference": references,
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

if __name__ == "__main__":
    logging.info("Server is running on port: 3000")
    app.run(host="0.0.0.0", port=3000, debug=True)
