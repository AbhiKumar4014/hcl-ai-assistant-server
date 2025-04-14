# HCL Software AI Assistant

## Description

This application provides an AI assistant developed exclusively by HCL Software, utilizing the Gemini model to answer questions based on HCL Software's internal resources and solutions.

## Prerequisites

- Python 3.9+
- pip
- Google Cloud Project with the Gemini API enabled
- A `.env` file containing your Google API key

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your `.env` file:**

    Create a `.env` file in the root directory of the project with the following content:

    ```
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
    ```

    Replace `YOUR_GOOGLE_API_KEY` with your actual Google API key.

## Usage

1.  **Run the Flask application:**

    ```bash
    python app/server.py
    ```

    This will start the Flask development server.

2.  **Access the API endpoints:**

    -   `/health-check`: Check the server's health.
    -   `/load`: Load the model data.
    -   `/ask`: Ask a question to the AI assistant.

## Production Deployment with Gunicorn

1.  **Install Gunicorn:**

    ```bash
    pip install gunicorn
    ```

2.  **Run the application with Gunicorn:**

    ```bash
    gunicorn --bind 0.0.0.0:3000 app.server:app
    ```

    This will start the Gunicorn server, binding it to port 3000.

## API Endpoints

### Health Check

-   **Endpoint:** `/health-check`
-   **Method:** GET
-   **Description:** Checks if the server is running and the model is loaded.
-   **Response:**

    ```json
    {
    "message": "Server is healthy and model loaded successfully"
    }
    ```

### Load Data

-   **Endpoint:** `/load`
-   **Method:** GET
-   **Description:** Loads the model data from the HCL Software sitemap.
-   **Response:**

    ```json
    {
    "message": "Model loaded successfully"
    }
    ```

### Ask Question

-   **Endpoint:** `/ask`
-   **Method:** GET or POST
-   **Description:** Asks a question to the AI assistant and receives a JSON response.
-   **Parameters:**
    -   `query` (required): The question to ask.
    -   `history` (optional): The conversation history.
-   **Request (GET):**

    ```
    /ask?query=What is HCL Software?&history=Previous conversation
    ```

-   **Request (POST):**

    ```json
    {
    "query": "What is HCL Software?",
    "history": "Previous conversation"
    }
    ```

-   **Response:**

    ```json
    {
    "answer": "HCL Software is...",
    "references": [
        {
            "Reference URL Title": "Reference URL Link"
        }
    ],
    "image_urls": [
        {
            "Image URL Title": "Image URL Link"
        }
    ]
    }
    ```

## Logging

The application uses the `logging` module to provide detailed logs. Logs are configured to output at the INFO level and include timestamps.

## Error Handling

The application includes comprehensive error handling, returning JSON responses with error messages and details when issues occur.
