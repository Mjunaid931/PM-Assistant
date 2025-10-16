# RAG PM Assistant

This project is a Retrieval-Augmented Generation (RAG) application that acts as a Product Manager Assistant. It uses a local Ollama model to answer questions based on the information provided in the `knowledge_base` directory.

## Prerequisites

*   **Python 3.12+**
*   **Ollama:** [https://ollama.com/](https://ollama.com/)
    *   Make sure Ollama is installed and running.
    *   Pull the required models:
        *   `ollama pull llama3`
        *   `ollama pull nomic-embed-text`
*   **Google Cloud Account (if using Gemini API):**
    *   Enable the Gemini API for your project.
    *   Create a service account and download the JSON key file.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1  # (PowerShell)
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the application:**

    *   **Ollama (Local):**
        *   Ensure Ollama is running.
        *   The application will automatically use the `llama3` model for the LLM and `nomic-embed-text` for embeddings.
    *   **Google Cloud (Optional):**
        *   Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account JSON key file:

        ```bash
        set GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service_account.json
        ```

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application in your browser:**

    *   Streamlit will provide a URL to access the application (usually `http://localhost:8501`).

## Data

*   The application uses the text files in the `knowledge_base` directory as its knowledge source.
*   Place your `.txt` files containing product specifications, pricing guides, etc., in this directory.

## Troubleshooting

*   **`ModuleNotFoundError`:** Make sure you have installed all the dependencies from `requirements.txt`.
*   **Ollama Errors:** Ensure Ollama is running and the required models are pulled.
*   **Embedding Dimension Mismatch:** Delete the `chroma_db` directory to recreate the vector store with the correct embedding dimension.
*   **Google Cloud Errors:** Verify that the Gemini API is enabled, your service account has the necessary permissions, and your environment variables are set correctly.

## License

[Your License]

