# 🌾 Farmer Query Assistant (RAG)
This application is designed to provide accurate and instant answers to farmer queries, trained on valuable Kisan Call Center (KCC) data. This project implements a specialized **Farmer Query Assistant** using a Retrieval-Augmented Generation (RAG) architecture. It is built to answer questions related to agriculture and horticulture, leveraging a fine-tuned LLM and a vector database for context-aware responses.

The entire application is packaged for serverless GPU deployment using **[Modal](https://modal.com/)**.

## 📋 Table of Contents

  * [Overview](#-overview)
  * [How it Works (Architecture)](#-how-it-works-architecture)
  * [Technology Stack](#-technology-stack)
  * [Project Structure](#-project-structure)
  * [Deployment with Modal](#-deployment-with-modal)
  * [How to Use](#-how-to-use)

-----

## Overview

The core of this project is a `modal.App` that serves a **Gradio** web interface. This UI allows users to select a category and ask a farming-related question.

### Key Features

  * **RAG Pipeline:** Uses a **ChromaDB** vector store to retrieve relevant Q\&A pairs from a knowledge base (`train.pkl`) before generating an answer.
  * **High-Performance LLM:** Utilizes the `meta-llama/Meta-Llama-3.1-8B` model, fine-tuned with PEFT adapters (`naveenng10/farmer-assistant`).
  * **Optimized Inference:** The model is loaded with 4-bit quantization (`BitsAndBytesConfig`) and runs on a **T4 GPU** for efficient inference.
  * **Strong Guardrails:**
    1.  Uses a keyword-based filter (`is_agriculture_related`) to immediately reject off-topic questions (sports, politics, etc.).
    2.  Provides a detailed system prompt to the LLM, instructing it to *only* answer farming-related queries.
  * **Vector Search:** Employs `sentence-transformers/all-MiniLM-L6-v2` for embedding queries and documents.
  * **Multilingual Support:** Includes a `Helsinki-NLP/opus-mt-mul-en` model to translate model outputs (and potentially inputs) to English, ensuring consistent response language.
  * **Serverless Deployment:** Fully configured for deployment on Modal, handling environment setup, model downloads, and scaling automatically.



## 🏗️ How it Works (Architecture)

The application follows a RAG pipeline with strict guardrails:

1.  **User Input:** A user provides a `category` and a `query` via the Gradio UI.
2.  **Guardrail 1 (Keyword Filter):** The query is first checked by the `is_agriculture_related` function. If it contains blocked keywords (e.g., "cricket", "politics"), the process stops and returns a default rejection message.
3.  **RAG - Retrieve:**
      * The user's `query` is encoded into a vector embedding.
      * ChromaDB is queried to find the top 3 most similar questions from its database (which was built from `train.pkl`).
      * A relevance check (`is_query_relevant`) is performed based on the distance of the results.
4.  **RAG - Augment:**
      * The similar questions and their corresponding answers are retrieved from the vector store's metadata.
      * This information is formatted into a `context` string.
5.  **Prompt Engineering:**
      * A final prompt is constructed containing:
        1.  **System Message:** A strict set of rules defining the assistant's role and limitations.
        2.  **RAG Context:** The similar Q\&A pairs found in step 4.
        3.  **User Query:** The original question from the user.
6.  **Generation:** The complete prompt is sent to the fine-tuned Llama 3.1 model (`FarmerAssistantModel`) to generate an answer.
7.  **Post-processing:** The generated text is passed through the `translate_to_english` function to ensure the final output is in English.
8.  **Response:** The final answer is displayed to the user in the Gradio interface.

-----

## 🛠️ Technology Stack

  * **Deployment & Infrastructure:** [Modal](https://modal.com/)
  * **Web UI:** [Gradio](https://www.gradio.app/)
  * **LLM:** `meta-llama/Meta-Llama-3.1-8B`
  * **Fine-tuning:** `peft` (adapters loaded from `naveenng10/farmer-assistant`)
  * **Quantization:** `bitsandbytes`
  * **Core AI/ML:** `torch`, `transformers`, `accelerate`
  * **Vector Database (RAG):** `chromadb`
  * **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Translation Model:** `Helsinki-NLP/opus-mt-mul-en`

-----

## 📁 Project Structure

For the Modal app to build correctly, your local directory should be structured as follows. The script assumes it is located in `src/inference/`.

```
your-project-root/
├── src/
│   ├── data/
│   │   ├── train.pkl             # Required: Pickle file with Q&A data
│   │   └── chroma_database/      # Will be created by the script if it doesn't exist
│   │
│   └── inference/
│       ├── farmer-assistant.py   # The main Modal script
│       └── constants.py          # Required: Must contain QUERY_TYPE list
│
└── ... (other files)
```

**Note:** The script uses relative paths (`os.path.dirname(__file__)`) to find the `src/data` directory and `constants.py`.

-----

## 🚀 Deployment with Modal

Follow these steps to deploy the Farmer Assistant.

### 1\. Prerequisites

  * A [Modal](https://modal.com/) account.
  * [Python 3.8+](https://www.python.org/) installed.
  * A Hugging Face account with an access token. You must have accepted the license terms for `meta-llama/Meta-Llama-3.1-8B`.

### 2\. Local Setup

1.  **Install the Modal client:**

    ```bash
    pip install modal
    ```

2.  **Set up Modal authentication:**

    ```bash
    modal setup
    ```

    This will open a browser window to link your local machine to your Modal account.

3.  **Create a Hugging Face Secret:**
    The application requires your Hugging Face token to download the Llama model. Create a secret in Modal named `huggingface-secret` with your token.

    ```bash
    modal secret create huggingface-secret HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"
    ```

### 3\. Run the Application

You can either run the app in "serve" mode for development or "deploy" it for a persistent endpoint.

  * **To Serve (Development Mode):**
    This command runs the app from your local directory and streams logs to your terminal. The app will hot-reload if you save changes to the file. It will stop when you press `Ctrl+C`.

    ```bash
    modal serve src/inference/farmer-assistant.py
    ```

  * **To Deploy (Production Mode):**
    This command builds the container image, uploads your files, and creates a permanent, shareable URL for your Gradio application.

    ```bash
    modal deploy src/inference/farmer-assistant.py
    ```

After running either command, Modal will output a URL (e.g., `https://your-username--farmer-assistant-gradio-app.modal.run`) where you can access the Gradio UI.

-----

## ❓ How to Use

1.  Open the Modal URL provided after deployment.
2.  You will see the **🌾 Farmer Query Assistant** interface.
3.  Select a **category** from the dropdown menu (e.g., "Plant Protection").
4.  Type your farming-related question in the **"Your Question"** text box.
5.  Click the **"Submit"** button.
6.  The assistant will process your query through the RAG pipeline and display the answer in the output box.
