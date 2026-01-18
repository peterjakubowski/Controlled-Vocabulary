# Controlled Vocabulary & Semantic Auto-Tagging

This repository explores the intersection of **Computer Vision**, **Large Language Models (LLMs)**, and **Controlled Vocabularies**.

The primary goal is to develop a "Semantic Auto-tagger" that can classify content (text and images) using complex, professional taxonomies (specifically **IPTC Media Topics**) while overcoming the technical limitations of current AI structured outputs.

## The Problem: The "Vocabulary Gap"

Most commercial Computer Vision models are trained on datasets that are either too generic or too rigid for professional Digital Asset Management (DAM) needs:

* **COCO**: Limited to ~80 object classes (e.g., "Car," "Person").
* **ImageNet**: Noun-heavy and biology-skewed (great for dog breeds, poor for abstract concepts).
* **LVMs (Large Vision Models)**: Models like GPT-4V or Gemini Vision offer "Open Vocabulary" capabilities but are prone to "creative interpretation." They might tag a cat as "Feline Companion"—a valid English phrase, but if the controlled vocabulary requires "Domestic Cat," the tag is effectively invalid.

This creates a compatibility issue where valid descriptions fail to map to the strict schema required by downstream systems.

## The Solution

This project demonstrates how to force an LLM (Google Gemini) to strictly adhere to a pre-defined taxonomy using:
1.  **Structured Outputs:** Constraining the model to return valid JSON matching specific schema.
2.  **RAG (Retrieval Augmented Generation):** Dynamically retrieving relevant vocabulary terms to fit within the model's context window.
3.  **Multimodal Inputs:** Processing both text and images.

## Repository Structure

### 1. Interactive Application

* **`app.py`**: A **Streamlit** application that provides a user-friendly interface for the auto-tagging pipeline. It allows users to input text or images and receive IPTC-compliant tags in real-time.

### 2. Jupyter Notebooks

The `notebooks/` directory contains experiments and pipelines demonstrating different techniques:

* **`media_topics_structured_output.ipynb`**: Demonstrates the baseline technique of using Google Gemini with structured outputs to classify text against the IPTC vocabulary.
* **`media_topics_RAG.ipynb`**: Addresses the context window limitation by using a Vector Database (**ChromaDB**) to retrieve only the most relevant parts of the 1,200+ term vocabulary before asking the LLM to classify the text.
* **`media_topics_image_tagging_pipeline.ipynb`**: Extends the pipeline to **Computer Vision**. It processes images, generates descriptions, and then maps those visual features to the controlled vocabulary.

### 3. Core Modules

* **`gemini.py`**: Handles interactions with the Google GenAI SDK, including client initialization and model prompting.
* **`media_topics.py`**: Utilities for downloading, parsing, and managing the IPTC Media Topics JSON taxonomy.
* **`db.py`**: Interface for database operations (graph/vector interactions).

## Getting Started

### Prerequisites

* Python 3.10+
* A Google AI Studio API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/peterjakubowski/Controlled-Vocabulary.git
    cd Controlled-Vocabulary
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your API key:
    ```env
    GOOGLE_AI_API_KEY=your_api_key_here
    ```

### Usage

**Running the Streamlit App:**

```bash
streamlit run app.py
```