# Controlled Vocabulary & Semantic Auto-Tagging

This repository explores the intersection of **Computer Vision**, **Large Language Models (LLMs)**, and **Controlled Vocabularies**.

The primary goal is to develop a "Semantic Auto-tagger" that can classify content using complex, professional taxonomies (like **IPTC Media Topics**) while overcoming the technical limitations of current AI structured outputs.

## The Problem: The "Vocabulary Gap"

Most commercial Computer Vision models are trained on datasets that are either too generic or too rigid for professional Digital Asset Management (DAM) needs:

* **COCO**: Limited to ~80 object classes (e.g., "Car," "Person").
* **ImageNet**: Noun-heavy and biology-skewed (great for dog breeds, poor for abstract concepts).
* ***LVMs (Large Vision Models)**: Models like GPT-4V or Gemini Vision offer "Open Vocabulary" capabilities but are prone to "creative interpretation." They might tag a cat as "Feline Companion"—a valid English phrase, but if the controlled vocabulary requires "Domestic Cat," the tag is effectively invalid. This creates a compatibility issue where valid descriptions fail strict metadata validation.

## The Solution: Constrained Decoding with LLMs

To solve this, we utilize LLMs with Structured Outputs to force the AI to map its understanding to a bespoke Controlled Vocabulary.

### The Technical Barrier

Modern LLM APIs (specifically Google Gemini and OpenAI) have strict limits on the size of the JSON schema used for structured outputs:

* **Google Gemini**: Limits structured output schemas to approximately 100 properties.
* **OpenAI**: Limits schemas to 1,000 properties.

Professional vocabularies (like IPTC) often contain thousands of terms, making a "one-shot" classification impossible.

### The Breakthrough: Multi-Pass Classification

This repository demonstrates a hierarchical, multi-pass approach to auto-tagging:

* **Pass 1 (Broad Concept)**: The LLM classifies the content into a high-level category (e.g., "Politics," "Sport," "Economy").
* **Pass 2 (Specific Topic)**: Based on the first result, the system dynamically loads the relevant subset of the vocabulary (e.g., only the "Politics" branch) and asks the LLM to select the specific leaf node.

This method allows us to access vocabularies of unlimited size without hitting the schema limits.

Example Flow: Society (Pass 1) -> Politics (Pass 2) -> Government (Pass 3) -> Local Election (Final Tag)

## Repository Structure

* `notebooks/`: Contains Jupyter notebooks demonstrating the classification experiments.

  * `media_topics_structured_output.ipynb`: Demonstrates the multi-pass classification technique using the Google Gemini API and the IPTC Media Topics vocabulary.

* `data/`: (Planned) Storage for vocabulary JSON files and sample datasets.

## Getting Started

1. **Install Requirements**:
    ```commandline
    pip install -r requirements.txt
    ```
2. **Environment Variables**: Create a `.env` file or export your API key: `export GOOGLE_API_KEY="your_key_here"`
3. **Run the Notebook**: Launch `media_topics_structured_output.ipynb` to see the multi-pass tagging in action on text samples.

## Roadmap

* [x] Text Classification: Proof of concept using IPTC Media Topics on text.
* [ ] Image Classification: Adapting the multi-pass workflow to Vision Models for image tagging.
* [ ] Visual Genome Integration: Exploring Scene Graphs for action/relationship tagging.
