import streamlit as st
from streamlit.connections import BaseConnection
from typing import Any
from dotenv import load_dotenv
from google import genai
import os
from PIL.Image import Image
from models import CaptionResponse
from media_topics import broad_topics_json

load_dotenv()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

SYSTEM_INSTRUCTION_CLASSIFY = """"# Role
* You are an expert digital asset librarian specializing in the IPTC Media Topics Controlled Vocabulary.
# Rules
* Given content, respond with a JSON object containing a list of all relevant keywords from the defined vocabulary.
* Limit your response to 50 keywords. 
# Vocabulary
{vocabulary_json}
"""

SYSTEM_INSTRUCTION_DESCRIBE = """# Role
* You are an expert digital asset librarian specializing in the IPTC Media Topics Controlled Vocabulary.
# Rules
* Given an image, you generate a detailed semantic description of everything that you see.
* You must adhere to the concepts as defined in the vocabulary.
# Broad Level Vocabulary Concepts
{broad_level_vocabulary} 
"""


# Define the response schema for classification
def load_json_response_schema(concepts: list[str]) -> dict:
    """Load a JSON schema for the given concepts."""

    response_schema = {
        '$defs': {
            'Tags': {
                'enum': concepts,
                'title': 'Tags',
                'type': 'string'
            }
        },
        'properties': {
            'keywords': {
                'items': {
                    '$ref': '#/$defs/Tags'
                },
                'title': 'Keywords',
                'type': 'array'
            }
        },
        'required': ['keywords'],
        'title': 'Metadata',
        'type': 'object'
    }

    return response_schema


class GeminiConnection(BaseConnection[genai.Client]):

    def _connect(self, **kwargs: Any) -> genai.Client:
        return genai.Client(api_key=GOOGLE_AI_API_KEY)

    # Function to classify media topics using GenAI
    def classify_media_topics(self, content: str, response_schema: dict, vocabulary_json: str, ttl: int = 3600):
        @st.cache_data(show_spinner="Classifying media topics...", ttl=ttl)
        def _classify_media_topics(content: str, response_schema: dict, vocabulary_json: str) -> list[str]:
            response = self._instance.models.generate_content(
                model="gemini-2.5-flash",
                contents=[content],
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION_CLASSIFY.format(vocabulary_json=vocabulary_json),
                    temperature=1.0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=-1
                    )
                )
            )
            return response.parsed.get('keywords', [])
        return _classify_media_topics(content=content, response_schema=response_schema, vocabulary_json=vocabulary_json)

    def generate_image_caption(self, image: Image, ttl: int = 3600):
        @st.cache_data(show_spinner="Generating image caption...", ttl=ttl)
        def _generate_image_caption(image: Image):
            response = self._instance.models.generate_content(
                model='gemini-3-flash-preview',
                contents=[image],
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION_DESCRIBE.format(broad_level_vocabulary=broad_topics_json),
                    response_mime_type="application/json",
                    response_json_schema=CaptionResponse.model_json_schema(),
                )
            )

            caption_response = CaptionResponse.model_validate_json(response.text)

            return caption_response

        return _generate_image_caption(image=image)


conn = st.connection("gemini", GeminiConnection)
