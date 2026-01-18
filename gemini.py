from dotenv import load_dotenv
from google import genai
import os
from PIL.Image import Image
from models import CaptionResponse

load_dotenv()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")


# Initialize the GenAI client
client = genai.Client(api_key=GOOGLE_AI_API_KEY)

SYSTEM_INSTRUCTION_CLASSIFY = """"# Instructions
* Label text with all relevant concepts from the IPTC Media Topics Controlled Vocabulary. 
* Respond with a JSON object containing an array of keywords from the defined vocabulary. 
# Vocabulary
{vocabulary_json}
"""

SYSTEM_INSTRUCTION_DESCRIBE = """# Role
* You are an expert digital asset librarian specializing in the IPTC Media Topics Controlled Vocabulary.
# Rules
* Given an image, you generate a detailed description of everything that you see.
* You should adhere to the concepts as defined in the vocabulary.
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


# Function to classify media topics using GenAI
def classify_media_topics(content: str | Image, response_schema: dict, vocabulary_json: str) -> genai.types.GenerateContentResponse:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[content],
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_CLASSIFY.format(vocabulary_json=vocabulary_json),
            temperature=1.0,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )
    return response


def generate_image_caption(image: Image):
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=["Write an exhaustive caption for this image", image],
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION_DESCRIBE,
            response_mime_type="application/json",
            response_json_schema=CaptionResponse.model_json_schema(),
        )
    )

    caption_response = CaptionResponse.model_validate_json(response.text)

    return caption_response
