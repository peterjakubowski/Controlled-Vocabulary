from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")


# Initialize the GenAI client
client = genai.Client(api_key=GOOGLE_AI_API_KEY)

SYSTEM_INSTRUCTION = """"# Instructions
* Label text with all relevant concepts from the IPTC Media Topics Controlled Vocabulary. 
* Respond with a JSON object containing an array of keywords from the defined vocabulary. 
# Vocabulary
{vocabulary_json}
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
def classify_media_topics(content: str, response_schema: dict, vocabulary_json: str) -> genai.types.GenerateContentResponse:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[content],
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION.format(vocabulary_json=vocabulary_json),
            temperature=1.0,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )
    return response
