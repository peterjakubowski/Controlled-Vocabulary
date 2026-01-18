from pydantic import BaseModel, Field
from enum import Enum


class InputType(Enum):
    TEXT = "Text"
    IMAGE = "Image"


class DataColumns(Enum):
    CONCEPT = "Concept"
    COUNT = "Count"
    DEFINITION = "Definition"


class CaptionResponse(BaseModel):
    caption: str = Field(..., description="A long, descriptive, and thorough caption for the image using "
                                          "concepts from the IPTC Media Topics controlled vocabulary.")
    concepts: list[str] = Field(..., description="Broad level concepts relevant to the image contents.")
