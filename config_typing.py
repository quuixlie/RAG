from dataclasses import dataclass
from typing import  Optional,Literal,TypedDict,Annotated
from pydantic import Field




class DatabaseKwargs(TypedDict):
    embedding_dimension: Literal[8,16,32,64,128,256,384,768,1536]

class EmbedderKwargs(TypedDict):
    api_key: str
    model_name: Literal["text-embedding-3-small"]
    dimension: Literal[8,16,32,64,128,256,384,768,1536]

class CrossEncoderKwargs(TypedDict):
    sentence_transformer_name: str
    device: Literal["cuda", "cpu"]


class TokenizerKwargs(TypedDict):
    chunk_size: Literal[256,512,1024,2048]

class LLMKwargs(TypedDict):
    api_key: str
    initial_prompt: Optional[str]
    model_name: Literal["gpt-3.5-turbo"]
    temperature: Annotated[float,Field(ge=0.0,le=1.0)]


class EvaluationKwargs(TypedDict):
    llm_kwargs: LLMKwargs

