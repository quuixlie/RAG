import logging
import os
from pydantic import BaseModel
from typing import Literal

import config_typing as ct


class Config(BaseModel):
    database_kwargs: ct.DatabaseKwargs
    # TODO :: Somehow create possible values from classes but i think it would be overkill
    rag_architecture_name: Literal["classic-rag", "brain-rag"]

    embedder_name: Literal["openai-embedder"]
    embedder_kwargs: ct.EmbedderKwargs

    cross_encoder_name: Literal["basic-cross-encoder"]
    cross_encoder_kwargs: ct.CrossEncoderKwargs

    tokenizer_name: Literal["fixed-size-tokenizer"]
    tokenizer_kwargs: ct.TokenizerKwargs

    llm_name: Literal["open-router", "chat-gpt"]
    llm_kwargs: ct.LLMKwargs

    evaluation_llm_name: Literal["chat-gpt", "open-router"]
    evaluation_kwargs: ct.EvaluationKwargs

    """
    Configuration base class for RAG (Retrieval-Augmented Generation).
    """

    def __repr__(self) -> str:
        return (f"Config (\n"
                f"  database_kwargs: {self.database_kwargs},\n"
                f"  rag_architecture_name: {self.rag_architecture_name},\n"
                f"  embedder_name: {self.embedder_name},\n"
                f"  embedder_kwargs: {self.embedder_kwargs},\n"
                f"  cross_encoder_name: {self.cross_encoder_name},\n"
                f"  cross_encoder_kwargs: {self.cross_encoder_kwargs},\n"
                f"  tokenizer_name: {self.tokenizer_name},\n"
                f"  tokenizer_kwargs: {self.tokenizer_kwargs}\n"
                f"  llm_name: {self.llm_name},\n"
                f"  llm_kwargs: {self.llm_kwargs}\n"
                f"  evaluation_llm_name: {self.evaluation_llm_name},\n"
                f"  evaluation_kwargs: {self.evaluation_kwargs}\n"
                f")")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    logging.critical("OPENAI_API_KEY environment variable not set - exiting.")
    exit(-1)

config: Config = Config(
    database_kwargs={
        "embedding_dimension": 8,  # 384,
    },
    rag_architecture_name='classic-rag',
    # embedder_name = "basic-embedder",
    # embedder_kwargs= {
    #     "sentence_transformer_name": "mixedbread-ai/mxbai-embed-large-v1",
    #     "device": "cuda",
    # },
    embedder_name="openai-embedder",
    embedder_kwargs={
        "api_key": OPENAI_API_KEY,
        "model_name": "text-embedding-3-small",
        "dimension": 8,
    },
    cross_encoder_name="basic-cross-encoder",
    cross_encoder_kwargs={
        "sentence_transformer_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "device": "cuda",
    },
    tokenizer_name="fixed-size-tokenizer",
    tokenizer_kwargs={
        "chunk_size": 1024,
    },
    llm_name="chat-gpt",
    llm_kwargs={
        "api_key": OPENAI_API_KEY,
        "initial_prompt": "You are a helpful assistant.",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0,
    },
    # llm_name = "open-router",
    # llm_kwargs = {
    #     "api_key": os.getenv("OPENROUTER_API_KEY"),
    #     "initial_prompt": "You are a helpful assistant. Answer the question based on the provided context.",
    #     "model_name": "deepseek/deepseek-chat-v3-0324:free",
    # }
    evaluation_llm_name="chat-gpt",
    evaluation_kwargs={
        "llm_kwargs": {
            "api_key": OPENAI_API_KEY,
            "initial_prompt": None,
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.0,
        },
    }
)