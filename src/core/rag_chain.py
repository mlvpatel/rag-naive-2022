"""The naive RAG chain for RagFlow (2022 generation).

Retrieve the top k chunks by dense similarity, stuff them into the prompt, and
generate. No query rewriting, no reranking, no fusion. This is the textbook
2022 pattern, packaged as a real service.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.config import get_settings
from src.embeddings.vectorstore import get_vectorstore

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer strictly from the provided context. If the context does "
            "not contain the answer, say you do not have that information. Do not "
            "invent facts.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


@lru_cache
def _make_llm():
    """Cached like the vector store: the client holds a connection pool, and
    rebuilding it per question throws that away for no gain."""
    s = get_settings()
    name = s.llm_model.lower()
    if name.startswith(("llama", "qwen", "mistral", "gemma", "deepseek", "phi")):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=0)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.llm_model, api_key=s.openai_api_key or None, temperature=0
    )


def _format(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def retrieve(question: str) -> List:
    """A single dense similarity search. This is the naive retrieval step."""
    s = get_settings()
    return get_vectorstore().similarity_search(question, k=s.top_k)


def answer_question(question: str) -> Dict:
    docs = retrieve(question)
    messages = _PROMPT.format_messages(context=_format(docs), question=question)
    chain = _make_llm() | StrOutputParser()
    answer = chain.invoke(messages)
    sources = sorted(
        {
            d.metadata.get("filename") or d.metadata.get("file_id", "unknown")
            for d in docs
        }
    )
    return {"answer": answer, "sources": sources}
