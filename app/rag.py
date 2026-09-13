"""
Explainable Retrieval-Augmented Generation.

- Embeds chunks with sentence-transformers.
- Stores vectors per-user in a FAISS index (kept on disk so it survives restarts).
- Answers a question ONLY from the user's own retrieved chunks, and always
  returns the exact supporting excerpt + document name + page/slide (FR7-FR9).
"""
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from sqlalchemy.orm import Session
from .models import Chunk, Document

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good enough for a baseline
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _index_path(user_id: int) -> str:
    return os.path.join(VECTORSTORE_DIR, f"user_{user_id}.faiss")


def _meta_path(user_id: int) -> str:
    return os.path.join(VECTORSTORE_DIR, f"user_{user_id}_meta.pkl")


def _load_or_create_index(user_id: int, dim: int):
    path = _index_path(user_id)
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexFlatIP(dim)   # cosine similarity via normalized inner product


def _load_meta(user_id: int) -> List[int]:
    """meta[i] = chunk_id stored at FAISS row i"""
    path = _meta_path(user_id)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def _save_meta(user_id: int, meta: List[int]):
    with open(_meta_path(user_id), "wb") as f:
        pickle.dump(meta, f)


def add_chunks_to_index(db: Session, user_id: int, chunks: List[Chunk]):
    """Embed a batch of Chunk rows and append them to the user's FAISS index."""
    if not chunks:
        return
    model = get_embed_model()
    texts = [c.text for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")

    index = _load_or_create_index(user_id, dim=vectors.shape[1])
    meta = _load_meta(user_id)

    start_row = index.ntotal
    index.add(vectors)
    for i, chunk in enumerate(chunks):
        chunk.vector_id = start_row + i
        meta.append(chunk.id)

    faiss.write_index(index, _index_path(user_id))
    _save_meta(user_id, meta)
    db.commit()


def retrieve(db: Session, user_id: int, question: str, top_k: int = 4) -> List[Chunk]:
    """Return the top-k most relevant chunks belonging to this user."""
    path = _index_path(user_id)
    if not os.path.exists(path):
        return []

    model = get_embed_model()
    q_vec = model.encode([question], normalize_embeddings=True).astype("float32")

    index = faiss.read_index(path)
    meta = _load_meta(user_id)
    if index.ntotal == 0:
        return []

    scores, indices = index.search(q_vec, min(top_k, index.ntotal))
    chunk_ids = [meta[i] for i in indices[0] if i != -1]
    chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
    # preserve similarity order
    order = {cid: rank for rank, cid in enumerate(chunk_ids)}
    chunks.sort(key=lambda c: order.get(c.id, 999))
    return chunks


def generate_grounded_answer(question: str, chunks: List[Chunk], db: Session) -> Tuple[str, bool]:
    """
    Calls Claude with a strict "answer only from the provided context, and say
    so if you can't" instruction. Returns (answer_text, grounded_flag).
    """
    if not chunks:
        return (
            "I couldn't find anything in your uploaded material that answers this question. "
            "Try uploading the relevant document, or rephrase your question.",
            False,
        )

    context_blocks = []
    for c in chunks:
        doc = db.query(Document).filter(Document.id == c.document_id).first()
        doc_name = doc.filename if doc else "Unknown document"
        context_blocks.append(f"[Source: {doc_name}, {c.page_or_slide}]\n{c.text}")
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a study assistant. Answer the student's question using ONLY the "
        "provided context extracted from their own uploaded material. "
        "If the context does not contain the answer, say clearly that you cannot "
        "find it in their material — never use outside knowledge. "
        "Keep the answer concise and reference which source(s) you used."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback for local dev without an API key: return the most relevant excerpt directly.
        return (f"[No LLM configured — showing best-matching excerpt]\n\n{chunks[0].text}", True)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = "".join(block.text for block in response.content if block.type == "text")
    return answer_text, True
