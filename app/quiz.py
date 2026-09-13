"""
Adaptive quiz generation.

Generates questions from the student's own indexed chunks using the LLM,
biased toward weak/prerequisite concepts identified by graph_module.py (FR13-FR14).
"""
import os
import json
from typing import List, Optional
from sqlalchemy.orm import Session

from .models import QuizQuestion, Chunk, Concept
from .graph_module import get_weak_concepts


def _llm_generate_mcq(context_text: str, concept_name: str) -> Optional[dict]:
    """Ask Claude to produce one multiple-choice question grounded in context_text."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Deterministic fallback for local dev without an API key.
        return {
            "question": f"Based on your notes, explain a key idea related to '{concept_name}'.",
            "options": ["See your notes", "Not applicable", "Not applicable", "Not applicable"],
            "correct_answer": "See your notes",
        }

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"Using ONLY the following study material, write one multiple-choice question "
        f"(4 options, exactly one correct) that tests understanding of the concept "
        f"'{concept_name}'. Respond as JSON with keys: question, options (list of 4), "
        f"correct_answer.\n\nMaterial:\n{context_text}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def generate_adaptive_quiz(db: Session, user_id: int, num_questions: int = 5) -> List[QuizQuestion]:
    """
    Builds a quiz biased toward the student's weakest concepts (and their
    prerequisites). Falls back to a random concept if no weaknesses are known yet
    (e.g. a brand-new student).
    """
    weak = get_weak_concepts(db, user_id)
    target_concepts = [w["concept_id"] for w in weak[:num_questions]]

    if not target_concepts:
        target_concepts = [c.id for c in db.query(Concept).limit(num_questions).all()]

    questions = []
    for concept_id in target_concepts:
        concept = db.query(Concept).filter(Concept.id == concept_id).first()
        if not concept:
            continue
        # Find a chunk associated with this concept's subject as grounding context.
        chunk = (
            db.query(Chunk)
            .join(Chunk.document)
            .filter(Chunk.document.has(subject=concept.subject))
            .first()
        )
        context_text = chunk.text if chunk else concept.name

        generated = _llm_generate_mcq(context_text, concept.name)
        if not generated:
            continue

        q = QuizQuestion(
            document_id=chunk.document_id if chunk else None,
            concept_id=concept.id,
            question_text=generated["question"],
            correct_answer=generated["correct_answer"],
            options=json.dumps(generated["options"]),
            difficulty=0.5,
        )
        db.add(q)
        questions.append(q)

    db.commit()
    return questions
