"""
Learning dependency graph + knowledge-gap detection.

Concepts are nodes; a directed edge (concept -> prerequisite) means the
concept depends on the prerequisite. When a student gets a question about
a concept wrong, we look at whether its prerequisites are also weak —
that's the most likely root cause to recommend for revision.
"""
from typing import List, Dict
import networkx as nx
from sqlalchemy.orm import Session

from .models import Concept, ConceptPrerequisite, Mastery, QuizAttempt, QuizQuestion

MASTERY_LEARNING_RATE = 0.2
DEFAULT_MASTERY = 0.5
WEAK_THRESHOLD = 0.5


def build_graph(db: Session) -> nx.DiGraph:
    g = nx.DiGraph()
    for c in db.query(Concept).all():
        g.add_node(c.id, name=c.name, subject=c.subject)
    for edge in db.query(ConceptPrerequisite).all():
        g.add_edge(edge.concept_id, edge.prerequisite_id)  # concept -> prerequisite
    return g


def get_mastery(db: Session, user_id: int, concept_id: int) -> float:
    row = db.query(Mastery).filter_by(user_id=user_id, concept_id=concept_id).first()
    return row.score if row else DEFAULT_MASTERY


def update_mastery_from_attempt(db: Session, user_id: int, concept_id: int, is_correct: bool):
    """Simple exponential-moving-average mastery update per concept."""
    row = db.query(Mastery).filter_by(user_id=user_id, concept_id=concept_id).first()
    current = row.score if row else DEFAULT_MASTERY
    target = 1.0 if is_correct else 0.0
    new_score = current + MASTERY_LEARNING_RATE * (target - current)
    new_score = max(0.0, min(1.0, new_score))

    if row:
        row.score = new_score
    else:
        row = Mastery(user_id=user_id, concept_id=concept_id, score=new_score)
        db.add(row)
    db.commit()
    return new_score


def record_quiz_attempt(db: Session, user_id: int, question_id: int, is_correct: bool):
    question = db.query(QuizQuestion).filter(QuizQuestion.id == question_id).first()
    attempt = QuizAttempt(user_id=user_id, question_id=question_id, is_correct=is_correct)
    db.add(attempt)
    db.commit()

    if question and question.concept_id:
        update_mastery_from_attempt(db, user_id, question.concept_id, is_correct)
        # Wrong answer -> nudge prerequisite mastery estimate down slightly too,
        # since a missed prerequisite is a plausible root cause.
        if not is_correct:
            g = build_graph(db)
            for prereq_id in g.successors(question.concept_id):
                update_mastery_from_attempt(db, user_id, prereq_id, is_correct=False)


def get_weak_concepts(db: Session, user_id: int) -> List[Dict]:
    """
    Returns weak concepts ranked worst-first, each annotated with its
    weakest prerequisite (the most likely root cause per FR11/FR12).
    """
    g = build_graph(db)
    results = []
    for concept in db.query(Concept).all():
        score = get_mastery(db, user_id, concept.id)
        if score < WEAK_THRESHOLD:
            prereqs = list(g.successors(concept.id))
            weakest_prereq = None
            if prereqs:
                prereq_scores = {p: get_mastery(db, user_id, p) for p in prereqs}
                weakest_id = min(prereq_scores, key=prereq_scores.get)
                weakest_prereq_obj = db.query(Concept).filter(Concept.id == weakest_id).first()
                weakest_prereq = {
                    "id": weakest_id,
                    "name": weakest_prereq_obj.name if weakest_prereq_obj else None,
                    "mastery": prereq_scores[weakest_id],
                }
            results.append({
                "concept_id": concept.id,
                "concept_name": concept.name,
                "mastery": score,
                "likely_root_cause_prerequisite": weakest_prereq,
            })
    results.sort(key=lambda r: r["mastery"])
    return results
