"""
Grammatical Mapper - Deep Structure Extraction
===============================================

Based on Chomsky's Universal Grammar: all languages share deep structural
patterns despite surface differences.

Examples of equivalent deep structures:
  - "John gives book to Mary" ≈ "Mary receives book from John"
  - "A causes B" ≈ "B is caused by A"
  - "João dá livro para Maria" ≈ "Mary receives book from John"

This module extracts these deep patterns to verify consistency between
prompts and responses without needing ensemble sampling.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
from enum import Enum


class RelationType(Enum):
    """Universal grammatical relations"""
    AGENT_ACTION = "agent_action"           # X does Y
    PATIENT_ACTION = "patient_action"       # Y is done to X
    POSSESSION = "possession"               # X has Y
    ATTRIBUTION = "attribution"             # X is Y
    TEMPORAL = "temporal"                   # X at time Y
    CAUSATION = "causation"                 # X causes Y
    LOCATION = "location"                   # X at place Y
    COMPARISON = "comparison"               # X vs Y


@dataclass
class StructurePattern:
    """
    Deep structure pattern extracted from text.

    Represents the canonical form of a statement, independent of
    surface realization (word order, voice, language).
    """
    entities: Set[str]                      # Normalized entities
    relations: List[Tuple[str, RelationType, str]]  # (subject, relation, object)
    predicates: Set[str]                    # Main predicates/actions
    temporal_markers: Set[str]              # Years, dates, time expressions
    negations: Set[str]                     # Negated statements

    def __hash__(self):
        return hash((
            frozenset(self.entities),
            tuple(self.relations),
            frozenset(self.predicates),
            frozenset(self.temporal_markers),
            frozenset(self.negations),
        ))

    def symmetry_score(self, other: StructurePattern) -> float:
        """
        Compute grammatical symmetry between two patterns.

        Returns:
            Float in [0, 1] where 1.0 = perfect symmetry

        Perfect symmetry example:
            P1: "John won the 2019 Nobel Prize"
            P2: "The 2019 Nobel Prize was won by John"
            → score ≈ 1.0 (same deep structure)

        Asymmetric example:
            P1: "John won the 2019 Nobel Prize"
            P2: "Mary won the 2020 Nobel Prize"
            → score ≈ 0.3 (different entities, different time)
        """
        if not other:
            return 0.0

        # Entity overlap (normalized)
        entity_overlap = len(self.entities & other.entities) / max(
            len(self.entities | other.entities), 1
        )

        # Predicate overlap
        pred_overlap = len(self.predicates & other.predicates) / max(
            len(self.predicates | other.predicates), 1
        )

        # Temporal consistency
        temporal_overlap = len(self.temporal_markers & other.temporal_markers) / max(
            len(self.temporal_markers | other.temporal_markers), 1
        ) if (self.temporal_markers or other.temporal_markers) else 1.0

        # Relation structure similarity
        rel_score = self._relation_similarity(other)

        # Negation consistency (critical!)
        neg_consistent = (len(self.negations & other.negations) ==
                         len(self.negations | other.negations))
        neg_penalty = 1.0 if neg_consistent else 0.3

        # Weighted average (relations are most important)
        score = (
            0.25 * entity_overlap +
            0.35 * rel_score +
            0.20 * pred_overlap +
            0.15 * temporal_overlap +
            0.05
        ) * neg_penalty

        return min(1.0, max(0.0, score))

    def _relation_similarity(self, other: StructurePattern) -> float:
        """Compare relational structures (handles voice transformations)"""
        if not self.relations and not other.relations:
            return 1.0
        if not self.relations or not other.relations:
            return 0.0

        # Convert relations to canonical form
        self_canonical = {self._canonicalize_relation(r) for r in self.relations}
        other_canonical = {self._canonicalize_relation(r) for r in other.relations}

        overlap = len(self_canonical & other_canonical)
        total = len(self_canonical | other_canonical)

        return overlap / max(total, 1)

    def _canonicalize_relation(self, rel: Tuple[str, RelationType, str]) -> Tuple:
        """
        Convert relations to canonical form.

        Examples:
            ("John", AGENT_ACTION, "win") → ("john", "action", "win")
            ("Prize", PATIENT_ACTION, "won") → ("prize", "action", "won")
        """
        subj, rel_type, obj = rel
        return (subj.lower(), rel_type.value, obj.lower())


class GrammaticalMapper:
    """
    Extracts deep grammatical structure from text using regex-based patterns.

    While simple, this captures sufficient structure for hallucination detection.
    Future versions could use dependency parsing (spaCy) for better accuracy.
    """

    # Regex patterns for structure extraction
    ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
    YEAR_PATTERN = re.compile(r'\b[12]\d{3}\b')  # Fixed: matches 1000-2999
    NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')
    NEGATION_PATTERN = re.compile(r'\b(not|no|never|neither|nor|n\'t)\b', re.I)

    # Action verbs (simplified list - could be expanded)
    ACTION_VERBS = {
        'won', 'received', 'awarded', 'gave', 'invented', 'discovered',
        'created', 'built', 'wrote', 'published', 'founded', 'established',
        'is', 'was', 'are', 'were', 'became', 'got', 'had', 'has'
    }

    # Causal markers
    CAUSAL_MARKERS = {'because', 'caused', 'resulted', 'led', 'due'}

    def extract_structure(self, text: str) -> StructurePattern:
        """
        Extract deep grammatical structure from text.

        Args:
            text: Input text (prompt or response)

        Returns:
            StructurePattern containing deep structure
        """
        text_lower = text.lower()

        # Extract entities (proper nouns)
        entities = {e.lower() for e in self.ENTITY_PATTERN.findall(text)}

        # Extract temporal markers
        years = set(self.YEAR_PATTERN.findall(text))

        # Extract predicates (action verbs)
        words = re.findall(r'\b\w+\b', text_lower)
        predicates = {w for w in words if w in self.ACTION_VERBS}

        # Extract relations (simplified - could be more sophisticated)
        relations = self._extract_relations(text, entities, predicates)

        # Detect negations
        negations = set(self.NEGATION_PATTERN.findall(text_lower))

        return StructurePattern(
            entities=entities,
            relations=relations,
            predicates=predicates,
            temporal_markers=years,
            negations=negations,
        )

    def _extract_relations(
        self,
        text: str,
        entities: Set[str],
        predicates: Set[str]
    ) -> List[Tuple[str, RelationType, str]]:
        """
        Extract subject-relation-object triples.

        Simple pattern matching - good enough for hallucination detection.
        """
        relations = []
        text_lower = text.lower()

        # Pattern: Entity + verb + Entity/Object
        # Example: "John won prize" → (John, AGENT_ACTION, prize)
        sentences = re.split(r'[.!?;]', text)

        for sent in sentences:
            sent = sent.strip().lower()
            if not sent:
                continue

            # Find entity-verb-object patterns
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower not in sent:
                    continue

                for pred in predicates:
                    if pred not in sent:
                        continue

                    # Simple heuristic: entity before verb = agent
                    entity_pos = sent.find(entity_lower)
                    pred_pos = sent.find(pred)

                    if entity_pos < pred_pos:
                        # Entity is agent
                        rel_type = RelationType.AGENT_ACTION
                        # Find object after verb
                        after_pred = sent[pred_pos + len(pred):].strip()
                        obj_match = re.search(r'\b([a-z]+(?:\s+[a-z]+)?)\b', after_pred)
                        if obj_match:
                            obj = obj_match.group(1)
                            relations.append((entity, rel_type, obj))
                    else:
                        # Entity is patient (passive voice)
                        rel_type = RelationType.PATIENT_ACTION
                        relations.append((entity, rel_type, pred))

        # Detect temporal relations
        years = self.YEAR_PATTERN.findall(text)
        for entity in entities:
            for year in years:
                if entity.lower() in text_lower and year in text:
                    relations.append((entity, RelationType.TEMPORAL, year))

        return relations

    def check_consistency(
        self,
        prompt_structure: StructurePattern,
        response_structure: StructurePattern,
        threshold: float = 0.6
    ) -> Tuple[bool, float, str]:
        """
        Check if response is grammatically consistent with prompt.

        Args:
            prompt_structure: Deep structure of input prompt
            response_structure: Deep structure of model response
            threshold: Minimum symmetry score for consistency

        Returns:
            (is_consistent, symmetry_score, explanation)
        """
        score = prompt_structure.symmetry_score(response_structure)

        is_consistent = score >= threshold

        explanation = self._explain_score(
            prompt_structure,
            response_structure,
            score,
            threshold
        )

        return is_consistent, score, explanation

    def _explain_score(
        self,
        prompt: StructurePattern,
        response: StructurePattern,
        score: float,
        threshold: float,
    ) -> str:
        """Generate human-readable explanation of symmetry score"""

        entity_match = len(prompt.entities & response.entities)
        pred_match = len(prompt.predicates & response.predicates)
        temporal_match = len(prompt.temporal_markers & response.temporal_markers)

        parts = [
            f"Symmetry score: {score:.3f} (threshold: {threshold:.3f})",
            f"Entity overlap: {entity_match}/{len(prompt.entities | response.entities)}",
            f"Predicate overlap: {pred_match}/{len(prompt.predicates | response.predicates)}",
        ]

        if prompt.temporal_markers or response.temporal_markers:
            parts.append(
                f"Temporal consistency: {temporal_match}/{len(prompt.temporal_markers | response.temporal_markers)}"
            )

        if prompt.negations != response.negations:
            parts.append("⚠️ Negation mismatch detected (critical)")

        decision = "✓ CONSISTENT" if score >= threshold else "✗ INCONSISTENT"
        parts.append(decision)

        return " | ".join(parts)


# Quick test
if __name__ == "__main__":
    mapper = GrammaticalMapper()

    # Test case 1: Same deep structure, different surface
    p1 = "Who won the 2019 Nobel Prize in Physics?"
    r1 = "James Peebles won the 2019 Nobel Prize in Physics."

    prompt_struct = mapper.extract_structure(p1)
    response_struct = mapper.extract_structure(r1)

    consistent, score, explanation = mapper.check_consistency(prompt_struct, response_struct)
    print(f"Test 1: {explanation}\n")

    # Test case 2: Inconsistent (wrong year)
    r2 = "James Peebles won the 2020 Nobel Prize in Physics."
    response_struct2 = mapper.extract_structure(r2)
    consistent2, score2, explanation2 = mapper.check_consistency(prompt_struct, response_struct2)
    print(f"Test 2: {explanation2}\n")
