from __future__ import annotations

import re

from aidoc.models import Entity, Keyphrase, QueryAnalysis, QueryToken
from aidoc.nlp.patterns import ENTITY_PATTERNS, DOMAIN_KEYWORDS, tokenize_words


QUERY_VERBS = {
    "show",
    "find",
    "retrieve",
    "compare",
    "summarize",
    "explain",
    "identify",
    "list",
    "which",
    "what",
    "where",
    "discuss",
    "discusses",
    "mention",
    "mentions",
    "require",
    "requires",
    "include",
    "includes",
}

PREPOSITIONS = {"for", "to", "from", "with", "of", "in", "on", "about", "over", "under"}


class QueryUnderstanding:
    def analyze(self, query: str) -> QueryAnalysis:
        tokens = self._tag_tokens(query)
        entities = self._extract_entities(query)
        keyphrases = self._extract_keyphrases(query)
        normalized = " ".join(token.text for token in tokens)
        intent = self._detect_intent(tokens)
        domain_hint = self._detect_domain(query)
        return QueryAnalysis(
            query=query,
            normalized_query=normalized,
            tokens=tokens,
            entities=entities,
            keyphrases=keyphrases,
            intent=intent,
            domain_hint=domain_hint,
        )

    def _tag_tokens(self, query: str) -> list[QueryToken]:
        tokens: list[QueryToken] = []
        words = tokenize_words(query)
        entity_map = self._entity_map(query)
        for word in words:
            pos = self._pos_tag(word)
            entity_label = entity_map.get(word)
            tokens.append(QueryToken(text=word, pos=pos, entity_label=entity_label))
        return tokens

    def _entity_map(self, query: str) -> dict[str, str]:
        lowered = query.lower()
        mapping: dict[str, str] = {}
        for label, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in lowered:
                    for word in pattern.split():
                        mapping[word] = label
        return mapping

    def _extract_entities(self, query: str) -> list[Entity]:
        lowered = query.lower()
        entities: list[Entity] = []
        for label, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in lowered:
                    entities.append(Entity(text=pattern, label=label, confidence=0.88))
        return entities

    def _extract_keyphrases(self, query: str) -> list[Keyphrase]:
        phrases = re.findall(r"[a-zA-Z][a-zA-Z0-9_\s]+", query.lower())
        ranked = []
        for phrase in phrases:
            clean = phrase.strip()
            if len(clean.split()) >= 2:
                ranked.append(Keyphrase(text=clean, score=min(1.0, len(clean.split()) / 6.0)))
        return ranked[:5]

    def _detect_intent(self, tokens: list[QueryToken]) -> str:
        if any(token.text in {"find", "show", "list", "retrieve", "which", "what", "where"} for token in tokens):
            return "retrieval"
        if any(token.text in {"explain", "summarize"} for token in tokens):
            return "explanation"
        if any(token.entity_label == "RISK" for token in tokens):
            return "risk_review"
        return "search"

    def _detect_domain(self, query: str) -> str:
        lowered = query.lower()
        best_label = "general"
        best_score = 0
        for label, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def _pos_tag(self, word: str) -> str:
        if word in QUERY_VERBS:
            return "VERB"
        if word in {"which", "what", "where"}:
            return "DET"
        if word in PREPOSITIONS:
            return "ADP"
        if word.endswith("ing") or word.endswith("ed"):
            return "VERB"
        if word.endswith("ly"):
            return "ADV"
        if word.endswith(("tion", "ment", "ness", "ity", "ence", "ance")):
            return "NOUN"
        if len(word) <= 3:
            return "DET"
        return "NOUN"
