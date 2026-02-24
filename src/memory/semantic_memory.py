"""
Semantic Memory - Neocortex Analogue
Slow consolidation, knowledge graph, concept hierarchies, preferences
"""

import sqlite3
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class Concept:
    """A concept in semantic memory"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = ""
    attributes: Dict = field(default_factory=dict)
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.5
    occurrence_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


@dataclass
class SemanticKnowledge:
    """Consolidated semantic knowledge"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fact: str = ""
    entity_type: str = ""  # person, place, object, concept
    entities: List[str] = field(default_factory=list)
    relation: str = ""
    confidence: float = 0.5
    evidence_count: int = 1
    source_trace_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class Preference:
    """User preference"""

    key: str
    value: Any
    category: str = "general"
    confidence: float = 0.5
    occurrence_count: int = 1
    updated_at: float = field(default_factory=time.time)


class SemanticMemory:
    """
    Slow-learning, stable knowledge storage

    Features:
    - Knowledge graph storage
    - Concept hierarchies
    - Preference learning
    - Slow consolidation (protects stability)
    - Generalization from examples
    """

    def __init__(self, db_path: str = "data/memory/semantic.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database"""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        # Facts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                fact TEXT NOT NULL,
                entity_type TEXT,
                entities TEXT,
                relation TEXT,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                source_traces TEXT,
                created_at REAL
            )
        """)

        # Concepts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                attributes TEXT,
                related_concepts TEXT,
                confidence REAL DEFAULT 0.5,
                occurrence_count INTEGER DEFAULT 1,
                created_at REAL,
                last_accessed REAL
            )
        """)

        # Preferences table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                category TEXT DEFAULT 'general',
                confidence REAL DEFAULT 0.5,
                occurrence_count INTEGER DEFAULT 1,
                updated_at REAL
            )
        """)

        # Indices
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_type)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prefs_category ON preferences(category)"
        )

        conn.commit()
        conn.close()

    def consolidate(self, episodic_trace) -> Optional[SemanticKnowledge]:
        """Extract semantic knowledge from episodic trace"""
        if not episodic_trace.content:
            return None

        # Extract facts from content
        facts = self._extract_facts(episodic_trace.content, episodic_trace.context)

        if not facts:
            return None

        # Store first fact as knowledge
        fact = facts[0]
        knowledge = SemanticKnowledge(
            fact=fact["fact"],
            entity_type=fact.get("entity_type", "unknown"),
            entities=fact.get("entities", []),
            relation=fact.get("relation", ""),
            confidence=min(0.9, 0.3 + episodic_trace.importance * 0.4),
            evidence_count=1,
            source_trace_ids=[episodic_trace.id],
        )

        self._store_knowledge(knowledge)

        # Extract and update concepts
        concepts = self._extract_concepts(episodic_trace.content)
        for concept_name in concepts:
            self._learn_concept(concept_name, episodic_trace.context)

        # Extract preferences
        preference = self._extract_preference(episodic_trace)
        if preference:
            self._update_preference(
                preference.key, preference.value, preference.category
            )

        return knowledge

    def _extract_facts(self, content: str, context: Dict) -> List[Dict]:
        """Extract facts from content"""
        facts = []

        # Simple extraction - in production, use NER
        words = content.split()

        # Look for patterns like "X is Y", "X has Y"
        for i, word in enumerate(words):
            if word.lower() in ["is", "are", "was", "were"]:
                if i > 0 and i < len(words) - 1:
                    entity = " ".join(words[max(0, i - 2) : i])
                    value = " ".join(words[i + 1 : min(i + 4, len(words))])

                    facts.append(
                        {
                            "fact": f"{entity} {word} {value}",
                            "entity_type": self._classify_entity(entity),
                            "entities": [entity, value],
                            "relation": "is_a",
                        }
                    )

        return facts

    def _classify_entity(self, entity: str) -> str:
        """Classify entity type"""
        entity_lower = entity.lower()

        # Simple heuristics
        if entity_lower.startswith(("the ", "a ", "an ")):
            entity_lower = entity_lower[4:]

        # Check for common patterns
        if entity_lower.endswith(("er", "or", "ist")):
            return "person"
        if entity_lower.endswith(("tion", "ment", "ness")):
            return "concept"
        if entity_lower.endswith(("ia", "land", "city", "town")):
            return "place"

        return "unknown"

    def _extract_concepts(self, content: str) -> Set[str]:
        """Extract concepts from content"""
        # Simple token-based extraction
        words = content.lower().split()

        # Filter common words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
        }

        concepts = {w for w in words if len(w) > 3 and w not in stop_words}
        return concepts

    def _extract_preference(self, episodic_trace) -> Optional[Preference]:
        """Extract preference from trace"""
        content = episodic_trace.content.lower()
        context = episodic_trace.context or {}

        # Look for preference indicators
        pref_indicators = [
            "prefer",
            "like",
            "dislike",
            "hate",
            "love",
            "want",
            "dont want",
        ]

        for indicator in pref_indicators:
            if indicator in content:
                # Extract key from context or content
                key = context.get("preference_key")
                value = context.get("preference_value")

                if key and value is not None:
                    return Preference(
                        key=key,
                        value=value,
                        category=context.get("preference_category", "general"),
                    )

        return None

    def _store_knowledge(self, knowledge: SemanticKnowledge):
        """Store knowledge in database with transaction"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO facts
                    (id, fact, entity_type, entities, relation, confidence, evidence_count, source_traces, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        knowledge.id,
                        knowledge.fact,
                        knowledge.entity_type,
                        json.dumps(knowledge.entities),
                        knowledge.relation,
                        knowledge.confidence,
                        knowledge.evidence_count,
                        json.dumps(knowledge.source_trace_ids),
                        knowledge.created_at,
                    ),
                )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _learn_concept(self, name: str, context: Dict = None):
        """Learn or update a concept with transaction"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                # Check existing
                cursor = conn.execute(
                    "SELECT id, occurrence_count, confidence FROM concepts WHERE name = ?",
                    (name,),
                )
                row = cursor.fetchone()

                if row:
                    # Update existing
                    new_count = row[1] + 1
                    new_confidence = min(1.0, row[2] + 0.1)
                    conn.execute(
                        """
                        UPDATE concepts 
                        SET occurrence_count = ?, confidence = ?, last_accessed = ?
                        WHERE name = ?
                    """,
                        (new_count, new_confidence, time.time(), name),
                    )
                else:
                    # Create new
                    concept = Concept(name=name)
                    if context:
                        concept.category = context.get("category", "general")

                    conn.execute(
                        """
                        INSERT INTO concepts
                        (id, name, category, attributes, related_concepts, confidence, occurrence_count, created_at, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            concept.id,
                            concept.name,
                            concept.category,
                            json.dumps({}),
                            json.dumps([]),
                            concept.confidence,
                            concept.occurrence_count,
                            concept.created_at,
                            concept.last_accessed,
                        ),
                    )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _update_preference(self, key: str, value: Any, category: str = "general"):
        """Update preference with transaction"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT occurrence_count, confidence FROM preferences WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()

                if row:
                    new_count = row[0] + 1
                    new_confidence = min(1.0, row[1] + 0.1)
                    conn.execute(
                        """
                        UPDATE preferences
                        SET value = ?, occurrence_count = ?, confidence = ?, updated_at = ?
                        WHERE key = ?
                    """,
                        (
                            json.dumps(value),
                            new_count,
                            new_confidence,
                            time.time(),
                            key,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO preferences (key, value, category, confidence, occurrence_count, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (key, json.dumps(value), category, 0.5, 1, time.time()),
                    )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def generalize(self, traces: List) -> Optional[SemanticKnowledge]:
        """Create generalization from multiple episodes"""
        if len(traces) < 2:
            return None

        # Find common patterns
        contents = [t.content for t in traces if t.content]
        if not contents:
            return None

        # Simple generalization: use most common words
        all_words = []
        for content in contents:
            words = content.lower().split()
            all_words.extend([w for w in words if len(w) > 3])

        from collections import Counter

        word_counts = Counter(all_words)
        common = word_counts.most_common(5)

        if not common:
            return None

        # Create generalized fact
        common_words = " and ".join([w for w, _ in common[:3]])
        knowledge = SemanticKnowledge(
            fact=f"Frequently occurring: {common_words}",
            entity_type="generalization",
            entities=[w for w, _ in common],
            relation="co_occurs",
            confidence=min(0.95, 0.5 + len(traces) * 0.1),
            evidence_count=len(traces),
            source_trace_ids=[t.id for t in traces],
        )

        self._store_knowledge(knowledge)
        return knowledge

    def retrieve(self, query: str, limit: int = 5) -> List[SemanticKnowledge]:
        """Retrieve relevant knowledge"""
        conn = sqlite3.connect(self.db_path)

        query_lower = query.lower()
        query_words = query_lower.split()

        cursor = conn.execute(
            """
            SELECT id, fact, entity_type, entities, relation, confidence, evidence_count, source_traces, created_at
            FROM facts
            ORDER BY confidence DESC
            LIMIT ?
        """,
            (limit * 2,),
        )

        results = []
        for row in cursor.fetchall():
            fact_lower = row[1].lower()
            relevance = sum(1 for w in query_words if w in fact_lower)

            results.append(
                {
                    "id": row[0],
                    "fact": row[1],
                    "entity_type": row[2],
                    "entities": json.loads(row[3]) if row[3] else [],
                    "relation": row[4],
                    "confidence": row[5],
                    "evidence_count": row[6],
                    "relevance": relevance,
                }
            )

        conn.close()

        # Sort by combined score
        results.sort(key=lambda x: x["confidence"] * 0.5 + x["relevance"], reverse=True)
        return results[:limit]

    def get_preferences(self, category: str = None) -> Dict:
        """Get preferences"""
        conn = sqlite3.connect(self.db_path)

        if category:
            cursor = conn.execute(
                """
                SELECT key, value, category, confidence FROM preferences WHERE category = ?
            """,
                (category,),
            )
        else:
            cursor = conn.execute("""
                SELECT key, value, category, confidence FROM preferences
            """)

        results = {}
        for row in cursor.fetchall():
            try:
                value = json.loads(row[1])
            except:
                value = row[1]
            results[row[0]] = {"value": value, "category": row[2], "confidence": row[3]}

        conn.close()
        return results

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get concept by name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT id, name, category, attributes, related_concepts, confidence, occurrence_count, created_at, last_accessed
            FROM concepts WHERE name = ?
        """,
            (name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return Concept(
                id=row[0],
                name=row[1],
                category=row[2],
                attributes=json.loads(row[3]) if row[3] else {},
                related_concepts=json.loads(row[4]) if row[4] else [],
                confidence=row[5],
                occurrence_count=row[6],
                created_at=row[7],
                last_accessed=row[8],
            )
        return None

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)

        facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        concepts_count = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        prefs_count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]

        conn.close()

        return {
            "facts": facts_count,
            "concepts": concepts_count,
            "preferences": prefs_count,
        }
