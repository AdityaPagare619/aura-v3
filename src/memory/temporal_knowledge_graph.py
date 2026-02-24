"""
Temporal Knowledge Graph - Graphiti-style Temporal Knowledge Graphs
Knowledge graph with validity intervals for temporal reasoning
"""

import sqlite3
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class FactStatus(Enum):
    """Status of a fact in the knowledge graph"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    VALIDATED = "validated"
    QUESTIONABLE = "questionable"


class RelationType(Enum):
    """Types of relations in the knowledge graph"""

    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"
    BEFORE = "before"
    AFTER = "after"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"


@dataclass
class TemporalFact:
    """A fact with temporal validity"""

    id: str
    subject: str
    predicate: str
    object: str

    valid_from: float = field(default_factory=time.time)
    valid_to: Optional[float] = None

    confidence: float = 0.5
    status: FactStatus = FactStatus.ACTIVE

    source: str = ""
    evidence: List[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """An entity in the knowledge graph"""

    id: str
    name: str
    entity_type: str

    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class TemporalQuery:
    """Query for temporal knowledge"""

    subject: str = ""
    predicate: str = ""
    object: str = ""
    valid_at: Optional[float] = None
    valid_in_range: Tuple[float, float] = None
    include_superseded: bool = False


class TemporalKnowledgeGraph:
    """
    Graphiti-style temporal knowledge graph

    Features:
    - Validity intervals for facts (valid from/to timestamps)
    - Temporal queries (what was true at time X)
    - Fact expiration and cleanup
    - Dynamic knowledge updates
    - Efficient graph traversal
    - Works offline on Android/Termux
    """

    def __init__(
        self,
        db_path: str = "data/memory/temporal_graph.db",
        default_validity_days: int = 365,
    ):
        self.db_path = db_path
        self.default_validity_days = default_validity_days

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT,
                properties TEXT,
                aliases TEXT,
                created_at REAL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from REAL,
                valid_to REAL,
                confidence REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                source TEXT,
                evidence TEXT,
                created_at REAL,
                updated_at REAL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                from_entity TEXT NOT NULL,
                to_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                valid_from REAL,
                valid_to REAL,
                properties TEXT,
                created_at REAL,
                UNIQUE(from_entity, to_entity, relation_type)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_subject 
            ON facts(subject, predicate)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_validity 
            ON facts(valid_from, valid_to)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name 
            ON entities(name)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_entities 
            ON relations(from_entity, to_entity)
        """)

        conn.commit()
        conn.close()

    def add_entity(
        self,
        name: str,
        entity_type: str = "",
        properties: Dict[str, Any] = None,
        aliases: List[str] = None,
    ) -> Entity:
        """Add an entity to the graph"""
        entity = Entity(
            id=f"ent_{uuid.uuid4().hex[:12]}",
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            aliases=aliases or [],
        )

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO entities 
                       (id, name, entity_type, properties, aliases, created_at, last_accessed, access_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entity.id,
                        entity.name,
                        entity.entity_type,
                        json.dumps(entity.properties),
                        json.dumps(entity.aliases),
                        entity.created_at,
                        entity.last_accessed,
                        entity.access_count,
                    ),
                )
        finally:
            conn.close()

        for alias in entity.aliases:
            self._add_entity_alias(entity.id, alias)

        return entity

    def _add_entity_alias(self, entity_id: str, alias: str):
        """Add alias for entity"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO entities (id, name, entity_type, created_at, last_accessed) VALUES (?, ?, 'alias', ?, ?)",
                    (entity_id, alias, time.time(), time.time()),
                )
        finally:
            conn.close()

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """SELECT id, name, entity_type, properties, aliases, created_at, last_accessed, access_count
               FROM entities WHERE name = ? AND entity_type != 'alias'""",
            (name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return Entity(
                id=row[0],
                name=row[1],
                entity_type=row[2],
                properties=json.loads(row[3]) if row[3] else {},
                aliases=json.loads(row[4]) if row[4] else [],
                created_at=row[5],
                last_accessed=row[6],
                access_count=row[7],
            )
        return None

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        valid_from: float = None,
        valid_to: float = None,
        confidence: float = 0.5,
        source: str = "",
        evidence: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> TemporalFact:
        """Add a temporal fact to the graph"""
        valid_from = valid_from or time.time()

        if valid_to is None and self.default_validity_days:
            valid_to = valid_from + (self.default_validity_days * 86400)

        fact = TemporalFact(
            id=f"fact_{uuid.uuid4().hex[:12]}",
            subject=subject,
            predicate=predicate,
            object=object,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            source=source,
            evidence=evidence or [],
            metadata=metadata or {},
        )

        self._supersede_old_facts(subject, predicate, valid_from)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """INSERT INTO facts 
                       (id, subject, predicate, object, valid_from, valid_to, confidence, 
                        status, source, evidence, created_at, updated_at, access_count, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fact.id,
                        fact.subject,
                        fact.predicate,
                        fact.object,
                        fact.valid_from,
                        fact.valid_to,
                        fact.confidence,
                        fact.status.value,
                        fact.source,
                        json.dumps(fact.evidence),
                        fact.created_at,
                        fact.updated_at,
                        fact.access_count,
                        json.dumps(fact.metadata),
                    ),
                )
        finally:
            conn.close()

        self._ensure_entity_exists(subject)
        self._ensure_entity_exists(object)

        return fact

    def _supersede_old_facts(self, subject: str, predicate: str, valid_from: float):
        """Mark old facts as superseded when new fact is added"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """UPDATE facts 
                       SET status = 'superseded', updated_at = ?
                       WHERE subject = ? AND predicate = ? 
                       AND status = 'active' AND valid_from < ?""",
                    (time.time(), subject, predicate, valid_from),
                )
        finally:
            conn.close()

    def _ensure_entity_exists(self, name: str):
        """Ensure entity exists in graph"""
        if not self.get_entity(name):
            self.add_entity(name)

    def query(
        self,
        query: TemporalQuery,
        limit: int = 50,
    ) -> List[TemporalFact]:
        """Query facts with temporal conditions"""
        conn = sqlite3.connect(self.db_path)

        conditions = ["1=1"]
        params = []

        if query.subject:
            conditions.append("subject = ?")
            params.append(query.subject)

        if query.predicate:
            conditions.append("predicate = ?")
            params.append(query.predicate)

        if query.object:
            conditions.append("object = ?")
            params.append(query.object)

        if query.valid_at:
            conditions.append("valid_from <= ?")
            params.append(query.valid_at)
            conditions.append("(valid_to IS NULL OR valid_to > ?)")
            params.append(query.valid_at)

        if query.valid_in_range:
            conditions.append("valid_from <= ?")
            params.append(query.valid_in_range[1])
            conditions.append("(valid_to IS NULL OR valid_to >= ?)")
            params.append(query.valid_in_range[0])

        if not query.include_superseded:
            conditions.append("(status = 'active' OR status = 'validated')")

        where_clause = " AND ".join(conditions)

        cursor = conn.execute(
            f"""SELECT id, subject, predicate, object, valid_from, valid_to, confidence,
                       status, source, evidence, created_at, updated_at, access_count, metadata
                FROM facts 
                WHERE {where_clause}
                ORDER BY confidence DESC, valid_from DESC
                LIMIT ?""",
            params + [limit],
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                TemporalFact(
                    id=row[0],
                    subject=row[1],
                    predicate=row[2],
                    object=row[3],
                    valid_from=row[4],
                    valid_to=row[5],
                    confidence=row[6],
                    status=FactStatus(row[7]),
                    source=row[8],
                    evidence=json.loads(row[9]) if row[9] else [],
                    created_at=row[10],
                    updated_at=row[11],
                    access_count=row[12],
                    metadata=json.loads(row[13]) if row[13] else {},
                )
            )

        conn.close()
        return results

    def what_was_true_at(
        self,
        subject: str,
        timestamp: float = None,
    ) -> List[TemporalFact]:
        """Get all facts that were true for a subject at a given time"""
        timestamp = timestamp or time.time()

        query = TemporalQuery(
            subject=subject,
            valid_at=timestamp,
            include_superseded=True,
        )

        return self.query(query)

    def get_facts_about(
        self,
        subject: str,
        include_history: bool = True,
    ) -> Dict[str, List[TemporalFact]]:
        """Get all facts about a subject grouped by predicate"""
        facts = self.query(
            TemporalQuery(subject=subject, include_superseded=include_history),
            limit=200,
        )

        grouped = defaultdict(list)
        for fact in facts:
            grouped[fact.predicate].append(fact)

        return dict(grouped)

    def update_fact(
        self,
        fact_id: str,
        new_object: str,
        confidence: float = None,
    ) -> Optional[TemporalFact]:
        """Update a fact (creates new version)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """SELECT subject, predicate FROM facts WHERE id = ?""", (fact_id,)
        )
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        subject, predicate = row[0], row[1]

        new_fact = self.add_fact(
            subject=subject,
            predicate=predicate,
            object=new_object,
            confidence=confidence or 0.5,
        )

        conn.close()
        return new_fact

    def expire_fact(self, fact_id: str):
        """Manually expire a fact"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """UPDATE facts SET status = 'expired', updated_at = ? WHERE id = ?""",
                    (time.time(), fact_id),
                )
        finally:
            conn.close()

    def validate_fact(self, fact_id: str):
        """Mark fact as validated"""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """UPDATE facts SET status = 'validated', updated_at = ? WHERE id = ?""",
                    (time.time(), fact_id),
                )
        finally:
            conn.close()

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: RelationType,
        valid_from: float = None,
        valid_to: float = None,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Add a relation between entities"""
        relation_id = f"rel_{uuid.uuid4().hex[:12]}"

        self._ensure_entity_exists(from_entity)
        self._ensure_entity_exists(to_entity)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO relations
                       (id, from_entity, to_entity, relation_type, valid_from, valid_to, properties, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        relation_id,
                        from_entity,
                        to_entity,
                        relation_type.value,
                        valid_from or time.time(),
                        valid_to,
                        json.dumps(properties or {}),
                        time.time(),
                    ),
                )
        finally:
            conn.close()

        return relation_id

    def get_relations(
        self,
        entity: str = None,
        relation_type: RelationType = None,
    ) -> List[Dict[str, Any]]:
        """Get relations for an entity"""
        conn = sqlite3.connect(self.db_path)

        conditions = []
        params = []

        if entity:
            conditions.append("(from_entity = ? OR to_entity = ?)")
            params.extend([entity, entity])

        if relation_type:
            conditions.append("relation_type = ?")
            params.append(relation_type.value)

        where = " AND ".join(conditions) if conditions else "1=1"

        cursor = conn.execute(
            f"""SELECT id, from_entity, to_entity, relation_type, valid_from, valid_to, properties, created_at
                FROM relations WHERE {where}""",
            params,
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "type": row[3],
                    "valid_from": row[4],
                    "valid_to": row[5],
                    "properties": json.loads(row[6]) if row[6] else {},
                    "created_at": row[7],
                }
            )

        conn.close()
        return results

    def cleanup_expired(self, older_than_days: int = 30) -> int:
        """Clean up expired facts"""
        cutoff = time.time() - (older_than_days * 86400)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM facts WHERE valid_to IS NOT NULL AND valid_to < ? AND status IN ('expired', 'superseded')",
            (cutoff,),
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} expired facts")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        conn = sqlite3.connect(self.db_path)

        entities_count = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE entity_type != 'alias'"
        ).fetchone()[0]

        facts_total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        facts_active = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE status = 'active'"
        ).fetchone()[0]
        facts_validated = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE status = 'validated'"
        ).fetchone()[0]

        relations_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

        cursor = conn.execute("""
            SELECT predicate, COUNT(*) as cnt 
            FROM facts 
            GROUP BY predicate 
            ORDER BY cnt DESC 
            LIMIT 10
        """)
        top_predicates = [{"predicate": r[0], "count": r[1]} for r in cursor.fetchall()]

        conn.close()

        return {
            "entities": entities_count,
            "facts_total": facts_total,
            "facts_active": facts_active,
            "facts_validated": facts_validated,
            "relations": relations_count,
            "top_predicates": top_predicates,
        }


class TemporalReasoner:
    """
    Temporal reasoning over the knowledge graph
    """

    def __init__(self, graph: TemporalKnowledgeGraph):
        self.graph = graph

    def get_timeline(
        self,
        subject: str,
        predicate: str = None,
    ) -> List[Tuple[float, str, str]]:
        """Get timeline of facts for a subject"""
        facts = self.graph.query(
            TemporalQuery(subject=subject, include_superseded=True),
            limit=100,
        )

        if predicate:
            facts = [f for f in facts if f.predicate == predicate]

        timeline = [(f.valid_from, f.predicate, f.object) for f in facts]
        timeline.sort(key=lambda x: x[0])

        return timeline

    def resolve_conflicts(
        self,
        subject: str,
        predicate: str,
    ) -> Optional[TemporalFact]:
        """Resolve conflicting facts about subject and predicate"""
        facts = self.graph.query(
            TemporalQuery(subject=subject, predicate=predicate),
            limit=10,
        )

        if not facts:
            return None

        valid_facts = [
            f for f in facts if f.status in (FactStatus.ACTIVE, FactStatus.VALIDATED)
        ]

        if valid_facts:
            return max(valid_facts, key=lambda f: f.confidence)

        return facts[0]

    def get_evolution(
        self,
        subject: str,
    ) -> List[Dict[str, Any]]:
        """Get how knowledge about subject evolved over time"""
        facts = self.graph.query(
            TemporalQuery(subject=subject, include_superseded=True),
            limit=200,
        )

        grouped = defaultdict(list)
        for fact in facts:
            grouped[fact.predicate].append(fact)

        evolution = []
        for predicate, fact_list in grouped.items():
            fact_list.sort(key=lambda f: f.created_at)

            versions = []
            for fact in fact_list:
                versions.append(
                    {
                        "object": fact.object,
                        "confidence": fact.confidence,
                        "status": fact.status.value,
                        "valid_from": fact.valid_from,
                        "created_at": fact.created_at,
                    }
                )

            evolution.append(
                {
                    "predicate": predicate,
                    "versions": versions,
                }
            )

        return evolution


def get_temporal_graph(
    db_path: str = "data/memory/temporal_graph.db",
) -> TemporalKnowledgeGraph:
    """Get or create temporal knowledge graph instance"""
    return TemporalKnowledgeGraph(db_path)


__all__ = [
    "TemporalKnowledgeGraph",
    "TemporalReasoner",
    "TemporalFact",
    "TemporalQuery",
    "Entity",
    "FactStatus",
    "RelationType",
    "get_temporal_graph",
]
