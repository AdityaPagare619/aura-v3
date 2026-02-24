"""
Unified Memory Retrieval
Retrieves from all memory systems with phased search
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MemorySource(Enum):
    """Memory source types"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SKILL = "skill"
    ANCESTOR = "ancestor"


@dataclass
class MemoryQuery:
    """Query for memory retrieval"""

    query: str
    embedding: List[float] = None
    urgent: bool = False  # Working memory only
    procedural: bool = False  # Skill memory
    limit: int = 5
    min_importance: float = 0.0
    memory_types: List[str] = None  # Filter by type
    context: Dict = field(default_factory=dict)


@dataclass
class MemoryResult:
    """Single memory result"""

    id: str
    content: str
    source: MemorySource
    importance: float = 0.5
    relevance: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemoryResponse:
    """Response from memory retrieval"""

    results: List[MemoryResult]
    source: MemorySource  # Primary source
    confidence: float = 0.0
    total_found: int = 0


class UnifiedMemoryRetrieval:
    """
    Unified retrieval from all memory systems

    Phase order:
    1. Working memory (instant)
    2. Episodic memory (fast)
    3. Semantic memory (medium)
    4. Skill memory (if procedural)
    5. Ancestor memory (lazy, if needed)
    """

    def __init__(
        self,
        working_memory=None,
        episodic_memory=None,
        semantic_memory=None,
        skill_memory=None,
        ancestor_memory=None,
    ):
        self.working = working_memory
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.skill = skill_memory
        self.ancestor = ancestor_memory

    async def retrieve(self, query: MemoryQuery) -> MemoryResponse:
        """Unified retrieval from all memory systems"""

        all_results = []
        primary_source = None

        # Phase 1: Working memory (instant, if urgent)
        if query.urgent and self.working:
            working_results = self._search_working(query)
            if working_results:
                all_results.extend(working_results)
                primary_source = MemorySource.WORKING

                if working_results:
                    return MemoryResponse(
                        results=working_results,
                        source=MemorySource.WORKING,
                        confidence=0.9,
                        total_found=len(working_results),
                    )

        # Phase 2: Episodic memory (fast)
        episodic_results = []
        if self.episodic:
            episodic_results = await self._search_episodic_async(query)
            all_results.extend(episodic_results)
            if episodic_results and not primary_source:
                primary_source = MemorySource.EPISODIC

        # Phase 3: Semantic memory (medium)
        semantic_results = []
        if self.semantic:
            semantic_results = await self._search_semantic_async(query)
            all_results.extend(semantic_results)
            if semantic_results and not primary_source:
                primary_source = MemorySource.SEMANTIC

        # Phase 4: Skill memory (if procedural)
        skill_results = []
        if query.procedural and self.skill:
            skill_results = await self._search_skill_async(query)
            all_results.extend(skill_results)
            if skill_results and not primary_source:
                primary_source = MemorySource.SKILL

        # Phase 5: Ancestor memory (lazy, if needed)
        ancestor_results = []
        if not episodic_results and not semantic_results and self.ancestor:
            ancestor_results = await self._search_ancestor_async(query)
            all_results.extend(ancestor_results)
            if ancestor_results and not primary_source:
                primary_source = MemorySource.ANCESTOR

        # Fuse results
        fused = self._fuse_results(all_results, query)

        return MemoryResponse(
            results=fused[: query.limit],
            source=primary_source or MemorySource.EPISODIC,
            confidence=self._calculate_confidence(fused, query),
            total_found=len(fused),
        )

    def _search_working(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search working memory"""
        if not self.working:
            return []

        results = []

        # Handle both list and object with get_all method
        working_items = []
        if hasattr(self.working, "get_all"):
            working_items = self.working.get_all()
        elif isinstance(self.working, list):
            working_items = self.working
        else:
            # Try to iterate
            try:
                working_items = list(self.working)
            except:
                return []

        for item in working_items:
            content = str(item.content)
            relevance = self._calculate_relevance(content, query.query)

            if relevance > 0:
                results.append(
                    MemoryResult(
                        id=item.id,
                        content=content,
                        source=MemorySource.WORKING,
                        importance=item.importance,
                        relevance=relevance,
                        metadata=item.metadata,
                    )
                )

        return sorted(results, key=lambda x: x.importance, reverse=True)

    async def _search_episodic_async(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search episodic memory"""
        if not self.episodic:
            return []

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        traces = await loop.run_in_executor(
            None,
            lambda: self.episodic.retrieve(query.embedding or [], top_k=query.limit),
        )

        results = []
        for trace in traces:
            relevance = self._calculate_relevance(trace.content, query.query)

            results.append(
                MemoryResult(
                    id=trace.id,
                    content=trace.content,
                    source=MemorySource.EPISODIC,
                    importance=trace.importance,
                    relevance=relevance,
                    metadata={"context": trace.context, "emotional": trace.emotional},
                )
            )

        return results

    async def _search_semantic_async(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search semantic memory"""
        if not self.semantic:
            return []

        loop = asyncio.get_event_loop()
        knowledge = await loop.run_in_executor(
            None, lambda: self.semantic.retrieve(query.query, limit=query.limit)
        )

        results = []
        for item in knowledge:
            results.append(
                MemoryResult(
                    id=item.get("id", ""),
                    content=item.get("fact", ""),
                    source=MemorySource.SEMANTIC,
                    importance=item.get("confidence", 0.5),
                    relevance=item.get("relevance", 0),
                    metadata={"entity_type": item.get("entity_type")},
                )
            )

        return results

    async def _search_skill_async(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search skill memory"""
        if not self.skill:
            return []

        try:
            skills = self.skill.list_skills()
        except Exception as e:
            logger.warning(f"Failed to list skills: {e}")
            return []

        results = []
        for skill in skills:
            relevance = self._calculate_relevance(skill.name, query.query)

            if relevance > 0:
                results.append(
                    MemoryResult(
                        id=skill.id,
                        content=f"Skill: {skill.name} - {skill.description}",
                        source=MemorySource.SKILL,
                        importance=skill.success_rate,
                        relevance=relevance,
                        metadata={
                            "success_rate": skill.success_rate,
                            "attempts": skill.attempts,
                        },
                    )
                )

        return sorted(results, key=lambda x: x.importance, reverse=True)[: query.limit]

    async def _search_ancestor_async(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search ancestor memory"""
        if not self.ancestor:
            return []

        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(
            None,
            lambda: self.ancestor.search(
                query.query,
                memory_type=query.memory_types[0] if query.memory_types else None,
                min_importance=query.min_importance,
                limit=query.limit,
            ),
        )

        results = []
        for memory in memories:
            results.append(
                MemoryResult(
                    id=memory.id,
                    content=memory.summary,
                    source=MemorySource.ANCESTOR,
                    importance=memory.importance,
                    relevance=getattr(memory, "_relevance", 0),
                    metadata={"tags": memory.tags},
                )
            )

        return results

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate simple relevance score"""
        if not content or not query:
            return 0.0

        content_lower = content.lower()
        query_lower = query.lower()

        # Word overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if not query_words:
            return 0.0

        overlap = len(query_words & content_words)
        return overlap / len(query_words)

    def _fuse_results(
        self, results: List[MemoryResult], query: MemoryQuery
    ) -> List[MemoryResult]:
        """Fuse results from multiple sources"""

        # Score and sort
        for result in results:
            # Combined score
            result.relevance = (
                result.importance * 0.5
                + result.relevance * 0.3
                + (1.0 if result.source == MemorySource.WORKING else 0.5) * 0.2
            )

        # Sort by combined score
        return sorted(results, key=lambda x: x.relevance, reverse=True)

    def _calculate_confidence(
        self, results: List[MemoryResult], query: MemoryQuery
    ) -> float:
        """Calculate confidence in results"""

        if not results:
            return 0.0

        # Higher confidence with more relevant results
        avg_relevance = sum(r.relevance for r in results) / len(results)

        # Boost for working memory (instant recall)
        if any(r.source == MemorySource.WORKING for r in results):
            avg_relevance += 0.2

        return min(1.0, avg_relevance)

    def get_all_stats(self) -> Dict:
        """Get stats from all memory systems"""
        stats = {}

        if self.working:
            stats["working"] = {"count": len(self.working.items)}

        if self.episodic:
            stats["episodic"] = self.episodic.get_stats()

        if self.semantic:
            stats["semantic"] = self.semantic.get_stats()

        if self.skill:
            stats["skill"] = self.skill.get_stats()

        if self.ancestor:
            stats["ancestor"] = self.ancestor.get_stats()

        return stats
