"""
AURA v3 Memory Package
Biologically inspired memory systems with advanced persistence
"""

from src.memory.neural_memory import (
    NeuralMemory,
    Neuron,
    Synapse,
    MemoryCluster,
    MemoryType,
    MemoryStrength,
    get_neural_memory,
)

from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory
from src.memory.memory_retrieval import UnifiedMemoryRetrieval
from src.memory.importance_scorer import ImportanceScorer

from src.memory.knowledge_graph import (
    KnowledgeGraph,
    AppNode,
    RelationshipEdge,
    RelationshipType,
    ValidityInterval,
    TopologyMapper,
    QueryEngine,
    KnowledgeGraphError,
    get_knowledge_graph,
    get_topology_mapper,
    get_query_engine,
    initialize_from_app_discovery,
)

from src.memory.sqlite_state_machine import (
    SQLiteStateMachine,
    MemoryStateMachine,
    State,
    Transition,
    StateSnapshot,
    TransitionType,
    get_state_machine,
    get_memory_state_machine,
)

from src.memory.local_vector_store import (
    LocalVectorStore,
    LocalEmbeddingGenerator,
    QuantizedVectorStore,
    VectorEntry,
    SearchResult,
    get_vector_store,
)

from src.memory.temporal_knowledge_graph import (
    TemporalKnowledgeGraph,
    TemporalReasoner,
    TemporalFact,
    TemporalQuery,
    Entity,
    FactStatus,
    RelationType,
    get_temporal_graph,
)

from src.memory.memory_optimizer import (
    MemoryOptimizer,
    MemoryArchiver,
    MemoryCompressor,
    MemoryCleanup,
    SmartMemoryManager,
    CompressionResult,
    MemoryMetrics,
    get_memory_optimizer,
)

__all__ = [
    # Neural Memory
    "NeuralMemory",
    "Neuron",
    "Synapse",
    "MemoryCluster",
    "MemoryType",
    "MemoryStrength",
    "get_neural_memory",
    # Other Memory Systems
    "EpisodicMemory",
    "SemanticMemory",
    "UnifiedMemoryRetrieval",
    "ImportanceScorer",
    # Knowledge Graph
    "KnowledgeGraph",
    "AppNode",
    "RelationshipEdge",
    "RelationshipType",
    "ValidityInterval",
    "TopologyMapper",
    "QueryEngine",
    "KnowledgeGraphError",
    "get_knowledge_graph",
    "get_topology_mapper",
    "get_query_engine",
    "initialize_from_app_discovery",
    # SQLite State Machine
    "SQLiteStateMachine",
    "MemoryStateMachine",
    "State",
    "Transition",
    "StateSnapshot",
    "TransitionType",
    "get_state_machine",
    "get_memory_state_machine",
    # Local Vector Store
    "LocalVectorStore",
    "LocalEmbeddingGenerator",
    "QuantizedVectorStore",
    "VectorEntry",
    "SearchResult",
    "get_vector_store",
    # Temporal Knowledge Graph
    "TemporalKnowledgeGraph",
    "TemporalReasoner",
    "TemporalFact",
    "TemporalQuery",
    "Entity",
    "FactStatus",
    "RelationType",
    "get_temporal_graph",
    # Memory Optimizer
    "MemoryOptimizer",
    "MemoryArchiver",
    "MemoryCompressor",
    "MemoryCleanup",
    "SmartMemoryManager",
    "CompressionResult",
    "MemoryMetrics",
    "get_memory_optimizer",
]
