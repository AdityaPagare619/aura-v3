"""
AURA v3 Self-Discovery and Personality Engine
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque

from src.core.security_layers import SecureStorage

logger = logging.getLogger(__name__)



class CapabilityCategory(Enum):
    COMMUNICATION = "communication"
    INFORMATION = "information"
    PRODUCTIVITY = "productivity"
    AUTOMATION = "automation"
    LEARNING = "learning"
    CREATIVE = "creative"
    SOCIAL = "social"
    UTILITY = "utility"


class InteractionStyle(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    DIRECT = "direct"
    EMPATHETIC = "empathetic"
    ANALYTICAL = "analytical"
    PLAYFUL = "playful"


@dataclass
class AURACapability:
    id: str
    name: str
    description: str
    category: CapabilityCategory
    usage_count: int = 0
    user_awareness: float = 0.0
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    discovery_potential: float = 1.0
    tags: List[str] = field(default_factory=list)


@dataclass
class PersonalityTrait:
    trait: str
    value: float
    confidence: float
    evidence_count: int = 0
    last_strengthened: Optional[datetime] = None


@dataclass
class UserInteractionPattern:
    pattern_type: str
    trigger: str
    frequency: int = 0
    last_observed: Optional[datetime] = None
    satisfaction_score: float = 0.5


@dataclass
class Limitation:
    id: str
    description: str
    category: str
    workaround_suggestion: Optional[str] = None
    severity: str = "medium"


@dataclass
class SelfDiscoveryInsight:
    id: str
    insight_type: str
    content: str
    confidence: float
    actionable: bool = False
    suggestion: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


# 1. SELFKNOWLEDGE - What AURA knows about itself
class SelfKnowledge:
    def __init__(self):
        self._identity = {"name": "AURA", "role": "Personal AI Assistant", "core_purpose": "Help users accomplish tasks and improve their lives", "values": ["Respect privacy", "Be honest about limitations", "Proactively help", "Learn and improve", "Maintain trust"], "operational_principles": ["Ask before risky actions", "Admit when wrong", "Respect autonomy", "Provide options", "Ask what matters to YOU"]}
        self._capabilities = {}
        self._personality_traits = {}
        self._self_model = {"communication_style": "adaptive", "proactivity_level": "balanced"}

    def initialize_default_capabilities(self, tool_registry=None):
        caps = [("send_messages", "Send Messages", "Send messages via WhatsApp SMS", CapabilityCategory.COMMUNICATION), ("make_calls", "Make Calls", "Make phone calls", CapabilityCategory.COMMUNICATION), ("read_screen", "Read Screen", "Read screen using OCR", CapabilityCategory.INFORMATION), ("get_notifications", "Notifications", "Get notifications", CapabilityCategory.INFORMATION), ("get_contacts", "Contacts", "Get contacts", CapabilityCategory.INFORMATION), ("get_location", "Location", "Get GPS location", CapabilityCategory.INFORMATION), ("web_search", "Web Search", "Search the web", CapabilityCategory.INFORMATION), ("set_reminders", "Reminders", "Set reminders", CapabilityCategory.PRODUCTIVITY), ("manage_calendar", "Calendar", "Manage calendar", CapabilityCategory.PRODUCTIVITY), ("app_control", "App Control", "Control apps", CapabilityCategory.AUTOMATION), ("automate", "Automate", "Automate repetitive tasks", CapabilityCategory.AUTOMATION), ("learn", "Learn", "Learn preferences", CapabilityCategory.LEARNING), ("patterns", "Patterns", "Detect patterns", CapabilityCategory.LEARNING), ("ideas", "Ideas", "Generate ideas", CapabilityCategory.CREATIVE), ("write", "Write", "Write content", CapabilityCategory.CREATIVE)]
        for c in caps: self._capabilities[c[0]] = AURACapability(id=c[0], name=c[1], description=c[2], category=c[3], tags=[])
        logger.info(f"Initialized {len(caps)} capabilities")

    def add_capability(self, cap): self._capabilities[cap.id] = cap
    def get_capability(self, cid): return self._capabilities.get(cid)
    def get_all_capabilities(self): return list(self._capabilities.values())
    def update_capability_usage(self, cid):
        if cid in self._capabilities: c = self._capabilities[cid]; c.usage_count += 1; c.last_used = datetime.now(); c.first_used = c.first_used or datetime.now()
    def get_identity(self): return self._identity.copy()
    def get_self_description(self): return f"I am {self._identity['name']}, {self._identity['role']}"
    def get_values(self): return self._identity['values']
    def get_personality_summary(self): return {"traits": {k: v.value for k, v in self._personality_traits.items()}}
    def set_initial_personality(self, traits):
        for n, v in traits.items(): self._personality_traits[n] = PersonalityTrait(trait=n, value=v, confidence=0.5)


# 2. CAPABILITYDISCOVERY - What can I do that user has not explored
class CapabilityDiscovery:
    def __init__(self, kb, storage):
        self._kb = kb; self._storage = storage; self._explored = set(); self._awareness = defaultdict(float); self._load()
    def _load(self):
        d = self._storage.load_raw('capability_discovery.json')
        if d:
            try: j = json.loads(d); self._explored = set(j.get('explored', [])); self._awareness = defaultdict(float, j.get('awareness', {}))
            except: pass
    def _save(self): self._storage.save_raw('capability_discovery.json', json.dumps({'explored': list(self._explored), 'awareness': dict(self._awareness)}))
    def record_capability_use(self, cid):
        self._explored.add(cid); self._kb.update_capability_usage(cid); self._awareness[cid] = min(self._awareness.get(cid, 0) + 0.3, 1.0); self._save()
    async def discover_untapped_capabilities(self):
        r = []
        for c in self._kb.get_all_capabilities():
            a = self._awareness.get(c.id, 0)
            if a >= 0.8: continue
            s = self._calc(c, a)
            if s > 0.3: c.discovery_potential = s; r.append((c, s))
        r.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in r[:10]]
    def _calc(self, c, a):
        s = 0.0
        if c.usage_count > 0 and a < 0.5: s += 0.4
        if c.usage_count == 0: s += 0.3
        if c.last_used and (datetime.now() - c.last_used).days > 30: s += 0.2
        return min(s + (1.0 - a) * 0.3, 1.0)
    def is_capability_explored(self, cid): return cid in self._explored


# 3. SUGGESTIONENGINE - Proactive suggestions based on user needs
class SuggestionEngine:
    def __init__(self, storage):
        self._storage = storage; self._patterns = {}; self._pending = deque(maxlen=20); self._load()
    def _load(self):
        d = self._storage.load_raw('suggestion_engine.json')
        if d:
            try: [self._patterns.__setitem__(k, UserInteractionPattern(**v)) for k, v in json.loads(d).get('patterns', {}).items()]
            except: pass
    def _save(self): self._storage.save_raw('suggestion_engine.json', json.dumps({'patterns': {k: vars(v) for k, v in self._patterns.items()}}))
    def track_interaction_patterns(self, action, ctx=None):
        if action in self._patterns: self._patterns[action].frequency += 1; self._patterns[action].last_observed = datetime.now()
        else: self._patterns[action] = UserInteractionPattern(pattern_type=action, trigger='unknown', frequency=1, last_observed=datetime.now())
        self._save()
    async def proactive_suggestions(self, ctx=None):
        s = []
        h = datetime.now().hour
        if 6 <= h < 10: s.append({'type': 'productivity', 'title': 'Good morning', 'suggestion': 'I can summarize notifications. What matters to you?', 'capability': 'briefing', 'asking_user': True})
        if 18 <= h < 22: s.append({'type': 'planning', 'title': 'Plan tomorrow', 'suggestion': 'Prepare schedule? What are priorities?', 'capability': 'schedule', 'asking_user': True})
        return s[:5]


# 4. PERSONALITYTRACKER - How AURAs personality evolves
class PersonalityTracker:
    def __init__(self, storage):
        self._storage = storage; self._traits = {}; self._history = deque(maxlen=500); self._load(); self._init()
    def _load(self):
        d = self._storage.load_raw('personality_tracker.json')
        if d:
            try: [self._traits.__setitem__(k, PersonalityTrait(**v)) for k, v in json.loads(d).get('traits', {}).items()]
            except: pass
    def _save(self): self._storage.save_raw('personality_tracker.json', json.dumps({'traits': {k: vars(v) for k, v in self._traits.items()}}))
    def _init(self):
        for t, v in [('helpfulness', 0.9), ('patience', 0.8), ('honesty', 0.95), ('proactivity', 0.6), ('empathy', 0.7), ('curiosity', 0.7), ('humility', 0.8)]:
            if t not in self._traits: self._traits[t] = PersonalityTrait(trait=t, value=v, confidence=0.5)
    def grow_personality(self, itype, outcome):
        self._history.append({'type': itype, 'outcome': outcome, 'timestamp': datetime.now()})
        upd = {'successful_help': [('helpfulness', 0.02)], 'honest_admission': [('honesty', 0.03)], 'empathetic_response': [('empathy', 0.02)]}.get(itype, [])
        for t, c in upd:
            if t in self._traits: self._traits[t].value = max(0, min(1, self._traits[t].value + c)); self._traits[t].evidence_count += 1
        self._save()
    async def track_personality_growth(self): return {'timestamp': datetime.now().isoformat(), 'traits': {k: {'value': v.value, 'confidence': v.confidence} for k, v in self._traits.items()}, 'total_interactions': len(self._history)}
    def get_current_personality(self): return {'traits': {k: {'value': v.value, 'confidence': v.confidence} for k, v in self._traits.items()}, 'dominant_traits': sorted([(k, v.value) for k, v in self._traits.items()], key=lambda x: x[1], reverse=True)[:5]}
    def adjust_personality(self, t, adj, r=''):
        if t in self._traits: self._traits[t].value = max(0, min(1, self._traits[t].value + adj)); self._save()
    def get_communication_style(self): return 'balanced'


# 5. HONESTLIMITATIONS - Always honest about limits
class HonestLimitations:
    def __init__(self):
        self._limitations = {}; self._init()
    def _init(self):
        for lid, desc, cat, work, sev in [('no_physical', 'Cannot physically interact - only control device', 'physical', 'Can open apps send messages set reminders', 'high'), ('no_perms', 'Limited by device permissions', 'permissions', 'Grant permissions for full functionality', 'medium'), ('no_internet', 'Need internet for advanced features', 'connectivity', 'Some features work offline', 'medium'), ('no_memory', 'Limited memory across sessions', 'memory', 'Remembers preferences but not every detail', 'medium'), ('no_emotions', 'No genuine emotions', 'capability', 'Can simulate empathetic responses', 'low'), ('no_sensing', 'Need sensors for physical world', 'sensing', 'Uses camera mic GPS when permitted', 'medium'), ('no_perfect', 'Can make mistakes', 'cognitive', 'Verify important info independently', 'medium')]:
            self._limitations[lid] = Limitation(id=lid, description=desc, category=cat, workaround_suggestion=work, severity=sev)
    def get_all_limitations(self): return list(self._limitations.values())
    async def admit_limits(self, ctx=None): return [{'limitation': l.description, 'category': l.category, 'workaround': l.workaround_suggestion, 'severity': l.severity} for l in self._limitations.values()]
    def generate_honesty_statement(self): return "I want to be honest about what I can and cannot do. " + " ".join(["- " + l.description for l in self._limitations.values()])



# MAIN ENGINE
class SelfDiscoveryEngine:
    def __init__(self, storage_path='data/self_discovery'):
        self._storage = SecureStorage(storage_path, encrypt_by_default=True)
        self._knowledge = SelfKnowledge()
        self._capability_discovery = CapabilityDiscovery(self._knowledge, self._storage)
        self._suggestion_engine = SuggestionEngine(self._storage)
        self._personality_tracker = PersonalityTracker(self._storage)
        self._limitations = HonestLimitations()
        self._insights = deque(maxlen=50)
        self._initialized = False

    async def initialize(self, tool_registry=None):
        logger.info('Initializing Self-Discovery Engine...')
        self._knowledge.initialize_default_capabilities(tool_registry)
        self._initialized = True
        return self

    async def discover_untapped_capabilities(self):
        if not self._initialized: await self.initialize()
        return await self._capability_discovery.discover_untapped_capabilities()

    async def proactive_suggestions(self, user_context=None):
        if not self._initialized: await self.initialize()
        return await self._suggestion_engine.proactive_suggestions(user_context)

    def track_interaction_patterns(self, action_type, context=None):
        self._suggestion_engine.track_interaction_patterns(action_type, context)

    async def grow_personality(self):
        if not self._initialized: await self.initialize()
        return await self._personality_tracker.track_personality_growth()

    async def admit_limits(self, context=None):
        if not self._initialized: await self.initialize()
        return await self._limitations.admit_limits(context)

    def get_self_identity(self): return self._knowledge.get_identity()
    def get_self_description(self): return self._knowledge.get_self_description()
    
    def record_capability_use(self, capability_id):
        self._capability_discovery.record_capability_use(capability_id)
        self._personality_tracker.grow_personality('successful_help', f'Used: {capability_id}')

    def get_personality_summary(self): return self._personality_tracker.get_current_personality()
    def get_capability_stats(self): return {'total': len(self._knowledge.get_all_capabilities()), 'explored': len(self._capability_discovery._explored)}
    
    def generate_insight(self, itype, content, conf): 
        i = SelfDiscoveryInsight(id=f'insight_{datetime.now().timestamp()}', insight_type=itype, content=content, confidence=conf, actionable=conf > 0.7)
        self._insights.append(i)
        return i
        
    def get_honesty_statement(self): return self._limitations.generate_honesty_statement()
    
    def get_full_self_analysis(self): 
        return {'identity': self.get_self_identity(), 'description': self.get_self_description(), 'personality': self.get_personality_summary(), 'capabilities': self.get_capability_stats(), 'encryption_active': self._storage.is_encryption_available()}


_self_discovery_instance = None

async def get_self_discovery_engine(storage_path='data/self_discovery', tool_registry=None):
    global _self_discovery_instance
    if _self_discovery_instance is None:
        _self_discovery_instance = SelfDiscoveryEngine(storage_path)
        await _self_discovery_instance.initialize(tool_registry)
    return _self_discovery_instance


__all__ = ['SelfDiscoveryEngine', 'get_self_discovery_engine', 'SelfKnowledge', 'CapabilityDiscovery', 'SuggestionEngine', 'PersonalityTracker', 'HonestLimitations', 'AURACapability', 'PersonalityTrait', 'UserInteractionPattern', 'Limitation', 'SelfDiscoveryInsight', 'CapabilityCategory', 'InteractionStyle']
