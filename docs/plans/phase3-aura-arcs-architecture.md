# AURA v3 Phase 3: Arcs/Sub-Agents Architecture

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Version:** 1.0  
**Date:** February 2026  
**Status:** Architecture Design Document  
**Prerequisite:** Phase 2 (Refinement & Resurrection) must be complete

---

## Executive Summary

Phase 3 transforms AURA from a reactive assistant into an **autonomous background intelligence** that works continuously for the user. The core innovation is the **Arc System** — a collection of specialized sub-agents that operate independently, self-manage their sleep/wake states, and coordinate through AURA Core.

### Key Innovations

1. **Independent Arcs**: 5 specialized sub-agents, each with distinct responsibilities
2. **Smart Self-Management**: Arcs decide autonomously when to sleep/wake based on context
3. **RAM-Optimized Design**: Only active arcs loaded into memory (8GB constraint)
4. **Boss Detection**: AURA recognizes when the user is present and adapts behavior
5. **Toggle Visibility**: User can watch AURA work or let it work invisibly

### Why This Matters

Current AURA is reactive — it waits for user input. Phase 3 AURA is **proactive at scale** — multiple arcs work in parallel, researching, preparing, and optimizing while the user focuses on life. When the user returns, AURA presents curated results: "While you were gaming, I found 3 outfit options for your wedding, rescheduled your meeting conflict, and noticed your sleep has been poor — here's a suggestion."

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AURA v3 ARCS SYSTEM                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         AURA CORE (Orchestrator)                       │  │
│  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │ Personality  │  │   Memory    │  │  Boss Det.  │  │  Arc Sched.  │  │  │
│  │  │   Engine     │  │   Router    │  │   System    │  │   Manager    │  │  │
│  │  └──────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                          ┌─────────▼─────────┐                               │
│                          │    ARC MESSAGE    │                               │
│                          │       BUS         │                               │
│                          └─────────┬─────────┘                               │
│         ┌──────────────────────────┼──────────────────────────┐              │
│         │              │           │           │              │              │
│         ▼              ▼           ▼           ▼              ▼              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│  │   HEALTH   │ │   SOCIAL   │ │    LIFE    │ │  LEARNING  │ │  RESEARCH  │ │
│  │    ARC     │ │    ARC     │ │    ARC     │ │    ARC     │ │    ARC     │ │
│  │            │ │            │ │            │ │            │ │            │ │
│  │ - Fitness  │ │ - Messages │ │ - Calendar │ │ - Study    │ │ - Web      │ │
│  │ - Sleep    │ │ - Social   │ │ - Tasks    │ │ - Memory   │ │ - Facts    │ │
│  │ - Diet     │ │ - Contacts │ │ - Routines │ │ - Skills   │ │ - Trends   │ │
│  │ - Wellness │ │ - Events   │ │ - Planning │ │ - Habits   │ │ - Analysis │ │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ │
│        │              │              │              │              │         │
│        └──────────────┴──────────────┴──────────────┴──────────────┘         │
│                                    │                                          │
│                          ┌─────────▼─────────┐                               │
│                          │   SHARED STATE    │                               │
│                          │   (SQLite + RAM)  │                               │
│                          └───────────────────┘                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The Five Arcs

Each Arc is a self-contained agent with:
- Its own **tick loop** (heartbeat)
- Its own **state machine** (sleeping/waking/active/hibernating)
- Its own **priority queue** for tasks
- Its own **memory partition** (scoped to its domain)

### Arc 1: Health Arc

**Responsibility:** Physical and mental wellness tracking, proactive health suggestions.

| Capability | Description |
|------------|-------------|
| Fitness Tracking | Monitor step counts, workouts, activity patterns |
| Sleep Analysis | Track sleep duration/quality, suggest improvements |
| Diet Monitoring | Track meals, suggest balanced nutrition |
| Health Reminders | Medication reminders, hydration, posture breaks |
| Wellness Insights | Correlate activity with mood, energy levels |

**Wake Triggers:**
- Morning (start-of-day routine)
- After meals (diet logging window)
- Pre-sleep (sleep prep suggestions)
- Activity detection (workout started)
- Low activity alert (sedentary too long)

**Sleep Triggers:**
- User in meeting
- Late night (outside sleep-tracking window)
- User gaming/entertainment (non-health context)

### Arc 2: Social Arc

**Responsibility:** Relationship management, communication assistance, social event awareness.

| Capability | Description |
|------------|-------------|
| Message Monitoring | Track unread messages, prioritize important contacts |
| Social Media | Analyze engagement, suggest content (with permission) |
| Contact Management | Track relationship health, remind to connect |
| Event Awareness | Birthdays, anniversaries, social gatherings |
| Communication Style | Adapt to context (professional vs casual) |

**Wake Triggers:**
- Unread message from VIP contact
- Upcoming birthday/anniversary (3 days ahead)
- Social event approaching
- User opens messaging app
- Long silence from close contact

**Sleep Triggers:**
- User in focus mode/DND
- User sleeping
- No pending social tasks
- User in professional meeting

### Arc 3: Life Arc

**Responsibility:** Daily planning, calendar management, task orchestration, routine optimization.

| Capability | Description |
|------------|-------------|
| Calendar Management | Schedule conflicts, meeting prep, reminders |
| Task Tracking | Todo lists, deadlines, progress tracking |
| Daily Planning | Morning brief, day structure, priorities |
| Routine Optimization | Learn patterns, suggest improvements |
| Conflict Resolution | Detect and resolve scheduling conflicts |

**Wake Triggers:**
- Start of day (morning brief)
- 30 min before meetings
- Task deadline approaching
- Calendar conflict detected
- User asks about schedule

**Sleep Triggers:**
- End of work hours (unless urgent tasks)
- User sleeping
- Weekend (reduced activity unless events)

### Arc 4: Learning Arc

**Responsibility:** Knowledge management, study support, skill development, habit formation.

| Capability | Description |
|------------|-------------|
| Study Reminders | Spaced repetition, review scheduling |
| Knowledge Capture | Save insights, organize notes |
| Skill Tracking | Progress on learning goals |
| Habit Formation | Track streaks, reinforce behaviors |
| Content Curation | Relevant articles, videos for interests |

**Wake Triggers:**
- Scheduled study time
- Knowledge gap detected in conversation
- Habit check-in time
- Content matching user interests found
- User learning new skill

**Sleep Triggers:**
- User in entertainment mode
- User sleeping
- No active learning goals

### Arc 5: Research Arc (Background)

**Responsibility:** Autonomous research, fact-finding, trend analysis, preparation.

| Capability | Description |
|------------|-------------|
| Topic Research | Deep dives on user interests |
| Fact Verification | Check claims, gather evidence |
| Trend Analysis | Market, tech, personal interest trends |
| Purchase Research | Compare products user is considering |
| Event Preparation | Research for upcoming meetings/events |

**Wake Triggers:**
- User saves item for later
- Upcoming important event
- Research request queued
- New topic of interest detected
- Price drop on tracked item

**Sleep Triggers:**
- High device temperature
- Low battery
- User actively engaged
- No pending research tasks

---

## AURA Core (Orchestrator)

AURA Core is not just another Arc — it's the **central intelligence** that:

1. **Routes requests** to appropriate Arcs
2. **Manages Arc lifecycle** (load/unload from RAM)
3. **Maintains personality** across all Arc responses
4. **Handles user interaction** directly
5. **Synthesizes reports** from multiple Arcs
6. **Detects boss presence** and coordinates pauses

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AURA CORE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐ │
│  │  Personality  │   │    Memory     │   │     Boss Detection    │ │
│  │    Engine     │   │    Router     │   │       System          │ │
│  │               │   │               │   │                       │ │
│  │ - Mood state  │   │ - Arc routing │   │ - Screen state        │ │
│  │ - Tone adapt  │   │ - Cross-arc   │   │ - App foreground      │ │
│  │ - User prefs  │   │   queries     │   │ - User input activity │ │
│  │ - Boundaries  │   │ - Conflict    │   │ - Presence timeout    │ │
│  │               │   │   resolution  │   │                       │ │
│  └───────────────┘   └───────────────┘   └───────────────────────┘ │
│                                                                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐ │
│  │     Arc       │   │   Response    │   │     User Session      │ │
│  │   Scheduler   │   │  Synthesizer  │   │      Manager          │ │
│  │               │   │               │   │                       │ │
│  │ - Priority Q  │   │ - Multi-arc   │   │ - Active convo        │ │
│  │ - RAM budget  │   │   combine     │   │ - Context preserve    │ │
│  │ - Load/unload │   │ - Personality │   │ - Interruption        │ │
│  │ - Wake/sleep  │   │   injection   │   │   handling            │ │
│  │               │   │               │   │                       │ │
│  └───────────────┘   └───────────────┘   └───────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## RAM Management Strategy

### The 8GB Constraint

Mobile devices with 8GB RAM must also run:
- Android OS (~2-3 GB)
- Other apps (~2-3 GB)
- System services (~500 MB - 1 GB)

**AURA budget: ~1.5-2 GB maximum**

### Arc Memory Footprint (Estimated)

| Component | Loaded (MB) | Sleeping (MB) |
|-----------|-------------|---------------|
| AURA Core | 150-200 | N/A (always loaded) |
| Health Arc | 80-120 | 5 (state only) |
| Social Arc | 100-150 | 10 (contact cache) |
| Life Arc | 80-100 | 5 (next event only) |
| Learning Arc | 60-80 | 5 (state only) |
| Research Arc | 100-150 | 5 (state only) |
| LLM (Shared) | 500-800 | N/A (always loaded) |
| Memory System | 100-200 | N/A (lazy load) |

### Loading Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                    ARC LOADING PRIORITY                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  TIER 0 - Always Loaded (~800 MB)                              │
│  ├── AURA Core (orchestration, personality)                   │
│  ├── LLM Engine (shared inference)                            │
│  └── Memory Router (cross-arc access)                         │
│                                                                │
│  TIER 1 - Context-Loaded (~200-400 MB max at once)            │
│  ├── Life Arc (loaded during work hours)                      │
│  ├── Health Arc (loaded during active/morning/evening)        │
│  └── Social Arc (loaded when messages pending)                │
│                                                                │
│  TIER 2 - On-Demand (~100-150 MB each)                        │
│  ├── Learning Arc (loaded during study sessions)              │
│  └── Research Arc (loaded when idle/charging)                 │
│                                                                │
│  RULE: Maximum 2 TIER 1 + 1 TIER 2 arcs loaded simultaneously │
│  RULE: Total AURA footprint never exceeds 1.5 GB              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Unloading Algorithm

```python
def decide_unload(arc: Arc, context: SystemContext) -> bool:
    """
    Decide if an Arc should be unloaded from RAM.
    
    Factors:
    1. Time since last activity (idle > 5 min)
    2. No pending tasks in priority queue
    3. RAM pressure from system
    4. Not in wake trigger state
    5. User not actively using Arc's domain
    """
    
    # Never unload if actively processing
    if arc.state == ArcState.ACTIVE:
        return False
    
    # Calculate unload score (higher = more likely to unload)
    score = 0.0
    
    # Idle time factor (max 0.3)
    idle_minutes = (now() - arc.last_activity).minutes
    score += min(0.3, idle_minutes / 20)
    
    # Empty queue factor (0.2)
    if arc.priority_queue.is_empty():
        score += 0.2
    
    # RAM pressure factor (max 0.3)
    if context.ram_pressure > 0.8:
        score += 0.3 * context.ram_pressure
    
    # No upcoming triggers (0.2)
    if not arc.has_upcoming_triggers(window_minutes=15):
        score += 0.2
    
    return score > 0.6
```

---

## Arc State Machine

Each Arc operates as an independent state machine:

```
                        ┌──────────────────┐
                        │                  │
                        ▼                  │
┌─────────┐  wake   ┌─────────┐  work   ┌─────────┐
│ ASLEEP  │────────▶│ WAKING  │────────▶│ ACTIVE  │
│         │         │         │         │         │
│ (5 MB)  │         │ (50 MB) │         │(100 MB) │
└────┬────┘         └─────────┘         └────┬────┘
     │                                       │
     │                                       │ idle
     │                                       ▼
     │              ┌─────────┐         ┌─────────┐
     │◀─────────────│ SLEEPING│◀────────│ DROWSY  │
     │    timeout   │         │  quiet  │         │
     │              │ (10 MB) │         │ (50 MB) │
     │              └─────────┘         └─────────┘
     │
     │  system pressure
     ▼
┌─────────────┐
│ HIBERNATING │
│             │
│   (1 MB)    │
│ (disk only) │
└─────────────┘
```

### State Definitions

| State | RAM Usage | Description |
|-------|-----------|-------------|
| **ASLEEP** | ~5 MB | Arc code loaded, no active data, only wake triggers monitored |
| **WAKING** | ~50 MB | Loading Arc data, preparing for activity |
| **ACTIVE** | ~100-150 MB | Full Arc loaded, processing tasks, LLM access |
| **DROWSY** | ~50 MB | Finished work, still loaded, watching for quick reactivation |
| **SLEEPING** | ~10 MB | Saving state, preparing for unload |
| **HIBERNATING** | ~1 MB | State saved to disk, Arc module unloaded, only stub remains |

### Transitions

```python
class ArcStateMachine:
    """
    Arc state machine with context-aware transitions.
    """
    
    async def transition(self, arc: Arc, event: ArcEvent) -> ArcState:
        current = arc.state
        
        match (current, event):
            # Wake from sleep
            case (ArcState.ASLEEP, ArcEvent.WAKE_TRIGGER):
                await self._load_arc_data(arc)
                return ArcState.WAKING
            
            # Become active after waking
            case (ArcState.WAKING, ArcEvent.READY):
                await self._connect_to_bus(arc)
                return ArcState.ACTIVE
            
            # Go drowsy after finishing work
            case (ArcState.ACTIVE, ArcEvent.IDLE):
                self._start_drowsy_timer(arc, timeout=120)
                return ArcState.DROWSY
            
            # Re-activate from drowsy
            case (ArcState.DROWSY, ArcEvent.WORK_ARRIVED):
                return ArcState.ACTIVE
            
            # Fall asleep from drowsy
            case (ArcState.DROWSY, ArcEvent.TIMEOUT):
                await self._save_state(arc)
                return ArcState.SLEEPING
            
            # Fully asleep
            case (ArcState.SLEEPING, ArcEvent.UNLOAD_COMPLETE):
                await self._unload_arc_data(arc)
                return ArcState.ASLEEP
            
            # Emergency hibernate
            case (_, ArcEvent.SYSTEM_PRESSURE):
                await self._hibernate_arc(arc)
                return ArcState.HIBERNATING
            
            # Restore from hibernation
            case (ArcState.HIBERNATING, ArcEvent.RESTORE):
                await self._restore_from_disk(arc)
                return ArcState.WAKING
            
            case _:
                return current
```

---

## Boss Detection System

"Boss Detection" is AURA recognizing when the user is actively present and adapting behavior accordingly.

### Detection Signals

```
┌────────────────────────────────────────────────────────────────┐
│                    BOSS DETECTION SIGNALS                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  HIGH CONFIDENCE (Boss is here):                               │
│  ├── Screen turned ON                                          │
│  ├── AURA app opened/foregrounded                              │
│  ├── User typed something                                      │
│  ├── Voice input detected                                      │
│  └── Touch on AURA interface                                   │
│                                                                │
│  MEDIUM CONFIDENCE (Boss might be here):                       │
│  ├── Phone picked up (accelerometer)                           │
│  ├── Other app opened                                          │
│  └── Notification interaction                                  │
│                                                                │
│  LOW CONFIDENCE (Boss probably away):                          │
│  ├── Screen off for > 1 minute                                 │
│  ├── Phone stationary                                          │
│  └── No interaction for > 5 minutes                            │
│                                                                │
│  BOSS DEFINITELY AWAY:                                         │
│  ├── Phone locked + screen off > 10 minutes                    │
│  ├── Sleep mode engaged                                        │
│  └── Do Not Disturb active                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Boss Detection Algorithm

```python
class BossDetector:
    """
    Detects user presence and adapts Arc behavior accordingly.
    
    Integrates with existing ContextDetector for sensor data.
    """
    
    def __init__(self, context_detector: ContextDetector):
        self.context_detector = context_detector
        self.presence_score = 0.0  # 0.0 = away, 1.0 = definitely present
        self.last_interaction = None
        self.presence_history = deque(maxlen=60)  # 1 minute of history
    
    async def update(self) -> BossPresence:
        """
        Called every second (or on event) to update presence state.
        """
        ctx = await self.context_detector.detect()
        
        signals = []
        
        # High confidence signals
        if ctx.get('activity', {}).get('details', {}).get('screen_on'):
            signals.append(('screen_on', 0.8))
        
        if self._recent_user_input():
            signals.append(('user_input', 1.0))
        
        # Medium confidence signals
        if self._phone_moving(ctx):
            signals.append(('phone_moving', 0.4))
        
        # Calculate weighted presence score
        if signals:
            weights = [s[1] for s in signals]
            self.presence_score = min(1.0, sum(weights) / len(weights) + 0.2)
        else:
            # Decay presence over time
            decay_rate = 0.05  # per second
            self.presence_score = max(0.0, self.presence_score - decay_rate)
        
        self.presence_history.append(self.presence_score)
        
        return self._classify_presence()
    
    def _classify_presence(self) -> BossPresence:
        if self.presence_score > 0.7:
            return BossPresence.ACTIVE
        elif self.presence_score > 0.3:
            return BossPresence.IDLE
        elif self.presence_score > 0.1:
            return BossPresence.AWAY
        else:
            return BossPresence.DEEP_AWAY


class BossPresence(Enum):
    ACTIVE = "active"      # User is actively using device
    IDLE = "idle"          # User present but not interacting
    AWAY = "away"          # User probably stepped away
    DEEP_AWAY = "deep_away"  # User definitely away (sleeping, etc.)
```

### Arc Behavior by Boss Presence

| Arc | ACTIVE | IDLE | AWAY | DEEP_AWAY |
|-----|--------|------|------|-----------|
| Health | Respond to queries | Subtle reminders | Silent tracking | Sleep analysis |
| Social | Message assistance | Notification summary | Background monitor | Silent |
| Life | Active planning | Upcoming reminders | Background prep | Silent except alarms |
| Learning | Study assistance | Review prompts | Content curation | Silent |
| Research | User-requested only | Pause | Active research | Heavy research |

---

## Toggle Visibility System

Users can choose to watch AURA work or let it work invisibly.

### Visibility Modes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        VISIBILITY MODES                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MODE 1: INVISIBLE (Default)                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  - Arcs work completely in background                            │   │
│  │  - No UI updates, no notifications during work                   │   │
│  │  - User sees summary reports at designated times                 │   │
│  │  - "While you were away, I did X, Y, Z"                          │   │
│  │  - Minimal battery impact, maximum efficiency                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  MODE 2: AMBIENT (Subtle)                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  - Small indicator shows AURA is working (pulsing dot)           │   │
│  │  - Notification badge updates when significant discoveries       │   │
│  │  - Pull-down to see current activity                             │   │
│  │  - Can pause with single tap                                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  MODE 3: VISIBLE (Transparent)                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  - Real-time activity feed showing all Arc work                  │   │
│  │  - See thoughts, searches, discoveries as they happen            │   │
│  │  - Interactive: can guide, pause, prioritize                     │   │
│  │  - "AURA is researching wedding outfits... 3 options found"      │   │
│  │  - Higher battery usage due to UI updates                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  MODE 4: DEBUG (Developer)                                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  - Full Arc state machine visualization                          │   │
│  │  - RAM usage per Arc                                             │   │
│  │  - Message bus traffic                                           │   │
│  │  - Wake/sleep trigger logs                                       │   │
│  │  - For development and troubleshooting only                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Summary Reports

When visibility is INVISIBLE or AMBIENT, AURA generates summary reports:

```python
@dataclass
class ActivityReport:
    """
    Summary of what AURA did while user was away.
    """
    period_start: datetime
    period_end: datetime
    user_state_during: str  # "gaming", "sleeping", "working", etc.
    
    discoveries: List[Discovery]      # Things AURA found
    preparations: List[Preparation]   # Things AURA prepared
    reminders: List[Reminder]         # Things user should know
    actions_taken: List[Action]       # Things AURA did
    suggestions: List[Suggestion]     # Things AURA recommends


# Example Report:
"""
While you were gaming (3h 24m):

DISCOVERIES:
- Found 3 outfit options matching your saved style for the wedding
- Price dropped on the headphones you were tracking ($299 -> $249)

PREPARATIONS:
- Prepared meeting summary for tomorrow's 10am call
- Organized your notes from last week into study guide

REMINDERS:
- Mom's birthday is in 2 days - no gift ordered yet
- You haven't exercised in 3 days

ACTIONS:
- Rescheduled conflicting meeting (moved 2pm to 3pm, both parties confirmed)
- Sent auto-reply to non-urgent emails

SUGGESTIONS:
- Your sleep has been poor this week. Consider earlier bedtime tonight?
"""
```

---

## Integration with Existing Systems

Phase 3 builds on Phase 2 foundations without replacing them.

### Integration Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION WITH EXISTING AURA                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CONTEXT DETECTION (src/context/detector.py)                            │
│  └── Arcs use ContextDetector for wake/sleep decisions                  │
│  └── Boss Detection wraps ContextDetector                               │
│  └── No changes to ContextDetector — just consumption                   │
│                                                                         │
│  TOOL REGISTRY (src/tools/registry.py + handlers.py)                    │
│  └── Arcs access tools through existing bind_handlers pattern           │
│  └── Each Arc has scoped tool access (Health Arc can't send messages)   │
│  └── Tool execution goes through unified handlers.py                    │
│                                                                         │
│  MEMORY SYSTEM (src/memory/*)                                           │
│  └── Each Arc has its own memory namespace                              │
│  └── Cross-arc queries go through Memory Router in Core                 │
│  └── Existing vector store, episodic, semantic memory used              │
│                                                                         │
│  LLM MANAGER (src/llm/production_llm.py)                                │
│  └── Single LLM instance shared by all Arcs                             │
│  └── Priority queue for LLM access (user queries > arc work)            │
│  └── Token budget per Arc (prevent runaway usage)                       │
│                                                                         │
│  PERSONALITY ENGINE (src/core/adaptive_personality.py)                  │
│  └── Core maintains single personality state                            │
│  └── Arc responses pass through personality layer                       │
│  └── Consistent AURA voice across all Arcs                              │
│                                                                         │
│  NEUROMORPHIC ENGINE (src/core/neuromorphic_engine.py)                  │
│  └── Arcs integrate as specialized sub-agents                           │
│  └── MultiAgentOrchestrator coordinates with Arc Scheduler              │
│  └── Resource budget shared with Arc RAM management                     │
│                                                                         │
│  VOICE PIPELINE (src/voice/real_time_pipeline.py)                       │
│  └── User voice input routes through Core to relevant Arcs              │
│  └── Arc responses pass through TTS for voice output                    │
│  └── Boss Detection uses microphone silence detection                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Backward Compatibility

- All existing `src/agents/healthcare/` components become part of Health Arc
- `SubAgentCoordinator` evolves into Arc Scheduler
- Existing proactive services (`src/services/`) integrate with Life Arc
- User-facing API unchanged — Arcs are an implementation detail

---

## File Structure Proposal

```
src/agents/arcs/
├── __init__.py                    # Arc system exports
├── base_arc.py                    # Base Arc class with state machine
├── arc_scheduler.py               # Arc loading/unloading, priority queue
├── arc_message_bus.py             # Inter-arc communication
├── boss_detector.py               # User presence detection
├── visibility_manager.py          # Visibility mode management
├── report_generator.py            # Activity summary reports
│
├── core/                          # AURA Core (orchestrator)
│   ├── __init__.py
│   ├── core_arc.py                # Main orchestration logic
│   ├── personality_bridge.py      # Connect personality to arcs
│   ├── memory_router.py           # Cross-arc memory queries
│   └── response_synthesizer.py    # Combine multi-arc responses
│
├── health/                        # Health Arc
│   ├── __init__.py
│   ├── health_arc.py              # Arc implementation
│   ├── triggers.py                # Wake/sleep triggers
│   └── capabilities/              # Health-specific capabilities
│       ├── fitness.py
│       ├── sleep.py
│       ├── diet.py
│       └── wellness.py
│
├── social/                        # Social Arc
│   ├── __init__.py
│   ├── social_arc.py              # Arc implementation
│   ├── triggers.py                # Wake/sleep triggers
│   └── capabilities/
│       ├── messages.py
│       ├── contacts.py
│       ├── events.py
│       └── social_media.py
│
├── life/                          # Life Arc
│   ├── __init__.py
│   ├── life_arc.py                # Arc implementation
│   ├── triggers.py                # Wake/sleep triggers
│   └── capabilities/
│       ├── calendar.py
│       ├── tasks.py
│       ├── routines.py
│       └── planning.py
│
├── learning/                      # Learning Arc
│   ├── __init__.py
│   ├── learning_arc.py            # Arc implementation
│   ├── triggers.py                # Wake/sleep triggers
│   └── capabilities/
│       ├── study.py
│       ├── knowledge.py
│       ├── habits.py
│       └── skills.py
│
└── research/                      # Research Arc
    ├── __init__.py
    ├── research_arc.py            # Arc implementation
    ├── triggers.py                # Wake/sleep triggers
    └── capabilities/
        ├── web_research.py
        ├── fact_check.py
        ├── trends.py
        └── purchase_research.py
```

---

## Implementation Phases

### Phase 3.1: Foundation (Week 1-2)

**Goal:** Create Arc infrastructure without migrating existing functionality.

| Task | Description | Files |
|------|-------------|-------|
| 3.1.1 | Create `base_arc.py` with state machine | `src/agents/arcs/base_arc.py` |
| 3.1.2 | Create `arc_scheduler.py` with RAM management | `src/agents/arcs/arc_scheduler.py` |
| 3.1.3 | Create `arc_message_bus.py` for inter-arc communication | `src/agents/arcs/arc_message_bus.py` |
| 3.1.4 | Create `boss_detector.py` integrating with ContextDetector | `src/agents/arcs/boss_detector.py` |
| 3.1.5 | Create `visibility_manager.py` with mode switching | `src/agents/arcs/visibility_manager.py` |
| 3.1.6 | Write unit tests for Arc infrastructure | `tests/agents/arcs/` |

**Verification:**
```bash
python -c "from src.agents.arcs import BaseArc, ArcScheduler, BossDetector; print('Foundation OK')"
```

### Phase 3.2: Health Arc (Week 3)

**Goal:** Create first complete Arc by migrating existing healthcare agent.

| Task | Description | Files |
|------|-------------|-------|
| 3.2.1 | Create `health_arc.py` extending BaseArc | `src/agents/arcs/health/health_arc.py` |
| 3.2.2 | Migrate `src/agents/healthcare/` capabilities | `src/agents/arcs/health/capabilities/` |
| 3.2.3 | Implement Health Arc triggers | `src/agents/arcs/health/triggers.py` |
| 3.2.4 | Test Health Arc state transitions | `tests/agents/arcs/health/` |
| 3.2.5 | Test Health Arc RAM footprint | `tests/agents/arcs/test_memory.py` |

**Verification:**
```bash
python -c "
import asyncio
from src.agents.arcs.health import HealthArc
arc = HealthArc()
asyncio.run(arc.wake())
print(f'Health Arc active, state: {arc.state}')
"
```

### Phase 3.3: Life Arc (Week 4)

**Goal:** Create Life Arc for calendar/tasks/planning.

| Task | Description | Files |
|------|-------------|-------|
| 3.3.1 | Create `life_arc.py` extending BaseArc | `src/agents/arcs/life/life_arc.py` |
| 3.3.2 | Migrate calendar/task services | `src/agents/arcs/life/capabilities/` |
| 3.3.3 | Implement daily planning capability | `src/agents/arcs/life/capabilities/planning.py` |
| 3.3.4 | Implement Life Arc triggers | `src/agents/arcs/life/triggers.py` |
| 3.3.5 | Test Life Arc state transitions | `tests/agents/arcs/life/` |

### Phase 3.4: Social Arc (Week 5)

**Goal:** Create Social Arc for relationships/communication.

| Task | Description | Files |
|------|-------------|-------|
| 3.4.1 | Create `social_arc.py` extending BaseArc | `src/agents/arcs/social/social_arc.py` |
| 3.4.2 | Implement message monitoring capability | `src/agents/arcs/social/capabilities/messages.py` |
| 3.4.3 | Implement contact management | `src/agents/arcs/social/capabilities/contacts.py` |
| 3.4.4 | Implement Social Arc triggers | `src/agents/arcs/social/triggers.py` |
| 3.4.5 | Test Social Arc state transitions | `tests/agents/arcs/social/` |

### Phase 3.5: Learning & Research Arcs (Week 6)

**Goal:** Create remaining Arcs.

| Task | Description | Files |
|------|-------------|-------|
| 3.5.1 | Create `learning_arc.py` extending BaseArc | `src/agents/arcs/learning/learning_arc.py` |
| 3.5.2 | Implement study/knowledge capabilities | `src/agents/arcs/learning/capabilities/` |
| 3.5.3 | Create `research_arc.py` extending BaseArc | `src/agents/arcs/research/research_arc.py` |
| 3.5.4 | Implement research capabilities | `src/agents/arcs/research/capabilities/` |
| 3.5.5 | Test Learning/Research Arcs | `tests/agents/arcs/learning/`, `tests/agents/arcs/research/` |

### Phase 3.6: AURA Core Integration (Week 7-8)

**Goal:** Connect all Arcs through Core orchestrator.

| Task | Description | Files |
|------|-------------|-------|
| 3.6.1 | Create `core_arc.py` orchestrator | `src/agents/arcs/core/core_arc.py` |
| 3.6.2 | Implement personality bridge | `src/agents/arcs/core/personality_bridge.py` |
| 3.6.3 | Implement memory router | `src/agents/arcs/core/memory_router.py` |
| 3.6.4 | Implement response synthesizer | `src/agents/arcs/core/response_synthesizer.py` |
| 3.6.5 | Implement report generator | `src/agents/arcs/report_generator.py` |
| 3.6.6 | Integration tests with multiple Arcs | `tests/agents/arcs/test_integration.py` |

### Phase 3.7: Visibility & Polish (Week 9)

**Goal:** Implement visibility modes and polish user experience.

| Task | Description | Files |
|------|-------------|-------|
| 3.7.1 | Implement INVISIBLE mode | `src/agents/arcs/visibility_manager.py` |
| 3.7.2 | Implement AMBIENT mode with UI indicators | UI components |
| 3.7.3 | Implement VISIBLE mode activity feed | UI components |
| 3.7.4 | Implement DEBUG mode | `src/agents/arcs/debug_visualizer.py` |
| 3.7.5 | End-to-end testing | `tests/agents/arcs/test_e2e.py` |

---

## Verification Checkpoints

### Phase 3.1 Verification

```bash
# Test base infrastructure
python -c "
from src.agents.arcs.base_arc import BaseArc, ArcState
from src.agents.arcs.arc_scheduler import ArcScheduler
from src.agents.arcs.boss_detector import BossDetector
print('Base infrastructure: OK')
"
```

### Phase 3.2-3.5 Verification (per Arc)

```bash
# Test individual Arc
python -c "
import asyncio
from src.agents.arcs.health import HealthArc

async def test():
    arc = HealthArc()
    assert arc.state == ArcState.ASLEEP
    await arc.wake()
    assert arc.state == ArcState.ACTIVE
    await arc.sleep()
    assert arc.state == ArcState.ASLEEP
    print('Health Arc: OK')

asyncio.run(test())
"
```

### Phase 3.6 Verification

```bash
# Test full Arc system
python -c "
import asyncio
from src.agents.arcs import ArcSystem

async def test():
    system = ArcSystem()
    await system.initialize()
    
    # Verify Core is running
    assert system.core.is_active()
    
    # Verify RAM budget respected
    assert system.get_total_ram_mb() < 1500
    
    # Test Arc coordination
    response = await system.process('What should I do today?')
    assert response is not None
    print('Arc System: OK')

asyncio.run(test())
"
```

### Phase 3.7 Verification

```bash
# Test visibility modes
python -c "
import asyncio
from src.agents.arcs import ArcSystem, VisibilityMode

async def test():
    system = ArcSystem()
    await system.initialize()
    
    # Test mode switching
    system.set_visibility(VisibilityMode.INVISIBLE)
    system.set_visibility(VisibilityMode.AMBIENT)
    system.set_visibility(VisibilityMode.VISIBLE)
    
    # Test report generation
    report = await system.generate_report()
    assert report is not None
    print('Visibility: OK')

asyncio.run(test())
"
```

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| RAM overflow on 8GB device | High | Strict RAM budgets, aggressive unloading, hibernation |
| Arc coordination deadlocks | Medium | Timeout on all inter-arc calls, independent failure |
| LLM bottleneck with multiple Arcs | Medium | Priority queue, token budgets, batch processing |
| Boss detection false positives | Low | Conservative thresholds, user correction mechanism |
| Backward compatibility breaks | High | Extensive testing, migration path, feature flags |

---

## Success Criteria

Phase 3 is complete when:

1. **All 5 Arcs implemented** with state machines and triggers
2. **RAM usage under 1.5 GB** with all Arcs potentially active
3. **Boss detection working** with <2s latency on presence changes
4. **Visibility modes functional** including activity reports
5. **Integration tests pass** for multi-arc coordination
6. **No regression** in existing AURA functionality
7. **Battery impact acceptable** (<5% per hour during background operation)

---

## Appendix A: Arc Message Protocol

Inter-arc communication uses a simple message protocol:

```python
@dataclass
class ArcMessage:
    """
    Message passed between Arcs via the message bus.
    """
    id: str
    source_arc: str           # "health", "social", "life", "learning", "research", "core"
    target_arc: str           # Same options, or "broadcast"
    message_type: str         # "request", "response", "event", "query"
    priority: int             # 1-10, higher = more important
    payload: Dict[str, Any]   # Message-specific data
    timestamp: datetime
    requires_response: bool
    timeout_ms: int = 5000    # Response timeout
```

### Message Types

| Type | Description | Example |
|------|-------------|---------|
| **request** | Ask another Arc to do something | Health -> Social: "Get emergency contacts" |
| **response** | Reply to a request | Social -> Health: "Here are 3 contacts" |
| **event** | Notify of something that happened | Life -> Core: "Meeting started" |
| **query** | Ask for information without action | Research -> Learning: "What topics is user studying?" |

---

## Appendix B: Wake Trigger Specification

Each Arc defines its wake triggers as a list of conditions:

```python
@dataclass
class WakeTrigger:
    """
    Condition that can wake an Arc from sleep.
    """
    name: str
    check_interval_seconds: int
    condition: Callable[[], Awaitable[bool]]
    priority: int  # Higher priority = faster wake
    
    
# Example: Health Arc triggers
HEALTH_TRIGGERS = [
    WakeTrigger(
        name="morning_routine",
        check_interval_seconds=60,
        condition=lambda: 6 <= datetime.now().hour <= 8,
        priority=5
    ),
    WakeTrigger(
        name="activity_detected",
        check_interval_seconds=30,
        condition=lambda: accelerometer.is_moving(),
        priority=3
    ),
    WakeTrigger(
        name="sedentary_alert",
        check_interval_seconds=300,
        condition=lambda: time_since_movement() > timedelta(hours=1),
        priority=7
    ),
]
```

---

*This document defines the vision for AURA Phase 3 — transforming AURA into an autonomous background intelligence that truly becomes a life companion.*
