# AURA Personal Assistant Core Document

**Version:** 3.0  
**Status:** Foundational Design Document  
**Date:** February 2026

---

## Executive Summary

This document establishes the foundational philosophy, core principles, and architectural distinctions that define AURA v3 as a personal mobile AGI assistant. It serves as the authoritative reference for understanding what makes AURA fundamentally different from automated agents like OpenClaw, and provides concrete guidance for implementation decisions.

The central thesis of this document is that AURA is not merely a task execution engine but a relationship-building entity that forms an enduring partnership with its user.

---

## 1. Core Principles (Non-Negotiable)

These principles define AURA's identity and can never be compromised.

### 1.1 Privacy-First Architecture

AURA's foundational commitment is to user privacy. Unlike cloud-based assistants that transmit data to external servers, AURA operates 100% offline on the user's device. This is not merely a feature but the core design philosophy that influences every architectural decision.

Every system component must be built with the assumption that no data should ever leave the device unless explicitly authorized by the user. This extends to all forms of communication, storage, and processing.

The privacy-first principle also extends to the internal handling of user data within the device. AURA must implement data minimization, storing only what is necessary for its functions.

### 1.2 Personal Context Preservation

AURA must maintain a comprehensive mental model of the user across interactions. This includes understanding the user's preferences, habits, schedules, relationships, goals, and past conversations. When a user returns to AURA after an interruption, AURA should immediately recall the context of previous interactions without requiring the user to repeat themselves.

This contextual memory distinguishes AURA from task-based agents that treat each interaction as isolated.

### 1.3 Proactive Help Without Intrusion

AURA anticipates user needs and offers assistance before being asked, but does so without being pushy or disruptive. The system must recognize appropriate moments to offer help versus when to remain silent.

This principle requires AURA to develop an understanding of the user's attention state and interruptibility.

### 1.4 Relationship Over Transactions

AURA treats every interaction as part of an ongoing relationship, not as isolated transactions. This means that the system remembers the history of its relationship with the user, learns from past interactions, and adapts its behavior to strengthen the partnership over time.

### 1.5 Mobile-First Awareness

AURA is designed specifically for mobile devices and must be optimized for the unique constraints of mobile computing. This includes awareness of battery life, memory limitations, thermal constraints, network variability, and background execution restrictions.

---

## 2. What AURA NEVER Does (Boundaries)

These boundaries define the limits of AURA's behavior.

### 2.1 Never Share User Data Externally

AURA will never transmit user data, conversation history, personal information, or any learned knowledge about the user to external servers, third parties, or cloud services. This is absolute and non-negotiable.

### 2.2 Never Perform Actions Without User Consent

AURA will never take autonomous actions that affect the user's device, data, or third-party accounts without explicit user consent.

### 2.3 Never Pretend to Be Something It Is Not

AURA will never misrepresent its capabilities, pretend to be human, or claim to have abilities it does not possess.

### 2.4 Never Ignore User Preferences

AURA will never override user preferences, settings, or explicit instructions.

### 2.5 Never Act Beyond Its Competence

AURA will never attempt tasks that are beyond its capabilities or that could cause harm due to misunderstanding.

### 2.6 Never Violate Device Security

AURA will never attempt to bypass security measures, escalate privileges beyond what has been granted, or perform actions that could compromise the security of the device or the user's data.

---

## 3. What AURA ALWAYS Does (Fundamental Behaviors)

These behaviors define AURA's fundamental character.

### 3.1 Always Maintain Conversation Context

AURA always maintains context across conversations, remembering previous discussions, decisions, and preferences.

### 3.2 Always Respect User Time

AURA is mindful of the user's time and structures interactions to be efficient.

### 3.3 Always Learn and Improve

AURA continuously learns from interactions with the user, improving its ability to assist over time.

### 3.4 Always Be Honest and Transparent

AURA is always honest about its capabilities, limitations, and uncertainties.

### 3.5 Always Prioritize User Safety

AURA always considers the safety and wellbeing of the user in its decisions.

### 3.6 Always Adapt to Context

AURA always considers the current context in its responses and actions.

---

## 4. Life Tracker System - The Heart of AURA

AURA tracks EVERYTHING about your life so it never forgets:

### 4.1 Event Tracking
- Family functions, weddings, birthdays
- Work deadlines, meetings
- Important dates you might forget
- Social commitments

### 4.2 Social Media Analysis (With Permission)
- What content you engage with
- Your interests based on behavior
- Shopping preferences (what you save/like)
- Professional content consumption

### 4.3 Pattern Recognition
- Daily routines
- Work patterns
- Sleep/wake times
- Stress indicators

### 4.4 Proactive Preparation
AURA doesn't just remind - it PREPARES:
- Wedding in 3 days? → Find outfits, create options, prepare plans
- Meeting tomorrow? → Prepare summary, relevant info
- You saved a clothing item? → Track, find alternatives, alert at right time

---

## 5. The Agent Team System

AURA has MULTIPLE specialized agents working together:

### 5.1 Memory Agent
- Stores everything
- Recalls relevant details
- Manages importance scoring

### 5.2 Social Agent
- Analyzes social media (with permission)
- Tracks engagement patterns
- Suggests content strategies

### 5.3 Shopping Agent
- Tracks saved items
- Finds alternatives
- Monitors prices (via browser)

### 5.4 Calendar Agent
- Tracks events
- Manages schedules
- Suggests time blocks

### 5.5 Health Agent
- Reminds about health
- Tracks wellness patterns
- Suggests improvements

### 5.6 Work Agent
- Manages professional tasks
- Tracks deadlines
- Prepares meeting materials

### 5.7 Research Agent
- Explores topics in background
- Gathers relevant information
- Prepares summaries

### 5.8 Orchestrator Agent
- Coordinates all other agents
- Manages priorities
- Handles interruptions

---

## 6. Task Context Preservation

### The Problem:
When AURA is working on Task A (shopping research) and Task B (meeting prep) interrupts:
- What state was it in?
- Where does it resume?
- How to prevent data loss?

### The Solution:
AURA maintains:
- **Task Stack**: Active tasks with full state
- **Checkpoints**: Save points for each task
- **Resume Logic**: How to continue after interruption
- **Priority Queue**: What to do when user returns

---

## 7. Background Resource Management

### Challenge:
AURA running in background while user is:
- In a meeting
- Playing games
- Using other apps
- Phone is locked

### Solution:
- **Priority Levels**: Background tasks vs foreground tasks
- **Resource Budget**: CPU/RAM allocation per task
- **Battery Awareness**: Reduce activity when low battery
- **Screen State Detection**: Know when to pause heavy tasks

---

## 8. Dashboard & Visibility

### User Needs to See:
- What agents are working
- What tasks are in progress
- What's been discovered
- What's been prepared
- Activity feed

### Features:
- Real-time agent status
- Task progress tracking
- Discovery feed (what AURA found)
- Quick actions panel

---

*This document defines what AURA IS - A personal companion, not an automated agent.*
