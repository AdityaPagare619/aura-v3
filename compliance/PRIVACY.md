# AURA v3 Privacy Policy

This document outlines AURA's privacy practices and your rights.

---

## Table of Contents

1. [Privacy Commitment](#privacy-commitment)
2. [Data Collection](#data-collection)
3. [Data Usage](#data-usage)
4. [Data Sharing](#data-sharing)
5. [Your Rights](#your-rights)
6. [Data Retention](#data-retention)
7. [Verification](#verification)

---

## Privacy Commitment

### Our Promise

AURA is built on a fundamental principle: **Your data stays on your device**.

We believe that personal AI assistants should not require sacrificing privacy. That's why AURA is designed to work 100% offline.

### Privacy by Design

| Principle | Implementation |
|-----------|---------------|
| **Minimization** | Only collect what's needed |
| **Local Processing** | All AI runs on-device |
| **No Tracking** | Zero telemetry, zero analytics |
| **User Control** | You own your data |
| **Transparency** | Open source, verifiable |

---

## Data Collection

### What We Collect

**Nothing**. AURA does not collect any data from you.

### What AURA Processes (Locally)

When you interact with AURA, the following happens on your device:

| Data Type | Processing | Storage |
|-----------|------------|---------|
| Voice input | Converted to text, processed locally | Only if you enable history |
| Text input | Processed by local LLM | Only if you enable history |
| User preferences | Learned locally | `./data/profiles/` |
| Conversation history | Stored locally | `./data/conversations/` |
| Learned patterns | Stored locally | `./data/memory/` |

### What Stays on Your Device

```
┌─────────────────────────────────────────────────────┐
│              Your Device Only                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ Input → Processing → Output                 │   │
│  │                                              │   │
│  │ Voice    Text    AI        Response  Action │   │
│  │ Input ─► Input ─► Engine ─► Output ─► Done │   │
│  │                                              │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ✗ Cloud      ✗ External Server    ✗ Third Party  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Data Usage

### How Your Data Is Used

If you enable conversation history, AURA uses your data for:

1. **Conversation Context**
   - Remembering previous messages
   - Maintaining conversation flow

2. **Personalization**
   - Learning your preferences
   - Adapting communication style

3. **Improvement**
   - Training the local model
   - Improving responses

### How Data Is NOT Used

- **Never sold**: We don't sell your data
- **Never shared**: No data leaves your device
- **Never analyzed**: No external analysis
- **Never marketed**: No advertising

---

## Data Sharing

### We Do Not Share Your Data

**Zero. Nada. None.**

- No data sharing with third parties
- No data sharing with affiliates
- No data sharing with advertisers
- No data sharing with governments (even if requested*)

> *Note: Since we don't have your data, we literally cannot share it.

### Network Requests

In normal operation, AURA makes **zero network requests**.

To verify:

```bash
# Monitor network connections
# (requires root or separate monitoring tool)
tcpdump -i any -c 100

# Or check AURA logs
grep -E "http|https|request" logs/aura.log
# Should return: (nothing)
```

### What About Updates?

Update checks are **optional**:

```yaml
# config.yaml
updates:
  auto_check: false  # Disable update checks
  allow_downloads: false
```

If enabled, updates only download from:
- Official GitHub releases
- Checksums verified

---

## Your Rights

### You Own Your Data

Under AURA's architecture, you have complete control:

| Right | How to Exercise |
|-------|-----------------|
| **Access** | View `./data/` directory |
| **Export** | Copy `./data/` to backup |
| **Delete** | Delete `./data/` directory |
| **Rectify** | Edit profile files |
| **Portability** | Standard JSON format |
| **Object** | Disable storage entirely |

### How to Exercise Your Rights

#### Access Your Data

```bash
# View all data
ls -la data/

# View conversation history
cat data/conversations/2024/2024-01-15.json
```

#### Export Your Data

```bash
# Create backup
tar czf aura_data_backup.tar.gz data/

# Or use rsync
rsync -av data/ backup/
```

#### Delete Your Data

```bash
# Delete all data
rm -rf data/*

# Or use factory reset
rm -rf data/ cache/ logs/
```

#### Disable Storage

```yaml
# config.yaml - no data stored
privacy:
  store_conversations: false
```

---

## Data Retention

### Retention Policy

You control how long data is kept:

```yaml
# config.yaml
storage:
  # Keep last 30 days of conversations
  conversation_retention_days: 30
  
  # Keep last 1000 interactions
  memory:
    episodic_limit: 1000
    semantic_limit: 5000
```

### Automatic Cleanup

AURA automatically cleans old data:

```python
# Pseudocode
if conversation.age > retention_period:
    delete(conversation)
```

### Manual Cleanup

```bash
# Delete conversations before date
rm data/conversations/2023/*

# Clear all memory
rm -rf data/memory/*

# Clear cache
rm -rf cache/*
```

---

## Verification

### Verifying Privacy Claims

You can verify AURA's privacy claims:

#### 1. Network Verification

```bash
# Run AURA and monitor network
# (Use separate machine or network monitoring)

# If ANY network traffic is found:
# - Check config: network.allow_outgoing
# - File a bug report
```

#### 2. Source Code Verification

```bash
# Review source for network calls
grep -r "requests\." src/
grep -r "urllib" src/
grep -r "socket" src/

# Should only find:
# - localhost references
# - comments about disabled features
```

#### 3. Log Verification

```bash
# Check logs for network activity
grep -i "network\|http\|request\|telemetry" logs/aura.log

# Should return nothing in normal operation
```

### Privacy Audit

For advanced verification:

1. **Static Analysis**
   - Review source code
   - Check dependencies

2. **Dynamic Analysis**
   - Run in sandbox
   - Monitor all system calls

3. **Network Analysis**
   - Capture all packets
   - Verify zero external calls

---

## Questions?

### Common Questions

**Q: Does AURA send my data anywhere?**
A: No. AURA operates 100% offline.

**Q: Can you see my conversations?**
A: No. All data stays on your device.

**Q: What if I lose my phone?**
A: Your data is on the device. Enable encryption for protection.

**Q: Do you respond to legal requests?**
A: Since we don't have your data, we have nothing to share.

**Q: How do you make money?**
A: We may offer premium features/services, but never by selling data.

---

## Contact

For privacy questions:
- Email: privacy@aura.example.com
- GitHub: Open an issue (non-sensitive only)

---

## Changes to This Policy

We may update this policy. Changes will be documented in CHANGELOG.md.

Last updated: February 2026
