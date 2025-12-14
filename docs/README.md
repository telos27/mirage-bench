# MIRAGE-Bench Documentation

## Contents

### Verifiers

- [Logic-Based Verifier](./logic-based-verifier.md) - API-free verifier using keyword/pattern matching
- [Generic Hallucination Detector](./generic-hallucination-detector.md) - Type-agnostic common sense approach

### Architecture

| Verifier | Description | Cost |
|----------|-------------|------|
| Pure LLM | Default LLM-as-judge | High |
| Logic-based | Keyword/pattern matching | Free |
| Soufflé | Datalog rules | Free |
| Neuro-Symbolic | LLM extraction + Datalog reasoning | Medium |
| Generic | LLM common sense | Medium |
| Soufflé-Generic | Heuristic + Datalog + LLM hybrid | Low |

### Agent Verifier (In Development)

General-purpose AI agent verification system with 6-layer architecture:

| Layer | Name | Description |
|-------|------|-------------|
| 1 | Common Knowledge | Universal truths, logic, formats |
| 2 | Domain Best Practices | Agent-type specific patterns |
| 3 | Business Policies | Per-deployment rules |
| 4 | User Preferences | Per-user settings |
| 5 | Session Context | Conversation history |
| 6 | Active Request | Current prompt constraints |

Location: `/agent_verifier/`

## Session Logs

- [2025-12-13: Logic Verifier Implementation](./sessions/2025-12-13-logic-verifier-implementation.md)
