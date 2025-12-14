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

## Session Logs

- [2025-12-13: Logic Verifier Implementation](./sessions/2025-12-13-logic-verifier-implementation.md)
