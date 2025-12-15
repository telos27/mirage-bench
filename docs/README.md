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

### Agent Verifier

General-purpose AI agent verification system with 6-layer architecture. **All 6 layers implemented with 453 tests.**

| Layer | Name | Description | Status |
|-------|------|-------------|--------|
| 1 | Common Knowledge | Universal truths (Datalog) | ✅ |
| 2 | Domain Best Practices | Domain-specific rules | ✅ |
| 3 | Business Policies | Per-deployment rules | ✅ |
| 4 | User Preferences | Per-user settings | ✅ |
| 5 | Session History | Multi-turn context | ✅ |
| 6 | Prompt Constraints | Prompt-specific rules | ✅ |

**Quick Start:**
```python
from agent_verifier import quick_verify

result = quick_verify(
    prompt="Write hello world",
    output="print('Hello, World!')",
)
print(result.verdict)  # "pass" or "fail"
```

Documentation: [`/agent_verifier/README.md`](../agent_verifier/README.md)

## Session Logs

- [2025-12-13: Logic Verifier Implementation](./sessions/2025-12-13-logic-verifier-implementation.md)
