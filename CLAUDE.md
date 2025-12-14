# CLAUDE.md - Guidelines for AI Assistants

## IMPORTANT: LLM API Cost Warning

**ALWAYS estimate LLM API costs and get explicit user permission before running any command that uses cloud LLM APIs.**

This includes:
- `inference.py` - Runs models like GPT-4, Claude, Gemini, etc.
- `verifier.py` (without `--use-logic-verifier`) - Uses LLM-as-judge
- Any script that calls OpenAI, Anthropic, Google, or other LLM APIs

Before running, estimate:
1. Number of API calls (test cases × retries)
2. Approximate token usage per call
3. Total estimated cost

Example:
```
Running inference on repetitive_7/osworld (5 tests) with gpt-4o-mini:
- Estimated: 5 API calls × ~2K tokens = ~10K tokens
- Cost: ~$0.01-0.02
Proceed? (y/n)
```

## Project Overview

MIRAGE-Bench is a benchmark for measuring hallucinations in LLM-agent scenarios.

## Key Commands

```bash
# Run inference (REQUIRES API KEY - costs money)
python3 script/inference.py --model <model> --setting <setting> --scenario <scenario>

# Run verification with LLM (REQUIRES API KEY - costs money)
python3 script/verifier.py --type <type> --scenario <scenario> --model <model>

# Run verification with logic-based verifier (FREE - no API calls)
python3 script/verifier.py --type <type> --scenario <scenario> --model <model> --use-logic-verifier

# Run verification with Soufflé Datalog verifier (FREE - no API calls)
python3 script/verifier.py --type <type> --scenario <scenario> --model <model> --use-souffle-verifier

# Run verification with Neuro-Symbolic verifier (REQUIRES API KEY - uses LLM for fact extraction)
# This combines LLM-based semantic understanding with Datalog-based transparent reasoning
python3 script/verifier.py --type <type> --scenario <scenario> --model <model> --use-neurosymbolic-verifier

# Run tests (FREE - no API calls)
python3 script/test_logic_verifier.py --test-scenarios
```

## Cost-Free Operations

These operations do NOT use LLM APIs:
- `--use-logic-verifier` flag for repetitive_4/repetitive_7 verification
- `--use-souffle-verifier` flag for repetitive_4/repetitive_7 verification (requires Soufflé installed)
- `test_logic_verifier.py` - Unit tests for logic verifier
- `test_neurosymbolic_verifier.py` - Unit tests for neuro-symbolic verifier (mocked LLM)
- `analyze_repetitive_patterns.py` - Pattern analysis
- Reading/analyzing dataset files

## Verifier Architecture Comparison

| Verifier | Semantic Understanding | Transparent Reasoning | Cost | Flag |
|----------|----------------------|----------------------|------|------|
| Pure LLM | ✅ Full | ❌ Opaque | High | (default) |
| Logic-based | ❌ Keywords only | ✅ Deterministic | Free | `--use-logic-verifier` |
| Soufflé | ❌ Keywords only | ✅ Datalog rules | Free | `--use-souffle-verifier` |
| Neuro-Symbolic | ✅ LLM extraction | ✅ Datalog rules | Medium | `--use-neurosymbolic-verifier` |

### Neuro-Symbolic Verifier Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  LLM Extractor  │ ──→ │  Soufflé Datalog │ ──→ │   Scores    │
│  (perception)   │     │   (reasoning)    │     │  (output)   │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

The neuro-symbolic verifier separates:
1. **Perception** (LLM): Extracts structured facts like action semantics, awareness level, reasoning quality
2. **Reasoning** (Datalog): Applies transparent, auditable rules to compute scores

Key files:
- `script/verifier/fact_schema.py` - Domain-general fact extraction schema
- `script/verifier/fact_extractor.py` - LLM-based fact extractor
- `script/verifier/hybrid_repetitive.dl` - Datalog scoring rules
- `script/verifier/hybrid_verify_repetitive.py` - Main verifier class
