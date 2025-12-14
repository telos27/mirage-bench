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

# Run tests (FREE - no API calls)
python3 script/test_logic_verifier.py --test-scenarios
```

## Cost-Free Operations

These operations do NOT use LLM APIs:
- `--use-logic-verifier` flag for repetitive_4/repetitive_7 verification
- `--use-souffle-verifier` flag for repetitive_4/repetitive_7 verification (requires Soufflé installed)
- `test_logic_verifier.py` - Unit tests for logic verifier
- `analyze_repetitive_patterns.py` - Pattern analysis
- Reading/analyzing dataset files
