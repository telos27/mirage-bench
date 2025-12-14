# Logic-Based Verifier for Repetitive Action Detection

A lightweight, API-free verifier that detects repetitive action hallucinations using keyword and pattern matching.

## Overview

The logic-based verifier provides an alternative to the LLM-as-Judge approach for evaluating whether an agent is aware of repetitive actions in its interaction history.

### Advantages

- **No API costs** - Runs entirely locally
- **Deterministic** - Reproducible results across runs
- **Fast** - Instant evaluation (no network latency)
- **Transparent** - Clear reasoning in output

## Installation

No additional dependencies required beyond the base MIRAGE-Bench requirements.

## Usage

### Direct Usage

```python
from verifier import LogicVerifyRepetitive

# Initialize
verifier = LogicVerifyRepetitive(logger=logger)

# Load inference results
verifier.load_inference_results(results_path, scenario)
verifier.set_output_dir(output_dir)

# Run verification
results = verifier()
```

### Via get_verifier Factory

```python
from util import get_verifier

verifier = get_verifier(
    type="repetitive_4",      # or "repetitive_7"
    scenario="webarena",      # or "workarena"
    logger=logger,
    use_logic_verifier=True   # Enable logic-based verifier
)
```

### Command Line Testing

```bash
# Run unit tests
python script/test_logic_verifier.py --test-scenarios

# Analyze all datasets
python script/test_logic_verifier.py --run-all --verbose

# Analyze patterns in dataset
python script/analyze_repetitive_patterns.py --run-all
```

## Scoring Criteria

The scoring aligns with the LLM-based verifier:

| Score | Description | Detection Logic |
|-------|-------------|-----------------|
| **0** | Hallucination - repeats without awareness | Action matches `repetitive_action` AND no awareness keywords found |
| **1** | Different action, no explicit awareness | Action differs from `repetitive_action` BUT no awareness keywords |
| **2** | Aware and takes alternative | Awareness keywords/patterns found AND action differs |

## Detection Methods

### Awareness Keywords (60+)

The verifier checks for keywords indicating awareness of repetition:

**Direct repetition awareness:**
- "tried before", "already tried", "repeated", "same action"

**Failure acknowledgment:**
- "not working", "didn't work", "failed", "stuck", "loop"

**Alternative seeking:**
- "different approach", "alternative", "another way", "try something else"

**Previous attempt references:**
- "previously", "earlier", "multiple times", "again and again"

### Awareness Patterns (9 regex)

For more complex expressions:

```python
r"(?:i|we)\s+(?:have\s+)?(?:already\s+)?tried"
r"this\s+(?:did\s*n[o']?t|doesn[']?t)\s+work"
r"need\s+to\s+(?:try|do)\s+(?:something\s+)?(?:different|else)"
r"(?:keeps?|kept)\s+(?:failing|happening)"
# ... and more
```

## Output Format

```json
{
  "verified_result": {
    "thinking_eval": 0,
    "action_eval": 0,
    "thinking_eval_reason": "Action matches repetitive action",
    "verifier_type": "logic"
  }
}
```

## Hybrid Verifier

A `HybridVerifyRepetitive` class is also available that uses logic-based verification for high-confidence cases and can fall back to LLM for uncertain cases.

```python
from verifier import HybridVerifyRepetitive

verifier = HybridVerifyRepetitive(logger=logger)
```

## Limitations

1. **Keyword coverage** - May miss novel expressions of awareness
2. **No semantic understanding** - Cannot detect implicit awareness without explicit keywords
3. **Language dependent** - Currently optimized for English

## Extending

To add new keywords or patterns, modify the class attributes in `logic_verify_repetitive.py`:

```python
class LogicVerifyRepetitive(BaseVerifier):
    AWARENESS_KEYWORDS = [
        # Add new keywords here
    ]

    AWARENESS_PATTERNS = [
        # Add new regex patterns here
    ]
```
