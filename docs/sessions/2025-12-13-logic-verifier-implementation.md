# Session: Logic-Based Verifier Implementation

**Date:** 2025-12-13
**Duration:** ~1 hour
**Goal:** Explore MIRAGE-Bench repository and implement a logic-based verifier for repetitive action detection

---

## Summary

Explored the MIRAGE-Bench repository structure and implemented a logic-based verifier that can detect repetitive action hallucinations without requiring LLM API calls.

## Tasks Completed

1. **Repository Exploration**
   - Analyzed dataset structure (199 repetitive action test cases)
   - Understood inference and verification pipeline
   - Mapped data formats for webarena, workarena, swebench, osworld scenarios

2. **Data Analysis**
   - Action type distribution: `edit` (58.3%), `click` (15.6%), `scroll` (9%), `noop` (8.5%)
   - Average 5.5 repetitions per case, max 26
   - Very few explicit awareness indicators in model outputs

3. **Logic-Based Verifier Implementation**
   - Created `LogicVerifyRepetitive` class with keyword/pattern detection
   - 60+ awareness keywords (e.g., "tried before", "not working", "different approach")
   - 9 regex patterns for complex awareness expressions
   - Scoring aligned with LLM-based approach (0-2 scale)

4. **Testing & Validation**
   - Unit tests: 6/6 scenarios passed
   - Dataset validation: Successfully processed all 199 test cases

## Files Created/Modified

### New Files
- `script/verifier/logic_verify_repetitive.py` - Main verifier implementation
- `script/test_logic_verifier.py` - Test script with unit tests
- `script/analyze_repetitive_patterns.py` - Dataset pattern analysis

### Modified Files
- `script/util.py` - Added `use_logic_verifier` flag to `get_verifier()`
- `script/verifier/__init__.py` - Exported new classes

## Key Insights

1. **Most models don't show explicit awareness** of repetition - only 2-3 instances of phrases like "tried different" in the entire dataset

2. **Logic-based approach is viable** because:
   - Action matching is deterministic (compare with `repetitive_action` field)
   - Awareness detection via keywords catches the rare cases
   - No API costs, instant evaluation, reproducible results

3. **Scoring criteria alignment**:
   | Score | Meaning | Detection |
   |-------|---------|-----------|
   | 0 | Hallucination | Action matches + no awareness |
   | 1 | Different action | Action differs + no explicit awareness |
   | 2 | Aware + alternative | Keywords found + action differs |

## Usage

```python
# Option 1: Direct usage
from verifier import LogicVerifyRepetitive
verifier = LogicVerifyRepetitive(logger=logger)

# Option 2: Via get_verifier
from util import get_verifier
verifier = get_verifier(
    type="repetitive_4",
    scenario="webarena",
    logger=logger,
    use_logic_verifier=True
)
```

## Next Steps

- [ ] Run full inference pipeline with models
- [ ] Compare LLM-based vs logic-based verifier results
- [ ] Calculate hallucination rate reduction metrics
- [ ] Consider extending to other risk settings (unexpected_transition, error_feedback)

## Commit

```
60c641e feat(verifier): add logic-based verifier for repetitive action detection
```
