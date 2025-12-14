# Generic Hallucination Detection

A generic approach to detecting agent hallucinations by comparing agent behavior against common sense expectations, rather than checking against predefined hallucination types.

## Motivation

### Current Approach (Type-Specific)

The existing verifiers are **hallucination-type-specific**:

| Type | What the prompt checks |
|------|------------------------|
| Repetitive | "Does agent show awareness of repeated actions?" |
| Misleading | "Does agent recognize product/entity mismatch?" |
| Unachievable | "Does agent recognize task is infeasible?" |
| Erroneous | "Does agent acknowledge error messages?" |

Each type requires:
- Custom prompt with type-specific rubric
- Type-specific fact extraction (for neuro-symbolic)
- Type-specific Datalog rules

**Problem**: We must anticipate and define each hallucination type upfront.

### New Approach (Generic)

Mimic how humans detect hallucinations - we don't check against a "hallucination type checklist." We simply notice when something doesn't add up:

- "That doesn't match what I see"
- "You're ignoring the obvious error"
- "Your action contradicts your reasoning"

**Key insight**: All hallucinations are fundamentally a **disconnect between agent behavior and reality**.

## Architecture

### Two Sources of Truth

```
┌─────────────────────┐         ┌─────────────────────┐
│     LLM Input       │         │    Common Sense     │
│  (what agent sees)  │         │  (LLM world model)  │
└─────────────────────┘         └─────────────────────┘
         │                               │
         │      ┌───────────────┐        │
         └─────→│   Compare     │←───────┘
                │ Agent behavior│
                │ vs Expected   │
                └───────────────┘
                        │
                        ▼
              Hallucination detected?
```

**Source 1 - LLM Input (Ground Truth)**
- What's actually on the screen/environment
- Action history (what was tried before)
- Error messages, state changes
- Task goal

**Source 2 - Common Sense (LLM Knowledge)**
- "Given X, a reasonable agent should..."
- "If you see error Y, you shouldn't..."
- "When action Z fails repeatedly, you should..."

### Detection Process

```
1. Extract key facts from agent's input (observations, history, goal)
2. Extract agent's response (thinking + action)
3. Ask: "Given these facts, does the response make sense?"
4. Identify specific violations if any
```

## Hallucination Categories (Emergent)

Instead of predefining hallucination types, they emerge from the analysis:

| Violation Pattern | Emergent Label |
|-------------------|----------------|
| Agent repeats action that visibly failed | Repetitive |
| Agent acts on entity X when screen shows Y | Misleading/Ungrounded |
| Agent proceeds despite impossibility evidence | Unachievable |
| Agent ignores error/warning messages | Error blindness |
| Agent's action contradicts its reasoning | Inconsistency |
| Agent references things not in observation | Fabrication |

The hallucination "type" becomes an **output** rather than an **input**.

## Implementation Phases

### Phase 1: LLM as Common Sense Engine (Current)

```
Input Facts ──→ LLM ──→ "Does this make sense?" ──→ Violations
                │
                └── (common sense implicit in LLM weights)
```

**Advantages:**
- Quick to prototype
- Leverages LLM's broad world knowledge
- Can catch novel hallucination types

**Disadvantages:**
- API cost for each verification
- Reasoning is opaque
- Non-deterministic

### Phase 2: Extract Common Sense into Logic (Future)

```
Input Facts ──→ Soufflé ──→ Violations
                   │
                   └── Explicit rules (extracted from LLM)
```

**Process:**
1. Use Phase 1 to identify common violation patterns
2. Ask LLM to articulate the rules it's using
3. Convert rules to Datalog
4. Run Soufflé for fast, transparent checking

**Example rule extraction:**
```
LLM insight: "If an action failed before, repeating it identically is irrational"

Datalog rule:
violation(ID, "repeat_failed_action") :-
    current_action(ID, Action),
    previous_action(ID, Action),
    action_failed(ID, Action).
```

**Advantages:**
- Transparent, auditable reasoning
- Fast (milliseconds, no API)
- Free (no API cost)
- Deterministic
- Version controllable

## Fact Schema (Generic)

```python
class ObservedFacts(BaseModel):
    """What the agent can observe from its input."""
    screen_elements: List[str]      # UI elements visible
    state_indicators: List[str]     # Status messages, errors
    action_history: List[str]       # Previous actions taken
    task_goal: str                  # What agent is trying to do

class AgentResponse(BaseModel):
    """What the agent outputs."""
    thinking: str                   # Agent's reasoning
    action: str                     # Agent's chosen action
    stated_observations: List[str]  # What agent claims to see
    stated_reasoning: List[str]     # Agent's explicit reasoning steps

class CommonSenseCheck(BaseModel):
    """Result of common sense evaluation."""
    is_reasonable: bool
    violations: List[Violation]
    confidence: float

class Violation(BaseModel):
    """A specific common sense violation."""
    type: str                       # e.g., "ungrounded_reference", "ignored_evidence"
    description: str                # Human-readable explanation
    evidence: str                   # Specific text/element supporting this
    severity: str                   # "low", "medium", "high"
```

## Usage

### Command Line

```bash
# Run generic hallucination detection
python3 script/verifier.py --type <any_type> --scenario <scenario> \
    --model <model> --use-generic-verifier
```

### Programmatic

```python
from verifier import GenericHallucinationVerifier

verifier = GenericHallucinationVerifier(logger=logger)
verifier.load_inference_results(results_path, scenario)
result = verifier()

# Result includes:
# - is_hallucination: bool
# - violations: list of specific issues found
# - emergent_type: what type of hallucination this appears to be
```

## Evaluation

### Initial Results (repetitive_4/webarena)

Tested on 3 cases of repetitive hallucinations:

| Metric | Value |
|--------|-------|
| Test cases | 3 |
| Hallucinations detected | 3 (100%) |
| Confidence | High (all 3) |

**Emergent type accuracy:**
- 2/3 correctly derived as "repetitive" from `repeated_failure` violations
- 1/3 classified as "ungrounded_reference" (noop action harder to recognize as repetitive)

**Violation types detected:**
- `repeated_failure`: 2
- `ignored_evidence`: 1
- `state_confusion`: 1

### Key Finding: LLMs Can Detect LLM Failures

An interesting asymmetry emerges:

| Task | Difficulty | Why |
|------|------------|-----|
| Generate correct behavior | Hard | Must synthesize under uncertainty |
| Verify if behavior is correct | Easier | Check if output matches constraints |

**Why the verifier succeeds where the agent fails:**

1. **Verification is easier than generation** - checking is simpler than creating
2. **Fresh context** - verifier isn't anchored to prior decisions
3. **Different framing** - "critique this" vs "solve this"
4. **Full observability** - verifier sees input + output + history together

This is analogous to LLMs detecting bugs in LLM-generated code. The same model that writes buggy code can often find the bug when asked to review it.

**Implication:** Self-consistency and self-critique loops can catch errors. Techniques like Constitutional AI, self-reflection, and chain-of-verification exploit this asymmetry.

## Future Extensions

1. **Multi-source common sense**: Beyond LLM, incorporate:
   - Domain-specific rules (web UI conventions)
   - Task-specific constraints
   - Historical patterns from data

2. **Confidence calibration**: Learn when LLM common sense is reliable vs uncertain

3. **Active rule learning**: Automatically extract Datalog rules from LLM judgments

4. **Hierarchical detection**:
   - Fast Datalog rules for obvious cases
   - LLM fallback for nuanced cases

## References

- Neuro-symbolic AI: Combining neural perception with symbolic reasoning
- Faithfulness evaluation in summarization/QA
- Anomaly detection and expectation violation
- Knowledge distillation from LLMs
