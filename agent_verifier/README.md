# Agent Verifier

A general-purpose AI agent verification system using a 6-layer architecture for detecting hallucinations and policy violations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VerificationEngine                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 6: Active Request    │ Current prompt constraints     │
│  Layer 5: Session Context   │ Conversation history           │
│  Layer 4: User Preferences  │ Per-user settings              │
│  Layer 3: Business Policies │ Per-deployment rules           │
│  Layer 2: Domain Practices  │ Agent-type patterns            │
│  Layer 1: Common Knowledge  │ Universal truths (Datalog)     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from agent_verifier import (
    VerificationEngine,
    EngineConfig,
    VerificationRequest,
    CommonKnowledgeLayer,
)

# Create engine with Layer 1
engine = VerificationEngine(EngineConfig(enabled_layers=[1]))
engine.register_layer(CommonKnowledgeLayer())

# Verify an LLM output
request = VerificationRequest(
    request_id="req-001",
    deployment_id="my-app",
    prompt="Click the submit button",
    llm_output="I'll click the submit button now.",
    llm_model="gpt-4",
)

result = engine.verify(request)
print(f"Verdict: {result.verdict}")  # "pass" or "fail"
print(f"Violations: {len(result.violations)}")
```

## Components

### VerificationEngine
Main orchestrator that runs requests through configured layers.

```python
config = EngineConfig(
    enabled_layers=[1, 2, 3],  # Which layers to run
    fail_fast=True,            # Stop on first critical error
    include_reasoning=True,    # Include reasoning chain in results
)
engine = VerificationEngine(config)
```

### CommonKnowledgeLayer (Layer 1)
Checks universal truths using Datalog rules:
- **Ungrounded references**: Agent references something not in context
- **Ignored errors**: Agent ignores error messages in input
- **Repeated failures**: Agent repeats an action that already failed
- **Target not visible**: Agent targets something not present

```python
layer = CommonKnowledgeLayer()

# Add custom extracted rules (from LLM rule extraction)
layer.add_extracted_rule("""
custom_violation(id, msg) :-
    some_condition(id),
    !exception(id),
    msg = "Custom violation detected".
""")
```

### Fact Extractors
Extract structured facts from agent inputs and outputs.

```python
from agent_verifier.extractors import (
    create_web_extractor,
    HeuristicOutputExtractor,
    HeuristicCombinedExtractor,
)

# Create domain-specific input extractor
input_extractor = create_web_extractor()  # For web browser agents
# Also: create_code_extractor(), create_chat_extractor()

# Extract facts from agent input
input_facts = input_extractor.extract("""
    ## Goal:
    Click the login button

    [100] button 'Login'
    [101] link 'Sign Up'
""")
print(input_facts.task_goal)        # "Click the login button"
print(input_facts.visible_elements) # ["Login", "Sign Up"]

# Extract facts from LLM output
output_extractor = HeuristicOutputExtractor()
output_facts = output_extractor.extract("I'll click('100') to log in")
print(output_facts.action_target)   # "100"

# Combined extraction
combined = HeuristicCombinedExtractor(input_extractor)
facts = combined.extract(prompt, output)
```

### Prompt Constraint Extractor
Extract constraints from system prompts for Layer 6 verification.

```python
from agent_verifier.extractors import PromptConstraintExtractor

extractor = PromptConstraintExtractor()
constraints = extractor.extract(
    system_prompt="You must always cite sources. Never make up facts.",
    user_message="Be concise in your response."
)

print(constraints.must_do)      # ["cite sources"]
print(constraints.must_not)     # ["make up facts"]
print(constraints.style_requirements)  # ["concise"]

# Convert to rules for Datalog
rules = extractor.extract_as_rules(system_prompt)
```

### DatalogEngine
Soufflé wrapper for deterministic, transparent reasoning.

```python
from agent_verifier import DatalogEngine

engine = DatalogEngine()
engine.add_fact("context_element", "case-1", "submit button")
engine.add_fact("output_reference", "case-1", "cancel button")

result = engine.run_program("rules/common_knowledge.dl")
violations = result.get_relation("output_violation")
```

## Verification Result

```python
result = engine.verify(request)

# Overall verdict
result.verdict        # "pass" or "fail"

# Violations found
for v in result.violations:
    print(f"Layer {v.layer}: {v.violation_type}")
    print(f"  Severity: {v.severity}")
    print(f"  Message: {v.message}")

# Reasoning chain (for transparency)
for step in result.reasoning:
    print(f"Layer {step.layer}: {step.description}")

# Performance
print(f"Latency: {result.latency_ms}ms")
print(f"Layers checked: {result.layers_checked}")
```

## Project Structure

```
agent_verifier/
├── engine/
│   └── verifier.py          # VerificationEngine
├── extractors/
│   ├── base.py              # Base extractor interfaces
│   ├── heuristic_input.py   # Input extractor with domain plugins
│   ├── heuristic_output.py  # Output fact extractor
│   └── prompt_constraints.py # Prompt constraint extraction
├── layers/
│   ├── base_layer.py        # Abstract BaseLayer
│   └── layer1_common.py     # CommonKnowledgeLayer
├── reasoning/
│   ├── datalog_engine.py    # Soufflé wrapper
│   └── rules/
│       └── common_knowledge.dl
├── schemas/
│   ├── request.py           # VerificationRequest
│   ├── result.py            # VerificationResult, Violation
│   ├── facts.py             # ExtractedFacts
│   ├── rules.py             # Rule, PolicySpec
│   └── session.py           # Session, Turn
├── storage/
│   ├── models.py            # SQLAlchemy models
│   └── sqlite_store.py      # SQLiteStore
└── tests/
    ├── test_extractors.py   # Extractor tests
    ├── test_prompt_constraints.py
    ├── test_integration.py  # End-to-end tests
    └── ...
```

## Requirements

- Python 3.10+
- Soufflé (for Datalog reasoning)

Install Soufflé:
```bash
# Ubuntu/Debian
apt-get install souffle

# macOS
brew install souffle
```

## Running Tests

```bash
python3 -m pytest agent_verifier/tests/ -v
```

## Implementation Status

- [x] Phase 1: Foundation (Sessions 1-3)
  - [x] Core schemas
  - [x] SQLite storage
  - [x] DatalogEngine
  - [x] VerificationEngine
  - [x] Layer 1 (Common Knowledge)
  - [x] Heuristic fact extractors (input, output)
  - [x] Prompt constraint extractor
  - [x] End-to-end integration tests
- [ ] Phase 1: Continued (Session 3.5)
  - [ ] LLM rule extraction tool
- [ ] Phase 2: Configuration Layers (Sessions 4-6)
  - [ ] Layer 2: Domain Practices
  - [ ] Layer 3: Business Policies
  - [ ] Layer 4: User Preferences
- [ ] Phase 3: Runtime Layers (Sessions 7-9)
  - [ ] Layer 5: Session Context
  - [ ] Layer 6: Prompt Constraints
  - [ ] REST API
- [ ] Phase 4: Polish (Session 10)
