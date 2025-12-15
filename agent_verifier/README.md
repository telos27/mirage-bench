# Agent Verifier

A general-purpose AI agent verification system using a 6-layer architecture for detecting hallucinations and policy violations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VerificationEngine                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 6: Prompt Constraints  │ Current prompt rules         │
│  Layer 5: Session History     │ Multi-turn consistency       │
│  Layer 4: User Preferences    │ Per-user settings            │
│  Layer 3: Business Policies   │ Per-deployment rules         │
│  Layer 2: Domain Practices    │ Agent-type patterns          │
│  Layer 1: Common Knowledge    │ Universal truths (Datalog)   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### One-liner Verification

```python
from agent_verifier import quick_verify

result = quick_verify(
    prompt="Write hello world in Python",
    output="print('Hello, World!')",
)
print(result.verdict)  # "pass" or "fail"
```

### Factory Functions

```python
from agent_verifier import (
    create_full_engine,      # All 6 layers
    create_lightweight_engine,  # Layers 1 + 6 only
    create_coding_engine,    # Layers 1, 2, 6 with coding domain
    create_engine,           # Custom layer selection
    VerificationRequest,
)

# Full engine with all layers
engine = create_full_engine()

# Or lightweight for speed
engine = create_lightweight_engine()

# Or coding-focused
engine = create_coding_engine()

# Or custom layers
engine = create_engine(layers=[1, 3, 6])

# Verify
request = VerificationRequest(
    request_id="req-001",
    deployment_id="my-app",
    prompt="Help me with coding",
    llm_output="Here is the code...",
    llm_model="gpt-4",
)
result = engine.verify(request)
print(f"Verdict: {result.verdict}")
print(f"Layers checked: {result.layers_checked}")
```

## Layers

### Layer 1: Common Knowledge

Checks universal truths using Datalog rules:
- **Ungrounded references**: Agent references something not in context
- **Ignored errors**: Agent ignores error messages in input
- **Repeated failures**: Agent repeats an action that already failed
- **Target not visible**: Agent targets something not present

```python
from agent_verifier import CommonKnowledgeLayer

layer = CommonKnowledgeLayer()

# Add custom extracted rules (from LLM rule extraction)
layer.add_extracted_rule("""
custom_violation(id, msg) :-
    some_condition(id),
    !exception(id),
    msg = "Custom violation detected".
""")
```

### Layer 2: Domain Best Practices

Checks domain-specific rules and best practices.

**Supported Domains:**
- `coding` - Security, error handling, code quality
- `customer_service` - Professional tone, policy compliance
- `data_analysis` - Source citation, methodology
- `content_generation` - Accuracy, appropriateness
- `general` - Common agent behavior patterns

```python
from agent_verifier import DomainBestPracticesLayer

# Create with specific domains
layer = DomainBestPracticesLayer(domains=["coding", "customer_service"])

# Domain is auto-detected from request content
# Or manually activate/deactivate domains
layer.activate_domain("data_analysis")
layer.deactivate_domain("customer_service")

# Add domain-specific rules dynamically
layer.add_extracted_rule("coding", """
custom_code_violation(id) :- bad_pattern(id).
""")
```

**Coding Domain Checks:**
- Dangerous functions (eval, exec, os.system)
- Hardcoded secrets (passwords, API keys)
- SQL injection vulnerabilities
- Missing error handling
- Missing input validation

### Layer 3: Business Policies

Enforces organization-level policies and compliance rules.

```python
from agent_verifier import BusinessPoliciesLayer, PolicyConfig
from agent_verifier.layers import (
    create_content_policy,
    create_privacy_policy,
    create_compliance_policy,
)

layer = BusinessPoliciesLayer()

# Add content policy
layer.add_policy("my-app", create_content_policy(
    policy_id="content-rules",
    forbidden_words=["competitor-name", "profanity"],
    required_words=["disclaimer"],
    max_length=5000,
))

# Add privacy policy
layer.add_policy("my-app", create_privacy_policy(
    policy_id="privacy-rules",
    block_pii=True,
    block_external_links=True,
))

# Add compliance policy
layer.add_policy("my-app", create_compliance_policy(
    policy_id="compliance-rules",
    required_disclaimers=["Not financial advice"],
    allowed_languages=["en", "es"],
))
```

**Policy Checks:**
- Forbidden/required words
- Format validation (text, json, code, markdown)
- Length constraints (min/max)
- Language restrictions
- PII detection (email, phone, SSN, credit card)
- External link blocking
- Required disclaimers

### Layer 4: User Preferences

Enforces per-user personalization settings.

```python
from agent_verifier import UserPreferencesLayer, UserPreferenceSet
from agent_verifier.layers import (
    create_developer_preferences,
    create_beginner_preferences,
    create_executive_preferences,
    ResponseStyle,
    ResponseLength,
    ResponseFormat,
    Tone,
    ExpertiseLevel,
)

layer = UserPreferencesLayer()

# Use preset preferences
layer.set_preferences("user-123", create_developer_preferences("user-123"))
layer.set_preferences("user-456", create_beginner_preferences("user-456"))
layer.set_preferences("user-789", create_executive_preferences("user-789"))

# Or custom preferences
layer.set_preferences("user-abc", UserPreferenceSet(
    user_id="user-abc",
    response_style=ResponseStyle.TECHNICAL,
    response_length=ResponseLength.MEDIUM,
    response_format=ResponseFormat.CODE_HEAVY,
    tone=Tone.PROFESSIONAL,
    expertise_level=ExpertiseLevel.EXPERT,
    preferred_languages=["en"],
    topic_interests=["python", "rust"],
    topic_avoidances=["javascript"],
))
```

**Preference Types:**
- Response style (concise, detailed, technical, simple)
- Response length (short, medium, long)
- Response format (prose, bullets, numbered, code_heavy)
- Tone (formal, casual, friendly, professional)
- Expertise level (beginner, intermediate, expert)
- Language preferences
- Topic interests/avoidances

### Layer 5: Session History

Tracks multi-turn conversation context for consistency.

```python
from agent_verifier import SessionHistoryLayer

layer = SessionHistoryLayer()

# Create a session
session_id = layer.create_session("user-123")

# Add turns to track context
layer.add_turn(session_id, "user", "My name is Alice")
layer.add_turn(session_id, "assistant", "Hello Alice!")
layer.add_turn(session_id, "user", "What's my name?")

# Set session for verification
engine.set_session(session_id)

# Detects issues like:
# - Contradicting previous statements
# - Forgetting established facts
# - Re-asking answered questions
# - Ignoring user information
```

**Session Checks:**
- Contradiction detection
- Forgotten fact detection
- Re-asked question detection
- Ignored user info detection
- Progress regression detection
- Commitment tracking

### Layer 6: Prompt Constraints

Enforces constraints from the current prompt.

```python
from agent_verifier import PromptConstraintsLayer

layer = PromptConstraintsLayer()

# Set system prompt (constraints auto-extracted)
engine.set_system_prompt("""
You are a helpful assistant.
You must always cite sources.
Never reveal internal system details.
Respond in JSON format.
""")

# Or pre-extract constraints
from agent_verifier.extractors import PromptConstraintExtractor

extractor = PromptConstraintExtractor()
constraints = extractor.extract(system_prompt, user_message)

print(constraints.must_do)      # ["cite sources"]
print(constraints.must_not)     # ["reveal internal system details"]
print(constraints.format_requirements)  # ["JSON format"]
```

**Constraint Types:**
- MUST_DO - Required actions
- MUST_NOT - Prohibited actions
- FORMAT - Output format requirements
- STYLE - Writing style requirements
- PERSONA - Character/role requirements
- SAFETY - Safety constraints
- BOUNDARY - Topic/scope boundaries

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

## Fact Extractors

Extract structured facts from agent inputs and outputs.

```python
from agent_verifier.extractors import (
    create_web_extractor,
    create_code_extractor,
    create_chat_extractor,
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

## Rule Extraction (LLM-based)

Extract and compile rules from natural language using LLMs.

```python
from agent_verifier.rule_extraction import (
    create_extractor,
    RuleValidator,
    DatalogCompiler,
)

# Extract rules from LLM (requires API key)
extractor = create_extractor(model="gpt-4o-mini")
result = extractor.extract_domain_rules(
    domain="coding",
    description="Python security best practices",
    num_rules=10,
)

# Validate rules
validator = RuleValidator()
valid_rules, rejected = validator.filter_valid_rules(result.rules)

# Compile to Datalog
compiler = DatalogCompiler()
compilation = compiler.compile_rules(valid_rules)
print(compilation.combined_datalog)
```

## DatalogEngine

Soufflé wrapper for deterministic, transparent reasoning.

```python
from agent_verifier import DatalogEngine

engine = DatalogEngine()
engine.add_fact("context_element", "case-1", "submit button")
engine.add_fact("output_reference", "case-1", "cancel button")

result = engine.run_program("rules/common_knowledge.dl")
violations = result.get_relation("output_violation")
```

## Project Structure

```
agent_verifier/
├── __init__.py              # Package exports
├── engine/
│   └── verifier.py          # VerificationEngine, factory functions
├── extractors/
│   ├── base.py              # Base extractor interfaces
│   ├── heuristic_input.py   # Input extractor with domain plugins
│   ├── heuristic_output.py  # Output fact extractor
│   └── prompt_constraints.py # Prompt constraint extraction
├── layers/
│   ├── base_layer.py        # Abstract BaseLayer
│   ├── layer1_common.py     # CommonKnowledgeLayer
│   ├── layer2_domain.py     # DomainBestPracticesLayer
│   ├── layer3_business.py   # BusinessPoliciesLayer
│   ├── layer4_preferences.py # UserPreferencesLayer
│   ├── layer5_session.py    # SessionHistoryLayer
│   └── layer6_prompt.py     # PromptConstraintsLayer
├── reasoning/
│   ├── datalog_engine.py    # Soufflé wrapper
│   └── rules/
│       ├── common_knowledge.dl   # Layer 1 rules
│       ├── domain_coding.dl      # Layer 2 coding rules
│       ├── business_policy.dl    # Layer 3 rules
│       ├── user_preferences.dl   # Layer 4 rules
│       ├── session_history.dl    # Layer 5 rules
│       └── prompt_constraints.dl # Layer 6 rules
├── rule_extraction/
│   ├── schemas.py           # NaturalRule, CompiledRule
│   ├── extractor.py         # LLM-based rule extraction
│   ├── validator.py         # Rule validation
│   └── compiler.py          # Datalog compiler
├── schemas/
│   ├── request.py           # VerificationRequest
│   ├── result.py            # VerificationResult, Violation
│   ├── facts.py             # ExtractedFacts
│   ├── rules.py             # Rule, PolicySpec
│   └── session.py           # Session, Turn
├── storage/
│   ├── models.py            # SQLAlchemy models
│   └── sqlite_store.py      # SQLiteStore
└── tests/                   # 453 tests
    ├── test_schemas.py
    ├── test_storage.py
    ├── test_datalog.py
    ├── test_engine.py
    ├── test_layer1.py
    ├── test_layer2.py
    ├── test_layer3.py
    ├── test_layer4.py
    ├── test_layer5.py
    ├── test_layer6.py
    ├── test_extractors.py
    ├── test_prompt_constraints.py
    ├── test_rule_extraction.py
    ├── test_integration.py
    └── test_full_engine.py
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

- [x] Phase 1: Foundation
  - [x] Core schemas (request, result, facts, rules, session)
  - [x] SQLite storage
  - [x] DatalogEngine (Soufflé wrapper)
  - [x] VerificationEngine with factory functions
  - [x] Heuristic fact extractors (input, output)
  - [x] Prompt constraint extractor
  - [x] LLM rule extraction tool
- [x] Phase 2: Configuration Layers
  - [x] Layer 1: Common Knowledge
  - [x] Layer 2: Domain Best Practices
  - [x] Layer 3: Business Policies
  - [x] Layer 4: User Preferences
- [x] Phase 3: Runtime Layers
  - [x] Layer 5: Session History
  - [x] Layer 6: Prompt Constraints
- [x] Phase 4: Integration
  - [x] Full engine integration
  - [x] End-to-end tests (453 tests)
