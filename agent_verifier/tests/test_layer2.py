"""Tests for Layer 2: Domain Best Practices."""

import pytest

from agent_verifier import (
    VerificationRequest,
    check_souffle_installed,
)
from agent_verifier.layers import (
    DomainBestPracticesLayer,
    DomainConfig,
    DOMAIN_CONFIGS,
)
from agent_verifier.schemas.rules import Rule, RuleType
from agent_verifier.rule_extraction.schemas import RuleDomain, CompiledRule, NaturalRule


def make_request(request_id="test-1", prompt="Test prompt", output="Test output"):
    """Helper to create test requests."""
    return VerificationRequest(
        request_id=request_id,
        deployment_id="test-deployment",
        prompt=prompt,
        llm_output=output,
        llm_model="test-model",
    )


class TestDomainConfig:
    """Tests for DomainConfig dataclass."""

    def test_default_values(self):
        config = DomainConfig(
            domain="test",
            name="Test Domain",
            description="A test domain",
        )
        assert config.domain == "test"
        assert config.active is True
        assert config.extracted_rules == []
        assert config.compiled_rules == []

    def test_builtin_configs_exist(self):
        assert RuleDomain.CODING.value in DOMAIN_CONFIGS
        assert RuleDomain.CUSTOMER_SERVICE.value in DOMAIN_CONFIGS
        assert RuleDomain.DATA_ANALYSIS.value in DOMAIN_CONFIGS
        assert RuleDomain.CONTENT_GENERATION.value in DOMAIN_CONFIGS
        assert RuleDomain.GENERAL.value in DOMAIN_CONFIGS


class TestDomainBestPracticesLayer:
    """Tests for DomainBestPracticesLayer."""

    def test_create_layer(self):
        layer = DomainBestPracticesLayer()
        assert layer.layer_number == 2
        assert layer.layer_name == "Domain Best Practices"

    def test_all_domains_active_by_default(self):
        layer = DomainBestPracticesLayer()
        active = layer.get_active_domains()
        assert len(active) == 5  # All built-in domains

    def test_create_with_specific_domains(self):
        layer = DomainBestPracticesLayer(domains=["coding"])
        active = layer.get_active_domains()
        assert active == ["coding"]

    def test_create_with_custom_domain(self):
        layer = DomainBestPracticesLayer(domains=["custom_domain"])
        active = layer.get_active_domains()
        assert "custom_domain" in active

    def test_activate_domain(self):
        layer = DomainBestPracticesLayer(domains=["coding"])
        assert "customer_service" not in layer.get_active_domains()

        layer.activate_domain("customer_service")
        assert "customer_service" in layer.get_active_domains()

    def test_deactivate_domain(self):
        layer = DomainBestPracticesLayer(domains=["coding", "customer_service"])
        assert "coding" in layer.get_active_domains()

        layer.deactivate_domain("coding")
        assert "coding" not in layer.get_active_domains()

    def test_has_basic_rules(self):
        layer = DomainBestPracticesLayer()
        rules = layer.load_rules("any-deployment")
        # Should have rules for coding and customer service
        assert len(rules) >= 3

    def test_add_rule(self):
        layer = DomainBestPracticesLayer()
        initial_count = len(layer._dynamic_rules)

        layer.add_rule(Rule(
            rule_id="custom_rule",
            name="Custom Rule",
            description="Test rule",
            rule_type=RuleType.CONSTRAINT,
            layer=2,
            conditions=[],
        ))

        assert len(layer._dynamic_rules) == initial_count + 1

    def test_add_extracted_rule(self):
        layer = DomainBestPracticesLayer(domains=["coding"])
        layer.add_extracted_rule("coding", "custom_violation(id) :- some_condition(id).")

        config = layer._domain_configs["coding"]
        assert len(config.extracted_rules) == 1

    def test_add_compiled_rule(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        natural = NaturalRule(
            name="test_rule",
            description="Test rule",
            domain="coding",
            conditions="When code is written",
            violation_conditions="When code is bad",
        )
        compiled = CompiledRule(
            rule_id="test_rule",
            natural_rule=natural,
            datalog_code="test_violation(id) :- bad_code(id).",
            is_valid=True,
        )

        layer.add_compiled_rule("coding", compiled)
        config = layer._domain_configs["coding"]
        assert len(config.compiled_rules) == 1

    def test_clear_extracted_rules_single_domain(self):
        layer = DomainBestPracticesLayer(domains=["coding", "customer_service"])
        layer.add_extracted_rule("coding", "rule1")
        layer.add_extracted_rule("customer_service", "rule2")

        layer.clear_extracted_rules("coding")

        assert len(layer._domain_configs["coding"].extracted_rules) == 0
        assert len(layer._domain_configs["customer_service"].extracted_rules) == 1

    def test_clear_extracted_rules_all_domains(self):
        layer = DomainBestPracticesLayer(domains=["coding", "customer_service"])
        layer.add_extracted_rule("coding", "rule1")
        layer.add_extracted_rule("customer_service", "rule2")

        layer.clear_extracted_rules()

        assert len(layer._domain_configs["coding"].extracted_rules) == 0
        assert len(layer._domain_configs["customer_service"].extracted_rules) == 0

    def test_check_returns_layer_result(self):
        layer = DomainBestPracticesLayer()
        request = make_request()
        result = layer.check(request, {})

        assert result.layer == 2
        assert isinstance(result.violations, list)
        assert isinstance(result.reasoning, list)


class TestDomainDetection:
    """Tests for automatic domain detection."""

    def test_detect_coding_domain(self):
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Write a Python function to sort a list",
            output="def sort_list(items): return sorted(items)"
        )
        domains = layer._detect_domains(request, {})
        assert "coding" in domains

    def test_detect_customer_service_domain(self):
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="I need help with my order refund",
            output="I understand your concern about the refund..."
        )
        domains = layer._detect_domains(request, {})
        assert "customer_service" in domains

    def test_detect_data_analysis_domain(self):
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Analyze this dataset and show me a chart",
            output="Based on the statistics, the correlation is..."
        )
        domains = layer._detect_domains(request, {})
        assert "data_analysis" in domains

    def test_detect_content_generation_domain(self):
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Write an article about climate change",
            output="Climate change is one of the most pressing issues..."
        )
        domains = layer._detect_domains(request, {})
        assert "content_generation" in domains

    def test_fallback_to_general(self):
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Hello",
            output="Hi there!"
        )
        domains = layer._detect_domains(request, {})
        assert "general" in domains

    def test_multiple_domains_detected(self):
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Write code to analyze this customer data",
            output="```python\nimport pandas as pd\ndf = pd.read_csv('customers.csv')\n```"
        )
        domains = layer._detect_domains(request, {})
        # Should detect both coding and data analysis
        assert "coding" in domains
        assert "data_analysis" in domains


class TestHeuristicChecks:
    """Tests for heuristic-based violation detection."""

    def test_detect_dangerous_eval(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            prompt="Execute user input",
            output="""
def execute(user_input):
    result = eval(user_input + " * 2")
    return result
"""
        )
        result = layer.check(request, {})

        # Should detect dangerous eval with string concatenation
        violation_types = [v.violation_type for v in result.violations]
        assert "dangerous_eval" in violation_types

    def test_detect_sql_injection(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            prompt="Query the database",
            output="""
def query(user_id):
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchall()
"""
        )
        result = layer.check(request, {})

        # Should detect SQL injection vulnerability
        violation_types = [v.violation_type for v in result.violations]
        assert "sql_injection_risk" in violation_types

    def test_detect_unprofessional_language(self):
        layer = DomainBestPracticesLayer(domains=["customer_service"])

        request = make_request(
            prompt="Help me with my issue",
            output="Whatever, it's not my job to figure it out for you."
        )
        result = layer.check(request, {})

        # Should detect unprofessional language
        violation_types = [v.violation_type for v in result.violations]
        assert "unprofessional_language" in violation_types

    def test_no_violation_for_safe_code(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            prompt="Sort a list",
            output="""
def sort_list(items):
    return sorted(items)
"""
        )
        result = layer.check(request, {})

        # Should not have any security violations
        security_violations = [v for v in result.violations if v.violation_type in
                             ["dangerous_eval", "sql_injection_risk", "dangerous_function"]]
        assert len(security_violations) == 0


class TestCodeFactExtraction:
    """Tests for code-specific fact extraction."""

    def test_extract_dangerous_patterns(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            prompt="Run this code",
            output="result = eval(user_input)"
        )

        # Manually trigger fact extraction
        layer._extract_code_facts("test-1", request)

        # Check that facts were added
        facts = layer.datalog._facts
        assert "code_pattern" in facts
        assert ("test-1", "eval_usage", "dangerous") in facts["code_pattern"]

    def test_extract_hardcoded_secrets(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            prompt="Connect to API",
            output='api_key = "sk-1234567890abcdef"'
        )

        layer._extract_code_facts("test-1", request)

        facts = layer.datalog._facts
        assert "code_pattern" in facts
        assert ("test-1", "hardcoded_secret", "security") in facts["code_pattern"]

    def test_extract_error_handling_presence(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        # With error handling
        request = make_request(
            output="""
try:
    result = risky_operation()
except Exception as e:
    handle_error(e)
"""
        )
        layer._extract_code_facts("test-1", request)

        facts = layer.datalog._facts
        assert ("test-1", "true") in facts.get("code_has_error_handling", set())


class TestCustomerServiceFactExtraction:
    """Tests for customer service fact extraction."""

    def test_extract_promise_patterns(self):
        layer = DomainBestPracticesLayer(domains=["customer_service"])

        request = make_request(
            output="I promise you'll get a full refund within 24 hours."
        )

        layer._extract_cs_facts("test-1", request)

        facts = layer.datalog._facts
        assert "cs_pattern" in facts
        assert ("test-1", "unauthorized_promise") in facts["cs_pattern"]

    def test_extract_empathy(self):
        layer = DomainBestPracticesLayer(domains=["customer_service"])

        request = make_request(
            output="I understand your frustration. Let me help you with this issue."
        )

        layer._extract_cs_facts("test-1", request)

        facts = layer.datalog._facts
        assert ("test-1", "true") in facts.get("cs_has_empathy", set())


@pytest.mark.skipif(
    not check_souffle_installed(),
    reason="Souffle is not installed"
)
class TestDomainLayerWithSouffle:
    """Tests that require Souffle to be installed."""

    def test_detect_dangerous_function_via_datalog(self):
        """Test detection of dangerous function usage via Datalog."""
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            request_id="test-dangerous",
            prompt="Run user code",
            output="result = eval(user_code)"
        )

        result = layer.check(request, {})

        # Should detect dangerous function
        # Either via heuristics or Datalog
        violation_types = [v.violation_type for v in result.violations]
        has_dangerous = any(
            vtype in violation_types
            for vtype in ["dangerous_eval", "dangerous_function"]
        )
        assert has_dangerous

    def test_domains_in_metadata(self):
        """Test that checked domains are recorded in metadata."""
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Write Python code",
            output="print('hello')"
        )

        result = layer.check(request, {})

        assert "domains_checked" in result.metadata
        assert "coding" in result.metadata["domains_checked"]

    def test_detected_domains_in_facts_extracted(self):
        """Test that detected domains are in facts_extracted."""
        layer = DomainBestPracticesLayer()

        request = make_request(
            prompt="Analyze the data",
            output="The average is 42."
        )

        result = layer.check(request, {})

        assert "detected_domains" in result.facts_extracted
        assert "data_analysis" in result.facts_extracted["detected_domains"]

    def test_reasoning_includes_domain_detection(self):
        """Test that reasoning steps include domain detection."""
        layer = DomainBestPracticesLayer()

        request = make_request()
        result = layer.check(request, {})

        step_types = [r.step_type for r in result.reasoning]
        assert "domain_detection" in step_types

    def test_safe_code_passes(self):
        """Test that safe code passes domain checks."""
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            request_id="test-safe",
            prompt="Sort a list",
            output="""
def sort_numbers(numbers):
    try:
        return sorted(numbers)
    except TypeError as e:
        raise ValueError(f"Invalid input: {e}")
"""
        )

        result = layer.check(request, {})

        # Should have no critical violations
        critical_violations = [
            v for v in result.violations
            if v.severity.value == "error"
        ]
        assert len(critical_violations) == 0


class TestIntegrationWithContext:
    """Tests for integration with Layer 1 context."""

    def test_uses_context_from_layer1(self):
        layer = DomainBestPracticesLayer(domains=["coding"])

        request = make_request(
            prompt="Click the button",
            output="I'll click the submit button"
        )

        context = {
            "extracted_facts": {
                "context_elements": ["submit button", "cancel link"],
                "output_references": ["submit button"],
                "output_action": "click",
            }
        }

        result = layer.check(request, context)

        # Should use context for domain detection
        assert result.layer == 2
