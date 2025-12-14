"""
Tests for LLM-based rule extraction components.

Tests schemas, validator, and compiler. Uses mocked LLM responses
to avoid actual API calls during testing.
"""

import json
import pytest
from unittest.mock import Mock, patch

from agent_verifier.rule_extraction.schemas import (
    NaturalRule,
    RuleSeverity,
    RuleDomain,
    CompiledRule,
    ExtractionResult,
    ValidationResult,
    CompilationResult,
)
from agent_verifier.rule_extraction.extractor import (
    RuleExtractor,
    LLMClient,
    create_extractor,
    DOMAIN_DESCRIPTIONS,
)
from agent_verifier.rule_extraction.validator import (
    RuleValidator,
    validate_rules,
    filter_valid_rules,
)
from agent_verifier.rule_extraction.compiler import (
    DatalogCompiler,
    compile_rules_to_datalog,
    compile_rule,
    DATALOG_TEMPLATES,
)


# ============================================================================
# Schema Tests
# ============================================================================

class TestNaturalRule:
    """Tests for NaturalRule schema."""

    def test_create_basic_rule(self):
        """Test creating a basic rule."""
        rule = NaturalRule(
            name="no_hardcoded_secrets",
            description="Never include hardcoded secrets in code",
            domain="coding",
            conditions="When generating code",
            violation_conditions="Code contains API keys or passwords",
        )

        assert rule.name == "no_hardcoded_secrets"
        assert rule.rule_id == "no_hardcoded_secrets"
        assert rule.severity == RuleSeverity.ERROR
        assert rule.domain == "coding"

    def test_rule_id_generation(self):
        """Test automatic rule_id generation."""
        rule = NaturalRule(
            name="My Complex Rule Name",
            description="Test",
            domain="test",
            conditions="test",
            violation_conditions="test",
        )

        # Should be snake_case, lowercase
        assert rule.rule_id == "my_complex_rule_name"
        assert "_" in rule.rule_id
        assert rule.rule_id.islower()

    def test_rule_id_sanitization(self):
        """Test rule_id sanitizes special characters."""
        rule = NaturalRule(
            name="rule-with-dashes!@#",
            description="Test",
            domain="test",
            conditions="test",
            violation_conditions="test",
        )

        assert "-" not in rule.rule_id
        assert "!" not in rule.rule_id
        assert "@" not in rule.rule_id

    def test_rule_to_dict(self):
        """Test converting rule to dictionary."""
        rule = NaturalRule(
            name="test_rule",
            description="A test rule",
            domain="coding",
            conditions="When testing",
            violation_conditions="Test fails",
            severity=RuleSeverity.WARNING,
            tags=["test", "example"],
        )

        d = rule.to_dict()

        assert d["name"] == "test_rule"
        assert d["severity"] == "warning"
        assert d["tags"] == ["test", "example"]

    def test_rule_from_dict(self):
        """Test creating rule from dictionary."""
        data = {
            "name": "test_rule",
            "description": "A test rule",
            "domain": "coding",
            "conditions": "When testing",
            "violation_conditions": "Test fails",
            "severity": "warning",
        }

        rule = NaturalRule.from_dict(data)

        assert rule.name == "test_rule"
        assert rule.severity == RuleSeverity.WARNING


class TestCompiledRule:
    """Tests for CompiledRule schema."""

    def test_create_compiled_rule(self):
        """Test creating a compiled rule."""
        natural_rule = NaturalRule(
            name="test",
            description="Test rule",
            domain="coding",
            conditions="Always",
            violation_conditions="Test fails",
        )

        compiled = CompiledRule(
            rule_id="test",
            natural_rule=natural_rule,
            datalog_code="violation(id) :- test(id).",
            input_relations=["test"],
        )

        assert compiled.is_valid
        assert len(compiled.validation_errors) == 0
        assert "test" in compiled.input_relations


class TestExtractionResult:
    """Tests for ExtractionResult schema."""

    def test_extraction_result(self):
        """Test extraction result structure."""
        rules = [
            NaturalRule(
                name="rule1",
                description="First rule",
                domain="coding",
                conditions="c1",
                violation_conditions="v1",
            ),
            NaturalRule(
                name="rule2",
                description="Second rule",
                domain="coding",
                conditions="c2",
                violation_conditions="v2",
            ),
        ]

        result = ExtractionResult(
            domain="coding",
            description="Test domain",
            rules=rules,
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=200,
            total_cost=0.001,
        )

        assert len(result.rules) == 2
        assert result.total_cost > 0


# ============================================================================
# Validator Tests
# ============================================================================

class TestRuleValidator:
    """Tests for RuleValidator."""

    def test_validate_complete_rule(self):
        """Test validating a complete, well-formed rule."""
        validator = RuleValidator()

        rule = NaturalRule(
            name="no_hardcoded_secrets",
            description="Never include hardcoded API keys, passwords, or secrets in code.",
            domain="coding",
            conditions="When generating or modifying code that handles authentication",
            violation_conditions="Code contains string literals that appear to be API keys or passwords",
            examples=["api_key = 'sk-123'"],
            rationale="Security best practice",
            tags=["security"],
        )

        result = validator.validate_rule(rule)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_incomplete_rule(self):
        """Test validating a rule with missing fields."""
        validator = RuleValidator()

        rule = NaturalRule(
            name="bad_rule",
            description="",  # Empty
            domain="coding",
            conditions="",  # Empty
            violation_conditions="",  # Empty
        )

        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("description" in e.lower() for e in result.errors)

    def test_validate_vague_rule(self):
        """Test detecting vague language."""
        validator = RuleValidator()

        rule = NaturalRule(
            name="vague_rule",
            description="Maybe sometimes avoid doing bad things etc",
            domain="coding",
            conditions="Perhaps when needed",
            violation_conditions="Could be a problem",
        )

        result = validator.validate_rule(rule)

        assert len(result.warnings) > 0
        assert any("vague" in w.lower() for w in result.warnings)

    def test_validate_bad_name_format(self):
        """Test detecting invalid rule name format."""
        validator = RuleValidator()

        rule = NaturalRule(
            name="BadCamelCase",  # Should be snake_case
            description="A valid description for this rule.",
            domain="coding",
            conditions="When something happens",
            violation_conditions="When there is a failure",
        )

        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert any("snake_case" in e.lower() for e in result.errors)

    def test_filter_valid_rules(self):
        """Test filtering valid from invalid rules."""
        rules = [
            NaturalRule(
                name="good_rule",
                description="A well-formed rule description.",
                domain="coding",
                conditions="When testing this functionality",
                violation_conditions="When the test fails completely",
            ),
            NaturalRule(
                name="bad_rule",
                description="",
                domain="coding",
                conditions="",
                violation_conditions="",
            ),
        ]

        valid, rejected = filter_valid_rules(rules)

        assert len(valid) == 1
        assert len(rejected) == 1
        assert valid[0].name == "good_rule"

    def test_redundancy_detection(self):
        """Test detecting redundant rules."""
        validator = RuleValidator()

        # Create nearly identical rules to ensure high similarity
        rules = [
            NaturalRule(
                name="rule_one",
                description="Do not include hardcoded secrets in the generated code output",
                domain="coding",
                conditions="When writing code",
                violation_conditions="Code has secrets",
            ),
            NaturalRule(
                name="rule_two",
                description="Do not include hardcoded secrets in the generated code output",  # Identical
                domain="coding",
                conditions="When writing code",
                violation_conditions="Code has secrets",
            ),
        ]

        results = validator.validate_rules(rules, check_redundancy=True)

        # At least one should have redundancy warning
        all_warnings = [w for r in results for w in r.warnings]
        assert any("redundant" in w.lower() for w in all_warnings)


# ============================================================================
# Extractor Tests (with mocked LLM)
# ============================================================================

class TestRuleExtractor:
    """Tests for RuleExtractor with mocked LLM."""

    def _create_mock_llm(self, response_json: list[dict]) -> Mock:
        """Create a mock LLM client."""
        mock = Mock(spec=LLMClient)
        mock.model = "gpt-4o-mini"
        mock.complete.return_value = (
            json.dumps(response_json),
            {"prompt_tokens": 100, "completion_tokens": 200},
        )
        return mock

    def test_extract_domain_rules(self):
        """Test extracting rules for a domain."""
        mock_response = [
            {
                "name": "no_hardcoded_secrets",
                "description": "Never include hardcoded secrets",
                "conditions": "When generating code",
                "violation_conditions": "Code contains secrets",
                "severity": "error",
                "examples": ["api_key = 'sk-123'"],
                "rationale": "Security",
                "tags": ["security"],
            },
            {
                "name": "validate_user_input",
                "description": "Always validate user input",
                "conditions": "When handling user input",
                "violation_conditions": "Input not validated",
                "severity": "warning",
                "examples": [],
                "rationale": "Security",
                "tags": ["security"],
            },
        ]

        mock_llm = self._create_mock_llm(mock_response)
        extractor = RuleExtractor(llm_client=mock_llm)

        result = extractor.extract_domain_rules(
            domain="coding",
            description="Python security",
            num_rules=5,
            refine=False,  # Skip refinement for this test
        )

        assert len(result.rules) == 2
        assert result.domain == "coding"
        assert result.rules[0].name == "no_hardcoded_secrets"
        assert result.rules[1].severity == RuleSeverity.WARNING

    def test_extract_with_refinement(self):
        """Test extraction with refinement pass."""
        initial_response = [
            {
                "name": "test_rule",
                "description": "Test",
                "conditions": "Always",
                "violation_conditions": "Never",
                "severity": "error",
                "examples": [],
                "rationale": "",
                "tags": [],
            },
        ]

        refined_response = [
            {
                "name": "test_rule",
                "description": "Refined test rule with better description",
                "conditions": "When testing the system functionality",
                "violation_conditions": "Test assertions fail unexpectedly",
                "severity": "error",
                "examples": ["assert False"],
                "rationale": "Ensures code correctness",
                "tags": ["testing"],
            },
        ]

        mock_llm = Mock(spec=LLMClient)
        mock_llm.model = "gpt-4o-mini"
        mock_llm.complete.side_effect = [
            (json.dumps(initial_response), {"prompt_tokens": 50, "completion_tokens": 100}),
            (json.dumps(refined_response), {"prompt_tokens": 50, "completion_tokens": 100}),
        ]

        extractor = RuleExtractor(llm_client=mock_llm)
        result = extractor.extract_domain_rules(
            domain="coding",
            description="Testing",
            num_rules=1,
            refine=True,
        )

        assert len(result.rules) == 1
        assert "Refined" in result.rules[0].description

    def test_extract_from_examples(self):
        """Test rule extraction from good/bad examples."""
        mock_response = [
            {
                "name": "proper_error_handling",
                "description": "Handle errors properly",
                "conditions": "When code might fail",
                "violation_conditions": "Errors are silently ignored",
                "severity": "error",
                "examples": [],
                "rationale": "From examples",
                "tags": [],
            },
        ]

        mock_llm = self._create_mock_llm(mock_response)
        extractor = RuleExtractor(llm_client=mock_llm)

        examples = [
            {
                "input": "Write file handling code",
                "good_output": "try: ... except IOError: log_error()",
                "bad_output": "try: ... except: pass",
                "reason": "Bad output ignores errors",
            },
        ]

        result = extractor.extract_rules_from_examples(
            domain="coding",
            examples=examples,
        )

        assert len(result.rules) == 1
        assert "error" in result.rules[0].description.lower()

    def test_cost_estimation(self):
        """Test cost estimation."""
        mock_response = [{"name": "test", "description": "test", "conditions": "c",
                         "violation_conditions": "v", "severity": "error",
                         "examples": [], "rationale": "", "tags": []}]

        mock_llm = self._create_mock_llm(mock_response)
        extractor = RuleExtractor(llm_client=mock_llm)

        result = extractor.extract_domain_rules(
            domain="coding",
            description="Test",
            num_rules=1,
            refine=False,
        )

        assert result.total_cost > 0
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 200


# ============================================================================
# Compiler Tests
# ============================================================================

class TestDatalogCompiler:
    """Tests for DatalogCompiler."""

    def test_compile_hardcoded_secrets_rule(self):
        """Test compiling a hardcoded secrets rule (template match)."""
        rule = NaturalRule(
            name="no_hardcoded_secrets",
            description="Never include hardcoded secrets in code",
            domain="coding",
            conditions="When generating code",
            violation_conditions="Code contains API keys or passwords",
        )

        compiler = DatalogCompiler(use_llm_fallback=False)
        result = compiler.compile_rule(rule)

        assert result.is_valid or "manual implementation" in result.datalog_code.lower()
        assert result.rule_id == "no_hardcoded_secrets"

    def test_compile_ungrounded_reference_rule(self):
        """Test compiling ungrounded reference rule (template match)."""
        rule = NaturalRule(
            name="no_ungrounded_references",
            description="Do not reference elements not present in context",
            domain="general",
            conditions="When producing output that references entities",
            violation_conditions="Reference to something not found in input",
        )

        compiler = DatalogCompiler(use_llm_fallback=False)
        result = compiler.compile_rule(rule)

        # Should match ungrounded_reference template
        assert "action_target" in result.datalog_code or "reference" in result.datalog_code.lower()

    def test_compile_ignored_error_rule(self):
        """Test compiling ignored error rule (template match)."""
        rule = NaturalRule(
            name="no_ignored_errors",
            description="Do not ignore errors in the context",
            domain="general",
            conditions="When errors are present",
            violation_conditions="Action ignores error messages",
        )

        compiler = DatalogCompiler(use_llm_fallback=False)
        result = compiler.compile_rule(rule)

        # Should have violation output
        assert "violation" in result.datalog_code.lower()

    def test_compile_multiple_rules(self):
        """Test compiling multiple rules."""
        rules = [
            NaturalRule(
                name="rule_one",
                description="First rule about references",
                domain="coding",
                conditions="Always",
                violation_conditions="Ungrounded reference found",
            ),
            NaturalRule(
                name="rule_two",
                description="Second rule about errors",
                domain="coding",
                conditions="Always",
                violation_conditions="Error ignored in output",
            ),
        ]

        compiler = DatalogCompiler(use_llm_fallback=False)
        result = compiler.compile_rules(rules)

        assert len(result.compiled_rules) + len(result.failed_rules) == 2
        assert result.combined_datalog != ""
        assert "violation" in result.combined_datalog

    def test_compile_with_llm_fallback(self):
        """Test compilation with LLM fallback."""
        mock_llm = Mock()
        mock_llm.complete.return_value = (
            '''
            .decl test_fact(case_id: symbol, value: symbol)
            .input test_fact

            violation(id, "custom_rule", "Custom violation") :-
                test_fact(id, _).
            ''',
            {"prompt_tokens": 50, "completion_tokens": 100},
        )

        rule = NaturalRule(
            name="custom_complex_rule",
            description="A complex rule that needs LLM compilation",
            domain="custom",
            conditions="Complex conditions that don't match templates",
            violation_conditions="Complex violation conditions",
        )

        compiler = DatalogCompiler(llm_client=mock_llm, use_llm_fallback=True)
        result = compiler.compile_rule(rule)

        # Should have used LLM
        assert "llm_generated" in result.datalog_code or result.is_valid

    def test_stub_generation(self):
        """Test stub rule generation when compilation fails."""
        rule = NaturalRule(
            name="unmatchable_rule",
            description="A rule that matches no template",
            domain="exotic",
            conditions="Exotic conditions",
            violation_conditions="Exotic violations",
        )

        compiler = DatalogCompiler(use_llm_fallback=False)
        result = compiler.compile_rule(rule)

        # Should generate a stub
        assert "manual implementation" in result.datalog_code.lower() or "placeholder" in result.datalog_code.lower()

    def test_input_relation_extraction(self):
        """Test extracting input relations from Datalog."""
        compiler = DatalogCompiler()

        datalog = '''
        .decl fact_a(x: symbol)
        .decl fact_b(x: symbol, y: symbol)
        .input fact_a
        .input fact_b

        violation(id, "test", "msg") :- fact_a(id), fact_b(id, _).
        '''

        relations = compiler._extract_input_relations(datalog)

        assert "fact_a" in relations
        assert "fact_b" in relations

    def test_datalog_validation(self):
        """Test Datalog syntax validation."""
        compiler = DatalogCompiler()

        # Valid Datalog
        valid, errors = compiler._validate_datalog('''
        .decl test(x: symbol)
        violation(id, "test", "msg") :- test(id).
        ''')
        assert valid

        # Missing declaration
        valid, errors = compiler._validate_datalog('''
        violation(id, "test", "msg") :- test(id).
        ''')
        assert not valid
        assert any("decl" in e.lower() for e in errors)

        # Unbalanced parens
        valid, errors = compiler._validate_datalog('''
        .decl test(x: symbol
        violation(id :- test(id).
        ''')
        assert not valid


class TestCompilerConvenience:
    """Tests for convenience functions."""

    def test_compile_rules_to_datalog(self):
        """Test convenience function."""
        rules = [
            NaturalRule(
                name="test_rule",
                description="Test ungrounded reference detection",
                domain="general",
                conditions="Always",
                violation_conditions="Reference not in context",
            ),
        ]

        result = compile_rules_to_datalog(rules)

        assert isinstance(result, CompilationResult)
        assert result.statistics["total"] == 1

    def test_compile_single_rule(self):
        """Test convenience function for single rule."""
        rule = NaturalRule(
            name="test_rule",
            description="Ignore error detection",
            domain="general",
            conditions="When errors present",
            violation_conditions="Ignored error in response",
        )

        result = compile_rule(rule)

        assert isinstance(result, CompiledRule)
        assert result.rule_id == "test_rule"


# ============================================================================
# Integration Tests
# ============================================================================

class TestRuleExtractionIntegration:
    """Integration tests for the full rule extraction pipeline."""

    def test_full_pipeline(self):
        """Test extract -> validate -> compile pipeline."""
        # Create mock LLM response with rules that will match templates
        mock_response = [
            {
                "name": "no_ungrounded_references",
                "description": "Do not reference elements that are not present in the context.",
                "conditions": "When generating any output that references entities",
                "violation_conditions": "Output reference is not found in input context",
                "severity": "error",
                "examples": ["Clicking button that doesn't exist"],
                "rationale": "Prevents hallucinated references",
                "tags": ["grounding", "references"],
            },
            {
                "name": "no_ignored_errors",
                "description": "Do not ignore error messages in the context.",
                "conditions": "When errors are present in observations",
                "violation_conditions": "Action ignores error and proceeds anyway",
                "severity": "warning",
                "examples": ["Ignoring file not found error"],
                "rationale": "Errors indicate problems that need addressing",
                "tags": ["errors", "awareness"],
            },
        ]

        mock_llm = Mock(spec=LLMClient)
        mock_llm.model = "gpt-4o-mini"
        mock_llm.complete.return_value = (
            json.dumps(mock_response),
            {"prompt_tokens": 100, "completion_tokens": 200},
        )

        # Extract
        extractor = RuleExtractor(llm_client=mock_llm)
        extraction_result = extractor.extract_domain_rules(
            domain="coding",
            description="Agent behavior",
            num_rules=5,
            refine=False,
        )

        assert len(extraction_result.rules) == 2

        # Validate
        validator = RuleValidator()
        validation_results = validator.validate_rules(extraction_result.rules)

        valid_rules = [r.rule for r in validation_results if r.is_valid]
        assert len(valid_rules) == 2

        # Compile - some may compile to templates, some may need stubs
        compiler = DatalogCompiler(use_llm_fallback=False)
        compilation_result = compiler.compile_rules(valid_rules)

        assert compilation_result.statistics["total"] == 2
        # Either compiled or failed (stubs) - both produce output
        total_processed = len(compilation_result.compiled_rules) + len(compilation_result.failed_rules)
        assert total_processed == 2

    def test_domain_descriptions_coverage(self):
        """Test that all predefined domains have descriptions."""
        for domain in RuleDomain:
            assert domain in DOMAIN_DESCRIPTIONS, f"Missing description for {domain}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
