"""
LLM-based rule extraction for generating verification rules.

This module provides tools to:
1. Extract common sense rules from LLMs for any domain
2. Validate extracted rules for completeness and consistency
3. Compile rules to Soufflé Datalog for transparent verification

Example usage:
    from agent_verifier.rule_extraction import (
        RuleExtractor,
        RuleValidator,
        DatalogCompiler,
        create_extractor,
        RuleDomain,
    )

    # Extract rules for a domain
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
"""

# Schemas
from .schemas import (
    NaturalRule,
    RuleSeverity,
    RuleDomain,
    CompiledRule,
    ExtractionResult,
    ValidationResult,
    CompilationResult,
)

# Extractor
from .extractor import (
    RuleExtractor,
    LLMClient,
    create_extractor,
    extract_rules_for_domain,
    DOMAIN_DESCRIPTIONS,
)

# Validator
from .validator import (
    RuleValidator,
    validate_rules,
    filter_valid_rules,
)

# Compiler
from .compiler import (
    DatalogCompiler,
    compile_rules_to_datalog,
    compile_rule,
    DATALOG_TEMPLATES,
)

__all__ = [
    # Schemas
    "NaturalRule",
    "RuleSeverity",
    "RuleDomain",
    "CompiledRule",
    "ExtractionResult",
    "ValidationResult",
    "CompilationResult",
    # Extractor
    "RuleExtractor",
    "LLMClient",
    "create_extractor",
    "extract_rules_for_domain",
    "DOMAIN_DESCRIPTIONS",
    # Validator
    "RuleValidator",
    "validate_rules",
    "filter_valid_rules",
    # Compiler
    "DatalogCompiler",
    "compile_rules_to_datalog",
    "compile_rule",
    "DATALOG_TEMPLATES",
]
