"""Tests for DatalogEngine."""

import pytest
import tempfile
from pathlib import Path

from agent_verifier.reasoning.datalog_engine import (
    DatalogEngine,
    DatalogResult,
    check_souffle_installed,
)


# Skip all tests if Soufflé is not installed
pytestmark = pytest.mark.skipif(
    not check_souffle_installed(),
    reason="Soufflé is not installed"
)


class TestDatalogResult:
    """Tests for DatalogResult."""

    def test_empty_result(self):
        result = DatalogResult(success=True)
        assert result.success
        assert result.outputs == {}
        assert result.error is None

    def test_get_relation(self):
        result = DatalogResult(
            success=True,
            outputs={
                "violation": [("case1", "error", "detail")],
                "empty": [],
            }
        )
        assert result.get_relation("violation") == [("case1", "error", "detail")]
        assert result.get_relation("empty") == []
        assert result.get_relation("nonexistent") == []

    def test_has_facts(self):
        result = DatalogResult(
            success=True,
            outputs={
                "has_data": [("a", "b")],
                "no_data": [],
            }
        )
        assert result.has_facts("has_data")
        assert not result.has_facts("no_data")
        assert not result.has_facts("nonexistent")

    def test_error_result(self):
        result = DatalogResult(success=False, error="Something went wrong")
        assert not result.success
        assert result.error == "Something went wrong"


class TestDatalogEngine:
    """Tests for DatalogEngine."""

    def test_create_engine(self):
        engine = DatalogEngine()
        assert engine.timeout == 30

    def test_escape_string(self):
        assert DatalogEngine.escape_string("hello") == "hello"
        assert DatalogEngine.escape_string('say "hi"') == 'say \\"hi\\"'
        assert DatalogEngine.escape_string("line1\nline2") == "line1 line2"
        assert DatalogEngine.escape_string("") == ""
        assert DatalogEngine.escape_string(None) == ""

    def test_escape_string_truncates(self):
        long_string = "a" * 1000
        escaped = DatalogEngine.escape_string(long_string, max_length=100)
        assert len(escaped) == 100

    def test_add_fact(self):
        engine = DatalogEngine()
        engine.add_fact("test", "value1", "value2")
        assert ("value1", "value2") in engine._facts["test"]

    def test_add_facts(self):
        engine = DatalogEngine()
        facts = [("a", "b"), ("c", "d")]
        engine.add_facts("rel", facts)
        assert len(engine._facts["rel"]) == 2

    def test_clear_facts(self):
        engine = DatalogEngine()
        engine.add_fact("test", "value")
        engine.clear_facts()
        assert len(engine._facts) == 0

    def test_run_simple_program(self):
        """Test running a simple Datalog program."""
        engine = DatalogEngine()

        # Add input facts
        engine.add_fact("parent", "alice", "bob")
        engine.add_fact("parent", "bob", "charlie")

        # Simple transitive closure program
        program = """
        .decl parent(x: symbol, y: symbol)
        .input parent

        .decl ancestor(x: symbol, y: symbol)
        .output ancestor

        ancestor(x, y) :- parent(x, y).
        ancestor(x, z) :- parent(x, y), ancestor(y, z).
        """

        result = engine.run_inline(program, output_relations=["ancestor"])

        assert result.success
        ancestors = result.get_relation("ancestor")
        assert ("alice", "bob") in ancestors
        assert ("bob", "charlie") in ancestors
        assert ("alice", "charlie") in ancestors  # Transitive

    def test_run_program_with_file(self):
        """Test running a program from a file."""
        engine = DatalogEngine()
        engine.add_fact("fact", "test")

        # Create temp program file
        program = """
        .decl fact(x: symbol)
        .input fact

        .decl has_fact(x: symbol)
        .output has_fact

        has_fact(x) :- fact(x).
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dl', delete=False) as f:
            f.write(program)
            temp_path = Path(f.name)

        try:
            result = engine.run_program(temp_path)
            assert result.success
            assert ("test",) in result.get_relation("has_fact")
        finally:
            temp_path.unlink()

    def test_run_program_not_found(self):
        engine = DatalogEngine()
        result = engine.run_program("/nonexistent/path.dl")
        assert not result.success
        assert "not found" in result.error

    def test_run_invalid_program(self):
        """Test handling of invalid Datalog syntax."""
        engine = DatalogEngine()

        invalid_program = """
        .decl broken(
        this is not valid datalog
        """

        result = engine.run_inline(invalid_program)
        assert not result.success
        assert result.error is not None

    def test_violation_detection(self):
        """Test a violation detection pattern."""
        engine = DatalogEngine()

        # Setup: element referenced but not visible
        engine.add_fact("test_case", "test-case")
        engine.add_fact("ref_made", "test-case", "missing-element")
        # Note: no visible fact for "missing-element"

        program = """
        .decl test_case(id: symbol)
        .input test_case

        .decl visible(case_id: symbol, elem: symbol)
        .input visible

        .decl ref_made(case_id: symbol, elem: symbol)
        .input ref_made

        .decl ungrounded(case_id: symbol, elem: symbol)
        .output ungrounded

        ungrounded(id, elem) :-
            ref_made(id, elem),
            !visible(id, elem).
        """

        result = engine.run_inline(program, output_relations=["ungrounded"])

        assert result.success
        ungrounded = result.get_relation("ungrounded")
        assert ("test-case", "missing-element") in ungrounded

    def test_multiple_output_relations(self):
        """Test parsing multiple output relations."""
        engine = DatalogEngine()
        # Use different relations to select different outputs
        engine.add_fact("type_a", "item1")
        engine.add_fact("type_b", "item2")

        program = """
        .decl type_a(x: symbol)
        .input type_a

        .decl type_b(x: symbol)
        .input type_b

        .decl output1(x: symbol)
        .output output1

        .decl output2(x: symbol)
        .output output2

        output1(x) :- type_a(x).
        output2(x) :- type_b(x).
        """

        result = engine.run_inline(program)

        assert result.success
        assert ("item1",) in result.get_relation("output1")
        assert ("item2",) in result.get_relation("output2")


class TestCheckSouffleInstalled:
    """Tests for check_souffle_installed helper."""

    def test_returns_bool(self):
        # This test runs regardless of whether Soufflé is installed
        result = check_souffle_installed()
        assert isinstance(result, bool)
