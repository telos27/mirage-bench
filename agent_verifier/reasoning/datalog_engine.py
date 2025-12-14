"""Soufflé Datalog engine wrapper for deterministic reasoning."""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatalogResult:
    """Result from running a Datalog program."""

    success: bool
    outputs: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)
    error: str | None = None

    def get_relation(self, name: str) -> list[tuple[str, ...]]:
        """Get tuples from a specific output relation."""
        return self.outputs.get(name, [])

    def has_facts(self, name: str) -> bool:
        """Check if a relation has any facts."""
        return len(self.outputs.get(name, [])) > 0


class DatalogEngine:
    """
    Wrapper for Soufflé Datalog engine.

    Provides deterministic, transparent reasoning via Datalog rules.
    All reasoning is auditable - facts in, derived facts out.

    Usage:
        engine = DatalogEngine()

        # Add input facts
        engine.add_fact("visible_element", "case1", "submit button")
        engine.add_fact("reference_made", "case1", "submit button")

        # Run rules
        result = engine.run_program("path/to/rules.dl")

        # Check derived facts
        violations = result.get_relation("violation")
    """

    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize the Datalog engine.

        Args:
            timeout_seconds: Timeout for Soufflé execution
        """
        self.timeout = timeout_seconds
        self._facts: dict[str, list[tuple[str, ...]]] = {}
        self._check_souffle_available()

    def _check_souffle_available(self) -> None:
        """Verify Soufflé is installed and accessible."""
        try:
            result = subprocess.run(
                ["souffle", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Soufflé returned error: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Soufflé is not installed. "
                "Install with: apt-get install souffle (Linux) or brew install souffle (macOS)"
            )

    @staticmethod
    def escape_string(s: str, max_length: int = 500) -> str:
        """
        Escape a string for use in Soufflé facts.

        Args:
            s: String to escape
            max_length: Maximum length (truncate if longer)

        Returns:
            Escaped string safe for Soufflé
        """
        if not s:
            return ""
        # Escape special characters
        s = s.replace('\\', '\\\\')
        s = s.replace('"', '\\"')
        s = s.replace('\n', ' ')
        s = s.replace('\t', ' ')
        s = s.replace('\r', '')
        # Truncate if too long
        return s[:max_length]

    def clear_facts(self) -> None:
        """Clear all input facts."""
        self._facts.clear()

    def add_fact(self, relation: str, *values: str) -> None:
        """
        Add a fact to an input relation.

        Args:
            relation: Name of the relation
            *values: Values for the tuple (will be escaped)
        """
        if relation not in self._facts:
            self._facts[relation] = []
        escaped = tuple(self.escape_string(str(v)) for v in values)
        self._facts[relation].append(escaped)

    def add_facts(self, relation: str, facts: list[tuple[str, ...]]) -> None:
        """
        Add multiple facts to a relation.

        Args:
            relation: Name of the relation
            facts: List of tuples to add
        """
        for fact in facts:
            self.add_fact(relation, *fact)

    def _write_facts_file(self, facts_dir: Path, relation: str, facts: list[tuple[str, ...]]) -> None:
        """Write facts to a TSV file for Soufflé input.

        Note: Soufflé fact files use tab-separated values WITHOUT quotes.
        Values with special characters should be pre-escaped.
        """
        filepath = facts_dir / f"{relation}.facts"
        with open(filepath, "w") as f:
            for fact in facts:
                line = "\t".join(str(v) for v in fact)
                f.write(line + "\n")

    def _parse_output_file(self, output_dir: Path, relation: str) -> list[tuple[str, ...]]:
        """Parse a Soufflé output CSV file."""
        filepath = output_dir / f"{relation}.csv"
        if not filepath.exists():
            return []

        results = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Parse tab-separated values, removing quotes
                values = []
                for part in line.split('\t'):
                    part = part.strip()
                    if part.startswith('"') and part.endswith('"'):
                        part = part[1:-1]
                    values.append(part)
                results.append(tuple(values))
        return results

    def _extract_input_relations(self, program: str) -> set[str]:
        """
        Extract all input relation names from a Datalog program.

        Looks for patterns like:
            .input relation_name
            .decl relation_name(...) / .input relation_name

        Args:
            program: The Datalog program text

        Returns:
            Set of input relation names
        """
        import re
        # Match ".input relation_name" declarations
        input_pattern = re.compile(r'\.input\s+(\w+)', re.MULTILINE)
        return set(input_pattern.findall(program))

    def _ensure_all_fact_files(self, facts_dir: Path, input_relations: set[str]) -> None:
        """
        Ensure all input relations have a fact file (even if empty).

        Soufflé requires fact files to exist for all declared input relations.

        Args:
            facts_dir: Directory containing fact files
            input_relations: Set of input relation names from the program
        """
        for relation in input_relations:
            filepath = facts_dir / f"{relation}.facts"
            if not filepath.exists():
                # Create empty fact file
                filepath.touch()

    def run_program(
        self,
        program_path: str | Path,
        output_relations: list[str] | None = None,
    ) -> DatalogResult:
        """
        Run a Datalog program with the current facts.

        Args:
            program_path: Path to the .dl file
            output_relations: List of output relation names to parse (optional)

        Returns:
            DatalogResult with success status and derived facts
        """
        program_path = Path(program_path)
        if not program_path.exists():
            return DatalogResult(
                success=False,
                error=f"Program not found: {program_path}"
            )

        # Read program to find input relations
        with open(program_path) as f:
            program_text = f.read()
        input_relations = self._extract_input_relations(program_text)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            facts_dir = tmpdir / "facts"
            output_dir = tmpdir / "output"
            facts_dir.mkdir()
            output_dir.mkdir()

            # Write all input facts
            for relation, facts in self._facts.items():
                self._write_facts_file(facts_dir, relation, facts)

            # Ensure all declared input relations have fact files (even if empty)
            self._ensure_all_fact_files(facts_dir, input_relations)

            # Run Soufflé
            try:
                result = subprocess.run(
                    [
                        "souffle",
                        "-F", str(facts_dir),
                        "-D", str(output_dir),
                        str(program_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if result.returncode != 0:
                    return DatalogResult(
                        success=False,
                        error=f"Soufflé error: {result.stderr}"
                    )

                # Parse outputs
                outputs = {}

                # If specific relations requested, parse those
                if output_relations:
                    for rel in output_relations:
                        outputs[rel] = self._parse_output_file(output_dir, rel)
                else:
                    # Otherwise, parse all .csv files in output
                    for csv_file in output_dir.glob("*.csv"):
                        rel_name = csv_file.stem
                        outputs[rel_name] = self._parse_output_file(output_dir, rel_name)

                return DatalogResult(success=True, outputs=outputs)

            except subprocess.TimeoutExpired:
                return DatalogResult(
                    success=False,
                    error=f"Soufflé timed out after {self.timeout}s"
                )
            except Exception as e:
                return DatalogResult(
                    success=False,
                    error=f"Soufflé execution failed: {e}"
                )

    def run_inline(
        self,
        program: str,
        output_relations: list[str] | None = None,
    ) -> DatalogResult:
        """
        Run an inline Datalog program (as a string).

        Args:
            program: Datalog program as a string
            output_relations: List of output relation names to parse

        Returns:
            DatalogResult with success status and derived facts
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dl', delete=False) as f:
            f.write(program)
            temp_path = Path(f.name)

        try:
            return self.run_program(temp_path, output_relations)
        finally:
            temp_path.unlink()


def check_souffle_installed() -> bool:
    """Check if Soufflé is installed without raising an error."""
    try:
        result = subprocess.run(
            ["souffle", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
