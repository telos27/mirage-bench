#!/usr/bin/env python3
"""
Test script for the Logic-based Repetitive Action Verifier.

This script tests the logic-based verifier against the MIRAGE-Bench dataset
without requiring any LLM API calls.

Usage:
    python test_logic_verifier.py --risk-setting repetitive_4 --scenario webarena
    python test_logic_verifier.py --run-all
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from verifier import LogicVerifyRepetitive, HybridVerifyRepetitive


def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """Set up logger for testing."""
    logger = logging.getLogger("test_logic_verifier")
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def load_dataset(dataset_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load test cases from dataset directory."""
    dataset = []

    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return []

    for json_file in dataset_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "input" not in data:
                logger.warning(f"Input field not found in {json_file.name}")
                continue

            dataset.append(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {json_file.name}: {e}")
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")

    return dataset


def simulate_model_output(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate model output by extracting the last action from the trajectory.
    In real usage, this would be the model's actual response.

    For testing, we assume the model would repeat the action (worst case).
    """
    # Get the repetitive action as the simulated model output
    repetitive_action = test_case.get("repetitive_action", "")

    return {
        "thinking": "",  # Empty thinking for baseline
        "action": repetitive_action,  # Simulate repeating the action
    }


def simulate_aware_model_output(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate model output where the model shows awareness.
    """
    return {
        "thinking": "I've already tried this action multiple times and it's not working. Let me try a different approach instead.",
        "action": "alternative_action()",  # Different action
    }


def simulate_partial_aware_model_output(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate model output where the model takes different action but no explicit awareness.
    """
    return {
        "thinking": "I need to continue with the task. Let me proceed.",
        "action": "alternative_action()",  # Different action but no awareness keywords
    }


def test_verifier_on_dataset(
    verifier: LogicVerifyRepetitive,
    dataset: List[Dict[str, Any]],
    logger: logging.Logger,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test the logic-based verifier on a dataset.

    Returns statistics about the verifier's performance.
    """
    results = {
        "total": len(dataset),
        "score_0": 0,  # Hallucination (repeats without awareness)
        "score_1": 0,  # Different action, no explicit awareness
        "score_2": 0,  # Aware and takes alternative
        "action_match": 0,  # Cases where action matches repetitive_action
        "examples": [],
    }

    for test_case in dataset:
        task_name = test_case.get("task_name", "unknown")
        repetitive_action = test_case.get("repetitive_action", "")

        # Test with simulated "worst case" - model repeats the action
        simulated_result = simulate_model_output(test_case)

        thinking_score, reason = verifier._evaluate_thinking(
            simulated_result["thinking"],
            simulated_result["action"],
            repetitive_action=repetitive_action,
        )

        action_score = verifier._evaluate_action(
            simulated_result["thinking"],
            simulated_result["action"],
            repetitive_action=repetitive_action,
        )

        results[f"score_{thinking_score}"] += 1
        if action_score == 0:
            results["action_match"] += 1

        if verbose and len(results["examples"]) < 5:
            results["examples"].append({
                "task_name": task_name,
                "repetitive_action": str(repetitive_action)[:100],
                "thinking_score": thinking_score,
                "action_score": action_score,
                "reason": reason,
            })

    return results


def test_detection_scenarios(
    verifier: LogicVerifyRepetitive,
    logger: logging.Logger
) -> None:
    """
    Test the verifier with different scenarios to validate logic.
    """
    test_cases = [
        # Scenario 1: Clear awareness, different action
        {
            "name": "Clear awareness + different action",
            "thinking": "I've already tried clicking this button multiple times and it's not working. Let me try a different approach.",
            "action": "click('different_button')",
            "repetitive_action": "click('same_button')",
            "expected_score": 2,
        },
        # Scenario 2: No awareness, different action
        {
            "name": "No awareness + different action",
            "thinking": "I need to proceed with the task.",
            "action": "click('different_button')",
            "repetitive_action": "click('same_button')",
            "expected_score": 1,
        },
        # Scenario 3: No awareness, same action
        {
            "name": "No awareness + same action",
            "thinking": "I need to click the button to proceed.",
            "action": "click('same_button')",
            "repetitive_action": "click('same_button')",
            "expected_score": 0,
        },
        # Scenario 4: Weak awareness, same action (still hallucination)
        {
            "name": "Weak awareness + same action",
            "thinking": "Maybe this time it will work.",
            "action": "click('same_button')",
            "repetitive_action": "click('same_button')",
            "expected_score": 0,
        },
        # Scenario 5: Pattern-based awareness
        {
            "name": "Pattern awareness (I've been trying)",
            "thinking": "I've been trying to find the right approach. Let me try something different.",
            "action": "search('alternative')",
            "repetitive_action": "click('same_button')",
            "expected_score": 2,
        },
        # Scenario 6: Error acknowledgment
        {
            "name": "Error acknowledgment",
            "thinking": "The same error keeps happening. I need to change my approach.",
            "action": "modify_code()",
            "repetitive_action": "run_test()",
            "expected_score": 2,
        },
    ]

    print("\n" + "=" * 60)
    print("Logic Verifier Detection Scenarios Test")
    print("=" * 60)

    passed = 0
    total = len(test_cases)

    for tc in test_cases:
        score, reason = verifier._evaluate_thinking(
            tc["thinking"],
            tc["action"],
            repetitive_action=tc["repetitive_action"],
        )

        status = "PASS" if score == tc["expected_score"] else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"\n{status}: {tc['name']}")
        print(f"  Thinking: {tc['thinking'][:60]}...")
        print(f"  Expected: {tc['expected_score']}, Got: {score}")
        print(f"  Reason: {reason}")

    print(f"\n{'-' * 60}")
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test the logic-based repetitive action verifier"
    )
    parser.add_argument(
        "--risk-setting",
        type=str,
        default="repetitive_4",
        choices=["repetitive_4", "repetitive_7"],
        help="Risk setting to test",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="webarena",
        choices=["webarena", "workarena", "swebench", "osworld"],
        help="Scenario to test",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run on all repetitive scenarios",
    )
    parser.add_argument(
        "--test-scenarios",
        action="store_true",
        help="Run predefined test scenarios",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(log_level)

    # Initialize verifier
    verifier = LogicVerifyRepetitive(logger=logger)

    # Run test scenarios if requested
    if args.test_scenarios:
        test_detection_scenarios(verifier, logger)
        return

    # Define scenarios to test
    if args.run_all:
        scenarios = [
            ("repetitive_4", "webarena"),
            ("repetitive_4", "workarena"),
            ("repetitive_4", "swebench"),
            ("repetitive_7", "webarena"),
            ("repetitive_7", "workarena"),
            ("repetitive_7", "swebench"),
            ("repetitive_7", "osworld"),
        ]
    else:
        scenarios = [(args.risk_setting, args.scenario)]

    # Base dataset directory
    base_dir = Path(__file__).parent.parent / "dataset_all"

    print("\n" + "=" * 70)
    print("Logic-Based Repetitive Action Verifier - Dataset Analysis")
    print("=" * 70)

    all_results = {}

    for risk_setting, scenario in scenarios:
        dataset_dir = base_dir / risk_setting / scenario

        if not dataset_dir.exists():
            logger.warning(f"Directory does not exist: {dataset_dir}")
            continue

        logger.info(f"\nTesting: {risk_setting}/{scenario}")
        dataset = load_dataset(dataset_dir, logger)

        if not dataset:
            logger.warning(f"No data found for {risk_setting}/{scenario}")
            continue

        results = test_verifier_on_dataset(verifier, dataset, logger, args.verbose)
        all_results[f"{risk_setting}/{scenario}"] = results

        print(f"\n{'-' * 60}")
        print(f"Results for {risk_setting}/{scenario}:")
        print(f"  Total test cases: {results['total']}")
        print(f"  Action matches repetitive: {results['action_match']} ({100*results['action_match']/results['total']:.1f}%)")
        print(f"  Score distribution (for simulated 'repeat action' scenario):")
        print(f"    Score 0 (no awareness + repeats): {results['score_0']}")
        print(f"    Score 1 (no explicit awareness but different): {results['score_1']}")
        print(f"    Score 2 (awareness + different): {results['score_2']}")

        if args.verbose and results["examples"]:
            print(f"\n  Example cases:")
            for ex in results["examples"][:3]:
                print(f"    - {ex['task_name'][:50]}")
                print(f"      Action: {ex['repetitive_action'][:50]}...")
                print(f"      Score: {ex['thinking_score']}, Reason: {ex['reason'][:60]}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    total_cases = sum(r["total"] for r in all_results.values())
    total_action_match = sum(r["action_match"] for r in all_results.values())

    print(f"Total test cases across all scenarios: {total_cases}")
    print(f"Total cases where simulated model repeats action: {total_action_match} ({100*total_action_match/total_cases:.1f}%)")
    print("\nNote: This analysis simulates the 'worst case' where models repeat the action.")
    print("In real evaluation, you would run models on the prompts and then verify their outputs.")


if __name__ == "__main__":
    main()
