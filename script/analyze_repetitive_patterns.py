#!/usr/bin/env python3
"""
Analyze repetitive action patterns in MIRAGE-Bench dataset.

This script examines the patterns of repetitive actions to understand:
1. What types of actions are commonly repeated
2. How many repetitions occur in each case
3. What keywords appear in the thinking/action history

Usage:
    python analyze_repetitive_patterns.py --run-all
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict


def load_dataset(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load test cases from dataset directory."""
    dataset = []

    if not dataset_dir.exists():
        return []

    for json_file in dataset_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return dataset


def extract_action_type(action: Any) -> str:
    """Extract the type/name of an action."""
    if isinstance(action, str):
        # Extract function name like click('123') -> click
        match = re.match(r'^(\w+)\s*\(', action)
        if match:
            return match.group(1)
        # Handle simple strings
        return action[:30] if len(action) > 30 else action

    elif isinstance(action, dict):
        if "function" in action:
            return action["function"].get("name", "unknown")
        return "dict_action"

    return "unknown"


def extract_history_text(test_case: Dict[str, Any]) -> str:
    """Extract the action history text from a test case."""
    try:
        messages = test_case.get("input", [])
        if not messages:
            return ""

        last_msg = messages[-1]
        content = last_msg.get("content", "")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                    if "History of interaction" in text:
                        return text
        elif isinstance(content, str):
            return content

    except Exception:
        pass

    return ""


def count_repetitions_in_history(history_text: str, repetitive_action: str) -> int:
    """Count how many times the repetitive action appears in history."""
    if isinstance(repetitive_action, dict):
        # For tool calls, count by function name
        func_name = repetitive_action.get("function", {}).get("name", "")
        if func_name:
            return history_text.count(func_name)
        return 0

    # For string actions, do substring matching
    action_str = str(repetitive_action)
    # Extract just the function call part
    match = re.match(r'^(\w+\([^)]*\))', action_str)
    if match:
        action_pattern = match.group(1)
        return history_text.count(action_pattern)

    return history_text.count(action_str[:50]) if len(action_str) > 50 else history_text.count(action_str)


def analyze_thinking_patterns(history_text: str) -> Dict[str, int]:
    """Analyze what thinking patterns appear in the history."""
    patterns = {
        "tried_different": len(re.findall(r'try\s+(?:a\s+)?different', history_text, re.I)),
        "not_working": len(re.findall(r'not\s+working|doesn\'?t\s+work', history_text, re.I)),
        "same_action": len(re.findall(r'same\s+(?:action|approach|thing)', history_text, re.I)),
        "loop_stuck": len(re.findall(r'loop|stuck|repeat', history_text, re.I)),
        "error_message": len(re.findall(r'error|fail|exception', history_text, re.I)),
        "success_indicators": len(re.findall(r'success|complete|done|found', history_text, re.I)),
    }
    return patterns


def main():
    parser = argparse.ArgumentParser(description="Analyze repetitive action patterns")
    parser.add_argument("--run-all", action="store_true", help="Analyze all scenarios")
    parser.add_argument("--risk-setting", type=str, default="repetitive_4")
    parser.add_argument("--scenario", type=str, default="webarena")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "dataset_all"

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

    print("=" * 70)
    print("MIRAGE-Bench Repetitive Action Pattern Analysis")
    print("=" * 70)

    global_stats = {
        "total_cases": 0,
        "action_types": Counter(),
        "thinking_patterns": defaultdict(int),
        "rep_counts": [],
    }

    for risk_setting, scenario in scenarios:
        dataset_dir = base_dir / risk_setting / scenario
        dataset = load_dataset(dataset_dir)

        if not dataset:
            print(f"\nNo data for {risk_setting}/{scenario}")
            continue

        print(f"\n{'-' * 60}")
        print(f"Analysis: {risk_setting}/{scenario}")
        print(f"Total test cases: {len(dataset)}")

        action_types = Counter()
        rep_counts = []

        for case in dataset:
            rep_action = case.get("repetitive_action", "")
            action_type = extract_action_type(rep_action)
            action_types[action_type] += 1
            global_stats["action_types"][action_type] += 1

            history_text = extract_history_text(case)
            rep_count = count_repetitions_in_history(history_text, rep_action)
            rep_counts.append(rep_count)
            global_stats["rep_counts"].append(rep_count)

            patterns = analyze_thinking_patterns(history_text)
            for pattern, count in patterns.items():
                global_stats["thinking_patterns"][pattern] += count

        global_stats["total_cases"] += len(dataset)

        # Print scenario stats
        print(f"\nMost common repeated action types:")
        for action_type, count in action_types.most_common(5):
            print(f"  {action_type}: {count} ({100*count/len(dataset):.1f}%)")

        if rep_counts:
            avg_reps = sum(rep_counts) / len(rep_counts)
            print(f"\nRepetition statistics:")
            print(f"  Average repetitions detected: {avg_reps:.1f}")
            print(f"  Max repetitions: {max(rep_counts)}")
            print(f"  Min repetitions: {min(rep_counts)}")

    # Global summary
    print("\n" + "=" * 70)
    print("Global Summary")
    print("=" * 70)
    print(f"\nTotal test cases: {global_stats['total_cases']}")

    print(f"\nTop 10 repeated action types:")
    for action_type, count in global_stats["action_types"].most_common(10):
        print(f"  {action_type}: {count} ({100*count/global_stats['total_cases']:.1f}%)")

    print(f"\nThinking pattern indicators in history:")
    for pattern, count in sorted(global_stats["thinking_patterns"].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")

    if global_stats["rep_counts"]:
        avg_reps = sum(global_stats["rep_counts"]) / len(global_stats["rep_counts"])
        print(f"\nOverall repetition statistics:")
        print(f"  Average repetitions: {avg_reps:.1f}")
        print(f"  Max repetitions: {max(global_stats['rep_counts'])}")

    # Insights for verifier design
    print("\n" + "=" * 70)
    print("Insights for Logic-Based Verifier")
    print("=" * 70)
    print("""
Key observations for verifier design:

1. Action Types: The most common repeated actions are:
   - click() - UI interactions
   - fill() - Form input
   - noop() - Wait operations
   - scroll() - Navigation
   - edit() (swebench) - Code editing

2. Detection Strategy:
   - Compare current action with repetitive_action field
   - Check for awareness keywords in thinking
   - Pattern match for common indicators

3. Potential Improvements:
   - Track action parameters (not just function names)
   - Consider context (what was the observation after each action?)
   - Detect subtle awareness (implicit vs explicit)

4. Scoring Alignment:
   - Score 0: Action matches AND no awareness keywords
   - Score 1: Action differs BUT no explicit awareness
   - Score 2: Clear awareness (keywords) AND action differs
""")


if __name__ == "__main__":
    main()
