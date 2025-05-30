import os
import re
import uuid
import subprocess
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

def correctness_reward_func(prompts, completions, answer, numerical_result, **kwargs) -> list[float]:
    """
    Compare the model’s executed Prolog answer to the known correct numeric result.
    Provide partial rewards for progress toward correctness during early training.
    This function depends on SWI-Prolog execution results.
    """
    # 1. Get the model's generated text and extract the Prolog snippet
    responses = [comp[0]["content"] for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # 2. Retrieve reference numeric results (passed from dataset)
    correct_values = numerical_result

    # 3. Debug print for the first sample only
    if len(responses) > 0:
        question = prompts[0][-1]["content"] if (prompts and prompts[0]) else "N/A"
        print(
            "-" * 20,
            f"Question:\n{question}",
            f"\nReference Prolog answer:\n{answer[0]}",
            f"\nReference Numerical Result:\n{correct_values[0]}",
            f"\nModel Response:\n{responses[0]}",
            f"\nExtracted Code:\n{extracted_responses[0]}"
        )

    # 4. Execute the model's Prolog code with SWI-Prolog
    model_values = []
    for code in extracted_responses:
        if code:
            mv = execute_prolog_code(code)
            if mv:
                model_values.append(mv)
            else:
                model_values.append(None)
                print("SWI-Prolog returned no output or an error.")
        else:
            model_values.append(None)
            print("No Prolog code extracted from the model.")

    # 5. Compare results and provide rewards
    rewards = []
    for mv, cv in zip(model_values, correct_values):
        if mv is None or cv is None:
            # Partial reward for at least attempting to generate some code
            rewards.append(0.5)
            print("Partial Reward: Model attempted code or code is None, no numeric match.")
            continue

        try:
            # If it's an unbound variable, e.g. "_12345", that's partial credit
            if mv.startswith("_"):
                rewards.append(0.5)
                print(f"Unbound variable in Prolog output: {mv}")
                continue

            mv_cleaned = mv.strip().split('\n')[-1]
            mv_float = float(mv_cleaned)
            cv_float = float(cv)
            print(f"Model Value: {mv_float}, Correct Value: {cv_float}")

            if abs(mv_float - cv_float) < 1e-6:
                # Full reward for correct numeric result
                rewards.append(2.0)
                print("Match: Model value matches correct value.")
            else:
                # Partial reward for producing a numeric result, but not correct
                rewards.append(1.0)
                print("Partial Reward: Model generated a numeric result, but it's incorrect.")
        except Exception as e:
            # Partial credit for at least generating code that runs
            rewards.append(0.5)
            print(f"Error converting model output to float: {e}\nModel: {mv}, Correct: {cv}")

    return rewards

def prolog_syntax_reward_func(completions, **kwargs) -> list[float]:
    """
    Partial reward for including Prolog-specific patterns:
      - ':-' (typical directives, e.g. :- use_module)
      - 'solve('
      - lines ending with '.'
      - 'use_module(library(clpq))'
    """
    pattern = r'(?::-|solve\s*\(|use_module|clpq|\.\s*$)'
    rewards = []
    for c in completions:
        text = c[0]["content"]
        hits = re.findall(pattern, text, re.MULTILINE)
        # Simple approach: #hits * 0.2, capped at 1.0
        score = min(len(hits) * 0.2, 1.0)
        rewards.append(score)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a CoT-like XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:
    """
    A custom function that attempts to parse how well the output
    adheres to your <reasoning>...</reasoning> <answer>...</answer> blocks.
    """
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
