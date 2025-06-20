"""
Module for evaluating Prolog code generation models (Multiple-Try inference).

This script provides utilities to:
- Convert conversation messages to text prompts.
- Extract Prolog code from XML-formatted responses.
- Execute Prolog code via SWI-Prolog subprocess.
- Analyze Prolog code structure using helper predicates.
- Compute various reward functions (semantic similarity, correctness, structure, format, XML).
- Run a multi-attempt evaluation loop over a dataset, logging results to Weights & Biases.
"""

import os
import re
import subprocess
import time
import uuid

import torch
import wandb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from vllm import SamplingParams

from unsloth import FastLanguageModel, is_bfloat16_supported


def conversation_to_prompt(messages: list[dict]) -> str:
    """
    Flatten a list of conversation messages into a prompt string.

    Keeps only the first system message (if any) and the first user message (if any).

    Args:
        messages: List of message dicts with "role" and "content" keys.

    Returns:
        A string containing "(SYSTEM) ..." and "(USER) ..." lines, or empty string.
    """
    system_msg = None
    user_msg = None

    for msg in messages:
        role = msg.get("role")
        if role == "system" and system_msg is None:
            system_msg = msg
        elif role == "user" and user_msg is None:
            user_msg = msg

        if system_msg and user_msg:
            break

    parts: list[str] = []
    if system_msg:
        parts.append(f"(SYSTEM) {system_msg['content']}")
    if user_msg:
        parts.append(f"(USER) {user_msg['content']}")

    return "\n".join(parts).strip()


def extract_xml_answer(text: str) -> str | None:
    """
    Extract the contents of the last complete <answer>...</answer> block from text.

    If an "<|endoftext|>" marker exists, truncate text before extracting.

    Args:
        text: The raw text containing XML-like tags.

    Returns:
        The content inside the <answer> tag, without leading/trailing whitespace;
        or None if no complete <answer> block is found.
    """
    try:
        eot_index = text.find("<|endoftext|>")
        truncated = text[:eot_index] if eot_index != -1 else text

        start_idx = truncated.find("<answer>")
        if start_idx == -1:
            return None

        end_idx = truncated.find("</answer>", start_idx)
        if end_idx == -1:
            return None

        return truncated[start_idx + len("<answer>"):end_idx].strip()
    except Exception:
        return None


def execute_prolog_code_subprocess(prolog_code: str, timeout: int = 5) -> str | None:
    """
    Write Prolog code to a temporary file and invoke SWI-Prolog to solve it.

    Args:
        prolog_code: String containing Prolog predicates and queries.
        timeout: Maximum seconds to wait for SWI-Prolog to finish.

    Returns:
        The final line of SWI-Prolog stdout (e.g., the numeric result), stripped;
        or None if execution fails or returns no output.
    """
    temp_filename = f"temp_{uuid.uuid4().hex}.pl"
    try:
        with open(temp_filename, "w") as f:
            f.write(prolog_code)

        result = subprocess.run(
            [
                "swipl",
                "-q",
                "-f",
                temp_filename,
                "-g",
                "solve(X), writeln(X), halt",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0 or not result.stdout:
            return None

        lines = result.stdout.strip().splitlines()
        return lines[-1].strip() if lines else None

    except Exception as e:
        print(f"Error executing Prolog code: {e}")
        return None

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def analyze_prolog_structure_subprocess(prolog_code: str) -> dict[str, int]:
    """
    Analyze Prolog code structure (predicate count, constraint count) via helper script.

    Args:
        prolog_code: String containing Prolog code.

    Returns:
        Dictionary with keys "predicate_count" and "constraint_count".
        Defaults to zeros if analysis fails.
    """
    temp_filename = f"temp_{uuid.uuid4().hex}.pl"
    try:
        with open(temp_filename, "w") as f:
            f.write(prolog_code)

        result = subprocess.run(
            [
                "swipl",
                "-q",
                "-f",
                "prolog_helpers.pl",
                "-g",
                f"analyze_code('{temp_filename}', PredCount, ConstCount), halt",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        predicate_count = 0
        constraint_count = 0

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("PREDICATE_COUNT:"):
                predicate_count = int(line.split(":", 1)[1].strip())
            elif line.startswith("CONSTRAINT_COUNT:"):
                constraint_count = int(line.split(":", 1)[1].strip())

        return {
            "predicate_count": predicate_count,
            "constraint_count": constraint_count,
        }

    except Exception as e:
        print(f"Error analyzing Prolog structure: {e}")
        return {"predicate_count": 0, "constraint_count": 0}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def check_structure_correctness(prolog_code: str) -> bool:
    """
    Check if Prolog code has at least one predicate (besides solve/1) and one constraint.

    Args:
        prolog_code: String containing Prolog code.

    Returns:
        True if predicate_count >= 1 and constraint_count >= 1, else False.
    """
    if not prolog_code:
        return False

    analysis = analyze_prolog_structure_subprocess(prolog_code)
    pred_count = analysis.get("predicate_count", 0)
    const_count = analysis.get("constraint_count", 0)

    return (pred_count >= 1) and (const_count >= 1)


def semantic_similarity_reward(
    completions: list[list[dict]],
    answer: list[str],
    semantic_model: SentenceTransformer,
    **kwargs,
) -> list[float]:
    """
    Compute semantic similarity reward between generated Prolog code and reference code.

    Args:
        completions: List of lists containing dicts with "content" keys.
        answer: List of reference Prolog code strings.
        semantic_model: Pre-loaded SentenceTransformer model.

    Returns:
        List of floats in [0, 1] representing the semantic similarity reward.
    """
    extracted_responses = [
        extract_xml_answer(comp[0]["content"]) for comp in completions
    ]
    rewards: list[float] = []

    for model_code, ref_code in zip(extracted_responses, answer):
        if not model_code or not ref_code:
            rewards.append(0.0)
            continue

        try:
            embedding_model = semantic_model.encode(model_code, convert_to_tensor=True)
            embedding_ref = semantic_model.encode(ref_code, convert_to_tensor=True)
            cosine_sim = util.cos_sim(embedding_model, embedding_ref).item()

            preds_model = set(re.findall(r"(\w+)\(", model_code))
            preds_ref = set(re.findall(r"(\w+)\(", ref_code))
            pred_overlap = len(preds_model & preds_ref) / max(1, len(preds_ref))

            reward_val = (cosine_sim + pred_overlap) / 2.0
            rewards.append(reward_val)
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            rewards.append(0.0)

    return rewards


def correctness_reward_func(
    prompts: list[list[dict]],
    completions: list[list[dict]],
    answer: list[str],
    numerical_result: list[str],
    **kwargs,
) -> list[float]:
    """
    Compute numeric correctness reward by executing Prolog code and comparing results.

    Args:
        prompts: List of message sequences (used for logging first prompt).
        completions: List of lists containing dicts with "content" keys.
        answer: List of reference Prolog code strings (for logging).
        numerical_result: List of gold numeric results as strings.

    Returns:
        List of float rewards per completion.
    """
    responses = [comp[0]["content"] for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    correct_values = numerical_result

    if responses:
        question = prompts[0][-1]["content"] if (prompts and prompts[0]) else "N/A"
        print("-" * 20)
        print(f"Question:\n{question}")
        print(f"Reference Prolog answer:\n{answer[0]}")
        print(f"Reference Numerical Result:\n{correct_values[0]}")
        print(f"Model Response:\n{responses[0]}")
        print(f"Extracted Code:\n{extracted_responses[0]}")

    model_values: list[str | None] = []
    for code in extracted_responses:
        if code:
            mv = execute_prolog_code_subprocess(code)
            if mv is None:
                print("SWI-Prolog returned no output or an error.")
            model_values.append(mv)
        else:
            model_values.append(None)
            print("No Prolog code extracted from the model.")

    rewards: list[float] = []
    for mv, cv in zip(model_values, correct_values):
        if mv is None or cv is None:
            rewards.append(0.5)
            print("Partial Reward: Code missing or no numeric match.")
            continue

        try:
            if mv.startswith("_"):
                rewards.append(0.5)
                print(f"Unbound variable in Prolog output: {mv}")
                continue

            mv_cleaned = mv.strip().split("\n")[-1]
            mv_float = float(mv_cleaned)
            cv_float = float(cv)
            print(f"Model Value: {mv_float}, Correct Value: {cv_float}")

            if abs(mv_float - cv_float) < 1e-6:
                rewards.append(2.0)
                print("Match: Model value is correct.")
            else:
                rewards.append(1.0)
                print("Partial Reward: Numeric result incorrect.")
        except Exception as e:
            rewards.append(0.5)
            print(f"Error converting output to float: {e}\nModel: {mv}, Correct: {cv}")

    return rewards


def prolog_structure_reward_func(
    completions: list[list[dict]],
    **kwargs,
) -> list[float]:
    """
    Compute a reward based on Prolog code structure.

    Args:
        completions: List of lists containing dicts with "content" keys.

    Returns:
        List of float scores per completion.
    """
    rewards: list[float] = []
    for comp in completions:
        text = comp[0]["content"]
        start_idx = text.find("<answer>")
        end_idx = text.find("</answer>", start_idx)

        if start_idx == -1 or end_idx == -1:
            rewards.append(0.0)
            continue

        extracted_code = text[start_idx + len("<answer>"):end_idx].strip()
        analysis = analyze_prolog_structure_subprocess(extracted_code)
        pred_count = analysis.get("predicate_count", 0)
        const_count = analysis.get("constraint_count", 0)

        score = min(pred_count * 0.25, 0.75) + min(const_count * 0.3, 0.9)
        hardcode_pattern = r"solve\([^)]*\)\s*:-.*(\b\w+\s*=\s*\d+|{\s*\w+\s*=\s*\d+\s*})"
        if re.search(hardcode_pattern, extracted_code, flags=re.DOTALL):
            score *= 0.2

        final_score = max(0.0, min(score, 2.0))
        rewards.append(final_score)

    return rewards


def prolog_syntax_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward based on presence of Prolog syntax keywords or constructs.

    Args:
        completions: List of lists containing dicts with "content" keys.

    Returns:
        List of float syntax scores per completion.
    """
    pattern = r"(?::-|solve\s*\(|use_module|clpq|\.\s*$)"
    rewards: list[float] = []
    for comp in completions:
        text = comp[0]["content"]
        hits = re.findall(pattern, text, flags=re.MULTILINE)
        score = min(len(hits) * 0.2, 1.0)
        rewards.append(score)
    return rewards


def strict_format_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward if the response exactly matches strict XML format:

    <reasoning>
      ...
    </reasoning>
    <answer>
      ...
    </answer>

    Full match yields 0.5, else 0.0.
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    rewards: list[float] = []
    for comp in completions:
        text = comp[0]["content"]
        match = re.match(pattern, text, flags=re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards


def soft_format_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward if the response contains <reasoning>...</reasoning> followed by <answer>...</answer>.

    Partial match yields 0.5, else 0.0.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    rewards: list[float] = []
    for comp in completions:
        text = comp[0]["content"]
        match = re.search(pattern, text, flags=re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards


def count_xml(text: str) -> float:
    """
    Count and score XML tag correctness:

    +0.125 for each of:
      - Exactly one "<reasoning>\n"
      - Exactly one "\n</reasoning>\n"
      - Exactly one "\n<answer>\n"
      - Exactly one "\n</answer>"

    Penalize small deviations in trailing text after </answer>.
    """
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        remainder = text.split("\n</answer>\n")[-1]
        count -= len(remainder) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        remainder = text.split("\n</answer>\n")[-1]
        count -= max(0, len(remainder) - 1) * 0.001
    return count


def xmlcount_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward based on count_xml metric for each completion.
    """
    contents = [comp[0]["content"] for comp in completions]
    return [count_xml(c) for c in contents]


def evaluate_prolog_generation(
    model: FastLanguageModel,
    tokenizer,
    dataset,
    max_new_tokens: int = 1024,
    max_attempts: int = 20,
) -> dict:
    """
    Main evaluation loop for Prolog generation on a dataset (Multiple-Try inference).

    For each sample:
        1. Convert conversation to text prompt.
        2. Loop up to max_attempts:
           a. Generate model output with Prolog code in XML format.
           b. Extract and execute Prolog code via SWI-Prolog.
           c. Attempt to parse the Prolog output directly as a float:
              - If parsing succeeds, mark numeric correctness and break.
              - If not, record failure and try again.
        3. Check structural correctness of the successful Prolog code.
        4. Compute semantic similarity if a reference answer exists.
        5. Log per-attempt and aggregate metrics to Weights & Biases.

    Args:
        model: An instance of FastLanguageModel with .fast_generate method.
        tokenizer: Corresponding tokenizer (not directly used here).
        dataset: Iterable of samples, each with keys "prompt", "answer", "numerical_result".
        max_new_tokens: Maximum number of tokens to generate per attempt.
        max_attempts: Maximum number of attempts per sample.

    Returns:
        A dict containing:
            - "accuracies": dict of final accuracies (strict, arithmetic, structure, full_correct).
            - "timing": dict of average times (generation, prolog, validation).
            - "details": list of rows from the W&B results table.
    """
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    metrics: dict[str, object] = {
        "total_samples": 0,
        "strict_correct": 0,
        "arithmetic_correct": 0,
        "structure_correct": 0,
        "full_correct_count": 0,
        "semantic_scores": [],
        "semantic_sum": 0.0,
        "total_semantic": 0.0,
        "attempts_list": [],
        "generation_times": [],
        "prolog_times": [],
        "validation_times": [],
    }

    results_table = wandb.Table(
        columns=[
            "Sample Index",
            "Question",
            "Reference Answer",
            "Gold Numerical Result",
            "Attempt Number",
            "Is Final Successful Attempt",
            "Model Output",
            "Extracted Code",
            "Execution Result",
            "Is Valid Prolog",
            "Produces Number",
            "Is Correct Number (vs Gold)",
            "Is Structure Valid",
            "Generation Time (s)",
            "Prolog Execution Time (s)",
            "Failure Reason",
        ]
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    start_time = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        sample_index = idx + 1
        has_reference = False
        
        prompt_text = conversation_to_prompt(sample["prompt"])
        print(f"\n[Sample {sample_index}] Flattened Prompt:\n{prompt_text}")

        gold_str = sample.get("numerical_result", "")
        gold_val = None
        if gold_str:
            try:
                gold_val = float(gold_str)
            except ValueError:
                print(
                    f"Warning: Could not convert gold numerical result "
                    f"'{gold_str}' to float for sample {sample_index}"
                )

        # Initialize all per-sample variables (in case of an exception)
        success = False
        final_line = None
        model_text = ""
        prolog_code = ""
        is_strict = False
        is_arithmetic = False
        is_structure = False
        is_full_correct = False
        raw_semantic = 0.0
        error_type = None

        attempts = 0
        successful_attempt_number = -1
        all_attempts: list[dict[str, object]] = []

        try:
            # === Attempt Loop ===
            while attempts < max_attempts and not success:
                attempts += 1

                gen_start = time.time()
                output_data = model.fast_generate(prompt_text, sampling_params)
                generation_time = time.time() - gen_start
                metrics["generation_times"].append(generation_time)

                gen_model_text = output_data[0].outputs[0].text
                gen_prolog_code = extract_xml_answer(gen_model_text)

                attempt_info: dict[str, object] = {
                    "attempt_number": attempts,
                    "model_output": gen_model_text,
                    "extracted_code": gen_prolog_code or "No code extracted",
                    "execution_result": None,
                    "is_valid_prolog": False,
                    "produces_number": False,
                    "is_correct_number": False,
                    "structure_valid": False,
                    "generation_time": generation_time,
                    "prolog_execution_time": None,
                    "reason_for_failure": None,
                }

                if not gen_prolog_code:
                    attempt_info["reason_for_failure"] = "No Prolog code extracted"
                    all_attempts.append(attempt_info)
                    print(f"Attempt {attempts}: No Prolog code extracted.")
                    continue

                prolog_start = time.time()
                gen_final_line = execute_prolog_code_subprocess(gen_prolog_code)
                prolog_exec_time = time.time() - prolog_start
                metrics["prolog_times"].append(prolog_exec_time)

                attempt_info["execution_result"] = gen_final_line
                attempt_info["is_valid_prolog"] = gen_final_line is not None
                attempt_info["prolog_execution_time"] = prolog_exec_time
                attempt_info["structure_valid"] = check_structure_correctness(gen_prolog_code)

                if gen_final_line:
                    try:
                        numeric_val = float(gen_final_line.strip())
                        attempt_info["produces_number"] = True
                        if gold_val is not None:
                            is_strict = abs(numeric_val - gold_val) < 1e-6
                            attempt_info["is_correct_number"] = is_strict

                        model_text = gen_model_text
                        prolog_code = gen_prolog_code
                        final_line = gen_final_line
                        print(f"Attempt {attempts}: Successful numeric output: {final_line}")

                        success = True
                        successful_attempt_number = attempts
                    except (ValueError, AttributeError):
                        attempt_info["reason_for_failure"] = "Output not parseable as float"
                        print(
                            f"Attempt {attempts}: Prolog output not parseable as float "
                            f"('{gen_final_line}')."
                        )
                    except Exception as e:
                        attempt_info["reason_for_failure"] = f"Error checking result: {e}"
                        print(f"Attempt {attempts}: Error checking result: {e}")
                else:
                    attempt_info["reason_for_failure"] = "Prolog execution did not return a result"

                all_attempts.append(attempt_info)

            # === End Attempt Loop ===

            valid_start = time.time()
            validation_time = time.time() - valid_start
            metrics["validation_times"].append(validation_time)

            if success and prolog_code:
                is_structure = check_structure_correctness(prolog_code)

            reference_answer = [sample.get("answer", "")]
            has_reference = bool(reference_answer[0].strip())
            raw_semantic = 0.0
            if success and has_reference:
                completion_wrapper = [[{"content": model_text}]]
                try:
                    semantic_rewards = semantic_similarity_reward(
                        completion_wrapper,
                        reference_answer,
                        semantic_model=semantic_model,
                    )
                    raw_semantic = semantic_rewards[0] if semantic_rewards else 0.0
                except Exception as e:
                    print(f"Semantic similarity error: {e}")
                    raw_semantic = 0.0

            is_full_correct = (is_strict or is_arithmetic) and is_structure

            print(f"\n--- Sample {sample_index} Summary ---")
            print(
                f"Result achieved in attempt: "
                f"{successful_attempt_number if success else 'N/A'} / {attempts}"
            )
            if success:
                print("-" * 40)
                print("Successful Model Output:")
                print(model_text.strip())
                print("-" * 40)
            print(f"Final Prolog Output: {final_line or 'None'}")
            print(f"Gold Value: {gold_str or 'None'}")
            print(
                f"Strict: {is_strict} | Arithmetic: {is_arithmetic} | "
                f"Structure: {is_structure} | Full: {is_full_correct}"
            )
            print(f"Semantic Score: {raw_semantic * 100:.2f}%")

        except Exception as e:
            # If any exception happened before or during attempts, record a placeholder attempt
            error_type = f"Processing error: {str(e)}"
            print(f"DEBUG: Exception during sample {sample_index} processing:", error_type)
            if not all_attempts:
                all_attempts.append({
                    "attempt_number": 1,
                    "model_output": "ERROR",
                    "extracted_code": "ERROR",
                    "execution_result": None,
                    "is_valid_prolog": False,
                    "produces_number": False,
                    "is_correct_number": False,
                    "structure_valid": False,
                    "generation_time": 0.0,
                    "prolog_execution_time": 0.0,
                    "reason_for_failure": error_type,
                })

        # --- Log each attempt to W&B table ---
        for attempt_data in all_attempts:
            results_table.add_data(
                sample_index,
                prompt_text,
                sample.get("answer", ""),
                gold_str or "",
                attempt_data["attempt_number"],
                attempt_data["attempt_number"] == successful_attempt_number,
                attempt_data["model_output"],
                attempt_data["extracted_code"],
                str(attempt_data["execution_result"]) if attempt_data["execution_result"] is not None else "",
                attempt_data["is_valid_prolog"],
                attempt_data["produces_number"],
                attempt_data["is_correct_number"],
                attempt_data["structure_valid"],
                f"{attempt_data['generation_time']:.3f}",
                f"{attempt_data['prolog_execution_time']:.3f}"
                if attempt_data["prolog_execution_time"] is not None
                else "",
                str(attempt_data["reason_for_failure"]) if attempt_data["reason_for_failure"] is not None else ""
            )

        # --- Update and log aggregate metrics ---
        metrics["total_samples"] += 1
        metrics["attempts_list"].append(attempts)
        if is_strict:
            metrics["strict_correct"] += 1
        if is_arithmetic:
            metrics["arithmetic_correct"] += 1
        if is_structure:
            metrics["structure_correct"] += 1
        if is_full_correct:
            metrics["full_correct_count"] += 1
        if has_reference:
            metrics["semantic_scores"].append(raw_semantic)
            metrics["semantic_sum"] += raw_semantic
            if raw_semantic >= 0.7:
                metrics["total_semantic"] += 1

        total = metrics["total_samples"]
        accuracies = {
            "strict": (metrics["strict_correct"] / total * 100) if total > 0 else 0.0,
            "arithmetic": (metrics["arithmetic_correct"] / total * 100) if total > 0 else 0.0,
            "structure": (metrics["structure_correct"] / total * 100) if total > 0 else 0.0,
            "full_correct": (metrics["full_correct_count"] / total * 100) if total > 0 else 0.0,
        }

        print(
            f"Accuracies => Prolog: {accuracies['strict']:.2f}%, "
            f"Arithmetic: {accuracies['arithmetic']:.2f}%, "
            f"Structure: {accuracies['structure']:.2f}%, "
            f"Fully Correct: {accuracies['full_correct']:.2f}%"
        )
        print("-" * 40)

        wandb.log(
            {
                "live/prolog_acc": accuracies["strict"],
                "live/arithmetic_acc": accuracies["arithmetic"],
                "live/structure_acc": accuracies["structure"],
                "live/full_correct_acc": accuracies["full_correct"],
                "live/semantic_score": raw_semantic * 100,
                "live/avg_attempts": sum(metrics["attempts_list"]) / len(metrics["attempts_list"])
                if metrics["attempts_list"]
                else 0,
                "time/generation_successful": generation_time if success else 0,
                "time/prolog_exec_successful": prolog_exec_time if success else 0,
                "time/validation": validation_time,
                "errors": 1 if error_type else 0,
                "sample_total_attempts": attempts,
            },
            step=sample_index,
        )

    # --- Final Aggregation and Logging ---
    elapsed = time.time() - start_time
    if metrics["total_samples"] == 0:
        print("WARNING: No samples processed during evaluation")
        return {"accuracies": {}, "timing": {}, "details": []}

    avg_times = {
        "generation": (
            sum(metrics["generation_times"]) / len(metrics["generation_times"])
            if metrics["generation_times"]
            else 0.0
        ),
        "prolog": (
            sum(metrics["prolog_times"]) / len(metrics["prolog_times"])
            if metrics["prolog_times"]
            else 0.0
        ),
        "validation": (
            sum(metrics["validation_times"]) / len(metrics["validation_times"])
            if metrics["validation_times"]
            else 0.0
        ),
    }

    total = metrics["total_samples"]
    final_accuracies = {
        "strict": (metrics["strict_correct"] / total * 100) if total > 0 else 0.0,
        "arithmetic": (metrics["arithmetic_correct"] / total * 100) if total > 0 else 0.0,
        "structure": (metrics["structure_correct"] / total * 100) if total > 0 else 0.0,
        "full_correct": (metrics["full_correct_count"] / total * 100) if total > 0 else 0.0,
    }
    avg_semantic = metrics["semantic_sum"] / total if total > 0 else 0.0
    final_semantic_accuracy = (metrics["total_semantic"] / total * 100) if total > 0 else 0.0

    wandb.log(
        {
            "detailed_results_per_attempt": results_table,
            "final/prolog_accuracy": final_accuracies["strict"],
            "final/arithmetic_accuracy": final_accuracies["arithmetic"],
            "final/structure_accuracy": final_accuracies["structure"],
            "final/full_correct_accuracy": final_accuracies["full_correct"],
            "final/semantic_accuracy": final_semantic_accuracy,
            "final/avg_semantic_score": avg_semantic,
            "final/total_time": elapsed,
            "final/avg_generation_time_per_attempt": avg_times["generation"],
            "final/avg_prolog_time_per_attempt": avg_times["prolog"],
        }
    )

    wandb.summary.update(
        {
            "prolog_accuracy": final_accuracies["strict"],
            "arithmetic_accuracy": final_accuracies["arithmetic"],
            "structure_accuracy": final_accuracies["structure"],
            "full_correct_accuracy": final_accuracies["full_correct"],
            "semantic_accuracy": final_semantic_accuracy,
            "avg_semantic_score": avg_semantic,
            "avg_generation_time_per_attempt": avg_times["generation"],
            "avg_prolog_time_per_attempt": avg_times["prolog"],
        }
    )

    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETE ".center(80))
    print("=" * 80)
    print(f"Prolog Accuracy: {final_accuracies['strict']:.2f}%")
    print(f"Arithmetic Accuracy: {final_accuracies['arithmetic']:.2f}%")
    print(f"Structure Accuracy: {final_accuracies['structure']:.2f}%")
    print(f"Fully Correct Accuracy: {final_accuracies['full_correct']:.2f}%")
    print(f"Semantic Accuracy (>= threshold): {final_semantic_accuracy:.2f}%")
    print(f"Average Semantic Score: {avg_semantic:.2f}%")
    print("\nAverage Times (per attempt):")
    print(f"  Generation: {avg_times['generation']:.3f}s")
    print(f"  Prolog Execution: {avg_times['prolog']:.3f}s")
    print(f"\nTotal Evaluation Time: {elapsed:.2f} seconds")

    return {
        "accuracies": final_accuracies,
        "timing": avg_times,
        "details": results_table.data,
    }


if __name__ == "__main__":
    wandb.init(
        project="gsm8k-prolog-prover-new-evaluation",
        name="sp-struct-rwd1-multipletry",
        settings=wandb.Settings(start_method="thread"),
        config={"environment": "colab"},
    )

    # Example usage assumes `model`, `tokenizer`, and `val_dataset` are defined elsewhere
    result_stats = evaluate_prolog_generation(
        model=model,
        tokenizer=tokenizer,
        dataset=val_dataset,
    )

    wandb.finish()
