"""
Module for evaluating Prolog code generation models on GSM8K dataset (Single-Try inference).

This script provides utilities to:
- Convert conversation messages to text prompts.
- Extract Prolog code from XML-formatted responses.
- Execute Prolog code via SWI-Prolog subprocess.
- Analyze Prolog code structure using helper predicates.
- Compute various reward functions (semantic similarity, correctness, structure, format, XML).
- Run a full evaluation loop over a dataset, logging results to Weights & Biases.
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

# Regex to ensure the string is a valid simple arithmetic expression.
# Matches only digits (\d), whitespace (\s), plus (+), minus (-), multiply (*), divide (/),
# decimal point (.), and parentheses (() and )). This prevents evaluating any unexpected
# or unsafe characters when using eval.
ARITHMETIC_PATTERN = re.compile(r'^[\d\s+\-*/().]+$')


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
        # Truncate at <|endoftext|> if present
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

    The Prolog file is created, then SWI-Prolog is called with:
        swipl -q -f temp_file -g "solve(X), writeln(X), halt"

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

    Writes the given Prolog code to a temp file, then calls:
        swipl -q -f prolog_helpers.pl -g "analyze_code('temp_file', PredCount, ConstCount), halt"

    The helper script should output lines like:
        PREDICATE_COUNT: <n>
        CONSTRAINT_COUNT: <m>

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

    Uses analyze_prolog_structure_subprocess to retrieve counts.

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

    - Extracts Prolog code from completions using extract_xml_answer.
    - Embeds model code and reference code with SentenceTransformer.
    - Computes cosine similarity.
    - Computes predicate name overlap ratio.
    - Final reward = (cosine_sim + predicate_overlap) / 2.0

    Args:
        completions: List of lists containing dicts with "content" keys. Each completion is [[{"content": text}, ...]].
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

    - Extracts code via extract_xml_answer.
    - Executes code via execute_prolog_code_subprocess.
    - Converts output to float and compares against gold numerical result.

    Reward scheme:
        - Exact match: 2.0
        - Incorrect numeric: 1.0
        - Missing or error: 0.5

    Args:
        prompts: List of message sequences (not used except for logging first prompt).
        completions: List of lists containing dicts with "content" keys.
        answer: List of reference Prolog code strings (for logging).
        numerical_result: List of gold numeric results as strings.

    Returns:
        List of float rewards per completion.
    """
    responses = [comp[0]["content"] for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    correct_values = numerical_result

    # Logging for the first sample (optional)
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
    Compute a reward based on Prolog code structure:

    - Parses <answer>...</answer> to extract code.
    - Uses analyze_prolog_structure_subprocess to count predicates and constraints.
    - Score = min(pred_count * 0.25, 0.75) + min(const_count * 0.3, 0.9)
    - Penalizes hard-coded solve/1 clauses by multiplying by 0.2.
    - Caps final score to [0.0, 2.0].

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

        # Base score from counts
        score = min(pred_count * 0.25, 0.75) + min(const_count * 0.3, 0.9)

        # Penalize hard-coded facts in solve/1
        hardcode_pattern = r"solve\([^)]*\)\s*:-.*(\b\w+\s*=\s*\d+|{\s*\w+\s*=\s*\d+\s*})"
        if re.search(hardcode_pattern, extracted_code, flags=re.DOTALL):
            score *= 0.2

        final_score = max(0.0, min(score, 2.0))
        rewards.append(final_score)

    return rewards


def prolog_syntax_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward based on presence of Prolog syntax keywords or constructs.

    - Looks for tokens: ":-", "solve(", "use_module", "clpq", or trailing ".".

    Each occurrence adds 0.2 up to a cap of 1.0.

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

    Args:
        completions: List of lists containing dicts with "content" keys.

    Returns:
        List of 0.5 or 0.0 per completion.
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

    Args:
        completions: List of lists containing dicts with "content" keys.

    Returns:
        List of 0.5 or 0.0 per completion.
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

    Args:
        text: Response string.

    Returns:
        Float score summing the above contributions.
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
        # Penalize extra lines or characters after </answer>
        remainder = text.split("\n</answer>\n")[-1]
        count -= max(0, len(remainder) - 1) * 0.001

    return count


def xmlcount_reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward based on count_xml metric for each completion.

    Args:
        completions: List of lists containing dicts with "content" keys.

    Returns:
        List of float XML count rewards.
    """
    contents = [comp[0]["content"] for comp in completions]
    return [count_xml(c) for c in contents]


def evaluate_prolog_generation(
    model: FastLanguageModel,
    tokenizer,
    dataset,
    max_new_tokens: int = 1024,
) -> dict:
    """
    Main evaluation loop for Prolog generation on a dataset.

    For each sample:
        1. Convert conversation to text prompt.
        2. Generate model output with Prolog code in XML format.
        3. Extract Prolog code and execute via SWI-Prolog.
        4. Check numeric correctness (strict and arithmetic).
        5. Check structural correctness.
        6. Compute semantic similarity if reference exists.
        7. Log per-sample and aggregate metrics to Weights & Biases.

    Args:
        model: An instance of FastLanguageModel with .fast_generate method.
        tokenizer: Corresponding tokenizer (not directly used here).
        dataset: Iterable of samples, each with keys "prompt", "answer", "numerical_result".
        max_new_tokens: Maximum number of tokens to generate per sample.

    Returns:
        A dict containing:
            - "accuracies": dict of final accuracies (strict, arithmetic, structure, full_correct).
            - "timing": dict of average times (generation, prolog, validation).
            - "details": list of rows from the W&B results table.
    """
    # Initialize semantic similarity model
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Metrics tracking
    metrics: dict[str, object] = {
        "total_samples": 0,
        "strict_correct": 0,
        "arithmetic_correct": 0,
        "structure_correct": 0,
        "full_correct_count": 0,
        "semantic_scores": [],
        "semantic_sum": 0.0,
        "total_semantic": 0.0,
        "generation_times": [],
        "prolog_times": [],
        "validation_times": [],
    }

    # Prepare W&B results table
    results_table = wandb.Table(
        columns=[
            "Sample Index",
            "Question",
            "Reference Answer",
            "Gold Numerical Result",
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

    # Sampling parameters for generation
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

        # Initialize default values per sample
        prompt_text = ""
        model_text = ""
        prolog_code = ""
        final_line = None
        gold_str = ""
        gold_val = None
        error_type = None

        is_strict = False
        is_arithmetic = False
        is_structure = False
        is_full_correct = False
        raw_semantic = 0.0

        generation_time = 0.0
        prolog_exec_time = 0.0
        validation_time = 0.0

        # Per-attempt info dictionary
        attempt_info: dict[str, object] = {
            "model_output": "",
            "extracted_code": "",
            "execution_result": None,
            "is_valid_prolog": False,
            "produces_number": False,
            "is_correct_number": False,
            "structure_valid": False,
            "generation_time": 0.0,
            "prolog_execution_time": 0.0,
            "reason_for_failure": None,
        }

        try:
            # 1. Flatten prompt
            messages = sample["prompt"]
            prompt_text = conversation_to_prompt(messages)
            print(f"\n[Sample {sample_index}] Flattened Prompt:\n{prompt_text}")

            # 2. Parse gold numerical value (if any)
            gold_str = sample.get("numerical_result", "")
            if gold_str:
                try:
                    gold_val = float(gold_str)
                except ValueError:
                    print(
                        f"Warning: Could not convert gold numerical result '{gold_str}' "
                        f"to float for sample {sample_index}"
                    )

            # 3. Generate model output
            gen_start = time.time()
            output_data = model.fast_generate(prompt_text, sampling_params)
            generation_time = time.time() - gen_start
            metrics["generation_times"].append(generation_time)

            model_text = output_data[0].outputs[0].text
            attempt_info["model_output"] = model_text

            # 4. Extract Prolog code
            prolog_code = extract_xml_answer(model_text)
            attempt_info["extracted_code"] = prolog_code or "No code extracted"

            if not prolog_code:
                attempt_info["reason_for_failure"] = "No Prolog code extracted"
                print(f"Sample {sample_index}: No Prolog code extracted.")
            else:
                # 5. Execute Prolog code
                prolog_start = time.time()
                final_line = execute_prolog_code_subprocess(prolog_code)
                prolog_exec_time = time.time() - prolog_start
                metrics["prolog_times"].append(prolog_exec_time)

                attempt_info["execution_result"] = final_line
                attempt_info["is_valid_prolog"] = final_line is not None
                attempt_info["prolog_execution_time"] = prolog_exec_time

                # 6. Check structural correctness
                structure_valid = check_structure_correctness(prolog_code)
                attempt_info["structure_valid"] = structure_valid

                # 7. Check numeric output and correctness
                if final_line:
                    try:
                        num_val = float(final_line)
                        attempt_info["produces_number"] = True
                        if gold_val is not None:
                            is_strict = abs(num_val - gold_val) < 1e-6
                            attempt_info["is_correct_number"] = is_strict
                        success = True
                        print(f"Sample {sample_index}: Numeric output: {final_line}")
                    except ValueError:
                        attempt_info["reason_for_failure"] = (
                            "Prolog output is not a valid number"
                        )
                        print(
                            f"Sample {sample_index}: Prolog code did not yield a "
                            f"numeric result ('{final_line}')."
                        )
                    except Exception as e:
                        attempt_info["reason_for_failure"] = f"Error checking result: {e}"
                        print(f"Sample {sample_index}: Error checking result: {e}")
                else:
                    attempt_info["reason_for_failure"] = "Prolog execution did not return a result"

                # 8. Validate arithmetic evaluation if strict fails
                valid_start = time.time()
                if not is_strict and final_line and ARITHMETIC_PATTERN.match(final_line.strip()):
                    try:
                        eval_val = float(eval(final_line.strip()))
                        is_arithmetic = abs(eval_val - gold_val) < 1e-6 if gold_val is not None else False
                        if not is_arithmetic:
                            error_type = "Arithmetic result incorrect"
                    except Exception as e:
                        error_type = f"Arithmetic conversion error: {e}"
                validation_time = time.time() - valid_start
                metrics["validation_times"].append(validation_time)

                is_structure = structure_valid
                attempt_info["structure_valid"] = is_structure

                # 9. Compute semantic similarity (if reference answer exists)
                reference_answer = [sample.get("answer", "")]
                has_reference = bool(reference_answer[0].strip())
                if final_line and has_reference:
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

                # 10. Determine if fully correct (strict/arithmetic + structure)
                is_full_correct = (is_strict or is_arithmetic) and is_structure

                # Print sample summary
                print(f"\n--- Sample {sample_index} Summary ---")
                if final_line:
                    print("-" * 40)
                    print("Model Output:")
                    print(model_text.strip())
                    print("-" * 40)
                print(f"Prolog Output: {final_line or 'None'}")
                print(f"Gold Value: {gold_str or 'None'}")
                print(
                    f"Strict: {is_strict} | Arithmetic: {is_arithmetic} | "
                    f"Structure: {is_structure} | Full: {is_full_correct}"
                )
                print(f"Semantic Score: {raw_semantic * 100:.2f}%")

        except Exception as e:
            error_type = f"Processing error: {e}"
            print(f"Exception during sample {sample_index} processing: {error_type}")
            attempt_info = {
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
            }

        # Log per-sample results to W&B table
        results_table.add_data(
            sample_index,
            prompt_text,
            sample.get("answer", ""),
            gold_str or "",
            attempt_info["model_output"],
            attempt_info["extracted_code"],
            str(attempt_info["execution_result"]) if attempt_info["execution_result"] else "",
            attempt_info["is_valid_prolog"],
            attempt_info["produces_number"],
            attempt_info["is_correct_number"],
            attempt_info["structure_valid"],
            f"{attempt_info['generation_time']:.3f}",
            f"{attempt_info['prolog_execution_time']:.3f}"
            if attempt_info["prolog_execution_time"] is not None
            else "",
            str(attempt_info["reason_for_failure"]) if attempt_info["reason_for_failure"] else "",
        )

        # Update aggregate metrics
        metrics["total_samples"] += 1
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

        # Compute running accuracies
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
                "time/generation": generation_time,
                "time/prolog_exec": prolog_exec_time,
                "time/validation": validation_time,
                "errors": 1 if error_type else 0,
            },
            step=sample_index,
        )

    # End of dataset loop
    elapsed = time.time() - start_time

    # Safety check
    if metrics["total_samples"] == 0:
        print("WARNING: No samples processed during evaluation")
        return {"accuracies": {}, "timing": {}, "details": []}

    # Compute average times
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

    # Compute final accuracies
    total = metrics["total_samples"]
    final_accuracies = {
        "strict": (metrics["strict_correct"] / total * 100) if total > 0 else 0.0,
        "arithmetic": (metrics["arithmetic_correct"] / total * 100) if total > 0 else 0.0,
        "structure": (metrics["structure_correct"] / total * 100) if total > 0 else 0.0,
        "full_correct": (metrics["full_correct_count"] / total * 100) if total > 0 else 0.0,
    }
    avg_semantic = (
        metrics["semantic_sum"] / total if total > 0 else 0.0
    )
    final_semantic_accuracy = (
        metrics["total_semantic"] / total * 100 if total > 0 else 0.0
    )

    # Log final results to W&B
    wandb.log(
        {
            "detailed_results": results_table,
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

    # Print final summary to console
    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETE ".center(80))
    print("=" * 80)
    print(f"Prolog Accuracy: {final_accuracies['strict']:.2f}%")
    print(f"Arithmetic Accuracy: {final_accuracies['arithmetic']:.2f}%")
    print(f"Structure Accuracy: {final_accuracies['structure']:.2f}%")
    print(f"Fully Correct Accuracy: {final_accuracies['full_correct']:.2f}%")
    print(f"Semantic Accuracy (>= threshold): {final_semantic_accuracy:.2f}%")
    print(f"Average Semantic Score: {avg_semantic:.2f}%")
    print("\nAverage Times:")
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
        name="sp-reflect-rwd1-singletry",
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
