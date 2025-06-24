"""
Module for evaluating Prolog code generation models (Agentic-Internal inference).

This script provides utilities to:
- Extract user prompts from dataset samples.
- Calculate an optimal token budget by measuring formatting overhead.
- Maintain an agentic reasoning loop that:
    - Generates XML-formatted <reasoning> and <answer> blocks using a language model.
    - Detects and executes Prolog code with a “run_prolog” tool, handling infinite-recursion risks.
    - Parses numeric outputs from Prolog and dynamically adjusts prompts on failures or duplicates.
    - Prunes conversation context to respect a token budget.
- Analyze Prolog code structure by counting predicates and constraints via a helper predicate.
- Check structural correctness (at least one user-defined predicate and one constraint).
- Compute semantic similarity between generated Prolog and reference code using SentenceTransformers.
- Track and log metrics (Prolog accuracy, structure accuracy, full correctness, semantic score, timing, attempts) to Weights & Biases.
- Save a complete console log and upload it as a W&B artifact for reproducibility.
"""

import os
import re
import subprocess
import time
import uuid
import json
import io
import sys
import contextlib
import pathlib
import datetime

import wandb
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

from unsloth import FastLanguageModel
from vllm import SamplingParams
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# SP-Struct + Agentic addition
#
# Agentic addition =
# You have one tool:
# <tools>
# {"name":"run_prolog","arguments":[{"code":"string"}]}
# </tools>
# +
# - Use the "run_prolog" tool to execute your answer in the <answer> section.

tool_spec_prompt = """
You are a specialized Prolog code–generating assistant.
You have one tool:

<tools>
{"name":"run_prolog","arguments":[{"code":"string"}]}
</tools>

Your task is to solve math problems by providing a structured answer in two clearly defined sections:

1. <reasoning>
   - Provide a clear, concise step-by-step explanation of how you arrive at the solution.

2. <answer>
   - Provide executable Prolog code using constraint logic programming to compute the numeric answer.
   - Always start with: ':- use_module(library(clpq)).'
   - Define any necessary numeric constants or intermediate values using predicates.
   - Final answer should be unified in solve(X) using curly-brace constraints, without printing commands.

Use this XML format strictly:
<reasoning>
(Your step-by-step reasoning here)
</reasoning>
<answer>
:- use_module(library(clpq)).

(Any predicates/constants defined here)

solve(X) :-
    (Intermediate computations using curly braces)
    {X = final_constraint_logic}.
</answer>

- Use the "run_prolog" tool to execute your answer in the <answer> section.
"""

# ─── Numeric Parsing ─────────────────────────────────────────────────────────
def _parse_numeric(text: str) -> Optional[str]:
    """
    Attempt to parse a string as a float. Strips trailing dot if present.

    Args:
        text: Raw text returned by Prolog.

    Returns:
        The stringified number if it can be converted to float, otherwise None.
    """
    s = text.strip()
    if s.endswith("."):
        s = s[:-1]
    try:
        _ = float(s)
        return s
    except ValueError:
        return None


# ─── Prolog Structure Checker ────────────────────────────────────────────────
def analyze_prolog_structure_subprocess(prolog_code: str) -> Dict[str, int]:
    """
    Write Prolog code to a temporary file, call helper script to count predicates
    and constraints, then delete the file.

    Args:
        prolog_code: String containing Prolog predicates and queries.

    Returns:
        A dictionary with keys "predicate_count" and "constraint_count".
    """
    tmp_path = f"temp_{uuid.uuid4().hex}.pl"
    with open(tmp_path, "w") as f:
        f.write(prolog_code)

    try:
        result = subprocess.run(
            ["swipl", "-q", "-f", "prolog_helpers.pl",
             "-g", f"analyze_code('{tmp_path}', P, C), halt"],
            capture_output=True,
            text=True,
            timeout=10
        )
        pc = cc = 0
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("PREDICATE_COUNT:"):
                pc = int(line.split(":", 1)[1].strip())
            elif line.startswith("CONSTRAINT_COUNT:"):
                cc = int(line.split(":", 1)[1].strip())
        return {"predicate_count": pc, "constraint_count": cc}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def check_structure_correctness(code: str) -> bool:
    """
    Check if Prolog code has at least one predicate (besides solve/1) and one constraint.

    Args:
        code: Prolog source code as a string.

    Returns:
        True if predicate_count >= 1 and constraint_count >= 1, else False.
    """
    counts = analyze_prolog_structure_subprocess(code)
    return counts.get("predicate_count", 0) >= 1 and counts.get("constraint_count", 0) >= 1


# ─── Recursion Risk Detection ────────────────────────────────────────────────
def detect_recursion_risks(code: str) -> bool:
    """
    Detect direct or mutual recursion patterns that may cause infinite loops.

    Args:
        code: Prolog source code as a string.

    Returns:
        True if potential recursion risk is detected, else False.
    """
    # Direct self-recursion: foo(...) :- ... foo(...).
    direct_recursion = re.search(r'([a-z]\w*)\s*\([^)]*\)\s*:-[^.]*\1\s*\(', code)

    # Mutual recursion: look for any predicate appearing in its own body.
    predicates = set(re.findall(r'([a-z]\w*)\s*\([^)]*\)\s*:-', code))
    mutual_recursion = False
    for pred in predicates:
        pattern = rf'{pred}\s*\([^)]*\)\s*:-[^.]*{pred}\s*\('
        if re.search(pattern, code):
            mutual_recursion = True
            break

    return bool(direct_recursion or mutual_recursion)


# ─── Prolog Execution Tool ───────────────────────────────────────────────────
def run_prolog(code: str, timeout: int = 5) -> Optional[str]:
    """
    Execute Prolog code via SWI-Prolog, adding necessary imports and solve/1 if missing.

    Args:
        code: Prolog source code (the <answer> block).
        timeout: Max seconds to wait for Prolog execution.

    Returns:
        The final line of SWI-Prolog stdout, or None on error/timeout.
    """
    if detect_recursion_risks(code):
        print(">>> WARNING: Potential infinite recursion detected in Prolog code")

    # Ensure CLP(Q) library is imported
    if ":- use_module(library(clpq))" not in code:
        code = ":- use_module(library(clpq)).\n\n" + code

    # If there's no solve/1, pick the first defined predicate and wrap it
    if "solve(" not in code:
        m = re.search(r"\b([a-z]\w*)\s*\(\s*X\s*\)\s*:-", code)
        if m:
            first_pred = m.group(1)
            code += f"\n\n% Added automatically\nsolve(X) :- {first_pred}(X)."

    tmp_path = f"temp_{uuid.uuid4().hex}.pl"
    with open(tmp_path, "w") as f:
        f.write(code)

    try:
        process = subprocess.run(
            ["swipl", "-q", "-f", tmp_path, "-g", "solve(X), writeln(X), halt"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = process.stdout.strip()
        return output.splitlines()[-1] if output else None
    except subprocess.TimeoutExpired:
        print(">>> TIMEOUT: Prolog execution took too long (likely infinite recursion)")
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


TOOLS = {"run_prolog": run_prolog}


# ─── Tool-Call Extraction ────────────────────────────────────────────────────
_tool_call_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> List[dict]:
    """
    Find and parse all <tool_call>...</tool_call> JSON objects in the generated text.

    Args:
        text: The assistant's raw output.

    Returns:
        A list of dictionaries representing tool calls; empty if none found.
    """
    calls = []
    for match in _tool_call_re.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


# ─── Token-Counting Helpers ──────────────────────────────────────────────────
# NOTE: TOKENIZER must be defined before using these helpers (e.g., HF tokenizer).

TOKENIZER = tokenizer  # to be assigned externally to a HuggingFace tokenizer instance

def _tok_count(text: str) -> int:
    """
    Count number of tokens in a plain text string using TOKENIZER.

    Args:
        text: Raw string.

    Returns:
        Token count (no BOS/EOS).
    """
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def _prompt_tokens(messages: List[dict]) -> int:
    """
    Compute token usage of a formatted conversation.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Total tokens after formatting each message as "(ROLE) content".
    """
    formatted = "\n\n".join(f"({m['role'].upper()}) {m['content']}" for m in messages)
    return _tok_count(formatted)


def print_tokens(stage: str, conversation: List[dict]) -> None:
    """
    Print current token usage and remaining budget.

    Args:
        stage: Label indicating current stage (e.g., "pre-gen").
        conversation: The list of messages in the context.
    """
    used = _prompt_tokens(conversation)
    remaining = TOKEN_BUDGET - used
    pct = (used / TOKEN_BUDGET) * 100
    print(f"[TOKENS:{stage}] used={used} ({pct:.1f}%) | rem={remaining} | budget={TOKEN_BUDGET}")


def _shrink_conv(conv: List[dict]) -> Tuple[List[dict], bool]:
    """
    If the conversation exceeds TOKEN_BUDGET, collapse middle messages into a summary.

    Args:
        conv: Full conversation as list of {"role", "content"}.

    Returns:
        (new_conv, pruned_flag)
        - new_conv: Possibly shortened list with summary inserted.
        - pruned_flag: True if summarization occurred.
    """
    if _prompt_tokens(conv) <= TOKEN_BUDGET:
        return conv, False

    # Keep first (system) and last 4 messages, summarize the rest
    must_keep = {0} | set(range(len(conv) - 4, len(conv)))
    summary_pieces = []
    new_conv = []

    for i, msg in enumerate(conv):
        if i in must_keep or msg["role"] == "system":
            new_conv.append(msg)
        else:
            snippet = msg["content"].replace("\n", " ")[:60]
            summary_pieces.append(f"[{msg['role']}:{snippet}…]")

    if summary_pieces:
        summary_msg = "Context too long, compressed: " + " ".join(summary_pieces)
        new_conv.insert(1, {"role": "system", "content": summary_msg})

    return new_conv, True


# ─── Agentic Loop ───────────────────────────────────────────────────────────
def agentic_loop(model: FastLanguageModel,
                 system_prompt: str,
                 user_query: str,
                 max_steps: int = 20) -> Tuple[Optional[str], Optional[str], int]:
    """
    Core loop for agentic reasoning + code execution. Generates up to max_steps rounds.

    At each step:
      - Generate a chunk of XML with <reasoning> and <answer>.
      - If the assistant calls run_prolog, execute and append output as a tool message.
      - Once an <answer> block is found, extract code and attempt numeric parse.
      - If a valid number is returned, exit early.

    Handles:
      - Empty generations (up to EMPTY_RETRIES, with resets).
      - Duplicate <answer> blocks (reshake temperature or reset).
      - Non-numeric results (retries with feedback, resets after 3 fails).

    Args:
        model: An instance of FastLanguageModel with .fast_generate method.
        system_prompt: Initial system message guiding the agent.
        user_query: The user's problem prompt.
        max_steps: Maximum reasoning steps before giving up.

    Returns:
        (prediction_string, final_code, steps_taken)
        - prediction_string: The numeric answer as a string, or None if failure.
        - final_code: The Prolog code from the successful <answer> block, or None.
        - steps_taken: Number of iterations completed (1-based).
    """
    BASE_TEMP = 0.20
    SHAKE_FACTOR = 1.15
    SHAKE_EVERY = 2
    ESC_AFTER = 5
    CAP_TEMP = 0.30
    MAX_DUP = 20
    EMPTY_RETRIES = 20  # max consecutive empty generations

    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    cur_temp = BASE_TEMP
    seen_answers = set()
    duplicate_count = 0
    empty_count = 0
    numeric_failures = 0

    for step in range(max_steps):
        params = SamplingParams(
            temperature=cur_temp,
            top_p=0.95,
            max_tokens=512,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

        # Pre-gen: if token usage exceeds 95% of budget, reset context
        if _prompt_tokens(conv) > TOKEN_BUDGET * 0.95:
            print(">>> PRE-GEN TOKEN BUDGET APPROACHING LIMIT: RESETTING CONTEXT")
            conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                {"role": "system", "content": "Please provide a concise solution - previous attempts used too many tokens."}
            ]
            print_tokens("pre-prune", conv)

        # Generate a new assistant turn
        prompt_text = "\n\n".join(f"({m['role'].upper()}) {m['content']}" for m in conv)
        print_tokens("pre-gen", conv)
        output = model.fast_generate(prompt_text, params)[0].outputs[0].text
        print(f"--- TURN {step + 1} ---\n{output}\n")

        # Handle empty generation
        if not output.strip():
            empty_count += 1
            print(f">>> Empty generation detected (#{empty_count})")
            if empty_count >= EMPTY_RETRIES:
                print(f">>> Too many empty generations ({empty_count}) - aborting this problem")
                return None, None, step + 1

            if empty_count >= 2:
                # Reset context after 2 consecutive empties
                print(">>> Multiple empty generations detected—resetting context")
                conv = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "system", "content": "Previous attempts were empty. Please try a fresh approach."}
                ]
                cur_temp = BASE_TEMP
                seen_answers.clear()
                duplicate_count = 0
                print_tokens("post-reset", conv)
            else:
                # Slightly increase temperature and add a hint
                cur_temp = min(cur_temp * SHAKE_FACTOR * 1.2, CAP_TEMP)
                print(f">>> Increasing temperature to {cur_temp:.2f} and trying again")
                conv.append({
                    "role": "system",
                    "content": "The previous generation was empty. Please try again with a complete solution."
                })
            continue

        # Reset empty counter on non-empty output
        empty_count = 0

        # Append assistant output and possibly prune
        conv.append({"role": "assistant", "content": output})
        print_tokens("post-gen", conv)
        conv, pruned = _shrink_conv(conv)
        if pruned:
            print(">>> TOKEN BUDGET EXCEEDED: RESETTING CONTEXT")
            conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                {"role": "system", "content": "Please retry - context was too long."}
            ]
            print_tokens("post-prune", conv)

        # Check for any tool calls (<tool_call> blocks)
        tool_calls = extract_tool_calls(output)
        if tool_calls:
            for call in tool_calls:
                res = TOOLS[call["name"]](**call["arguments"])
                conv.append({"role": "tool", "name": call["name"], "content": str(res)})
                print(f">>> TOOL {call['name']} → {res}")
            continue

        # Search for an <answer> block
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if not answer_match:
            continue

        # Extract and normalize code
        code = answer_match.group(1).strip()
        normalized = re.sub(r"\s+", " ", code)

        # Check for duplicates
        if normalized in seen_answers:
            duplicate_count += 1
            if duplicate_count % SHAKE_EVERY == 0 and duplicate_count < MAX_DUP:
                cur_temp = min(cur_temp * SHAKE_FACTOR, CAP_TEMP)
                print(f">>> Duplicate #{duplicate_count}—shaking temperature to {cur_temp:.2f}")
                continue
            if duplicate_count == ESC_AFTER:
                # After ESC_AFTER duplicates, force a skeleton answer
                reminder = (
                    "SYSTEM REMINDER:\n"
                    f"You have repeated the same <answer> {ESC_AFTER} times and it "
                    "still fails.  Emit ONLY this skeleton with the **correct "
                    "number**:\n"
                    "<answer>\n"
                    ":- use_module(library(clpq)).\n\n"
                    "solve(X) :-\n"
                    "    {X = NUMBER}.\n"
                    "</answer>\n"
                    "<tool_call>{\"name\":\"run_prolog\",\"arguments\":"
                    "{\"code\":\"...\"}}</tool_call>"
                )
                conv.append({"role": "system", "content": reminder})
                continue
            if duplicate_count >= 6:
                # Reset after too many duplicates
                print(">>> Multiple duplicate solutions detected—resetting context")
                conv = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "system", "content": "You've been generating the same incorrect solution repeatedly. Please start with a different approach."}
                ]
                cur_temp = BASE_TEMP
                seen_answers.clear()
                duplicate_count = 0
                print_tokens("post-duplicate-reset", conv)
                continue
            if duplicate_count >= MAX_DUP:
                print(">>> Too many duplicates—aborting")
                return None, code, step + 1
        else:
            seen_answers.add(normalized)
            duplicate_count = 0
            cur_temp = BASE_TEMP

        # Execute the extracted Prolog code
        res = run_prolog(code)
        print(f">>> run_prolog → {res}")

        # Attempt numeric parse
        num_str = _parse_numeric(res or "")
        if num_str is not None:
            return num_str, code, step + 1

        # If non-numeric, track failures and possibly reset context
        numeric_failures += 1
        if numeric_failures >= 3:
            print(">>> Multiple non-numeric results—resetting context")
            conv = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
                {"role": "system", "content": "Previous attempts produced invalid results. Please try a completely fresh approach."}
            ]
            cur_temp = BASE_TEMP
            seen_answers.clear()
            duplicate_count = 0
            numeric_failures = 0
            print_tokens("post-reset", conv)
            continue

        # Otherwise, inject concise feedback and retry
        feedback = (
            "The code failed to produce a numeric result.\n\n"
            "Let's fix it:\n\n"
            "1. Reflect on what went wrong.\n"
            "2. Recalculate\n"
            "3. Adjust your answer to:\n"
            "<answer>\n"
            ":- use_module(library(clpq)).\n\n"
            "solve(X) :-\n"
            "    {X = final_number}.\n"
            "</answer>\n\n"
            "<tool_call>{\n"
            '  "name": "run_prolog",\n'
            '  "arguments": {\n'
            '    "code": ":- use_module(library(clpq)).\\n\\nsolve(X) :- {X = final_number}."\n'
            "  }\n"
            "}</tool_call>"
        )
        print_tokens("pre-feedback", conv)
        print("\n>>> FEEDBACK INJECTED:\n" + feedback + "\n")
        conv.append({"role": "user", "content": feedback})
        print_tokens("post-feedback", conv)

    raise RuntimeError("Exhausted max_steps without finding a numeric result")


# ─── Conversation Helper ─────────────────────────────────────────────────────
def extract_problem(sample: dict) -> str:
    """
    Extract the user prompt from a dataset sample, handling both lists of messages
    or plain strings.

    Args:
        sample: A dictionary with key "prompt" that may be a string or a list of dicts.

    Returns:
        The user prompt as a single string.
    """
    p = sample.get("prompt")
    if isinstance(p, list):
        for msg in p:
            if msg.get("role") == "user":
                return msg["content"]
        return " ".join(msg.get("content", "") for msg in p)
    return p or ""


# ─── Token-Budget Estimation ─────────────────────────────────────────────────
def calculate_optimal_token_budget(model_max_tokens: int = 2048,
                                   safety_margin_pct: float = 5,
                                   max_samples: int = 10) -> int:
    """
    Empirically calculate a safe token budget by measuring formatting overhead.

    Uses a few samples from `val_dataset` to estimate how many extra tokens
    are consumed by the "(ROLE) content" formatting.

    Args:
        model_max_tokens: The maximum tokens supported by the model.
        safety_margin_pct: Percentage to inflate the maximum observed overhead.
        max_samples: Number of examples to measure.

    Returns:
        An integer token budget that leaves headroom for formatting.
    """
    print("\n=== DATASET SAMPLES VERIFICATION ===")
    print("Showing first 3 samples from dataset:")
    for idx, sample in enumerate(val_dataset):
        if idx >= 3:
            break
        question = extract_problem(sample)
        print(f"\nSample #{idx+1} content:")
        print("-" * 40)
        print(question)
        print("-" * 40)
    print("=== END VERIFICATION DISPLAY ===\n")

    print(f"=== COLLECTING SAMPLES FOR TOKEN BUDGET CALCULATION (max_samples={max_samples}) ===")
    sample_problems = []
    for idx, sample in enumerate(val_dataset):
        if idx >= max_samples:
            break
        sample_problems.append(extract_problem(sample))

    if not sample_problems:
        raise ValueError("No samples found in val_dataset for token budget calculation")

    overhead_factors = []
    print("\n=== TOKEN BUDGET ANALYSIS ===")
    for i, problem in enumerate(sample_problems):
        sample_conv = [
            {"role": "system", "content": tool_spec_prompt},
            {"role": "user", "content": f"Please solve this problem: {problem}"},
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\nAnalyzing the problem...\n</reasoning>\n"
                    "<answer>\n:- use_module(library(clpq)).\n\nsolve(X) :- {X = 42}.\n</answer>"
                )
            }
        ]
        raw_tokens = _prompt_tokens(sample_conv)
        formatted_prompt = "\n\n".join(f"({m['role'].upper()}) {m['content']}" for m in sample_conv)
        formatted_tokens = _tok_count(formatted_prompt)

        factor = formatted_tokens / raw_tokens
        overhead_factors.append(factor)
        print(f"Sample #{i+1} overhead: {factor:.4f}x ({raw_tokens}→{formatted_tokens} tokens)")

    min_factor = min(overhead_factors)
    max_factor = max(overhead_factors)
    avg_factor = sum(overhead_factors) / len(overhead_factors)
    safe_factor = max_factor * (1 + safety_margin_pct / 100)
    optimal_budget = int(model_max_tokens / safe_factor)

    print("\nFormatting overhead statistics:")
    print(f"  - Min: {min_factor:.4f}x")
    print(f"  - Avg: {avg_factor:.4f}x")
    print(f"  - Max: {max_factor:.4f}x")
    print(f"Safety margin: {safety_margin_pct}%")
    print(f"Applied factor: {safe_factor:.4f}x")
    print(f"Optimal token budget: {optimal_budget}")
    print(f"Headroom: {((model_max_tokens / optimal_budget) - 1) * 100:.1f}%")
    print("============================\n")

    return optimal_budget


# Set TOKEN_BUDGET using samples from `val_dataset`
TOKEN_BUDGET = calculate_optimal_token_budget(max_samples=375)


# ─── Evaluation Loop ─────────────────────────────────────────────────────────
def evaluate_agentic_prolog(model: FastLanguageModel,
                            dataset,
                            max_steps: int = 8) -> Dict[str, float]:
    """
    Evaluate the agentic internal Prolog approach on a dataset of math problems.

    For each sample:
      1. Print the question for debugging.
      2. Call agentic_loop to obtain (prediction, code, attempts).
      3. Check numeric correctness (strict if abs(pred - gold) < 1e-6).
      4. Check structure correctness with check_structure_correctness.
      5. Compute semantic similarity vs. reference <answer> if available.
      6. Log metrics to Weights & Biases and build a table.

    Args:
        model: An instance of FastLanguageModel.
        dataset: Iterable of samples, each containing "prompt" and "numerical_result".
        max_steps: Max reasoning steps per question.

    Returns:
        A dict of final metrics (prolog_accuracy, structure_accuracy, full_accuracy, semantic_accuracy, times, etc.).
    """
    sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    stats = {
        "total": 0,
        "strict": 0,
        "struct": 0,
        "full": 0,
        "sem_sum": 0.0,
        "sem_cnt": 0,
        "atts": [],
        "gtimes": [],
        "vtimes": []
    }
    wb_table = wandb.Table(columns=[
        "idx", "question", "gold", "prediction",
        "strict", "structure", "full",
        "attempts", "gen_time", "val_time", "semantic_%"
    ])

    overall_start = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating"), start=1):
        gold = float(str(sample["numerical_result"]).replace(",", ""))
        question = extract_problem(sample)

        print("\n" + "#" * 70)
        print(f"QUESTION {idx}: {question}")
        print("#" * 70 + "\n")

        # 1. Run agentic_loop
        t0 = time.time()
        try:
            pred, code, attempts = agentic_loop(
                model,
                tool_spec_prompt,
                f"Please solve this problem: {question}",
                max_steps=max_steps
            )
        except RuntimeError:
            pred, code, attempts = None, None, max_steps
        gen_time = time.time() - t0
        stats["gtimes"].append(gen_time)

        # 2. Numeric validation
        t1 = time.time()
        try:
            val = float(pred.split()[0]) if pred else None
            strict = val is not None and abs(val - gold) < 1e-6
        except Exception:
            val, strict = None, False
        val_time = time.time() - t1
        stats["vtimes"].append(val_time)

        # 3. Structure check
        struct_ok = check_structure_correctness(code) if code else False
        full_ok = strict and struct_ok

        # 4. Semantic similarity (if reference code exists)
        sem_pct = 0.0
        ref_xml = str(sample.get("answer", "")).strip()
        if code and ref_xml:
            m = re.search(r"<answer>(.*?)</answer>", ref_xml, re.DOTALL)
            ref_code = m.group(1).strip() if m else ref_xml
            try:
                sem_pct = util.cos_sim(
                    sem_model.encode(code, convert_to_tensor=True),
                    sem_model.encode(ref_code, convert_to_tensor=True)
                ).item() * 100
            except Exception:
                sem_pct = 0.0
            stats["sem_sum"] += sem_pct / 100
            stats["sem_cnt"] += 1

        # 5. Tally metrics
        stats["total"] += 1
        stats["strict"] += int(strict)
        stats["struct"] += int(struct_ok)
        stats["full"] += int(full_ok)
        stats["atts"].append(attempts)

        # 6. Print live accuracy
        p_acc = stats["strict"] / stats["total"] * 100
        s_acc = stats["struct"] / stats["total"] * 100
        f_acc = stats["full"] / stats["total"] * 100
        m_acc = (stats["sem_sum"] / stats["sem_cnt"] * 100) if stats["sem_cnt"] else 0.0

        print("\n" + "=" * 60)
        print(f" Q#{idx:<3} | Pred: {val} | Gold: {gold}")
        print(f" Flags Strict={strict} Struct={struct_ok} Full={full_ok}")
        print(f" Acc   Prolog={p_acc:.1f}% Struct={s_acc:.1f}% Full={f_acc:.1f}% Sem={m_acc:.1f}%")
        print("=" * 60)

        # 7. Add row to W&B
        wb_table.add_data(
            idx,
            (question[:120] + "…") if len(question) > 120 else question,
            gold,
            val,
            strict,
            struct_ok,
            full_ok,
            attempts,
            f"{gen_time:.2f}",
            f"{val_time:.2f}",
            f"{sem_pct:.1f}"
        )

        # 8. Log live metrics
        wandb.log({
            "live/prolog_acc": p_acc,
            "live/structure_acc": s_acc,
            "live/full_correct_acc": f_acc,
            "live/semantic_score": sem_pct,
            "live/avg_attempts": sum(stats["atts"]) / len(stats["atts"]),
            "time/generation": gen_time,
            "time/validation": val_time
        }, step=idx)

    # ─── Final Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    total = stats["total"]

    final_metrics = {
        "final/prolog_accuracy": (stats["strict"] / total * 100) if total else 0.0,
        "final/structure_accuracy": (stats["struct"] / total * 100) if total else 0.0,
        "final/full_correct_accuracy": (stats["full"] / total * 100) if total else 0.0,
        "final/semantic_accuracy": (stats["sem_sum"] / stats["sem_cnt"] * 100) if stats["sem_cnt"] else 0.0,
        "final/avg_generation_time": (sum(stats["gtimes"]) / len(stats["gtimes"])) if stats["gtimes"] else 0.0,
        "final/avg_validation_time": (sum(stats["vtimes"]) / len(stats["vtimes"])) if stats["vtimes"] else 0.0,
        "final/avg_attempts": (sum(stats["atts"]) / len(stats["atts"])) if stats["atts"] else 0.0,
        "final/total_time": elapsed
    }

    wandb.log({"detailed_results": wb_table, **final_metrics})
    wandb.summary.update(final_metrics)

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETE ".center(60, "="))
    print(f" Prolog Acc:   {final_metrics['final/prolog_accuracy']:.2f}%")
    print(f" Structure Acc:{final_metrics['final/structure_accuracy']:.2f}%")
    print(f" Full Acc:     {final_metrics['final/full_correct_accuracy']:.2f}%")
    print(f" Semantic Acc: {final_metrics['final/semantic_accuracy']:.2f}%")
    print(f" Total Time:   {elapsed:.2f}s")
    print("=" * 60)

    return final_metrics


# ─── Console Logging Helper ─────────────────────────────────────────────────
class Tee(io.TextIOBase):
    """
    Redirect stdout and stderr to both terminal and a log file.
    """
    def __init__(self, logfile_handle, terminal_handle):
        self.log = logfile_handle
        self.term = terminal_handle

    def write(self, text: str) -> int:
        self.term.write(text)
        self.term.flush()
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.term.flush()
        self.log.flush()


# ─── Main Driver ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize W&B
    wandb.init(
        project="gsm8k-prolog-prover-new-evaluation",
        name="sp-struct-rwd1-full-agentic-internal",
        settings=wandb.Settings(start_method="thread")
    )

    # Run evaluation
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("eval_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    log_path = out_dir / f"console_{ts}.txt"

    with log_path.open("w", encoding="utf-8") as fp, \
         contextlib.redirect_stdout(Tee(fp, sys.stdout)), \
         contextlib.redirect_stderr(Tee(fp, sys.stderr)):

        final_metrics = evaluate_agentic_prolog(model, val_dataset, max_steps=20)

    print(f"\nFull console saved to {log_path.resolve()}\n")

    # Upload the console log as a W&B artifact
    art = wandb.Artifact(
        name=f"evaluation-log-{ts}",
        type="evaluation_output",
        description="Saved stdout + stderr from the evaluation run"
    )
    art.add_file(str(log_path))
    wandb.run.log_artifact(art)
