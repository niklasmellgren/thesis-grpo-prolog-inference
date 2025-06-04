"""
Module for evaluating Prolog code generation models (Agentic-Independent inference).

This script provides utilities to:
- Parse numeric outputs from Prolog.
- Analyze Prolog code structure by counting predicates and constraints.
- Detect potential infinite recursion in generated Prolog code.
- Execute Prolog code via SWI-Prolog subprocess with timeouts.
- Maintain an agentic reasoning loop that:
    - Generates XML-formatted <reasoning> and <answer> blocks.
    - Handles empty generations, duplicates, and non-numeric outputs by bailing each attempt.
    - Prunes conversation context to respect a token budget.
- Extract and dispatch <tool_call> JSON objects to actual functions.
- Calculate an optimal token budget by empirically measuring formatting overhead.
- Evaluate the entire agentic-independent approach over a dataset:
    - Repeatedly run independent agentic tries until a valid numeric prediction is found or a step budget is exhausted.
    - Verify numeric correctness against gold answers.
    - Check structural correctness of the final Prolog code.
    - Compute semantic similarity between generated and reference Prolog code.
    - Log metrics (accuracy, attempts, timing, semantic) to Weights & Biases.
    - Save a full console log and upload it as a W&B artifact.
"""

import subprocess
import re
import time
import wandb
import json
import os
import uuid
from tqdm import tqdm
from unsloth import FastLanguageModel
from vllm import SamplingParams
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from typing import Optional, List, Tuple, Dict
import io
import sys
import contextlib
import pathlib
import datetime

# SP-Struct + agentic addition (You have one tool: <tools> ...)
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
    {X = final constraint logic}.
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


# ─── Prolog Structure Checker ───────────────────────────────────────────────
def analyze_prolog_structure_subprocess(prolog_code: str) -> Dict[str, int]:
    """
    Write Prolog code to a temporary file, call helper script to count predicates
    and constraints, then delete the file.

    Args:
        prolog_code: String containing Prolog predicates and queries.

    Returns:
        A dictionary with keys "predicate_count" and "constraint_count".
    """
    tmp = f"temp_{uuid.uuid4().hex}.pl"
    with open(tmp, "w") as f:
        f.write(prolog_code)
    try:
        res = subprocess.run(
            [
                "swipl",
                "-q",
                "-f",
                "prolog_helpers.pl",
                "-g",
                f"analyze_code('{tmp}', P, C), halt",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pc = cc = 0
        for L in res.stdout.splitlines():
            if L.startswith("PREDICATE_COUNT:"):
                pc = int(L.split(":", 1)[1])
            if L.startswith("CONSTRAINT_COUNT:"):
                cc = int(L.split(":", 1)[1])
        return {"predicate_count": pc, "constraint_count": cc}
    finally:
        os.remove(tmp)


def check_structure_correctness(code: str) -> bool:
    """
    Check if Prolog code has at least one user-defined predicate and one constraint.

    Args:
        code: Prolog source code as a string.

    Returns:
        True if predicate_count >= 1 and constraint_count >= 1, else False.
    """
    s = analyze_prolog_structure_subprocess(code)
    return s["predicate_count"] >= 1 and s["constraint_count"] >= 1


# ─── Recursion Risk Detection ────────────────────────────────────────────────
def detect_recursion_risks(code: str) -> bool:
    """
    Detect patterns that may cause infinite recursion in Prolog code.

    Args:
        code: Prolog source code as a string.

    Returns:
        True if potential recursion risk is detected, else False.
    """
    # Direct self-recursion: foo(...) :- ... foo(...).
    direct_recursion = re.search(r'([a-z]\w*)\s*\([^)]*\)\s*:-[^.]*\1\s*\(', code)
    # Mutual recursion: any predicate that appears in its own body.
    mutually_recursive = False
    predicates = set(re.findall(r'([a-z]\w*)\s*\([^)]*\)\s*:-', code))
    for pred in predicates:
        if re.search(rf'{pred}\s*\([^)]*\)\s*:-[^.]*{pred}\s*\(', code):
            mutually_recursive = True
            break
    return bool(direct_recursion or mutually_recursive)


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

    if ":- use_module(library(clpq))" not in code:
        code = ":- use_module(library(clpq)).\n\n" + code

    if "solve(" not in code:
        m = re.search(r"\b([a-z]\w*)\s*\(\s*X\s*\)\s*:-", code)
        if m:
            first_pred = m.group(1)
            code += f"\n\n% added automatically\nsolve(X) :- {first_pred}(X)."

    tmp = f"temp_{uuid.uuid4().hex}.pl"
    with open(tmp, "w") as f:
        f.write(code)

    try:
        r = subprocess.run(
            ["swipl", "-q", "-f", tmp, "-g", "solve(X), writeln(X), halt"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = r.stdout.strip()
        result = out.splitlines()[-1] if out else None
        return result
    except subprocess.TimeoutExpired:
        print(">>> TIMEOUT: Prolog execution took too long (likely infinite recursion)")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


TOOLS = {"run_prolog": run_prolog}


# ─── Tool-Call Extraction ────────────────────────────────────────────────────
_tool_call_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> List[dict]:
    """
    Find and parse all <tool_call>...</tool_call> JSON objects in the generated text.

    Args:
        text: The assistant’s raw output.

    Returns:
        A list of dictionaries representing tool calls; empty if none found.
    """
    calls = []
    for m in _tool_call_re.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
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


def print_tokens(stage: str, conv: List[dict]) -> None:
    """
    Print current token usage and remaining budget.

    Args:
        stage: Label indicating current stage (e.g., "pre-gen").
        conv: List of messages in the context.
    """
    used = _prompt_tokens(conv)
    rem = TOKEN_BUDGET - used
    pct = (used / TOKEN_BUDGET) * 100
    print(f"[TOKENS:{stage}] used={used} ({pct:.1f}%) | rem={rem} | budget={TOKEN_BUDGET}")


def _shrink_conv(conv: List[dict]) -> Tuple[List[dict], bool]:
    """
    If the conversation exceeds TOKEN_BUDGET, collapse middle messages into a summary.

    Args:
        conv: Full conversation as list of {"role", "content"} dicts.

    Returns:
        (new_conv, pruned_flag)
        - new_conv: Possibly shortened list with summary inserted.
        - pruned_flag: True if summarization occurred.
    """
    if _prompt_tokens(conv) <= TOKEN_BUDGET:
        return conv, False

    must_keep = {0} | set(range(len(conv) - 4, len(conv)))
    summary = []
    new_conv = []

    for i, msg in enumerate(conv):
        if i in must_keep or msg["role"] == "system":
            new_conv.append(msg)
        else:
            snippet = msg["content"].replace("\n", " ")[:60]
            summary.append(f"[{msg['role']}:{snippet}…]")

    if summary:
        summary_msg = "Context too long, compressed: " + " ".join(summary)
        new_conv.insert(1, {"role": "system", "content": summary_msg})

    return new_conv, True


# ─── Revised agentic_loop ─────────────────────────────────────────────────
def agentic_loop(
    model: FastLanguageModel,
    system_prompt: str,
    user_query: str,
    max_steps: int = 20,
    turn_offset: int = 0
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Core loop for agentic reasoning + code execution. Runs up to max_steps rounds.

    At each step:
      - Generate a chunk of XML with <reasoning> and <answer>.
      - If the assistant calls run_prolog, execute it and append the result.
      - Once an <answer> block is found, extract code and attempt numeric parse.
      - Returns as soon as a valid numeric string is found.

    Handles:
      - Empty generations (bail this attempt).
      - Duplicate <answer> blocks (bail this attempt).
      - Non-numeric outputs (bail this attempt).
      - Token budget pruning.

    Args:
        model: An instance of FastLanguageModel with .fast_generate method.
        system_prompt: Initial system message guiding the agent.
        user_query: The user’s problem prompt.
        max_steps: Maximum reasoning steps before giving up this attempt.
        turn_offset: Number of steps already consumed by previous tries.

    Returns:
        (prediction_string, final_code, steps_used)
        - prediction_string: The numeric answer as a string, or None if this attempt failed.
        - final_code: The Prolog code from the successful <answer> block, or None.
        - steps_used: Number of rounds consumed by this attempt.
    """
    BASE_TEMP = 0.20
    SHAKE_FACTOR = 1.15
    SHAKE_EVERY = 2
    ESC_AFTER = 5
    CAP_TEMP = 0.30
    MAX_DUP = 20
    EMPTY_RETRIES = 20

    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    cur_temp = BASE_TEMP
    seen = set()
    dup = 0
    empty_count = 0

    for step in range(max_steps):
        params = SamplingParams(
            temperature=cur_temp,
            top_p=0.95,
            max_tokens=512,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

        # Pre-generation: bail if token budget exceeded
        if _prompt_tokens(conv) > TOKEN_BUDGET * 0.95:
            print(">>> PRE-GEN TOKEN BUDGET EXCEEDED—ABORTING THIS ATTEMPT")
            return None, None, step + 1

        prompt = "\n\n".join(f"({m['role'].upper()}) {m['content']}" for m in conv)
        print_tokens("pre-gen", conv)
        out = model.fast_generate(prompt, params)[0].outputs[0].text

        current_turn = turn_offset + step + 1
        print(f"--- TURN: {current_turn}/{turn_offset+max_steps} ---\n{out}\n")

        # Empty generation handling
        if not out.strip():
            empty_count += 1
            print(f">>> Empty generation detected (#{empty_count})")
            if empty_count >= EMPTY_RETRIES:
                print(f">>> Too many empty generations—aborting this attempt")
                return None, None, step + 1
            print(">>> Aborting this attempt—resetting context for a new independent try")
            return None, None, step + 1

        empty_count = 0

        conv.append({"role": "assistant", "content": out})
        print_tokens("post-gen", conv)
        conv, pruned = _shrink_conv(conv)
        if pruned:
            print(">>> TOKEN BUDGET EXCEEDED AFTER GENERATION—ABORTING THIS ATTEMPT")
            return None, None, step + 1

        # Dispatch any tool calls
        calls = extract_tool_calls(out)
        if calls:
            for c in calls:
                res = TOOLS[c["name"]](**c["arguments"])
                conv.append({"role": "tool", "name": c["name"], "content": str(res)})
                print(f">>> TOOL {c['name']}→{res}")
            continue

        m = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
        if not m:
            continue

        code = m.group(1).strip()
        norm = re.sub(r"\s+", " ", code)
        if norm in seen:
            dup += 1
            if dup % SHAKE_EVERY == 0 and dup < MAX_DUP:
                cur_temp = min(cur_temp * SHAKE_FACTOR, CAP_TEMP)
                print(f">>> duplicate#{dup}, shaking temp to {cur_temp:.2f}")
                continue
            if dup == ESC_AFTER:
                esc = (
                    "SYSTEM REMINDER:\n"
                    f"You have repeated the same <answer> {ESC_AFTER} times and it still fails.  "
                    "Emit ONLY this skeleton with the **correct number**:\n"
                    "<answer>\n"
                    ":- use_module(library(clpq)).\n\n"
                    "solve(X) :-\n"
                    "    {X = NUMBER}.\n"
                    "</answer>\n"
                    "<tool_call>{\"name\":\"run_prolog\",\"arguments\":"
                    "{\"code\":\"...\"}}</tool_call>"
                )
                conv.append({"role": "system", "content": esc})
                continue
            if dup >= 6:
                print(">>> Aborting this attempt—resetting context for a new independent try")
                return None, None, step + 1
            if dup >= MAX_DUP:
                print(">>> too many duplicates—abort")
                return None, code, step + 1
        else:
            seen.add(norm)
            dup = 0
            cur_temp = BASE_TEMP

        res = run_prolog(code)
        print(f">>> run_prolog→ {res}")

        num = _parse_numeric(res or "")
        if num is not None:
            return num, code, step + 1

        # Track non-numeric failures
        if 'numeric_fails' not in locals():
            numeric_fails = 1
        else:
            numeric_fails += 1

        if numeric_fails >= 3:
            print(">>> Aborting this attempt—resetting context for a new independent try")
            return None, None, step + 1

        feedback_msg = (
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
        print("\n>>> FEEDBACK INJECTED:\n" + feedback_msg + "\n")
        conv.append({"role": "user", "content": feedback_msg})
        print_tokens("post-feedback", conv)

    raise RuntimeError("Exhausted max_steps")


# ─── Prompt & Helper ─────────────────────────────────────────────────────
def extract_problem(sample: dict) -> str:
    """
    Extract the user prompt from a dataset sample, handling both lists of messages
    or plain strings.

    Args:
        sample: A dict with key "prompt" that may be a string or a list of dicts.

    Returns:
        The user prompt as a single string.
    """
    p = sample.get("prompt")
    if isinstance(p, list):
        for m in p:
            if m.get("role") == "user":
                return m["content"]
        return " ".join(m.get("content", "") for m in p)
    return p or ""


# ─── Token-Budget Estimation ─────────────────────────────────────────────────
def calculate_optimal_token_budget(
    model_max_tokens: int = 2048,
    safety_margin_pct: float = 5,
    max_samples: int = 10
) -> int:
    """
    Calculate optimal token budget based on empirical measurements across multiple samples.
    Uses real examples from the dataset and measures actual formatting overhead.

    Args:
        model_max_tokens: Maximum tokens supported by the model.
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

    print(f"=== COLLECTING SAMPLES FOR TOKEN BUDGET CALCULATION ===")
    print(f"Using max_samples={max_samples}")

    sample_problems = []
    for idx, sample in enumerate(val_dataset):
        if idx >= max_samples:
            break
        question = extract_problem(sample)
        sample_problems.append(question)

    if not sample_problems:
        raise ValueError("No samples found in dataset for token budget calculation")

    print(f"Successfully collected {len(sample_problems)} samples")
    print("=" * 40)

    overhead_factors = []
    print("\n=== TOKEN BUDGET ANALYSIS ===")
    for i, problem in enumerate(sample_problems):
        sample_conv = [
            {"role": "system", "content": tool_spec_prompt},
            {"role": "user", "content": f"Please solve this problem: {problem}"}
        ]
        sample_conv.append(
            {
                "role": "assistant",
                "content": (
                    "<reasoning>\nAnalyzing the problem...\n</reasoning>\n"
                    "<answer>\n:- use_module(library(clpq)).\n\n"
                    "solve(X) :-\n    {X = 42}.\n</answer>"
                )
            }
        )

        raw_tokens = _prompt_tokens(sample_conv)
        formatted_prompt = "\n\n".join(
            f"({m['role'].upper()}) {m['content']}" for m in sample_conv
        )
        formatted_tokens = _tok_count(formatted_prompt)

        factor = formatted_tokens / raw_tokens
        overhead_factors.append(factor)
        print(f"Sample #{i+1} overhead factor: {factor:.4f}x ({raw_tokens} → {formatted_tokens} tokens)")

    min_factor = min(overhead_factors)
    max_factor = max(overhead_factors)
    avg_factor = sum(overhead_factors) / len(overhead_factors)
    safe_factor = max_factor * (1 + safety_margin_pct / 100)
    optimal_budget = int(model_max_tokens / safe_factor)

    print("\nFormatting overhead statistics:")
    print(f"  - Minimum: {min_factor:.4f}x")
    print(f"  - Average: {avg_factor:.4f}x")
    print(f"  - Maximum: {max_factor:.4f}x")
    print(f"Safety margin: {safety_margin_pct}%")
    print(f"Safe factor applied: {safe_factor:.4f}x")
    print(f"Optimal token budget: {optimal_budget}")
    print(f"This provides {((model_max_tokens / optimal_budget) - 1) * 100:.1f}% headroom")
    print("============================\n")

    return optimal_budget


# Usage: Change this value to use more or fewer samples
TOKEN_BUDGET = calculate_optimal_token_budget(max_samples=375)


# ─── Evaluation Loop ─────────────────────────────────────────────────────────
def evaluate_agentic_prolog(model, dataset, max_steps: int = 20) -> Dict[str, float]:
    """
    Evaluate the agentic-independent Prolog approach on a dataset of math problems.

    For each sample:
      1. Print the question for debugging.
      2. Repeatedly run agentic_loop independently until a valid numeric prediction is found
         or total step budget is exhausted.
      3. Verify numeric correctness against the gold answer.
      4. Check structure correctness of the final Prolog code.
      5. Compute semantic similarity vs. reference <answer> if available.
      6. Log metrics to Weights & Biases and build a results table.

    Args:
        model: An instance of FastLanguageModel.
        dataset: Iterable of samples, each containing "prompt" and "numerical_result".
        max_steps: Total maximum reasoning steps allowed per question across independent tries.

    Returns:
        A dict of final metrics:
            - final/prolog_accuracy
            - final/structure_accuracy
            - final/full_correct_accuracy
            - final/semantic_accuracy
            - final/avg_generation_time
            - final/avg_validation_time
            - final/avg_attempts
            - final/total_time
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
    wb_table = wandb.Table(
        columns=[
            "idx", "question", "gold", "prediction",
            "strict", "structure", "full",
            "attempts", "gen_time", "val_time", "semantic_%"
        ]
    )

    overall_start = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating"), start=1):
        gold = float(sample["numerical_result"])
        question = extract_problem(sample)

        print("\n" + "#" * 70)
        print(f"QUESTION {idx}: {question}")
        print("#" * 70 + "\n")

        # Keep retrying agentic_loop from scratch until we get a numeric result
        total_gen_t = 0.0
        pred = code = None
        total_steps_used = 0
        attempts = 0
        try_count = 0

        # ── Outer retry loop ──────────────────────────────────────────────────
        while True:
            try_count += 1

            remaining_steps = max_steps - total_steps_used
            if remaining_steps <= 0:
                print(f">>> Maximum total steps ({max_steps}) reached—moving to next question")
                break

            print(f">>> INDEPENDENT AGENTIC_TRY #{try_count}")
            t0 = time.time()
            try:
                p, c, steps_used = agentic_loop(
                    model,
                    tool_spec_prompt,
                    f"Please solve this problem: {question}",
                    max_steps=remaining_steps,
                    turn_offset=total_steps_used
                )
            except RuntimeError:
                print(">>> Agentic loop exhausted maximum steps")
                p, c, steps_used = None, None, remaining_steps

            total_steps_used += steps_used
            print(f">>> Total steps used: {total_steps_used}/{max_steps}")

            dt = time.time() - t0
            total_gen_t += dt
            print(f">>> TRY RESULT → pred={p!r}, steps={steps_used}, took {dt:.2f}s")

            if p is not None and _parse_numeric(p):
                pred, code, attempts = p, c, total_steps_used
                break

            if total_steps_used >= max_steps:
                print(f">>> Maximum total steps ({max_steps}) reached—moving to next question")
                break

            print(">>> Bailed—no valid numeric answer; retrying with fresh context\n")

        attempts = total_steps_used
        gen_t = total_gen_t
        stats["gtimes"].append(gen_t)

        t1 = time.time()
        try:
            val = float(pred.split()[0]) if pred else None
            strict = val is not None and abs(val - gold) < 1e-6
        except Exception:
            val, strict = None, False
        val_t = time.time() - t1
        stats["vtimes"].append(val_t)

        struct_ok = check_structure_correctness(code) if code else False
        full_ok = strict and struct_ok

        sem_pct = 0.0
        ref = str(sample.get("answer", "")).strip()
        if code and ref:
            m = re.search(r"<answer>(.*?)</answer>", ref, re.DOTALL)
            ref_code = m.group(1).strip() if m else ref
            try:
                sem_pct = util.cos_sim(
                    sem_model.encode(code, convert_to_tensor=True),
                    sem_model.encode(ref_code, convert_to_tensor=True)
                ).item() * 100
            except Exception:
                sem_pct = 0.0
            stats["sem_sum"] += sem_pct / 100
            stats["sem_cnt"] += 1

        stats["total"] += 1
        stats["strict"] += int(strict)
        stats["struct"] += int(struct_ok)
        stats["full"] += int(full_ok)
        stats["atts"].append(attempts)

        p_acc = stats["strict"] / stats["total"] * 100
        s_acc = stats["struct"] / stats["total"] * 100
        f_acc = stats["full"] / stats["total"] * 100
        m_acc = (stats["sem_sum"] / stats["sem_cnt"] * 100) if stats["sem_cnt"] else 0.0

        print("\n" + "=" * 60)
        print(f" Q#{idx:<3} | Pred: {val} | Gold: {gold}")
        print(f" Flags Strict={strict} Struct={struct_ok} Full={full_ok}")
        print(f" Acc   Prolog={p_acc:.1f}% Struct={s_acc:.1f}% Full={f_acc:.1f}% Sem={m_acc:.1f}%")
        print("=" * 60)

        wb_table.add_data(
            idx,
            question[:120] + ("…" if len(question) > 120 else ""),
            gold,
            val,
            strict,
            struct_ok,
            full_ok,
            attempts,
            f"{gen_t:.2f}",
            f"{val_t:.2f}",
            f"{sem_pct:.1f}"
        )

        wandb.log({
            "live/prolog_acc": p_acc,
            "live/structure_acc": s_acc,
            "live/full_correct_acc": f_acc,
            "live/semantic_score": sem_pct,
            "live/avg_attempts": sum(stats["atts"]) / len(stats["atts"]),
            "time/generation": gen_t,
            "time/validation": val_t
        }, step=idx)

    # ─── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    total = stats["total"]

    final = {
        "final/prolog_accuracy": (stats["strict"] / total * 100) if total else 0.0,
        "final/structure_accuracy": (stats["struct"] / total * 100) if total else 0.0,
        "final/full_correct_accuracy": (stats["full"] / total * 100) if total else 0.0,
        "final/semantic_accuracy": (stats["sem_sum"] / stats["sem_cnt"] * 100)
                                  if stats["sem_cnt"] else 0.0,
        "final/avg_generation_time": (sum(stats["gtimes"]) / len(stats["gtimes"]))
                                     if stats["gtimes"] else 0.0,
        "final/avg_validation_time": (sum(stats["vtimes"]) / len(stats["vtimes"]))
                                     if stats["vtimes"] else 0.0,
        "final/avg_attempts": (sum(stats["atts"]) / len(stats["atts"]))
                              if stats["atts"] else 0.0,
        "final/total_time": elapsed
    }

    wandb.log({"detailed_results": wb_table, **final})
    wandb.summary.update(final)

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETE ".center(60, "="))
    print(f" Prolog Acc:   {final['final/prolog_accuracy']:.2f}%")
    print(f" Structure Acc:{final['final/structure_accuracy']:.2f}%")
    print(f" Full Acc:     {final['final/full_correct_accuracy']:.2f}%")
    print(f" Semantic Acc: {final['final/semantic_accuracy']:.2f}%")
    print(f" Total Time:   {elapsed:.2f}s")
    print("=" * 60)

    return final


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
    wandb.init(
        project="gsm8k-prolog-prover-new-evaluation",
        name="sp-struct-rwd1-agentic-independent",
        settings=wandb.Settings(start_method="thread")
    )

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("eval_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    log_path = out_dir / f"console_{ts}.txt"

    with log_path.open("w", encoding="utf-8") as fp, \
         contextlib.redirect_stdout(Tee(fp, sys.stdout)), \
         contextlib.redirect_stderr(Tee(fp, sys.stderr)):

        final_metrics = evaluate_agentic_prolog(model, val_dataset, max_steps=20)

    print(f"\nFull console saved to {log_path.resolve()}\n")

    art = wandb.Artifact(
        name=f"evaluation-log-{ts}",
        type="evaluation_output",
        description="Combined stdout + stderr from the evaluation run",
    )
    art.add_file(str(log_path))
    wandb.run.log_artifact(art)
