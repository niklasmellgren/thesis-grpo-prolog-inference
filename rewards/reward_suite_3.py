import os
import re
import uuid
import subprocess
import math
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

_prompt_counter = 0

# ---------------------
# Reward Functions
# ---------------------
# 1) Semantic Similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
def semantic_similarity_reward(completions, answer, **kwargs) -> list[float]:
    """
    Computes a reward based on semantic similarity between generated and reference Prolog code,
    plus partial credit for overlap in predicate names.
    """
    extracted_responses = [extract_xml_answer(comp[0]["content"]) for comp in completions]
    rewards = []
    for model_code, ref_code in zip(extracted_responses, answer):
        if not model_code or not ref_code:
            rewards.append(0.0)
            continue

        # Compute embedding similarity
        embedding_model = semantic_model.encode(model_code, convert_to_tensor=True)
        embedding_ref = semantic_model.encode(ref_code, convert_to_tensor=True)
        cosine_sim = util.cos_sim(embedding_model, embedding_ref).item()

        # Overlap in predicate functors
        preds_model = set(re.findall(r'(\w+)\(', model_code))
        preds_ref = set(re.findall(r'(\w+)\(', ref_code))
        pred_overlap = len(preds_model & preds_ref) / max(1, len(preds_ref))

        reward_val = 0.5 * max(0.5, cosine_sim * 2.0) + 0.5 * (pred_overlap * 2.0)
        rewards.append(reward_val)
    return rewards

# 2) Correctness
def correctness_reward_func(prompts, completions, answer, numerical_result, **kwargs) -> list[float]:
    """
    Uses SWI-Prolog to run the code.
    Compares final numeric output to the correct answer and awards partial if numeric but incorrect.
    """
    responses = [comp[0]["content"] for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    correct_values = numerical_result

    # Optional debugging for the first sample
    if len(responses) > 0:
        question = prompts[0][-1]["content"] if (prompts and prompts[0]) else "N/A"
        print("-" * 20,
              f"Question:\n{question}",
              f"\nReference Prolog answer:\n{answer[0]}",
              f"\nReference Numerical Result:\n{correct_values[0]}",
              f"\nModel Response:\n{responses[0]}",
              f"\nExtracted Code:\n{extracted_responses[0]}")

    model_values = []
    for code in extracted_responses:
        if code:
            mv = execute_prolog_code_subprocess(code)
            if mv is None:
                print("SWI-Prolog returned no output or an error.")
            model_values.append(mv)
        else:
            model_values.append(None)
            print("No Prolog code extracted from the model.")

    rewards = []
    for mv, cv in zip(model_values, correct_values):
        if mv is None or cv is None:
            # Partial reward for trying
            rewards.append(0.5)
            print("Partial Reward: Code missing or no numeric match.")
            continue

        try:
            if mv.startswith("_"):
                rewards.append(0.5)
                print(f"Unbound variable in Prolog output: {mv}")
                continue
            mv_cleaned = mv.strip().split('\n')[-1]
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

# 3) Prolog Structure
def prolog_structure_reward_func(completions, **kwargs) -> list[float]:
    """
    Rewards the presence of multiple (non-solve) predicates and curly-brace constraints.
    Penalizes 'hardcoded' numeric solutions in solve.
    """
    rewards = []
    for idx, comp in enumerate(completions):
        text = comp[0]["content"]
        extracted_code = extract_xml_answer(text)
        print(f"\n=== SAMPLE {idx} ===")
        print("Extracted Code:", extracted_code)

        if not extracted_code:
            rewards.append(0.0)
            print("No code extracted -> reward 0.0")
            continue

        # Analyze structure
        analysis = analyze_prolog_structure_subprocess(extracted_code)
        predicate_count = analysis.get("predicate_count", 0)
        constraint_count = analysis.get("constraint_count", 0)
        print("Analysis:", analysis)
        print(f"Found {predicate_count} predicates, {constraint_count} constraints")

        # Basic scoring
        partial_score = 0.0
        partial_score += min(predicate_count * 0.3, 1.0)      # up to 1.0 from predicates
        partial_score += min(constraint_count * 0.5, 0.5)     # up to 0.5 from constraints

        # Hardcoded numeric solution penalty
        if re.search(r'solve\([^)]*\)\s*:-\s*\{[^}]*=\s*\d+\s*\}', extracted_code):
            print("Detected hardcoded numeric answer -> applying penalty.")
            partial_score *= 0.2

        final_score = max(0.0, min(partial_score, 2.0))
        print("Final Score:", final_score)
        rewards.append(final_score)

    return rewards

# 4) Syntax, Format, XML Counting
def prolog_syntax_reward_func(completions, **kwargs) -> list[float]:
    """
    Rewards presence of typical Prolog syntax: :-, solve(, use_module, clpq, lines ending with '.'.
    """
    pattern = r'(?::-|solve\s*\(|use_module|clpq|\.\s*$)'
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        hits = re.findall(pattern, text, re.MULTILINE)
        score = min(len(hits) * 0.2, 1.0)
        rewards.append(score)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Rewards if output strictly follows <reasoning>...</reasoning>\n<answer>\n...\n</answer>\n
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Rewards if the output uses a <reasoning>.*</reasoning><answer>.*</answer> structure (looser).
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        overflow = len(text.split("\n</answer>\n")[-1])
        count -= min(count, overflow * 0.001)   # never push below 0
    if text.count("\n</answer>") == 1:
        count += 0.125
        overflow = len(text.split("\n</answer>")[-1]) - 1
        count -= min(count, overflow * 0.001)
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# -------------------------------------------------------------
# Progressive reward (curriculum) wrapper
# -------------------------------------------------------------
def progressive_reward_func(
        prompts,
        completions,
        answer,
        numerical_result,
        *,                       # force keyword‑only
        step_idx: int = 0,       # GRPOTrainer injects this
        **kwargs,
) -> list[float]:
    """
    Combines all sub‑rewards and schedules their weights with a
    sigmoid curriculum.  Progress is measured in *prompts* seen.
    """

    global _prompt_counter

    # ---------- individual reward terms -----------------------
    semantic_rewards  = semantic_similarity_reward(completions, answer, **kwargs)
    correctness       = correctness_reward_func(prompts, completions, answer,
                                                numerical_result, **kwargs)
    structure_rewards = prolog_structure_reward_func(completions, **kwargs)
    syntax_rewards    = prolog_syntax_reward_func(completions, **kwargs)

    xml_count   = xmlcount_reward_func(completions, **kwargs)
    soft_fmt    = soft_format_reward_func(completions, **kwargs)
    strict_fmt  = strict_format_reward_func(completions, **kwargs)

    # --- log each sub‑reward’s batch mean to W&B ---
    wandb.log({
        "train/rewards/semantic_similarity_reward": sum(semantic_rewards) / len(semantic_rewards),
        "train/rewards/correctness_reward_func":     sum(correctness) / len(correctness),
        "train/rewards/prolog_structure_reward_func":sum(structure_rewards) / len(structure_rewards),
        "train/rewards/prolog_syntax_reward_func":   sum(syntax_rewards) / len(syntax_rewards),
        "train/rewards/soft_format_reward_func":     sum(soft_fmt) / len(soft_fmt),
        "train/rewards/strict_format_reward_func":   sum(strict_fmt) / len(strict_fmt),
        "train/rewards/xmlcount_reward_func":        sum(xml_count) / len(xml_count),
    }, step=_prompt_counter)
    # ----------------------------------------------------------

    # ---------- curriculum progress ---------------------------
    total_prompts    = len(train_dataset) * training_args.num_train_epochs
    linear_progress  = min(1.0, _prompt_counter / total_prompts)
    k, midpoint      = 12.0, 0.5
    sigmoid_progress = 1.0 / (1.0 + math.exp(-k * (linear_progress - midpoint)))

    # --- log the curriculum schedules ---
    wandb.log({
        "train/progress/linear_progress": linear_progress,
        "train/progress/sigmoid_progress": sigmoid_progress,
    }, step=_prompt_counter)

    # ---------- weight interpolation --------------------------
    early = dict(
        semantic    = 0.35,  # encourage broad, content‑level exploration  
        xml_format  = 0.25,  # enforce the <reasoning>/<answer> scaffolding  
        syntax      = 0.10,  
        correctness = 0.15,  # modest focus on solving (so it doesn’t dominate)  
        structure   = 0.15   # a little structure to keep things well‑formed  
    )
    late = dict(
        semantic    = 0.10,  # exploration of semantics tapers off  
        xml_format  = 0.10,  
        syntax      = 0.10,  
        correctness = 0.45,  # now we really push for right answers  
        structure   = 0.25   # and deeper logical skeletons  
    )

    weights = {k: early[k] + (late[k] - early[k]) * sigmoid_progress for k in early}

    # --- log the dynamic weights ---
    wandb.log({
        "train/weights/semantic":    weights["semantic"],
        "train/weights/xml_format":  weights["xml_format"],
        "train/weights/syntax":      weights["syntax"],
        "train/weights/correctness": weights["correctness"],
        "train/weights/structure":   weights["structure"],
    }, step=_prompt_counter)
    # ----------------------------------------------------------

    # ---- master debug print ----------------------------------
    print(
        f"Progressive weights at prompt {_prompt_counter}/{total_prompts} "
        f"(linear {linear_progress:.3f}, sigmoid {sigmoid_progress:.3f})"
    )
    print("  " + ", ".join(f"{k.capitalize()}: {v:.2f}" for k, v in weights.items()))

    # ---------- combine normalised sub‑rewards ----------------
    combined = []
    for i in range(len(completions)):
        norm_semantic  = min(semantic_rewards[i], 2.0) / 2.0
        norm_correct   = min(correctness[i],      2.0) / 2.0
        norm_structure = min(structure_rewards[i],2.0) / 2.0
        norm_syntax    = min(syntax_rewards[i],   1.0)

        fmt_raw   = xml_count[i] + soft_fmt[i] + strict_fmt[i]
        fmt_score = max(0.0, min(1.0, fmt_raw * 0.67))

        weighted = (
            weights["semantic"]    * norm_semantic +
            weights["xml_format"]  * fmt_score     +
            weights["syntax"]      * norm_syntax   +
            weights["correctness"] * norm_correct  +
            weights["structure"]   * norm_structure
        )
        final_reward = weighted * 2.0
        combined.append(final_reward)

        # # ---- detailed per‑sample print --------
        # if i < 8:
        #     print(f"Sample {i} breakdown:")
        #     print(f"  Semantic    {semantic_rewards[i]:.2f} → {norm_semantic:.2f}"
        #           f" × {weights['semantic']:.2f}")
        #     print(f"  XML format  {fmt_score:.2f} (raw {fmt_raw:.2f})"
        #           f" × {weights['xml_format']:.2f}")
        #     print(f"  Syntax      {syntax_rewards[i]:.2f} → {norm_syntax:.2f}"
        #           f" × {weights['syntax']:.2f}")
        #     print(f"  Correctness {correctness[i]:.2f} → {norm_correct:.2f}"
        #           f" × {weights['correctness']:.2f}")
        #     print(f"  Structure   {structure_rewards[i]:.2f} → {norm_structure:.2f}"
        #           f" × {weights['structure']:.2f}")
        #     print(f"  → final reward {final_reward:.2f}\n")

    _prompt_counter += 1   # advance the global counter
    return combined


