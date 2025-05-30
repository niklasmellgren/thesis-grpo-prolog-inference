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
from typing import Optional, List, Tuple
import io, sys, contextlib, pathlib, datetime

import tiktoken 

# Accepts 12   -7   3.14159   +0.5
# Rejects 611r5   12+   1,234   2e3   etc.
NUMERIC_RE = re.compile(r'^[+-]?\d+(?:\.\d+)?$')

def _parse_numeric(text: str) -> str | None:
    s = text.strip()
    if s.endswith('.'):
        s = s[:-1]
    return s if NUMERIC_RE.match(s) else None

# ─── Prolog Structure Checker ───────────────────────────────────────────────
def analyze_prolog_structure_subprocess(prolog_code: str) -> dict:
    tmp = f"temp_{uuid.uuid4().hex}.pl"
    with open(tmp, "w") as f:
        f.write(prolog_code)
    try:
        res = subprocess.run(
            ["swipl", "-q", "-f", "prolog_helpers.pl",
             "-g", f"analyze_code('{tmp}', P, C), halt"],
            capture_output=True, text=True, timeout=10
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
    s = analyze_prolog_structure_subprocess(code)
    return s["predicate_count"] >= 1 and s["constraint_count"] >= 1

# Add this detection function
def detect_recursion_risks(code: str) -> bool:
    """Detect patterns that may cause infinite recursion in Prolog code."""
    # Look for self-recursive predicates without clear termination conditions
    direct_recursion = re.search(r'([a-z]\w*)\s*\([^)]*\)\s*:-[^.]*\1\s*\(', code)
    # Look for mutual recursion
    mutual_recursion = False
    predicates = set(re.findall(r'([a-z]\w*)\s*\([^)]*\)\s*:-', code))
    for pred in predicates:
        if re.search(rf'{pred}\s*\([^)]*\)\s*:-[^.]*{pred}\s*\(', code):
            mutual_recursion = True
            break
    return bool(direct_recursion or mutual_recursion)

# ─── Prolog Execution Tool ────────────────────────────────────────────────
# Then modify run_prolog to use a try-except with the timeout
def run_prolog(code: str, timeout: int = 5) -> str:
    # Check for recursion risks and log warning
    if detect_recursion_risks(code):
        print(">>> WARNING: Potential infinite recursion detected in Prolog code")
    
    # Rest of your existing function for prolog setup
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
        # Run with timeout but handle it gracefully
        r = subprocess.run(
            ["swipl", "-q", "-f", tmp, "-g", "solve(X), writeln(X), halt"],
            capture_output=True, text=True, timeout=timeout
        )
        out = r.stdout.strip()
        result = out.splitlines()[-1] if out else None
        
        # Remove special handling for decimal results - just return as is
        return result
    except subprocess.TimeoutExpired:
        # Instead of letting this crash your evaluation, return None
        print(">>> TIMEOUT: Prolog execution took too long (likely infinite recursion)")
        return None
    finally:
        try: os.remove(tmp)
        except: pass

TOOLS = {"run_prolog": run_prolog}

# ─── Agentic Loop Helpers ─────────────────────────────────────────────────
_tool_call_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
def extract_tool_calls(text: str):
    calls=[]
    for m in _tool_call_re.finditer(text):
        try: calls.append(json.loads(m.group(1)))
        except: pass
    return calls

ENC = tiktoken.get_encoding("cl100k_base")
def _tok_count(text: str)->int:
    return len(ENC.encode(text, allowed_special="all", disallowed_special=()))
def _prompt_tokens(msgs): return sum(_tok_count(m["content"]) for m in msgs)

def print_tokens(stage: str, conv: List[dict]):
    used = _prompt_tokens(conv)
    rem = TOKEN_BUDGET - used
    percentage = (used / TOKEN_BUDGET) * 100
    print(f"[TOKENS:{stage}] used={used} ({percentage:.1f}%) | rem={rem} | budget={TOKEN_BUDGET}")

def _shrink_conv(conv:List[dict]) -> (List[dict], bool):
    """
    Returns (new_conv, pruned_flag). pruned_flag=True if we had to summarize.
    """
    if _prompt_tokens(conv) <= TOKEN_BUDGET:
        return conv, False
    # keep sys + last 4
    must_keep={0} | set(range(len(conv)-4,len(conv)))
    summary=[]
    new_conv=[]
    for i,msg in enumerate(conv):
        if i in must_keep or msg["role"]=="system":
            new_conv.append(msg)
        else:
            snippet=msg["content"].replace("\n"," ")[:60]
            summary.append(f"[{msg['role']}:{snippet}…]")
    if summary:
        summary_msg="Context too long, compressed: "+ " ".join(summary)
        new_conv.insert(1,{"role":"system","content":summary_msg})
    return new_conv, True    ### ⬅️ NEW: returns pruned_flag

# ─── Revised agentic_loop ─────────────────────────────────────────────────
def agentic_loop(model, system_prompt, user_query, max_steps=20, turn_offset=0):
    BASE_TEMP=0.20; SHAKE_FACTOR=1.15; SHAKE_EVERY=2
    ESC_AFTER=5; CAP_TEMP=0.30; MAX_DUP=20
    EMPTY_RETRIES=20 # New: Track empty generation retries

    conv=[{"role":"system","content":system_prompt},
          {"role":"user","content":user_query}]
    cur_temp=BASE_TEMP
    seen=set(); dup=0
    empty_count=0  # New: Counter for empty generations

    for step in range(max_steps):
        params=SamplingParams(
            # Remove seed parameter
            temperature=cur_temp, 
            top_p=0.95,
            max_tokens=512, 
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        
        # Add pre-generation token budget check
        # Check if we're close to the limit BEFORE generating
        if _prompt_tokens(conv) > TOKEN_BUDGET * 0.95:  # 95% of budget threshold
            print(">>> PRE-GEN TOKEN BUDGET APPROACHING LIMIT: PREEMPTIVELY RESETTING CONTEXT")
            conv=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_query},
                  {"role":"system","content":"Please provide a concise solution - previous attempts used too many tokens."}]
            print_tokens("pre-prune", conv)

        # Then proceed with generation
        prompt="\n\n".join(f"({m['role'].upper()}) {m['content']}" for m in conv)
        print_tokens("pre-gen", conv)
        out=model.fast_generate(prompt,params)[0].outputs[0].text
        
        # Modified print to include total steps counting
        current_turn = turn_offset + step + 1
        print(f"--- TURN: {current_turn}/{turn_offset+max_steps} ---\n{out}\n")
        
        # Modified empty generation handling
        if not out.strip():
            empty_count += 1
            print(f">>> Empty generation detected (#{empty_count})")
            
            if empty_count >= EMPTY_RETRIES:
                print(f">>> Too many empty generations ({empty_count}) - aborting this problem")
                return None, None, step+1
            
            # Reset context after 3 consecutive empty generations
            if empty_count >= 2:
                # print(">>> Multiple empty generations detected - resetting context")
                # conv = [{"role": "system", "content": system_prompt},
                #         {"role": "user", "content": user_query},
                #         {"role": "system", "content": "Previous attempts resulted in empty generations. Please try a fresh approach."}]
                # cur_temp = BASE_TEMP  # Start with fresh temperature
                # seen = set()  # Reset seen answers
                # dup = 0       # Reset duplicate counter
                # print_tokens("post-reset", conv)

                # bail out of this attempt entirely:
                print(">>> Aborting this attempt—resetting context for a new independent try")
                return None, None, step+1

            else:
                # Normal temperature increase for occasional empty generation
                cur_temp = min(cur_temp * SHAKE_FACTOR * 1.2, CAP_TEMP)
                print(f">>> Increasing temperature to {cur_temp:.2f} and trying again")
                
                # Add a system message to encourage better response
                conv.append({"role": "system", "content": "The previous generation was empty. Please try again with a complete solution."})
            
            continue  # Skip to next iteration without adding empty response
            
        # Reset empty counter when we get a non-empty response
        empty_count = 0

        conv.append({"role":"assistant","content":out})
        print_tokens("post-gen", conv)  # Add token usage reporting after generation
        conv, pruned = _shrink_conv(conv)   ###  NEW
        if pruned:
            # if we just pruned, reset to fresh prompt + concise hint
            print(">>> TOKEN BUDGET EXCEEDED: RESETTING CONTEXT")
            conv=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_query},
                  {"role":"system","content":"Please retry - context was too long."}]
            print_tokens("post-prune", conv)  # Add token usage after pruning

        # continue with tool calls / answer handling…

        # dispatch tool calls
        calls=extract_tool_calls(out)
        if calls:
            for c in calls:
                res=TOOLS[c["name"]](**c["arguments"])
                conv.append({"role":"tool","name":c["name"],"content":str(res)})
                print(f">>> TOOL {c['name']}→{res}")
            continue

        m=re.search(r"<answer>(.*?)</answer>",out,re.DOTALL)
        if not m: continue

        code=m.group(1).strip()
        norm=re.sub(r"\s+"," ",code)
        if norm in seen:
            dup+=1
            if dup%SHAKE_EVERY==0 and dup<MAX_DUP:
                cur_temp=min(cur_temp*SHAKE_FACTOR,CAP_TEMP)
                print(f">>> duplicate#{dup}, shaking temp to {cur_temp:.2f}")
                continue
            if dup==ESC_AFTER:
                esc = (
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
                conv.append({"role":"system","content": esc})
                continue
            # Add context reset after a higher number of duplicates (8)
            if dup >= 6:  # You can adjust this threshold as needed
                # print(">>> Multiple duplicate solutions detected - resetting context")
                # conv = [{"role": "system", "content": system_prompt},
                #         {"role": "user", "content": user_query},
                #         {"role": "system", "content": "You've been generating the same incorrect solution repeatedly. Please start with a different approach."}]
                # cur_temp = BASE_TEMP
                # seen = set()
                # dup = 0
                # print_tokens("post-duplicate-reset", conv)
                # continue

                # bail out of this attempt entirely:
                print(">>> Aborting this attempt—resetting context for a new independent try")
                return None, None, step+1

            if dup>=MAX_DUP:
                print(">>> too many duplicates—abort")
                return None, code, step+1
        else:
            seen.add(norm); dup=0; cur_temp=BASE_TEMP

        res = run_prolog(code)
        print(f">>> run_prolog→ {res}")

        # Continue with the regular numeric check
        num = _parse_numeric(res or "")
        if num is not None:
            return num, code, step+1

        # Track non-numeric result attempts
        if 'numeric_fails' not in locals():
            numeric_fails = 1
        else:
            numeric_fails += 1
            
        # Reset context after 3 non-numeric results
        if numeric_fails >= 3:
            # print(">>> Multiple non-numeric results detected - resetting context")
            # conv = [{"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_query},
            #         {"role": "system", "content": "Previous attempts produced invalid results. Please try a completely fresh approach."}]
            # cur_temp = BASE_TEMP
            # seen = set()
            # dup = 0
            # print_tokens("post-reset", conv)
            # continue

            # bail out of this attempt entirely:
            print(">>> Aborting this attempt—resetting context for a new independent try")
            return None, None, step+1


        # otherwise concise single retry
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
    
        print("\n>>> FEEDBACK INJECTED:\n" + feedback_msg + "\n")     # announce in console
        conv.append({"role":"user","content": feedback_msg})
        
        print_tokens("post-feedback", conv)

    raise RuntimeError("Exhausted max_steps")

# ─── Prompt & Helper ─────────────────────────────────────────────────────
def extract_problem(sample):
    p = sample.get("prompt")
    if isinstance(p, list):
        for m in p:
            if m.get("role") == "user":
                return m["content"]
        return " ".join(m.get("content","") for m in p)
    return p or ""

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

def calculate_optimal_token_budget(model_max_tokens=2048, safety_margin_pct=5, max_samples=10):
    """
    Calculate optimal token budget based on empirical measurements across multiple samples.
    Uses real examples from the dataset and measures actual formatting overhead.
    """
    # Print first 3 samples for verification regardless of how many we process
    print("\n=== DATASET SAMPLES VERIFICATION ===")
    print("Showing first 3 samples from dataset:")
    for idx, sample in enumerate(val_dataset):
        if idx >= 3:  # Always show just the first 3 for verification
            break
        question = extract_problem(sample)
        print(f"\nSample #{idx+1} content:")
        print("-" * 40)
        print(question)
        print("-" * 40)
    print("=== END VERIFICATION DISPLAY ===\n")
    
    # Get samples for actual token budget calculation
    print(f"=== COLLECTING SAMPLES FOR TOKEN BUDGET CALCULATION ===")
    print(f"Using max_samples={max_samples}")
    
    sample_problems = []
    for idx, sample in enumerate(val_dataset):
        if idx >= max_samples:
            break
        question = extract_problem(sample)
        sample_problems.append(question)
    
    # Safety check - we must have at least one sample from the dataset
    if not sample_problems:
        raise ValueError("No samples found in dataset for token budget calculation")
    
    print(f"Successfully collected {len(sample_problems)} samples")
    print("=" * 40)
    
    # Collect overhead factors from multiple samples
    overhead_factors = []
    
    print("\n=== TOKEN BUDGET ANALYSIS ===")
    print(f"Analyzing {len(sample_problems)} samples from dataset")
    
    for i, problem in enumerate(sample_problems):
        # Create sample conversation with this problem
        sample_conv = [
            {"role": "system", "content": tool_spec_prompt},
            {"role": "user", "content": f"Please solve this problem: {problem}"}
        ]
        
        # Add a plausible assistant response (simplified for measurement)
        sample_conv.append({"role": "assistant", "content": f"<reasoning>\nAnalyzing the problem...\n</reasoning>\n<answer>\n:- use_module(library(clpq)).\n\nsolve(X) :-\n    {{X = 42}}.\n</answer>"})
        
        # Measure raw vs formatted tokens
        raw_tokens = _prompt_tokens(sample_conv)
        formatted_prompt = "\n\n".join(f"({m['role'].upper()}) {m['content']}" for m in sample_conv)
        formatted_tokens = _tok_count(formatted_prompt)
        
        # Calculate overhead
        factor = formatted_tokens / raw_tokens
        overhead_factors.append(factor)
        
        print(f"Sample #{i+1} overhead factor: {factor:.4f}x ({raw_tokens} → {formatted_tokens} tokens)")
    
    # Statistical analysis of overhead factors
    min_factor = min(overhead_factors)
    max_factor = max(overhead_factors)
    avg_factor = sum(overhead_factors) / len(overhead_factors)
    
    # Use a conservative approach - the maximum observed overhead plus safety margin
    safe_factor = max_factor * (1 + safety_margin_pct/100)
    
    # Calculate optimal budget
    optimal_budget = int(model_max_tokens / safe_factor)
    
    print(f"\nFormatting overhead statistics:")
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

# ─── Evaluation loop (only tiny int-cast tweak) ───────────────────────────
def evaluate_agentic_prolog(model, dataset, max_steps: int = 20):
    sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    stats = dict(total=0, strict=0, struct=0, full=0,
                 sem_sum=0.0, sem_cnt=0, atts=[], gtimes=[], vtimes=[])
    wb_table = wandb.Table(columns=[
        "idx","question","gold","prediction",
        "strict","structure","full",
        "attempts","gen_time","val_time","semantic_%"
    ])

    overall_start = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating"), start=1):
        gold = float(sample["numerical_result"])
        question = extract_problem(sample)
        # ─── NEW: echo the question text ───────────────────────────
        print("\n" + "#"*70)
        print(f"QUESTION {idx}: {question}")
        print("#"*70 + "\n")
        # ───────────────────────────────────────────────────────────

        # keep retrying agentic_loop from scratch until we get a numeric result
        total_gen_t = 0.0
        pred = code = None
        total_steps_used = 0  # Track total steps across all attempts
        try_count = 0

        # ── START OUTER RETRY LOOP ─────────────────────────
        while True:
            try_count += 1
            
            # Calculate remaining steps for this question
            remaining_steps = max_steps - total_steps_used
            
            # Break if we've exhausted our step budget
            if remaining_steps <= 0:
                print(f">>> Maximum total steps ({max_steps}) reached - moving to next question")
                break
                
            print(f">>> INDEPENDENT AGENTIC_TRY #{try_count}")
            
            t0 = time.time()
            # pass a shifting seed if you want reproducible but distinct RNG per try
            try:  # Add this try-except block
                p, c, steps_used = agentic_loop(
                    model,
                    tool_spec_prompt,
                    f"Please solve this problem: {question}",
                    max_steps=remaining_steps,  # Only use remaining steps
                    # Remove seed parameter
                    turn_offset=total_steps_used  # Pass the current
                )
            except RuntimeError:
                # Handle the exception - use all remaining steps
                print(">>> Agentic loop exhausted maximum steps")
                p, c, steps_used = None, None, remaining_steps
            
            # Update total steps
            total_steps_used += steps_used
            print(f">>> Total steps used: {total_steps_used}/{max_steps}")
            
            dt = time.time() - t0
            total_gen_t += dt
            print(f">>> TRY RESULT → pred={p!r}, steps={steps_used}, took {dt:.2f}s")

            # did we get a valid integer answer?  If so, break out
            if p is not None and _parse_numeric(p):
                pred, code, attempts = p, c, total_steps_used  # Use total steps as attempts
                break
                
            # Second check if we've exhausted steps
            if total_steps_used >= max_steps:
                print(f">>> Maximum total steps ({max_steps}) reached - moving to next question")
                break

            print(">>> Bailed — no valid numeric answer; retrying with fresh context\n")

        # Use total_steps_used as the final attempts count
        attempts = total_steps_used
        gen_t = total_gen_t
        stats["gtimes"].append(gen_t)

        t1 = time.time()
        try:
            val = float(pred.split()[0]) if pred else None
            strict = val is not None and abs(val - gold) < 1e-6
        except:
            val, strict = None, False
        val_t = time.time() - t1
        stats["vtimes"].append(val_t)

        struct_ok = check_structure_correctness(code) if code else False
        full_ok   = strict and struct_ok

        # ─── ensure sem_pct is always defined ───
        sem_pct = 0.0                              # <<< add this line
        ref = sample.get("answer", "").strip()
        if code and ref:
            m = re.search(r"<answer>(.*?)</answer>", ref, re.DOTALL)
            ref_code = m.group(1).strip() if m else ref
            try:
                sem_pct = util.cos_sim(
                    sem_model.encode(code, convert_to_tensor=True),
                    sem_model.encode(ref_code, convert_to_tensor=True)
                ).item() * 100
            except:
                sem_pct = 0.0
            stats["sem_sum"] += sem_pct / 100
            stats["sem_cnt"] += 1

        # ── FIX #2: cast booleans to int for tallying ──
        stats["total"]  += 1
        stats["strict"] += int(strict)
        stats["struct"] += int(struct_ok)
        stats["full"]   += int(full_ok)
        # ------------------------------------------------
        stats["atts"].append(attempts)

        # console print
        p_acc = stats["strict"]/stats["total"]*100
        s_acc = stats["struct"]/stats["total"]*100
        f_acc = stats["full"]  /stats["total"]*100
        m_acc = (stats["sem_sum"]/stats["sem_cnt"]*100) if stats["sem_cnt"] else 0
        print("\n" + "="*60)
        print(f" Q#{idx:<3} | Pred: {val} | Gold: {gold}")
        print(f" Flags Strict={strict} Struct={struct_ok} Full={full_ok}")
        print(f" Acc   Prolog={p_acc:.1f}% Struct={s_acc:.1f}% Full={f_acc:.1f}% Sem={m_acc:.1f}%")
        print("="*60)

        # add row to WandB table
        wb_table.add_data(
            idx, question[:120] + ("…" if len(question)>120 else ""),
            gold, val, strict, struct_ok, full_ok,
            attempts, f"{gen_t:.2f}", f"{val_t:.2f}", f"{sem_pct:.1f}"
        )

        # live wandb metrics
        wandb.log({
            "live/prolog_acc": p_acc,
            "live/structure_acc": s_acc,
            "live/full_correct_acc": f_acc,
            "live/semantic_score": sem_pct,
            "live/avg_attempts": sum(stats["atts"])/len(stats["atts"]),
            "time/generation": gen_t,
            "time/validation": val_t
        }, step=idx)

    # --------- final summary ----------
    elapsed = time.time() - overall_start
    final = {
        "final/prolog_accuracy":   stats["strict"]/stats["total"]*100,
        "final/structure_accuracy":stats["struct"]/stats["total"]*100,
        "final/full_correct_accuracy": stats["full"]/stats["total"]*100,
        "final/semantic_accuracy": (stats["sem_sum"]/stats["sem_cnt"]*100)
                                   if stats["sem_cnt"] else 0.0,
        "final/avg_generation_time": sum(stats["gtimes"])/len(stats["gtimes"]),
        "final/avg_validation_time": sum(stats["vtimes"])/len(stats["vtimes"]),
        "final/avg_attempts":       sum(stats["atts"])/len(stats["atts"]),
        "final/total_time":         elapsed
    }

    # table + metrics
    wandb.log({"detailed_results": wb_table, **final})
    wandb.summary.update(final)

    # pretty print
    print("\n" + "="*60)
    print(" EVALUATION COMPLETE ".center(60, "="))
    print(f" Prolog Acc:   {final['final/prolog_accuracy']:.2f}%")
    print(f" Structure Acc:{final['final/structure_accuracy']:.2f}%")
    print(f" Full Acc:     {final['final/full_correct_accuracy']:.2f}%")
    print(f" Semantic Acc: {final['final/semantic_accuracy']:.2f}%")
    print(f" Total Time:   {elapsed:.2f}s")
    print("="*60)
    return final

# ─────────────────────────── 5. main driver ───────────────────────────────
if __name__ == "__main__":
    wandb.init(
        project="gsm8k-prolog-prover-new-evaluation",
        name="sp-struct-rwd1-agentic-independent-maxlimit",
        settings=wandb.Settings(start_method="thread")
    )
    # # load your model, val_dataset here...
    # _ = evaluate_agentic_prolog(model, small_val_dataset, max_steps=20)
    # wandb.finish()

    # ---------- helper -----------------------------------------------------------
    class Tee(io.TextIOBase):
        def __init__(self, logfile_handle, terminal_handle):
            self.log = logfile_handle
            self.term = terminal_handle        # original Jupyter stream

        def write(self, text):
            self.term.write(text)              # live display
            self.term.flush()
            self.log.write(text)               # save to file
            self.log.flush()
            return len(text)

        def flush(self):                       # tqdm calls this a lot
            self.term.flush()
            self.log.flush()
    # -----------------------------------------------------------------------------

    ts       = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir  = pathlib.Path("eval_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    log_path = out_dir / f"console_{ts}.txt"

    with log_path.open("w", encoding="utf-8") as fp, \
        contextlib.redirect_stdout(Tee(fp, sys.stdout)), \
        contextlib.redirect_stderr(Tee(fp, sys.stderr)):
        
        final_metrics = evaluate_agentic_prolog(
            model, val_dataset, max_steps=20
        )

    print(f"\n Full console saved to {log_path.resolve()}\n")

    # --------------- upload the file to Weights & Biases -----------------
    art = wandb.Artifact(
        name        = f"evaluation-log-{ts}",
        type        = "evaluation_output",
        description = "Combined stdout + stderr from the evaluation run",
    )
    art.add_file(str(log_path))
    wandb.run.log_artifact(art)

