import subprocess
import re
import time
import wandb
from tqdm import tqdm
from datasets import load_dataset

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from vllm import SamplingParams
import os, uuid
from sentence_transformers import SentenceTransformer, util

# Safer arithmetic evaluation pattern: accepts digits, spaces, and basic arithmetic symbols.
ARITHMETIC_PATTERN = re.compile(r'^[\d\s+\-*/().]+$')

##############################################################################
# Utility: Flatten a list of messages into a single string (system + user)
##############################################################################
def conversation_to_prompt(messages):
    """
    Keep only the first system message (if any) and the first user message (if any).
    """
    system_msg = None
    user_msg = None
    for msg in messages:
        role = msg.get("role")
        if role == "system" and system_msg is None:
            system_msg = msg
        elif role == "user" and user_msg is None:
            user_msg = msg
    prompt_text = ""
    if system_msg:
        prompt_text += f"(SYSTEM) {system_msg['content']}\n"
    if user_msg:
        prompt_text += f"(USER) {user_msg['content']}\n"
    return prompt_text.strip()

##############################################################################
# 1) Extract the last complete <answer>...</answer> block from text.
##############################################################################
def extract_xml_answer(text: str) -> str:
    try:
        # Truncate at <|endoftext|> if it exists.
        eot_index = text.find("<|endoftext|>")
        truncated_text = text[:eot_index] if eot_index != -1 else text
        start = truncated_text.find("<answer>")
        if start == -1:
            return None
        end = truncated_text.find("</answer>", start)
        if end == -1:
            return None
        return truncated_text[start+len("<answer>"):end].strip()
    except Exception:
        return None

##############################################################################
# 2) Execute Prolog code and return the final line of SWI-Prolog stdout.
##############################################################################
def execute_prolog_code_subprocess(prolog_code: str, timeout=5) -> str:
    temp_file = f"temp_{uuid.uuid4().hex}.pl"
    try:
        with open(temp_file, "w") as f:
            f.write(prolog_code)
        result = subprocess.run(
            ["swipl", "-q", "-f", temp_file, "-g", "solve(X), writeln(X), halt"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0 or not result.stdout:
            return None
        lines = result.stdout.strip().splitlines()
        return lines[-1].strip() if lines else None
    except Exception as e:
        print(f"Error executing Prolog code: {e}")
        return None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

##############################################################################
# 3) Analyze structure of generated Prolog code using prolog_helpers.pl.
##############################################################################
def analyze_prolog_structure_subprocess(prolog_code: str) -> dict:
    temp_file = f"temp_{uuid.uuid4().hex}.pl"
    try:
        with open(temp_file, "w") as f:
            f.write(prolog_code)
        result = subprocess.run(
            [
                "swipl", "-q", "-f", "prolog_helpers.pl",
                "-g", f"analyze_code('{temp_file}', PredCount, ConstCount), halt"
            ],
            capture_output=True, text=True, timeout=10
        )
        predicate_count = 0
        constraint_count = 0
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("PREDICATE_COUNT:"):
                predicate_count = int(line.split(":", 1)[1].strip())
            elif line.startswith("CONSTRAINT_COUNT:"):
                constraint_count = int(line.split(":", 1)[1].strip())
        return {"predicate_count": predicate_count, "constraint_count": constraint_count}
    except Exception as e:
        print("Error in analyze_prolog_structure_subprocess:", e)
        return {"predicate_count": 0, "constraint_count": 0}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

##############################################################################
# 4) Check structural correctness: valid if at least one predicate (other than solve/1)
#    and at least one curly-brace constraint exist.
##############################################################################
def check_structure_correctness(prolog_code: str) -> bool:
    if not prolog_code:
        return False
    analysis = analyze_prolog_structure_subprocess(prolog_code)
    pred_count = analysis.get("predicate_count", 0)
    const_count = analysis.get("constraint_count", 0)
    return (pred_count >= 1) and (const_count >= 1)

##############################################################################
# 5. Reward Functions
##############################################################################
# 5.1 Semantic Similarity Reward (Direct Approach):
def semantic_similarity_reward(completions, answer, semantic_model, **kwargs) -> list[float]:
    """
    Computes a semantic similarity score between generated and reference Prolog code.
    Returns a score on a [0,1] scale.
    """
    extracted_responses = [extract_xml_answer(comp[0]["content"]) for comp in completions]
    rewards = []
    for model_code, ref_code in zip(extracted_responses, answer):
        if not model_code or not ref_code:
            rewards.append(0.0)
            continue
        try:
            embedding_model = semantic_model.encode(model_code, convert_to_tensor=True)
            embedding_ref = semantic_model.encode(ref_code, convert_to_tensor=True)
            cosine_sim = util.cos_sim(embedding_model, embedding_ref).item()
            preds_model = set(re.findall(r'(\w+)\(', model_code))
            preds_ref = set(re.findall(r'(\w+)\(', ref_code))
            pred_overlap = len(preds_model & preds_ref) / max(1, len(preds_ref))
            reward_val = (cosine_sim + pred_overlap) / 2.0
            rewards.append(reward_val)
        except Exception as e:
            print(f"Error in semantic similarity: {str(e)}")
            rewards.append(0.0)
    return rewards

# 5.2 Correctness Reward (Numeric Evaluation)
def correctness_reward_func(prompts, completions, answer, numerical_result, **kwargs) -> list[float]:
    responses = [comp[0]["content"] for comp in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    correct_values = numerical_result
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

# 5.3 Prolog Structure Reward
def prolog_structure_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        start = text.find("<answer>")
        if start == -1:
            rewards.append(0.0)
            continue
        end = text.find("</answer>", start)
        if end == -1:
            rewards.append(0.0)
            continue
        extracted_code = text[start+len("<answer>"):end].strip()
        analysis = analyze_prolog_structure_subprocess(extracted_code)
        pred_count = analysis.get("predicate_count", 0)
        const_count = analysis.get("constraint_count", 0)
        score = min(pred_count * 0.25, 0.75) + min(const_count * 0.3, 0.9)
        hardcode_regex = r'solve\([^)]*\)\s*:-.*(\b\w+\s*=\s*\d+|{\s*\w+\s*=\s*\d+\s*})'
        if re.search(hardcode_regex, extracted_code, flags=re.DOTALL):
            score *= 0.2
        final_score = max(0.0, min(score, 2.0))
        rewards.append(final_score)
    return rewards

# 5.4 Prolog Syntax and XML Reward Functions (unchanged)
def prolog_syntax_reward_func(completions, **kwargs) -> list[float]:
    pattern = r'(?::-|solve\s*\(|use_module|clpq|\.\s*$)'
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        hits = re.findall(pattern, text, re.MULTILINE)
        score = min(len(hits) * 0.2, 1.0)
        rewards.append(score)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [comp[0]["content"] for comp in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [comp[0]["content"] for comp in completions]
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
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>\n")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [comp[0]["content"] for comp in completions]
    return [count_xml(c) for c in contents]

##############################################################################
# 6) Main Evaluation Function with Multiple Try Logic
##############################################################################
def extract_reasoning(text: str) -> str:
    try:
        start = text.find("<reasoning>")
        if start == -1:
            return None
        end = text.find("</reasoning>", start)
        if end == -1:
            return None
        return text[start+len("<reasoning>"):end].strip()
    except Exception:
        return None

def evaluate_prolog_generation(model, tokenizer, dataset, max_new_tokens=1024, max_attempts=20):
    # Initialize semantic similarity model
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize metrics
    metrics = {
        'total_samples': 0,
        'strict_correct': 0,
        'arithmetic_correct': 0,
        'structure_correct': 0,
        'full_correct': 0,
        'overall_count': 0,
        'strict_count': 0,
        'arithmetic_count': 0,
        'structure_count': 0,
        'full_correct_count': 0,
        'semantic_scores': [],
        'semantic_sum': 0.0,
        'total_semantic': 0.0,
        'attempts_list': [],
        'generation_times': [],
        'prolog_times': [],
        'validation_times': [],
    }

    # Initialize WandB table for PER-ATTEMPT details
    results_table = wandb.Table(columns=[
        "Sample Index", "Question", "Reference Answer", "Gold Numerical Result",
        "Attempt Number", "Is Final Successful Attempt",
        "Model Output", "Extracted Code", "Execution Result",
        "Is Valid Prolog", "Produces Number", "Is Correct Number (vs Gold)",
        "Is Structure Valid", "Generation Time (s)", "Prolog Execution Time (s)",
        "Failure Reason"
    ])

    # Add sampling parameters definition
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Add start time tracking
    start_time = time.time()

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        sample_index = idx + 1
        prompt_text = ""
        model_text = ""
        prolog_code = ""
        final_line = None
        gold_str = ""
        error_type = None
        is_strict = False
        is_arithmetic = False
        is_structure = False
        is_semantic = False
        is_full_correct = False
        semantic_score = 0.0
        raw_semantic = 0.0
        generation_time = 0.0
        prolog_exec_time = 0.0
        validation_time = 0.0
        attempts = 0
        success = False
        all_attempts = []
        successful_attempt_number = -1 # Track which attempt succeeded

        try:
            messages = sample["prompt"]
            prompt_text = conversation_to_prompt(messages)
            print("\n[1] Flattened Prompt:\n", prompt_text) # <-- Added prompt print
            gold_str = sample.get("numerical_result") # Get gold value once per sample
            gold_val = None
            if gold_str:
                try:
                    gold_val = float(gold_str)
                except ValueError:
                    print(f"Warning: Could not convert gold numerical result '{gold_str}' to float for sample {sample_index}")

            # --- Attempt Loop ---
            while attempts < max_attempts and not success:
                attempts += 1
                gen_start = time.time()
                output_data = model.fast_generate(prompt_text, sampling_params)
                generation_time_attempt = time.time() - gen_start
                metrics['generation_times'].append(generation_time_attempt)
                gen_model_text = output_data[0].outputs[0].text
                gen_prolog_code = extract_xml_answer(gen_model_text)

                # Initialize attempt_info
                attempt_info = {
                    'attempt_number': attempts,
                    'model_output': gen_model_text,
                    'extracted_code': gen_prolog_code if gen_prolog_code else "No code extracted",
                    'execution_result': None,
                    'is_valid_prolog': False,
                    'produces_number': False,
                    'is_correct_number': False, # Correctness vs gold standard
                    'structure_valid': False,
                    'generation_time': generation_time_attempt,
                    'prolog_execution_time': None,
                    'reason_for_failure': None
                }

                if not gen_prolog_code:
                    attempt_info['reason_for_failure'] = "No Prolog code extracted"
                    all_attempts.append(attempt_info)
                    print(f"Attempt {attempts}: No Prolog code extracted.")
                    continue

                # Execute code
                prolog_start = time.time()
                gen_final_line = execute_prolog_code_subprocess(gen_prolog_code)
                prolog_exec_time_attempt = time.time() - prolog_start
                metrics['prolog_times'].append(prolog_exec_time_attempt) # Still collect total time

                # Update attempt info
                attempt_info['execution_result'] = gen_final_line
                attempt_info['is_valid_prolog'] = gen_final_line is not None
                attempt_info['prolog_execution_time'] = prolog_exec_time_attempt
                attempt_info['structure_valid'] = check_structure_correctness(gen_prolog_code)

                # Check if execution yielded a number and if it's correct
                try:
                    if gen_final_line:
                        float_result = float(gen_final_line)
                        attempt_info['produces_number'] = True
                        if gold_val is not None:
                             attempt_info['is_correct_number'] = abs(float_result - gold_val) < 1e-6

                        # SUCCESS CONDITION: Execution produced a number
                        model_text = gen_model_text # Store the successful output
                        prolog_code = gen_prolog_code # Store the successful code
                        final_line = gen_final_line # Store the successful result
                        generation_time = generation_time_attempt # Store successful gen time
                        prolog_exec_time = prolog_exec_time_attempt # Store successful exec time
                        print(f"Attempt {attempts}: Successful numeric output: {final_line}")
                        success = True
                        successful_attempt_number = attempts # Record which attempt succeeded
                    else:
                        attempt_info['reason_for_failure'] = "Prolog execution did not return a result"
                except ValueError:
                    attempt_info['reason_for_failure'] = "Prolog output is not a valid number"
                    print(f"Attempt {attempts}: Prolog code did not yield a numeric result ('{gen_final_line}').")
                except Exception as e:
                    attempt_info['reason_for_failure'] = f"Error checking result: {str(e)}"
                    print(f"Attempt {attempts}: Error checking result: {str(e)}")

                all_attempts.append(attempt_info)

                if success:
                    break
            # --- End Attempt Loop ---

            # (5) Validate numeric correctness (for overall sample metrics)
            valid_start = time.time()
            is_strict = False
            is_arithmetic = False
            if success and gold_val is not None: # Check strict/arithmetic only if an attempt succeeded
                try:
                    prolog_val = float(final_line)
                    is_strict = abs(prolog_val - gold_val) < 1e-6
                except:
                     pass # is_strict remains False

                if not is_strict and ARITHMETIC_PATTERN.match(final_line.strip()):
                    try:
                        # Use a safer eval if needed, or stick to float conversion if sufficient
                        eval_val = float(final_line.strip()) # Simpler if only numbers expected
                        is_arithmetic = abs(eval_val - gold_val) < 1e-6
                    except Exception as e:
                        error_type = f"Arithmetic conversion error: {str(e)}"
            validation_time = time.time() - valid_start # Time for this specific check

            # (6) Structural correctness (for overall sample metrics)
            is_structure = False
            if success and prolog_code: # Check structure only if an attempt succeeded
                is_structure = check_structure_correctness(prolog_code)

            # (7) Semantic similarity calculation (based on successful attempt)
            reference_answer = [sample.get("answer", "")]
            has_reference = bool(reference_answer[0].strip()) if reference_answer else False
            raw_semantic = 0.0
            if success and has_reference:
                completion_wrapper = [[{"content": model_text}]]
                try:
                    semantic_rewards = semantic_similarity_reward(
                        completion_wrapper,
                        reference_answer,
                        semantic_model=semantic_model
                    )
                    raw_semantic = semantic_rewards[0] if semantic_rewards else 0.0
                except Exception as e:
                    print(f"Semantic similarity error: {str(e)}")
                    raw_semantic = 0.0

            # (8) Overall correctness metric
            is_full_correct = (is_strict or is_arithmetic) and is_structure

            # Print summary for the *sample*
            print(f"\n--- Sample {sample_index} Summary ---")
            print(f"Result achieved in attempt: {successful_attempt_number if success else 'N/A'} / {attempts}")
            if success: # <-- Add this block to print successful output
                print("-" * 40)
                print("Successful Model Output:")
                print(model_text.strip())
                print("-" * 40)
            print(f"Final Prolog Output: {final_line or 'None'}")
            print(f"Gold Value: {gold_str or 'None'}")
            print(f"Strict: {is_strict} | Arithmetic: {is_arithmetic} | Structure: {is_structure} | Full: {is_full_correct}")
            print(f"Semantic Score: {raw_semantic*100:.2f}%")


        except Exception as e:
            error_type = f"Processing error: {str(e)}"
            print(f"DEBUG: Exception during sample {sample_index} processing:", error_type)
            # Ensure all_attempts has at least a placeholder if error occurred before loop
            if not all_attempts:
                 all_attempts.append({'attempt_number': 1, 'reason_for_failure': error_type,
                                      'model_output': 'ERROR', 'extracted_code': 'ERROR',
                                      # ... add other keys with default/error values ...
                                     })

        # --- Log EACH attempt to WandB Table ---
        for attempt_data in all_attempts:
            results_table.add_data(
                sample_index,
                prompt_text,
                sample.get("answer", ""), # Reference Prolog code
                gold_str or "",           # Gold numerical result as string
                attempt_data['attempt_number'],
                # Mark True only if this attempt is the one that succeeded
                attempt_data['attempt_number'] == successful_attempt_number,
                attempt_data['model_output'],
                attempt_data['extracted_code'],
                str(attempt_data['execution_result']) if attempt_data['execution_result'] is not None else "",
                attempt_data['is_valid_prolog'],
                attempt_data['produces_number'],
                attempt_data['is_correct_number'], # Correctness vs Gold for this attempt
                attempt_data['structure_valid'],
                f"{attempt_data['generation_time']:.3f}",
                f"{attempt_data['prolog_execution_time']:.3f}" if attempt_data['prolog_execution_time'] is not None else "",
                str(attempt_data['reason_for_failure']) if attempt_data['reason_for_failure'] is not None else ""
            )

        # --- Update and Log Aggregate Metrics (per sample) ---
        metrics['total_samples'] += 1
        metrics['attempts_list'].append(attempts) # Log total attempts for this sample
        if is_strict: metrics['strict_correct'] += 1
        if is_arithmetic: metrics['arithmetic_correct'] += 1
        if is_structure: metrics['structure_correct'] += 1
        if is_full_correct: metrics['full_correct_count'] += 1
        if has_reference:
            metrics['semantic_scores'].append(raw_semantic)
            metrics['semantic_sum'] += raw_semantic
            if raw_semantic >= 0.7:  # Threshold for "good" semantic similarity
                metrics['total_semantic'] += 1

        # Calculate running accuracies
        accuracies = {
             'strict': (metrics['strict_correct'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
             'arithmetic': (metrics['arithmetic_correct'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
             'structure': (metrics['structure_correct'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
             'full_correct': (metrics['full_correct_count'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
        }

        # Print running accuracies for the current sample
        print(f"Accuracies => Prolog: {accuracies['strict']:.2f}%, "
              f"Arithmetic: {accuracies['arithmetic']:.2f}%, "
              f"Structure: {accuracies['structure']:.2f}%, "
              f"Fully Correct: {accuracies['full_correct']:.2f}%")
        print("-" * 40)

        # Log live aggregate metrics to WandB charts
        wandb.log({
            "live/prolog_acc": accuracies['strict'],
            "live/arithmetic_acc": accuracies['arithmetic'],
            "live/structure_acc": accuracies['structure'],
            "live/full_correct_acc": accuracies['full_correct'],
            "live/semantic_score": raw_semantic * 100, # Semantic score for the sample (if successful)
            "live/avg_attempts": sum(metrics['attempts_list']) / len(metrics['attempts_list']) if metrics['attempts_list'] else 0,
            "time/generation_successful": generation_time if success else 0,
            "time/prolog_exec_successful": prolog_exec_time if success else 0,
            "time/validation": validation_time,
            "errors": 1 if error_type else 0,
            "sample_total_attempts": attempts
        }, step=sample_index)

    # --- Final Calculations and Logging ---
    elapsed = time.time() - start_time

    # Safety check
    if metrics['total_samples'] == 0:
        print("WARNING: No samples processed during evaluation")
        return {"accuracies": {}, "timing": {}, "details": []}

    # Calculate final average times (using all collected times)
    avg_times = {
        'generation': sum(metrics['generation_times'])/len(metrics['generation_times']) if metrics['generation_times'] else 0.0,
        'prolog': sum(metrics['prolog_times'])/len(metrics['prolog_times']) if metrics['prolog_times'] else 0.0,
        'validation': sum(metrics['validation_times'])/len(metrics['validation_times']) if metrics['validation_times'] else 0.0
    }

    # Calculate final aggregate accuracies
    final_accuracies = {
        'strict': (metrics['strict_correct'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
        'arithmetic': (metrics['arithmetic_correct'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
        'structure': (metrics['structure_correct'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
        'full_correct': (metrics['full_correct_count'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0,
    }
    avg_semantic = metrics['semantic_sum'] / metrics['total_samples'] if metrics['total_samples'] > 0 else 0.0
    final_semantic_accuracy = (metrics['total_semantic'] / metrics['total_samples'] * 100) if metrics['total_samples'] > 0 else 0.0

    # Log the detailed PER-ATTEMPT table ONCE at the end
    wandb.log({
        "detailed_results_per_attempt": results_table,
        "final/prolog_accuracy": final_accuracies['strict'],
        "final/arithmetic_accuracy": final_accuracies['arithmetic'],
        "final/structure_accuracy": final_accuracies['structure'],
        "final/full_correct_accuracy": final_accuracies['full_correct'],
        "final/semantic_accuracy": final_semantic_accuracy,
        "final/avg_semantic_score": avg_semantic,
        "final/total_time": elapsed,
        "final/avg_generation_time_per_attempt": avg_times['generation'],
        "final/avg_prolog_time_per_attempt": avg_times['prolog'],
    })

    # Update WandB Summary with final aggregates
    wandb.summary.update({
        "prolog_accuracy": final_accuracies['strict'],
        "arithmetic_accuracy": final_accuracies['arithmetic'],
        "structure_accuracy": final_accuracies['structure'],
        "full_correct_accuracy": final_accuracies['full_correct'],
        "semantic_accuracy": final_semantic_accuracy,
        "avg_semantic_score": avg_semantic,
        "avg_generation_time_per_attempt": avg_times['generation'],
        "avg_prolog_time_per_attempt": avg_times['prolog'],
    })

    # Print final summary to console
    print("\n" + "="*80)
    print(" EVALUATION COMPLETE ".center(80))
    print("="*80)
    print(f"Prolog Accuracy: {final_accuracies['strict']:.2f}%")
    print(f"Arithmetic Accuracy: {final_accuracies['arithmetic']:.2f}%")
    print(f"Structure Accuracy: {final_accuracies['structure']:.2f}%")
    print(f"Fully Correct Accuracy: {final_accuracies['full_correct']:.2f}%")
    print(f"Semantic Accuracy (>= threshold): {final_accuracies['structure']:.2f}%")
    print(f"Average Semantic Score: {avg_semantic:.2f}")
    print(f"\nAverage Times (per attempt):")
    print(f"  Generation: {avg_times['generation']:.3f}s")
    print(f"  Prolog Execution: {avg_times['prolog']:.3f}s")
    print(f"\nTotal Evaluation Time: {elapsed:.2f} seconds")
    return {
        "accuracies": final_accuracies,
        "timing": avg_times,
        "details": results_table.data
    }

##############################################################################
# 7) Example usage
##############################################################################
if __name__ == "__main__":
    wandb.init(
        project="gsm8k-prolog-prover-new-evaluation",
        name="sp-reflect-rwd2-multipletry",
        settings=wandb.Settings(start_method="thread"),
        config={"environment": "colab"}
    )

    result_stats = evaluate_prolog_generation(
        model,
        tokenizer,
        val_dataset
    )
    wandb.finish()
