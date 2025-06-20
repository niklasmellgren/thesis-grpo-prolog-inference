This repository contains the code for Niklas Mellgren's 2025 master's thesis in Data Science – ICT Systems at the University of Southern Denmark (SDU).

### **Thesis title:**  
*Reinforcement Fine-Tuning Large Language Models to Use Prolog as a Tool*

### **Abstract:**
Using **Group Relative Policy Optimization (GRPO)**, this project fine-tunes **Qwen2.5-3B-Instruct** on a merged and cleaned version of `openai/gsm8k` and `Thomas-X-Yang/gsm8k-prolog`, resulting in the `niklasm222/gsm8k-prolog-prover` dataset.

Three main experimental axes were explored:

1. **Prompt structure**: From minimal XML formatting to reflexive, self-verifying scaffolds  
2. **Reward composition**: Combining execution correctness, syntax, semantic similarity, structural constraints, and curriculum shaping  
3. **Inference protocol**: Including single-shot, best-of-N multiple-try, and two agentic modes where Prolog is used as a tool inside dialogue (internal) or across fresh sessions (independent)

### Key findings:
- Joint tuning of prompt, reward, and inference **shapes the structure and quality of generated Prolog programs**
- **Highest accuracy** achieved using **external Prolog verification** in a multiple-try setting
- **Best generalization** (MMLU-Stem/Pro) came from **agentic inference**, where the model engages in self-repair using Prolog function-calling
