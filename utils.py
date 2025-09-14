import re


# url
base_url="https://api.deepseek.com"

# api keys
deepseek_api_key_2 = "sk-ec515d112a194762922b8b98cfea8c35"  # 我的谷歌账号注册的
# deepseek_api_key_2 = "sk-e2c481c08cf04736814434c9cc7ac6ba"  # 婉霞
# deepseek_api_key_2 = "sk-5e8c8ea10cca446097c515947b36d3ee"


gemini_api_key = "AIzaSyAkg5aPv3TgHdpqe90sDpSS6T5rVwNJZjU"  # 我的谷歌账号注册的
# gemini_api_key = "AQ.Ab8RN6IFlf4dNK5y0yb-N_BZZ-nk48NWj9Rhe3uU07iiPbzJhg"  # google cloud

custom_system_prompt = "Please reason step by step. *** PLEASE PUT YOUR FINAL ANSWER WITHIN \\boxed{}. ***"

formatting_reminder = " *** PLEASE MAKE SURE YOUR FINAL ANSWER IS IN THE FORMAT \\boxed{...}. ***"


def get_decomposition_prompts(problem_statement):
    
    step1_system_prompt = """
You are an elite mathematical strategist and analyst. Your primary function is to perform a deep Metacognitive Analysis of complex mathematical problems. You are to deconstruct the problem into its core components, identify underlying principles, and then formulate a high-level, executable strategic plan.

Your task is to produce a Metacognitive Analysis of the following problem. You must NOT provide a final solution or perform detailed calculations.

### Core Principles

*   **Analytical Depth:** Your analysis must go beyond a surface-level reading. Identify the mathematical field, key concepts, constraints, and the explicit goal.
*   **Strategic Foresight:** Your plan should be a viable path to a solution. This includes anticipating potential difficulties, identifying necessary lemmas, and choosing the most promising approach.
*   **Clarity and Brevity:** The analysis and plan must be clear, concise, and easily understood by another mathematical expert who will execute it.
"""
    
    step1_user_prompt = f"""
### Your Task

**Problem:**
==========
{problem_statement}
==========

**Metacognitive Analysis:**

**1. Problem Deconstruction:**
*   **Mathematical Domain:** Identify the primary field(s) of mathematics involved (e.g., Number Theory, Combinatorics, Euclidean Geometry).
*   **Given Conditions & Constraints:** List all the premises, conditions, and constraints provided in the problem statement in a structured format.
*   **Objective:** State the precise question to be answered or the proposition to be proven.

**2. Strategic Solution Plan (Method Sketch):**
Present a high-level, conceptual outline of your proposed solution path. This sketch should enable an expert to grasp the entire logical flow of the argument without needing the full details. It must include:
*   **Overall Strategy Narrative:** A brief description of the core idea behind your approach (e.g., "We will use proof by induction," "The strategy is to establish a coordinate system and use analytic geometry," "We will prove the contrapositive by assuming...").
*   **Key Lemmas and Intermediate Results:** State the full and precise mathematical formulations of any key lemmas or theorems you plan to prove or apply. These are the major milestones of the proof.
*   **Logical Skeleton:** If applicable, describe the key constructions, case splits, or transformations that form the backbone of your argument.
*   **Potential Challenges & Pitfalls:** Briefly note any steps that might be particularly tricky, prone to error, or require a non-obvious insight.

### Negative Constraints
*   **DO NOT** write the full, step-by-step solution.
*   **DO NOT** perform detailed algebraic manipulations or numerical calculations.
*   Your output should be strictly limited to the analysis and strategic plan as outlined above.

"""
    
    return step1_system_prompt, step1_user_prompt

def get_prompts_after_decomposition(problem_statement, step1_output):
    step2_system_prompt = f"""
You are an exceptionally rigorous mathematical solver. Your sole purpose is to take a pre-defined strategic plan and execute it with absolute precision and logical soundness. You must not deviate from, question, or reinterpret the provided plan.

Your task is to produce a complete and formally justified solution to the following mathematical problem, strictly following the `Solution Plan`.

### Core Principles

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Unyielding Adherence to Plan:** You MUST strictly follow the logical flow, lemmas, and constructions laid out in the `Solution Plan`. Do not introduce new methods, skip steps, or alter the proposed strategy in any way. Your role is execution, not creation.
*   **Honesty About Completeness:** If you cannot find a complete solution following the plan, you must **not** guess or create a solution that appears correct but contains hidden flaws. Instead, you should present only the significant partial results that you can rigorously prove by following the plan.
"""
    
    step2_user_prompt = f"""
### Your Task

**Problem:**
==========
{problem_statement}
==========

**Solution Plan:**
==========
{step1_output}
==========

**Detailed Solution:**

Present the full, step-by-step mathematical proof, meticulously following the guidance of the `Solution Plan`. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

[Your step-by-step reasoning, strictly following the plan, begins here...]

### Final Answer
After completing the detailed solution, state the final answer within \\boxed{{}}.

### Self-Correction Instruction
Before finalizing your output, carefully review your "Detailed Solution" to ensure it is clean, rigorous, and strictly adheres to all instructions provided above, especially the `Solution Plan`. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""
    
    return step2_system_prompt, step2_user_prompt


def get_monitoring_prompts(problem_statement, last_solution):
    step3_system_prompt = """
You are an expert mathematician and a meticulous grader for AIME-level computational problems. Your primary task is to rigorously verify the provided solution's **computational reasoning and numeric correctness**. A solution is to be judged correct **only if every step that affects the numeric outcome is correct and sufficiently justified.** A solution that reaches a correct final integer answer via arithmetic slips, incorrect algebraic manipulations, unverified casework, counting mistakes, or hidden assumptions must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to identify and report all issues in the provided solution. You must act strictly as a **verifier**, NOT a solver.  
*   You must **NOT attempt to correct, fix, or complete** any errors or missing arguments.  
*   Perform a **step-by-step** check of the entire solution and produce a **Detailed Verification Log**. For each step:
    *   If the step is correct, state briefly that it is correct.
    *   If the step contains an issue, explain the error and classify it (see section 2).

**2. How to Handle Issues in the Solution**
All issues must be classified into one of the following categories:

*   **a. Critical Error:**
    *   Definition: Any error that changes or potentially invalidates the numeric result. Examples include arithmetic mistakes, wrong algebraic transformations, misapplied formulas, incorrect combinatorial counts, invalid casework, or unjustified approximations that affect the integer outcome.
    *   **Procedure:**  
        *   Point out the exact error and explain why it invalidates the reasoning.  
        *   Do **not** check further steps that rely on this error.  
        *   You may still check other independent parts of the solution.

*   **b. Justification Gap:**
    *   Definition: Steps where the stated conclusion might be correct, but the reasoning is incomplete or not justified at AIME level.  
    *   **Procedure:**  
        *   Point out the missing justification.  
        *   Explicitly state that you will assume the step’s conclusion holds for the sake of checking subsequent steps.  

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**  
    *   **Final Verdict:** One clear sentence declaring overall validity (e.g., "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution contains several Justification Gaps.").  
    *   **List of Findings:** A bulleted list of every issue found. For each finding include:  
        *   **Location:** A direct quote of the key phrase or equation.  
        *   **Issue:** Short description and classification (**Critical Error** or **Justification Gap**).  

*   **b. Detailed Verification Log**  
    *   Provide a step-by-step verification.  
    *   Quote the relevant part of the solution before your check.  
    *   State clearly: **Correct**, **Critical Error**, or **Justification Gap**.  
    *   Do **not** supply corrections or alternative methods — only report the issues.  

**Important:**  
- Do not propose fixes or alternative solutions.  
- Do not attempt to supply missing reasoning.  
- Only check and report correctness of what is written.  
"""

    monitoring_reminder = """
### Monitoring Task Reminder ###

Your task is to act as an math grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""
    step3_user_prompt = f"""
### Your Task

**Original Problem:**
==========
{problem_statement}
==========

**Current Solution:**
==========
{last_solution}
==========

{monitoring_reminder}
"""
    return step3_system_prompt, step3_user_prompt


def get_controling_prompts(problem_statement, last_solution, monitor_output):
    step4_system_prompt = """
You are an expert mathematician and a careful corrector for AIME-level computational problems.  
You will be given three inputs:  
1) The Original Problem,  
2) The current Solution,  
3) A Verification Log (from a previous check), which labels each step as Correct / Justification Gap / Critical Error, and provides short notes.  

### Your Task ###
Using the Verification Log, **step by step correct the Original Solution**.  
- If a step is labeled **Correct**, keep it unchanged (you may lightly reformat for clarity).  
- If a step is labeled **Justification Gap**, supply the missing justification or intermediate calculations, enough for AIME-level rigor.  
- If a step is labeled **Critical Error**, replace it with a correct mathematical step (with explicit computations or reasoning) and update all dependent later steps accordingly.  
- Do **not** introduce new solution paths, alternative methods, or multiple approaches. Only repair the given solution chain.  

### Output Format ###

1. **Correction Summary**  
   - A single sentence declaring whether the solution has been fully corrected and what the final answer is.  
   - Example: “The solution has been fully corrected. Final Answer = 70.”  
   - Or, if not possible: “The solution cannot be fully corrected due to missing information in step X.”  

2. **Correction Log**  
   For each relevant step (especially those flagged in the Verification Log), provide an entry with:  
   - **Quoted Step:** The original line/equation (quoted or in a code block).  
   - **Verification Label:** Correct / Justification Gap / Critical Error.  
   - **Correction / Action:**  
     * If Correct → “Unchanged — correct.”  
     * If Justification Gap → Provide the missing computation/derivation briefly, ending with “Filled gap.”  
     * If Critical Error → Provide the corrected computation/derivation, briefly note why the original was wrong, and end with “Corrected.”  
   - If a step’s correction affects later steps, explicitly note “Affects subsequent steps: Yes/No.”  

3. **Full Corrected Solution**  
   - Present the entire solution in a clean, continuous write-up, combining unchanged and corrected steps.  
   - Show all necessary algebra, arithmetic, or combinatorial reasoning clearly.  
   - After completing the detailed solution, state the final answer within \\boxed{}.  
"""
    step4_user_prompt = f"""
### Your Task

**Original Problem:**
==========
{problem_statement}
==========

**Current Solution:**
==========
{last_solution}
==========

**Verification Log:**
==========
{monitor_output}
==========

"""
    return step4_system_prompt, step4_user_prompt



def extract_boxed_number(text: str):
    # 提取最后一个 \boxed{...} 中的大括号内容
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # 取最后一个
    else:
        return None

def only_keep_digits(extracted_answer):
    
    # 去掉非数字字符
    if extracted_answer is None:
        return None
    digits = re.findall(r'\d+', extracted_answer)
    if digits:
        return ''.join(digits)  # 如果有多个数字，连接起来
    else:
        return None

def extract_last_number(text: str):

    numbers = re.findall(r"\d+", text)
    if numbers:
        last_number = int(numbers[-1])
        return last_number
    else:
        return None