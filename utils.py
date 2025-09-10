# url
base_url="https://api.deepseek.com"

# api keys
deepseek_api_key_1 = "sk-ec515d112a194762922b8b98cfea8c35"  # 我的谷歌账号注册的
deepseek_api_key_2 = "sk-e2c481c08cf04736814434c9cc7ac6ba"  # 婉霞

# prompts
step1_decomposition_prompt = """Your task is to create a problem-solving plan for the given math problem. Please break it down into sequential steps and list the corresponding sub-tasks. 

Note: JUST provide the task decomposition plan. DO NOT solve the problem at this stage.
"""

question = "Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$."


def get_decomposition_prompts(problem_statement):
    
    step1_system_prompt = """You are an expert AI assistant specializing in mathematical reasoning. You possess advanced metacognitive capabilities. Your current task is to act as a "Problem Analyst". Your goal is to produce a rigorous Metacognitive Analysis of the following problem, which includes assessing its difficulty and outlining a strategic plan. You must NOT solve the problem."""
    
    step1_user_prompt = f"""
Your task is to produce a Metacognitive Analysis of the following problem.

### Your Task
Analyze the following problem and provide your output.

==========
**Problem:**

{problem_statement}

==========

**Strategic Solution Plan (Method Sketch):**
Present a high-level, conceptual outline of a viable solution path. This sketch should allow an expert to understand the logical flow of your proposed argument without reading the full detail. It should include:
*   A narrative of your overall strategy.
*   The full and precise mathematical statements of any key lemmas or major intermediate results you plan to use or prove.
*   If applicable, a description of any key constructions or case splits that will form the backbone of your argument.

**Constraint:** This plan should be a high-level guide, free of detailed calculations or full proof steps.
"""
    
    return step1_system_prompt, step1_user_prompt

def get_prompts_after_decomposition(problem_statement, step1_output):
    step2_system_prompt = f"""
You are an expert and exceptionally rigorous mathematical problem solver. Your goal is to produce a complete and formally justified solution, strictly following a provided plan.

Your task is to solve the following mathematical problem by meticulously following the provided `Solution Plan`.

### Core Principles of Your Solution
*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Strict Adherence to Plan:** Your solution MUST strictly follow the logical flow laid out in the `Solution Plan`. Do not deviate from the proposed strategy.
*   **Clarity for Verification:** The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. Use TeX for all mathematical variables and expressions (e.g., `Let $n$ be an integer.`).
"""
    
    step2_user_prompt = f"""
### Your Task
==========
**Problem:**

{problem_statement}

==========

**Solution Plan:**

{step1_output}

==========

**Detailed Solution:**
Present the full, step-by-step mathematical proof as guided by the plan. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

[Your step-by-step reasoning here...]

Finally, provide your final answer within \\boxed{{}}.
"""
    
    return step2_system_prompt, step2_user_prompt