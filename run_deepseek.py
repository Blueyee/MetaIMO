"""
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from pickle import FALSE
import sys
import json
from textwrap import indent
import requests
import argparse
import logging
from openai import OpenAI
import time
import re

# --- CONFIGURATION ---
# The model to use. "gemini-1.5-flash" is fast and capable.
#MODEL_NAME = "gemini-1.5-flash-latest" 
MODEL_NAME = "gemini-2.5-pro" 
# Use the Generative Language API endpoint, which is simpler for API key auth
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Global variables for logging
_log_file = None
original_print = print

def log_print(*args, **kwargs):
    """
    Custom print function that writes to both stdout and log file.
    """
    # Convert all arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Add timestamp to lines starting with ">>>>>"
    if message.startswith('>>>>>'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] {message}"
    
    # Print to stdout
    original_print(message)
    
    # Also write to log file if specified
    if _log_file is not None:
        _log_file.write(message + '\n')
        _log_file.flush()  # Ensure immediate writing

# Replace the built-in print function
print = log_print

def set_log_file(log_file_path):
    """Set the log file for output."""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """Close the log file if it's open."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

# def get_api_key():
#     """
#     Retrieves the Google API key from environment variables.
#     Exits if the key is not found.
#     """

#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("Error: GOOGLE_API_KEY environment variable not set.")
#         print("Please set the variable, e.g., 'export GOOGLE_API_KEY=\"your_api_key\"'")
#         sys.exit(1)
#     return api_key

def read_file_content(filepath):
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def build_request_payload(system_prompt, question_prompt, other_prompts=None):
    """
    Builds the JSON payload for the Gemini API request, using the
    recommended multi-turn format to include a system prompt.
    """
    payload = {
        "systemInstruction": {
            "role": "system",
            "parts": [
            {
                "text": system_prompt 
            }
            ]
        },
       "contents": [
        {
          "role": "user",
          "parts": [{"text": question_prompt}]
        }
      ],
      "generationConfig": {
        "temperature": 0.1,
        "topP": 1.0,
        "thinkingConfig": { "thinkingBudget": 32768} 
      },
    }

    if other_prompts:
        for prompt in other_prompts:
            payload["contents"].append({
                "role": "user",
                "parts": [{"text": prompt}]
            })

    return payload

def send_api_request(api_key, payload):
    """
    Sends the request to the Gemini API and returns the response.
    """
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key # API key now in header!
    }
    
    #print("Sending request to Gemini API...")
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        if response.status_code == 400:
            print(f"Possible reason for 400: Model '{MODEL_NAME}' might not be available or URL is incorrect for your setup.")
            print(f"Raw API Response (if available): {response.text}")
        #sys.exit(1)
        raise e

def extract_text_from_response(response_data):
    """
    Extracts the generated text from the API response JSON.
    Handles potential errors if the response format is unexpected.
    """
    try:
        return response_data['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError, TypeError) as e:
        print("Error: Could not extract text from the API response.")
        print(f"Reason: {e}")
        print("Full API Response:")
        print(json.dumps(response_data, indent=2))
        #sys.exit(1)
        raise e 

def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if(after):
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):

    dsol = extract_detailed_solution(solution)

    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_remider}
"""
    if(verbose):
        print(">>>>>>> Start verification.")
    p2 = build_request_payload(system_prompt=verification_system_prompt, 
        question_prompt=newst
        )
    
    if(verbose):
        print(">>>>>>> Verification prompt:")
        print(json.dumps(p2, indent=4))

    res = send_api_request(get_api_key(), p2)
    out = extract_text_from_response(res) 

    if(verbose):
        print(">>>>>>> Verification results:")
        print(json.dumps(out, indent=4))

    check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
            + "\n\n" + out 
    prompt = build_request_payload(system_prompt="", question_prompt=check_correctness)
    r = send_api_request(get_api_key(), prompt)
    o = extract_text_from_response(r) 

    if(verbose):
        print(">>>>>>> Is verification good?")
        print(json.dumps(o, indent=4))
        
    bug_report = ""

    if("yes" not in o.lower()):
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

        """p2["contents"].append(
            {"role": "model",
            "parts": [{"text": bug_report}]
            }
        )
        p2["contents"].append(
            {"role": "user",
            "parts": [{"text": check_verification_prompt}]
            }
        )

        if(verbose):
            print(">>>>>>> Review bug report prompt:")
            print(json.dumps(p2["contents"][-2:], indent=4))

        res = send_api_request(get_api_key(), p2)
        out = extract_text_from_response(res) 
    """

    if(verbose):
        print(">>>>>>>Bug report:")
        print(json.dumps(bug_report, indent=4))
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """

    p1 = build_request_payload(system_prompt="",    question_prompt=check_complete_prompt)
    r = send_api_request(get_api_key(), p1)
    o = extract_text_from_response(r)

    print(o)
    return "yes" in o.lower()


def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    p1  = build_request_payload(
            system_prompt=step1_prompt,
            question_prompt=problem_statement,
            #other_prompts=["* Please explore all methods for solving the problem, including casework, induction, contradiction, and analytic geometry, if applicable."]
            #other_prompts = ["You may use analytic geometry to solve the problem."]
            other_prompts = other_prompts
        )

    print(f">>>>>> Initial prompt.")
    print(json.dumps(p1, indent=4))

    response1 = send_api_request(get_api_key(), p1)
    output1 = extract_text_from_response(response1)

    print(f">>>>>>> First solution: ") 
    print(json.dumps(output1, indent=4))

    print(f">>>>>>> Self improvement start:")
    p1["contents"].append(
        {"role": "model",
        "parts": [{"text": output1}]
        }
    )
    p1["contents"].append(
        {"role": "user",
        "parts": [{"text": self_improvement_prompt}]
        }
    )

    response2 = send_api_request(get_api_key(), p1)
    solution = extract_text_from_response(response2)
    print(f">>>>>>> Corrected solution: ")
    print(json.dumps(solution, indent=4))
    
    #print(f">>>>>>> Check if solution is complete:"  )
    #is_complete = check_if_solution_claimed_complete(output1)
    #if not is_complete:
    #    print(f">>>>>>> Solution is not complete. Failed.")
    #    return None, None, None, None
    
    print(f">>>>>>> Vefify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print(f">>>>>>> Initial verification: ")
    print(json.dumps(verify, indent=4))
    print(f">>>>>>> verify results: {good_verify}")
    
    return p1, solution, verify, good_verify

############################################################################################
from utils import base_url
from utils import deepseek_api_key_2 as key

def build_messages(system_prompt, question_prompt, other_prompts=None):
    """
    创建 deepseek api 的 message
    """
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_prompt},
        ]
    return messages


def ask_api(model, messages):

    client = OpenAI(api_key=key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        # max_tokens=8192,
    )
    content = response.choices[0].message.content

    if model == "deepseek-reasoner":
        reasoning_content = response.choices[0].message.reasoning_content
        return reasoning_content, content

    # model == "deepseek-chat"
    return content


# step 1
def task_decomposition(model, problem_statement, other_prompts, verbose=True):
    
    from utils import get_decomposition_prompts
    #################
    step1_system_prompt, step1_user_prompt = get_decomposition_prompts(problem_statement)
    messages  = build_messages(
            system_prompt=step1_system_prompt,
            question_prompt=step1_user_prompt,
            other_prompts = other_prompts
        )
    print("="*20 + " Step 1, begins")
    if verbose:
        print(f">>>>>> Task decomposition prompt:")
        for p in messages:
            print(json.dumps(p, indent=4))
        
    #################
    print(f">>>>>> asking api ...")
    print("="*20)

    output = {}
    if model == "deepseek-chat":
        content = ask_api(model, messages)
        output["content"] = content

    elif model == "deepseek-reasoner":
        reasoning_content, content = ask_api(model, messages)
        output["reasoning_content"] = reasoning_content
        output["content"] = content

    else:
        print(f"Error: unknown model {model}")
        return None
    
    if verbose:
        print(f">>>>>>>> Decomposition results: ")
        print(json.dumps(output, indent=4))

    return output, messages


# step 2
def reasoning_after_decomposition(model, problem_statement, step1_output, other_prompts, verbose=True):
    
    from utils import get_prompts_after_decomposition
    #################
    step2_system_prompt, step2_user_prompt = get_prompts_after_decomposition(problem_statement, step1_output)
    messages  = build_messages(
            system_prompt=step2_system_prompt,
            question_prompt=step2_user_prompt,
            other_prompts = other_prompts
        )

    print("="*20 + " Step 2, begins")
    if verbose:
        print(f">>>>>> Reasoning after decomposition prompt:")
        for p in messages:
            print(json.dumps(p, indent=4))
        
    #################
    print(f">>>>>> asking api ...")
    print("="*20)
    
    output = {}
    if model == "deepseek-chat":
        content = ask_api(model, messages)
        output["content"] = content

    elif model == "deepseek-reasoner":
        reasoning_content, content = ask_api(model, messages)
        output["reasoning_content"] = reasoning_content
        output["content"] = content

    else:
        print(f"Error: unknown model {model}")
        return None

    if verbose:
        print(f">>>>>>>> reasoning results: ")
        print(json.dumps(output, indent=4))
    
    return output, messages


def extract_boxed_number(text: str):
    # 提取最后一个 \boxed{...} 中的大括号内容
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]  # 取最后一个
    else:
        return None


def verify(content, ground_truth):
    extracted_answer = extract_boxed_number(content)
    acc = True if extracted_answer == ground_truth else False
    return acc


def monitoring(model, problem_statement, last_solution, verbose=True):
    
    from utils import get_monitoring_prompts
    #################
    step3_system_prompt, step3_user_prompt = get_monitoring_prompts(problem_statement, last_solution)
    messages  = build_messages(
            system_prompt=step3_system_prompt,
            question_prompt=step3_user_prompt,
            other_prompts = other_prompts
        )

    print("="*20 + " Step 3, monitoring begins")
    if verbose:
        print(f">>>>>> Monitoring prompt:")
        for p in messages:
            print(json.dumps(p, indent=4))
    
    #################
    print(f">>>>>> asking api ...")
    print("="*20)
    
    output = {}
    if model == "deepseek-chat":
        content = ask_api(model, messages)
        output["content"] = content

    elif model == "deepseek-reasoner":
        reasoning_content, content = ask_api(model, messages)
        output["reasoning_content"] = reasoning_content
        output["content"] = content

    else:
        print(f"Error: unknown model {model}")
        return None

    if verbose:
        print(f">>>>>>>> Monitoring results: ")
        print(json.dumps(output, indent=4))
    
    return output, messages


def controling(model, problem_statement, last_solution, monitor_output, verbose=True):
    from utils import get_controling_prompts
    #################
    step4_system_prompt, step4_user_prompt = get_controling_prompts(problem_statement, last_solution, monitor_output)
    messages  = build_messages(
            system_prompt=step4_system_prompt,
            question_prompt=step4_user_prompt,
            other_prompts = other_prompts
        )

    print("="*20 + " Step 4, controlling begins")
    if verbose:
        print(f">>>>>> Controlling prompt:")
        for p in messages:
            print(json.dumps(p, indent=4))
    
    #################
    print(f">>>>>> asking api ...")
    print("="*20)
    
    output = {}
    if model == "deepseek-chat":
        content = ask_api(model, messages)
        output["content"] = content

    elif model == "deepseek-reasoner":
        reasoning_content, content = ask_api(model, messages)
        output["reasoning_content"] = reasoning_content
        output["content"] = content

    else:
        print(f"Error: unknown model {model}")
        return None

    if verbose:
        print(f">>>>>>>> Controlling results: ")
        print(json.dumps(output, indent=4))

    return output, messages


def agent(problem_statement, ground_truth, other_prompts=[], memory_file=None, resume_from_memory=False):

    current_iteration = 0
    solution = None
    
    verbose = True
    
    # model = "deepseek-chat"  # TODO
    model = "deepseek-reasoner"

    step1_output, step1_sent_messages = task_decomposition(model, problem_statement, other_prompts, verbose=verbose)

    step2_output, step2_sent_messages = reasoning_after_decomposition(model, problem_statement, step1_output["content"], other_prompts, verbose=verbose)
    
    
    error_count = 0
    correct_count = 0
    max_iterations = 10
    
    last_solution = step2_output["content"]
    
    acc_history = []
    
    for i in range(max_iterations):
        print(f">>>>>>> Iterations: {i}, correct_count: {correct_count}, error_count: {error_count}, acc_history: {acc_history}")

        acc = verify(last_solution, ground_truth)
        acc_history.append(acc)
        
        print(f">>>>>>> Current accuracy: {acc}, ground_truth: {ground_truth}")
        
        if acc:
            correct_count += 1
            error_count = 0
        else:
            correct_count = 0
            error_count += 1
        
        # if(correct_count >= 5):  # 判断的逻辑简化一下，只要对一次，就判为pass
        if(correct_count):  # 判断的逻辑简化一下，只要对一次，就判为pass
            print(">>>>>>> Correct solution found.")
            print(json.dumps(last_solution, indent=4))
            return True, acc_history, last_solution
        elif(error_count >= 10):
            print(">>>>>>> Failed in finding a correct solution.")
            return False, acc_history, last_solution

        # step 3: monitoring
        monitor_output, step3_sent_messages = monitoring(model, problem_statement, last_solution, verbose=verbose)
        
        # step 4: control
        control_output, step4_sent_messages = controling(model, problem_statement, last_solution, monitor_output["content"], verbose=verbose)
        
        output = control_output["content"]
        
        if "Full Corrected Solution" in output:
            idx = output.find("Full Corrected Solution")
            last_solution = output[idx + len("Full Corrected Solution"):].strip()
        else:
            last_solution = output
        
    
    return False, acc_history, last_solution



############################################################################################
        
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Meta Agent System')
    # parser.add_argument('--test_data_dir', type=str, default="./data/AIME2025/test.json", help='Path to the problems')
    parser.add_argument('--test_data_dir', type=str, default="./data/AIME2024/test.json", help='Path to the problems')

    # parser.add_argument('--test_data_dir', type=str, default="./data/GSM8K/test.json", help='Path to the problems')

    parser.add_argument('--log', '-l', type=bool, default=True, help='print to log file')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=5, help='Maximum number of runs (default: 10)')
    parser.add_argument('--memory', '-mem', type=str, help='Path to memory file for saving/loading state (optional)')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from memory file if provided')
    args = parser.parse_args()
    
    # ------------- 不重要的参数 -------------
    max_runs = args.max_runs
    memory_file = args.memory
    resume_from_memory = args.resume
    
    other_prompts = []
    if args.other_prompts:  
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)
    
    if memory_file:
        print(f"Memory file: {memory_file}")
        if resume_from_memory:
            print("Resume mode: Will attempt to load from memory file")

    # Set up logging if log file is specified
    if args.log:
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_filename = f"zz_log_{timestamp}.txt"  
        
        if not set_log_file(log_filename):
            sys.exit(1)
        print(f"Logging to file: {log_filename}")
        
    # ------------- 读取数据 -------------
    # 打开 test.json
    with open(args.test_data_dir, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems from {args.test_data_dir}")
    
    if args.test_data_dir.endswith("AIME2025/test.json"):
        # AIME2025 错误样本
        wrong_sample_IDs = [
            # "II_14", pass
            "I_14", 
            "I_15", 
            "II_13",
            ]
    elif args.test_data_dir.endswith("AIME2024/test.json"):
    
        # AIME2024 错误样本
        wrong_sample_IDs = [
            # "11",  # pass
            "29",
            ]
    else:
        wrong_sample_IDs = []
    
    for idx, item in enumerate(problems):
        
        if item['id'] not in wrong_sample_IDs:
            continue
        
        print("="*60)
        print(">>>>>>> Process problem id:", item['id'])
        print("="*60)
        
        print(json.dumps(item, indent=4))
        print("-"*40)
        
        problem_statement = item['prompt']
        ground_truth = item['answer']
    
        for i in range(max_runs):
            print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i+1}/{max_runs} ...")
            try:
                final_acc, acc_history, last_solution = agent(problem_statement, ground_truth, other_prompts)
                print(f">>>>>>> Final accuracy: {final_acc}")
                print(f">>>>>>> Accuracy history: {acc_history}")
            except Exception as e:
                print(f">>>>>>> Error in run {i}: {e}")
                continue
                
        # Close log file if it was opened
    close_log_file()


"""
指令:

python run_deepseek.py --test_data_dir ./data/AIME2025/test.json  --max_runs 5
python run_deepseek.py --test_data_dir ./data/AIME2024/test.json  --max_runs 5

"""