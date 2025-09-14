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

import gsm8k


# --- CONFIGURATION ---
# The model to use. "gemini-1.5-flash" is fast and capable.
# MODEL_NAME = "gemini-1.5-flash-latest" 

MODEL_NAME = "gemini-2.5-pro" 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# MODEL_NAME = "gemini-2.5-flash-lite" 
# API_URL = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{MODEL_NAME}:streamGenerateContent?key=API_KEY"

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

def save_memory(memory_file, problem_statement, other_prompts, current_iteration, max_runs, solution=None, verify=None):
    """
    Save the current state to a memory file.
    """
    memory = {
        "problem_statement": problem_statement,
        "other_prompts": other_prompts,
        "current_iteration": current_iteration,
        "max_runs": max_runs,
        "solution": solution,
        "verify": verify,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"Memory saved to {memory_file}")
        return True
    except Exception as e:
        print(f"Error saving memory to {memory_file}: {e}")
        return False

def load_memory(memory_file):
    """
    Load the state from a memory file.
    """
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        print(f"Memory loaded from {memory_file}")
        return memory
    except Exception as e:
        print(f"Error loading memory from {memory_file}: {e}")
        return None


def get_api_key():
    """
    Retrieves the Google API key from environment variables.
    Exits if the key is not found.
    """
    from utils import gemini_api_key
    api_key = gemini_api_key
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set the variable, e.g., 'export GOOGLE_API_KEY=\"your_api_key\"'")
        sys.exit(1)
    return api_key

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
    
    
from utils import *

def agent(problem_statement, ground_truth, other_prompts=[], memory_file=None, resume_from_memory=False):
    
    current_iteration = 0
    solution = None
    verify = None
    
    verbose = True
    
    from utils import custom_system_prompt, formatting_reminder
    
    payload  = build_request_payload(
            system_prompt=custom_system_prompt,
            question_prompt=problem_statement + formatting_reminder,
            other_prompts = other_prompts
        )

    if verbose:
        print(f">>>>>> prompt.")
        print(json.dumps(payload, indent=4))

    response = send_api_request(get_api_key(), payload)
    output = extract_text_from_response(response)

    if verbose:
        print(f">>>>>>> Solution: ") 
        print(json.dumps(output, indent=4))
        
        
    ##########
    # Gemini 有个问题，输出不放在box中，而且会自动在最后加一个check，导致提取最后一个数字也不行
    ##########
    
    # verify
    extracted_answer = extract_boxed_number(output)
    extracted_answer = only_keep_digits(extracted_answer)
    acc = True if extracted_answer == ground_truth else False
    if acc:
        print(f"extracted_answer: {extracted_answer}")
        print(f"Ground truth: {ground_truth}")
        print(f"Accuracy: {acc}")
        return True
    
    # verify
    acc = extract_last_number(output) == int(ground_truth)
    if acc:
        print(f"extracted_answer: {extract_last_number(output)}")
        print(f"Ground truth: {ground_truth}")
        print(f"Accuracy: True")
        return True
    
    # verify
    from gsm8k import compute_score
    acc = compute_score(output, ground_truth)
    if acc:
        print(f"Ground truth: {ground_truth}")
        print(f"Accuracy: {acc}")
        return True
    
    # verify : Final check
    if "inal check" in output :
        output_before_final_check = output.split("inal check")[-2]
        acc = extract_last_number(output_before_final_check) == int(ground_truth)
        if acc:
            print(f"extracted_answer: {extract_last_number(output_before_final_check)}")
            print(f"Ground truth: {ground_truth}")
            print(f"Accuracy: True")
            return True
    
    acc = False
    print(f"Ground truth: {ground_truth}")
    print(f"Accuracy: {acc}")
    return False
        
############################################################################################
        
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Meta Agent System')
    parser.add_argument('--test_data_dir', type=str, default="./data/AIME2025/test.json", help='Path to the problems')
    # parser.add_argument('--test_data_dir', type=str, default="./data/AIME2024/test.json", help='Path to the problems')
    # parser.add_argument('--test_data_dir', type=str, default="./data/GSM8K/test.json", help='Path to the problems')

    parser.add_argument('--log', '-l', type=bool, default=True, help='print to log file')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=1, help='Maximum number of runs (default: 10)')
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
    
    # # AIME2025 错误样本
    wrong_sample_IDs = [
        "I_13", 
        "I_14", 
        "I_15",
        ]
    
    # # AIME2024 错误样本
    wrong_sample_IDs = [
        "1", 
        "10",
        ]
    
    
    for idx, item in enumerate(problems):
        
        # if item['id'] not in wrong_sample_IDs:
        #     continue
        
        # sleep 60 seconds to avoid rate limit
        print("Sleeping 10 seconds to avoid rate limit...")
        time.sleep(10)
        
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
                agent(problem_statement, ground_truth, other_prompts)
            except Exception as e:
                print(f">>>>>>> Error in run {i}: {e}")
                continue
                
        # Close log file if it was opened
    close_log_file()
