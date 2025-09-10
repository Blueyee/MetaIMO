




"""
"""
from openai import OpenAI
import os
import json
import time

"""
sk-e2c481c08cf04736814434c9cc7ac6ba  婉霞
"""

def ask_api(content):

   
    client = OpenAI(api_key="sk-e2c481c08cf04736814434c9cc7ac6ba", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        # model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": content},
        ],
        stream=False,
        max_tokens=8192,
    )

    # deepseek-reasoner
    # reasoning_content = response.choices[0].message.reasoning_content
    # content = response.choices[0].message.content
    # return reasoning_content, content

    # deepseek-chat
    # print(response.choices[0].message.content)
    return response.choices[0].message.content



        
content = f"Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
    
# print(f"content: {content}")
api_answer = ask_api(content)
print(f"answer: {api_answer}")
    
        

"""
"""