import re



# 测试
# text = r"""### Step 6: Compute \\boxed{90}the Sum\n\n\\[\n\\text{Sum} = 21 + 49 = \\boxed{70}\n\\]"""
text = r"""### Step 6: Compute \\boxed0}the Sum\n\n\\[\n\\text{Sum} = 21 + 49 = \\boxed{70\n\\]"""

print(extract_boxed_number(text))  # 输出: 70

