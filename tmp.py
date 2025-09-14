
import re

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



def verify(output, ground_truth):

    # verify
    print("\nverify: 1")
    extracted_answer = extract_boxed_number(output)
    extracted_answer = only_keep_digits(extracted_answer)
    print(f"extracted_answer: {extracted_answer}")
    acc = True if extracted_answer == ground_truth else False
    if acc:
        print(f"extracted_answer: {extracted_answer}")
        print(f"Ground truth: {ground_truth}")
        print(f"Accuracy: {acc}")
        return True

    
    # verify
    print("\nverify: 2")
    print(f"extract_last_number(output): {extract_last_number(output)}")
    acc = extract_last_number(output) == int(ground_truth)
    if acc:
        print(f"extracted_answer: {extract_last_number(output)}")
        print(f"Ground truth: {ground_truth}")
        print(f"Accuracy: True")
        return True

    # verify
    print("\nverify: 3")
    from gsm8k import compute_score
    acc = compute_score(output, ground_truth)
    if acc:
        print(f"Ground truth: {ground_truth}")
        print(f"Accuracy: {acc}")
        return True
    
    # verify : Final check
    print("\nverify: 4")
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


if __name__ == "__main__":
    # 测试代码
    output = "5441 \\equiv 441 \\pmod{1000}$.\n\nThe number of triples is $N=3^{12}=531441$.\nThe remainder when $N$ is divided by 666 is 441.\n\nFinal check on the argument $A_m=3^{2m+1}$ for $m \\ge 3$.\n$A_k = N_k^{(1)} + N_k^{(2)}$.\n$N_k^{(1)} = 3^6 A_{k-3}$.\n$N_k^{(2)} = 3 \\times (\\#a \\in 3\\mathbb{Z}_{3^k}) \\times (\\# (b,c) \\in (\\mathbb{Z}_{3^k}^\\times)^2, b^3+c^3 \\equiv -a^3 \\pmod{3^k})$.\nThis path is complicated. The scaling argument seems much more robust.\nThe key relation is $a \\in \\{1, \\dots, 3^k\\}, a_0 \\in \\{1, \\dots, 3^{k-1}\\}, a=a_0+a_1 3^{k-1}$.\n$a^3 \\equiv a_0^3 \\pmod{3^k}$ for $k \\ge 2$.\nWe are working with $a^3+b^3+c^3 \\equiv 0 \\pmod{3^7}$.\nLet $a \\in \\{1, \\dots, 3^6\\}$.\nLet $a' \\in \\{1, \\dots, 3^7\\}$. $a' = a+k \\cdot 3^6$ for $a \\in \\{1, \\dots, 3^6\\}$ and $k \\in \\{0,1,2\\}$.\n$a'^3 = (a+k3^6)^3 = a^3+3a^2k3^6 + \\dots = a^3+a^2k3^7+\\dots$.\n$a'^3 \\equiv a^3 \\pmod{3^7}$.\nThis is correct. The scaling argument holds.\n\nFinal answer is $3^{12} \\pmod{1000}$.\n$3^{12} = (3^6)^2 = 729^2 \\equiv (-271)^2 \\pmod{1000}$.\n$271^2 = (270+1)^2 = 270^2+540+1 = 72900+541 \\equiv 900+541 = 1441 \\equiv 441 \\pmod{1000}$.\n\n"
    ground_truth = "441"
    verify(output, ground_truth)