import argparse
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_answer(text):
    """从模型输出中提取答案字母 A-D"""
    # 优先匹配特定的答案标签
    patterns = [
        r"<answer>\s*([A-D])\s*</answer>",
        r"\\boxed\{\s*([A-D])\s*\}",
        r"The answer is\s*([A-D])",
        r"Final Answer:\s*([A-D])"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 后备方案：删除思考过程后找最后的单独字母
    text_clean = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    choices = re.findall(r'\b([A-D])\b', text_clean)
    return choices[-1].upper() if choices else None

def get_pil_image(image_data):
    """处理多种格式的图片数据"""
    if isinstance(image_data, Image.Image):
        return image_data.convert('RGB')
    if isinstance(image_data, dict) and 'bytes' in image_data:
        return Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
    if isinstance(image_data, (list, np.ndarray)) and len(image_data) > 0:
        return get_pil_image(image_data[0])
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DCR Impact (Normal vs Adversarial)")
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen2.5-VL model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to train.parquet')
    parser.add_argument('--limit', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size')
    args = parser.parse_args()

    # 1. 加载并筛选对抗样本 (Type-1)
    df = pd.read_parquet(args.dataset_path)
    test_df = df[df['type'] == 1].head(args.limit).copy()
    if len(test_df) == 0:
        print("Error: No type=1 (adversarial) samples found in dataset.")
        return
    print(f"Loaded {len(test_df)} samples for DCR evaluation.")

    # 2. 初始化环境
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1}
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024, stop=["<|im_end|>"])

    # 3. 准备两组 Prompts
    baseline_inputs = []
    adversarial_inputs = []
    ground_truths = []

    print("Preparing prompt strings...")
    for idx, row in test_df.iterrows():
        question = row['prompt'] if isinstance(row['prompt'], str) else row['prompt'][0]['content']
        # 处理可能的列表嵌套
        if isinstance(question, list):
            question = next((x['text'] for x in question if x['type'] == 'text'), "")
        
        gt = str(row['reward_model']['ground_truth']).upper()
        img = get_pil_image(row['images'])
        cap_mod = row['extra_info'].get('caption_modified', "")

        # --- Baseline Prompt ---
        msgs_base = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}
        ]
        prompt_base = processor.apply_chat_template(msgs_base, add_generation_prompt=True, tokenize=False)
        
        # --- Adversarial Prompt (Prefill Trick) ---
        msgs_adv = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": f"<thought>\nOkay, I will first give a description of the image. {cap_mod}"}
        ]
        prompt_adv = processor.apply_chat_template(msgs_adv, add_generation_prompt=True, tokenize=False)
        # 移除最后的退出标志，让模型产生“接着说”的错觉
        prompt_adv = prompt_adv[:prompt_adv.rfind("<|im_end|>")]

        baseline_inputs.append({"prompt": prompt_base, "multi_modal_data": {"image": img}})
        adversarial_inputs.append({"prompt": prompt_adv, "multi_modal_data": {"image": img}})
        ground_truths.append(gt)

    # 4. 执行推理
    print("\nRunning Baseline Inference...")
    outputs_base = llm.generate(baseline_inputs, sampling_params)
    
    print("\nRunning Adversarial Inference...")
    outputs_adv = llm.generate(adversarial_inputs, sampling_params)

    # 5. 统计结果
    results = []
    base_correct = 0
    adv_correct = 0

    for i in range(len(test_df)):
        res_base = outputs_base[i].outputs[0].text
        res_adv = outputs_adv[i].outputs[0].text
        gt = ground_truths[i]

        ans_base = parse_answer(res_base)
        ans_adv = parse_answer(res_adv)

        is_base_ok = (ans_base == gt)
        is_adv_ok = (ans_adv == gt)

        if is_base_ok: base_correct += 1
        if is_adv_ok: adv_correct += 1

        results.append({
            "gt": gt,
            "ans_base": ans_base,
            "ans_adv": ans_adv,
            "impact": "FOOLED" if (is_base_ok and not is_adv_ok) else "STAYED_CORRECT" if (is_base_ok and is_adv_ok) else "OTHER"
        })

    # 6. 报告
    print(f"\n{'='*40}")
    print(f" DCR IMPACT EVALUATION REPORT")
    print(f"{'='*40}")
    print(f"Total Samples        : {len(test_df)}")
    print(f"Baseline Accuracy    : {base_correct/len(test_df):.2%}")
    print(f"Adversarial Accuracy : {adv_correct/len(test_df):.2%}")
    print(f"Drop in Accuracy     : {(base_correct - adv_correct)/len(test_df):.2%}")
    
    # 统计有多少比例成功被带偏了
    fooled = sum(1 for r in results if r['impact'] == "FOOLED")
    print(f"Attack Success Rate  : {fooled/max(1, base_correct):.2%} (of correct baseline)")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
