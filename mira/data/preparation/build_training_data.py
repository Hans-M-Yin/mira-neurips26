"""Construct training data with targeted visual-context perturbations.

The pipeline protects facts explicitly given in the question, identifies the
visual evidence needed to solve it, and edits relevant observations rather than
known conditions. Domain-specific examples cover geometry, charts, logic, and
real scenes. For multiple-choice tasks, edited captions target an incorrect
option. Blind verification checks both answer changes and whether the reasoning
uses the edited visual detail. The output schema remains compatible with the
existing training data format."""

from datetime import datetime

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import datasets
from openai import AzureOpenAI, OpenAI
import json
import argparse
if __package__:
    from .utils import encode_image, setup_logger
else:
    from utils import encode_image, setup_logger
from multiprocessing import Pool, cpu_count, Manager
import PIL.Image as Image
import re
import random
import yaml
import os
from tqdm import tqdm
import traceback
import numpy as np
import io
from pathlib import Path

# HDFS/FUSE does not reliably support flush/append on ordinary log files.
# Keep the multi-process runtime log on the local disk instead of the cwd.
LOCAL_LOG_DIR = Path("/tmp/verl_preprocess_logs")
LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_LOG_PATH = str(LOCAL_LOG_DIR / "build_training_data.log")
logger = setup_logger(__name__, LOCAL_LOG_PATH)

REQUEST_EXTRA_HEADERS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 1: caption generation with visual-evidence and known-condition constraints.
# ─────────────────────────────────────────────────────────────────────────────
prompts_create_caption = {
    "system_text": """You are a meticulous visual information extractor.

The user will provide you with an image and a question.
You will examine the image carefully and output a caption that describes only the visual content relevant to answering the question.

RULES:
1. The caption must contain concrete visual evidence needed to answer the question.
   - For charts: include specific values, trends, axis labels you can read from the image.
   - For geometry: include spatial relationships, shapes, and positional attributes VISIBLE in the image.
   - For logic/pattern: include counts, orientations, fill states, and rule patterns visible in the image.
2. IMPORTANT: Clearly distinguish between:
   - [Given in question]: facts explicitly stated in the question text (e.g., "AB=5cm").
   - [Observed from image]: facts you infer by looking at the image (e.g., "D appears to be the midpoint of BC").
   Description should focus on [Observed from image] information.
3. Do NOT answer the question in the caption.
4. The caption alone should be sufficient to answer the question without relying on the image.
5. MOST IMPORTANTLY: Finally provide your caption in <description></description> tags!
    """,
    "user_text": """
Question: {question}
Your response:"""
}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 2: edit visual details using domain-specific and option-targeted constraints.
# ─────────────────────────────────────────────────────────────────────────────
prompts_edit_caption = {
    "system_text": """You are a careful visual reasoning analyst.""",
    "user_text": """I have an image, a question (with a known correct answer), and a correct caption describing the image.
Your task is to introduce exactly K **plausible visual misinterpretation errors** into the caption.

============================
ABSOLUTE RULES (Never violate):
============================
RULE 0 [Known Condition Protection — MOST CRITICAL]:
    NEVER modify any fact that is explicitly given in the Question text itself (e.g., if the question says "∠A=110°", you cannot change 110°).
    You may ONLY modify observations that a viewer must DERIVE by looking at the image.

RULE 1 [Reasoning Path Tracing — Do this FIRST]:
    Before making any modification, analyze the solution path:
    - Step A: What is the correct answer and why?
    - Step B: Which visual observations from the caption are ABSOLUTELY NECESSARY to reach that answer?
    - Step C: Choose modification targets ONLY from the observations identified in Step B.
    A modification is INVALID if someone can still reach the correct answer while ignoring your modification.

RULE 2 [Option-Targeted for Multiple Choice]:
    If the question is multiple choice, your modifications must make the caption point logically and convincingly to one specific WRONG option (state which wrong option you are targeting).

RULE 3 [Plausibility — Simulate Real Perceptual Errors]:
    Modifications must simulate realistic visual misinterpretation, NOT random fabrication.
    Domain-specific plausible error types:
    - GEOMETRY: Misidentify point positions (inside/outside, on/off a line), mistake a property (isosceles→scalene, perpendicular→oblique, median→altitude), swap two visually similar points (e.g., B and D look close).
    - CHART: Misread a grid value by 1-2 scale units, confuse trend direction (slight increase→stable, sharp rise→slight drop), misread which series a data point belongs to.
    - LOGIC/PATTERN: Off-by-one count error, reverse rotation direction (clockwise→counter-clockwise), swap filled/empty state of one cell.
    - REAL SCENE: Misidentify a spatially adjacent object, confuse left/right, misread a label or sign by one character.

RULE 4 [Sentence Structure Preservation]:
    Keep the overall sentence structure similar to the original caption. Only change the specific incorrect detail.

============================
OUTPUT FORMAT (strictly follow):
============================
First, write your reasoning (Steps A, B, C from RULE 1) outside any tags.
Then, for each of the K modifications:

<modified>The FULL caption text after ALL K modifications have been applied</modified>
<answer><before>original phrase from caption</before><after>modified phrase replacing it</after></answer>
... (exactly K <answer> tags)
<think>Why this modification is on the critical reasoning path, and which wrong option it leads to (if applicable).</think>

============================
DOMAIN EXAMPLES:
============================

--- GEOMETRY EXAMPLE ---
Input:
Question: In △ABC, BD is the angle bisector of ∠ABC, intersecting AC at point D. Given ∠A=70°, ∠C=50°, find ∠BDC. (A) 95°  (B) 100°  (C) 105°  (D) 110°
Answer: B
Description: Triangle ABC with vertex A at the top, B at the lower-left, and C at the lower-right. Point D lies on segment AC, approximately 1/3 of the way from A toward C. Segment BD extends from B to D, dividing ∠ABC into two equal halves — i.e., BD is the angle bisector of ∠ABC.

Reasoning:
Step A: ∠BDC = 180° − ∠C − ∠DBC = 180° − 50° − 30° = 100°. The key step is ∠DBC = ½∠ABC, which relies on BD being the angle bisector.
Step B: Critical visual observation: "BD divides ∠ABC into two equal halves (angle bisector)." Note — ∠A=70° and ∠C=50° are GIVEN in the question text; DO NOT touch them (RULE 0).
Step C: If BD were an altitude instead of a bisector, the calculation would change. Targeting wrong option (A) 95°: reasoning with an altitude gives a different ∠DBC, leading to answer (A).

<modified>Triangle ABC with vertex A at the top, B at the lower-left, and C at the lower-right. Point D lies on segment AC, near its midpoint. Segment BD extends from B to D and is the altitude from vertex B to side AC, meeting AC at a right angle.</modified>
<answer><before>BD divides ∠ABC into two equal halves — i.e., BD is the angle bisector of ∠ABC</before><after>BD is the altitude from vertex B perpendicular to side AC, meeting AC at a right angle</after></answer>
<think>Changing "angle bisector" to "altitude" is a classic geometric property confusion. This is on the critical reasoning path because ∠DBC = ½∠ABC depends entirely on BD being the bisector. An altitude gives no bisection relationship, leading to wrong answer (A). ∠A and ∠C were NOT modified (RULE 0 satisfied).</think>

--- CHART EXAMPLE ---
Input:
Question: Which quarter had the highest sales revenue in 2020? (A) Q1  (B) Q2  (C) Q3  (D) Q4
Answer: C
Description: The bar chart shows quarterly sales for 2020. Q1 bar reaches approximately $45K, Q2 bar reaches approximately $52K, Q3 bar is the tallest at approximately $68K, and Q4 bar reaches approximately $61K. Q3 is clearly the highest bar.

Reasoning:
Step A: The answer is Q3 because its bar (~$68K) is taller than all others. The critical visual fact is the relative height of Q3 vs Q2.
Step B: The bar heights comparison is the only evidence needed — specifically Q2 (~$52K) vs Q3 (~$68K). These are read from the image, not given in the question.
Step C: Swap Q2 and Q3 bar heights to make Q2 appear highest. This leads to wrong answer (B).

<modified>The bar chart shows quarterly sales for 2020. Q1 bar reaches approximately $45K, Q2 bar is the tallest at approximately $68K, Q3 bar reaches approximately $62K, and Q4 bar reaches approximately $61K. Q2 is clearly the highest bar.</modified>
<answer><before>Q2 bar reaches approximately $52K, Q3 bar is the tallest at approximately $68K</before><after>Q2 bar is the tallest at approximately $68K, Q3 bar reaches approximately $62K</after></answer>
<think>Swapping Q2 and Q3 bar heights is a plausible reading error — adjacent bars, close heights. This directly shifts the answer from C to B. No question-given values were modified (RULE 0 satisfied).</think>

--- LOGIC/PATTERN EXAMPLE (中文题型示例) ---
Input:
Question: 下图九宫格中，第9格应填什么形状？(A)实心三角 (B)空心三角 (C)实心圆 (D)空心圆
Answer: A
Description: 九宫格中，每行的形状按照圆→三角→正方形的顺序排列，填充状态按照实心→空心→实心的规律循环。第三行已有空心圆（第7格）和实心三角（第8格），第9格应为实心三角。

Reasoning:
Step A: 填充规律为实心→空心→实心，第三行第1格空心圆、第2格实心三角，因此第3格应为实心。答案为实心三角(A)。
Step B: 关键视觉证据：从格子图中读取的"填充状态循环方向"。这是视觉观察，题目文字中没有直接给出。
Step C: 若将循环方向改为空心→实心→空心，则第9格变为空心三角→错误选项(B)。

<modified>九宫格中，每行的形状按照圆→三角→正方形的顺序排列，填充状态按照空心→实心→空心的规律循环。第三行已有实心圆（第7格）和空心三角（第8格），第9格应为空心三角。</modified>
<answer><before>填充状态按照实心→空心→实心的规律循环</before><after>填充状态按照空心→实心→空心的规律循环</after></answer>
<think>将填充循环方向从实心→空心→实心改为空心→实心→空心，是非常自然的视觉误读（尤其图片分辨率低时）。这将答案从(A)实心三角导向(B)空心三角——一个具体的错误干扰项。未修改任何题目文字给出的已知量（RULE 0 满足）。</think>

============================
NEGATIVE EXAMPLES (DO NOT DO THESE):
============================
1. Modifying a number that was given in the question text (violates RULE 0).
2. Changing background/irrelevant details (e.g., "the sky is blue" → "the sky is gray") when sky color has nothing to do with the answer.
3. Making a modification so extreme it's obviously wrong (e.g., changing "triangle" to "sphere").
4. The modified description semantically points to the same answer as the original.

============================
NOW PROCESS THE FOLLOWING:
============================
Question: {question}
Answer: {answer}
Description: {description}
K: {k}
Your output:
"""
}

# ─────────────────────────────────────────────────────────────────────────────
# Keep the dataset prompt compatible with the training format.
# ─────────────────────────────────────────────────────────────────────────────
prompt_dataset_system = """
You are a helpful assistant.\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer (single word or few words) MUST BE put in <answer></answer>.
"""
prompt_dataset_user = "{problem}"
think_token = "<think>"
prompt_dataset_assistant = [
    think_token + "Okay, I will give a description of the image first. {caption}",
    think_token + " {caption}",
    think_token + "In the image: {caption}"
]

prompt_dataset_assistant_chinese = [
    think_token + "好的，我将首先描述图片，然后仔细分析问题并回答。{caption}",
    think_token + "{caption}",
    think_token + "图片中，{caption}"
]


def init_client(**kwargs):
    """Initialise the Azure OpenAI client used for caption editing/verification."""
    global client
    client = AzureOpenAI(**kwargs, timeout=120)


def target_init_client(**kwargs):
    """Initialise the local Qwen3-VL caption-generation server client."""
    global target_client
    target_client = OpenAI(**kwargs, timeout=120)


def _as_image_list(images):
    """Normalise HF image fields to a non-empty list of PIL images/compatible items."""
    if images is None:
        return []
    return images if isinstance(images, list) else [images]


def get_prompt_create_caption(**kwargs):
    images = _as_image_list(kwargs.get('images'))
    question = kwargs.get('question')
    image_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}
        for image in images
    ]
    return [
        {
            "role": "system",
            "content": prompts_create_caption["system_text"],
        },
        {
            "role": "user",
            "content": image_content + [
                {
                    "type": "text",
                    "text": prompts_create_caption["user_text"].format(question=question)
                }
            ]
        }
    ]


def get_prompt_edit_caption(caption, **kwargs):
    images = _as_image_list(kwargs.get('images'))
    question = kwargs.get('question')
    answer = kwargs.get('answer')
    k = kwargs.get('k')
    return [
        {
            "role": "system",
            "content": prompts_edit_caption["system_text"],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompts_edit_caption["user_text"].format(
                        question=question, answer=answer, description=caption, k=k
                    )
                },
            ] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}
                for image in images
            ]
        }
    ]


def sample_prob_distribution(class_weights):
    a, b, c = class_weights.values()
    if not (abs(a + b + c - 1.0) < 1e-10):
        raise ValueError(f"{a + b + c} ({a} {b} {c}) not equal 1.")
    r = random.random()
    if r < a:
        return 0  # No caption
    elif r < a + b:
        return 1  # Wrong caption
    else:
        return 2  # Correct caption


def sample_modify_k(k_list, weights):
    assert sum(weights) == 1, "weights must sum to 1"
    assert len(k_list) == len(weights), f"k_list and weights must have the same length"
    return random.choices(k_list, weights=weights, k=1)[0]


def process_caption(caption_response):
    if caption_response.choices[0].finish_reason != 'stop':
        return None, 0, None
    response = caption_response.choices[0].message.content
    caption = re.findall("<description>(.*?)</description>", response, re.DOTALL)
    if len(caption) > 0:
        caption = caption[0]
    else:
        print(f"No caption find:{response}")
        caption = response
    if not caption or not caption.strip():
        logger.error("No valid caption found. This item will changed into TYPE 0.")
        return None, 0, None
    return caption, None, caption


def process_edit_response(edit_response, **kwargs):
    k = kwargs.get('k')
    response = edit_response.choices[0].message.content
    modifications = []

    caption_edited = re.findall("<modified>(.*?)</modified>", response, re.DOTALL)
    if len(caption_edited) < 1:
        logger.error(f"No modified caption found. Model original output: {response}")
        return None, None, response, k
    caption_edited = caption_edited[0]
    splits = re.findall("<answer>(.*?)</answer>", response, re.DOTALL)

    for split in splits:
        before = re.findall("<before>(.*?)</before>", split, re.DOTALL)
        after = re.findall("<after>(.*?)</after>", split, re.DOTALL)
        if len(before) != 1 or len(after) != 1:
            logger.error(f"Not suitable modifications. {split}")
        if len(before) >= 1 and len(after) >= 1:
            modifications.append([before[0], after[0]])
    if len(modifications) != k:
        logger.error(f"Not enough modifications. Wish: {k}, get: {len(modifications)}")
        k = len(modifications)
    if len(modifications) < 1:
        return None, None, response, k
    return caption_edited, modifications, response, k


# ─────────────────────────────────────────────────────────────────────────────
# Blind verification checks both the answer flip and its visual cause.
# Check 1: determine whether the blind-test answer changed.
# Check 2: verify that the reasoning cites the edited visual detail.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Retry Helper for API Robustness
# ─────────────────────────────────────────────────────────────────────────────
import time

# Shared by all multiprocessing workers.  The next-slot schedule, rather than
# a per-worker sleep, enforces one aggregate request stream for caption,
# editing, verification, and retries.
rate_limit_lock = None
rate_limit_next_slot = None
rate_limit_qpm = None


def init_rate_limiter(lock, next_slot, qpm):
    global rate_limit_lock, rate_limit_next_slot, rate_limit_qpm
    rate_limit_lock = lock
    rate_limit_next_slot = next_slot
    rate_limit_qpm = qpm


def wait_for_rate_limit():
    if rate_limit_lock is None or rate_limit_next_slot is None or not rate_limit_qpm:
        return
    interval = 60.0 / rate_limit_qpm
    with rate_limit_lock:
        now = time.monotonic()
        scheduled_at = max(now, rate_limit_next_slot.value)
        rate_limit_next_slot.value = scheduled_at + interval
    delay = scheduled_at - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def init_worker(azure_client_kwargs, local_client_kwargs, lock, next_slot, qpm):
    """Create Azure/local clients in each process and attach the Azure QPM limiter."""
    init_client(**azure_client_kwargs)
    target_init_client(**local_client_kwargs)
    init_rate_limiter(lock, next_slot, qpm)

def safe_chat_create(client_instance, messages, model, max_retries=5, base_delay=3, **kwargs):
    """Call the remote service with exponential backoff.

    Retry API failures such as timeouts, rate limits, and temporary service errors
    instead of terminating the worker immediately."""
    for attempt in range(max_retries):
        try:
            request_kwargs = {
                'messages': messages,
                'model': model,
                **kwargs,
            }
            # The cache/logid headers are required by the Azure gateway, but
            # must not be sent to the locally deployed Qwen3-VL server.
            if client_instance is not target_client:
                wait_for_rate_limit()
                request_kwargs['extra_headers'] = REQUEST_EXTRA_HEADERS
            return client_instance.chat.completions.create(
                **request_kwargs
            )
        except Exception as e:
            wait_time = base_delay * (2 ** attempt)
            logger.warning(f"[API Error] Attempt {attempt+1}/{max_retries} failed for model {model}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    logger.error(f"[API Error] All {max_retries} attempts failed for model {model}.")
    return None

def verify_edited_caption(question, client, client_config, edited_caption, original_caption, modifications):
    """
    Upgraded verification:
    1. Blind test: does the edited caption lead to a different answer?
    2. Causal test: does the blind model's reasoning actually reference the modified points?

    Returns True if the modification is valid (answer changed AND modification is causally referenced).
    Returns False if invalid (should downgrade to type=0).
    """
    verify_prompt_user = """I have a question related to an image. The image is not available, but I have a caption describing the image.
Please answer the question based ONLY on the caption below.
Think step by step, then give your final answer in <answer></answer> tags.
Also, explicitly cite which part of the caption you used to reach your answer in <key_evidence></key_evidence> tags.

[Caption]: {caption}
[Question]: {question}
"""
    original_messages = [
        {
            "role": "user",
            "content": verify_prompt_user.format(question=question, caption=original_caption)
        }
    ]
    edited_messages = [
        {
            "role": "user",
            "content": verify_prompt_user.format(question=question, caption=edited_caption)
        }
    ]
    try:

        # Use the configured Azure model for verification as well.  This avoids
        # an undeclared second model and keeps all traffic under the same QPM cap.
        check_model = guide_model
        # Check 1: Answer comparison
        resp_orig = safe_chat_create(
            client_instance=client,
            messages=original_messages,
            model=check_model,
            **client_config
        )
        if not resp_orig:
            return True  # Fail open: conservatively keep it if API fails
            
        text_orig = resp_orig.choices[0].message.content
        first_answer_match = re.search("<answer>(.*?)</answer>", text_orig, re.DOTALL)
        if not first_answer_match:
            logger.error("Verify: no answer tag in original response.")
            return True  # Fail open: accept if we can't verify
        first_answer = first_answer_match.group(1).strip()

        resp_edit = safe_chat_create(
            client_instance=client,
            messages=edited_messages,
            model=check_model,  # Changed to use same model for consistency
            **client_config
        )
        if not resp_edit:
            return True
            
        text_edit = resp_edit.choices[0].message.content
        second_answer_match = re.search("<answer>(.*?)</answer>", text_edit, re.DOTALL)
        if not second_answer_match:
            logger.error("Verify: no answer tag in edited response.")
            return True
        second_answer = second_answer_match.group(1).strip()

        logger.info(f"Verify answers: orig='{first_answer}' edit='{second_answer}'")

        # Check 1 fail: answer didn't change → modification is ineffective
        if first_answer == second_answer:
            logger.info("Verify FAIL: answer unchanged after edit. Downgrading.")
            return False

        # Check 2: Causal test — did the blind model actually use the modified point?
        # if modifications:
        #     key_evidence_match = re.search("<key_evidence>(.*?)</key_evidence>", text_edit, re.DOTALL)
        #     if key_evidence_match:
        #         key_evidence = key_evidence_match.group(1).strip().lower()
        #         # Check if any of the 'after' modification strings appear in the key evidence
        #         causal_hit = False
        #         for before, after in modifications:
        #             # Use a substring of the after-text for matching (first ~20 chars)
        #             after_snippet = after[:30].lower() if len(after) > 30 else after.lower()
        #             before_snippet = before[:30].lower() if len(before) > 30 else before.lower()
        #             if after_snippet in key_evidence or before_snippet in key_evidence:
        #                 causal_hit = True
        #                 break
        #         if not causal_hit:
        #             logger.info("Verify FAIL: answer changed but modification not causally referenced. Downgrading.")
        #             return False
        #     else:
        #         # No key_evidence tag produced — accept if answer changed (fall back to Check 1 only)
        #         logger.info("Verify: no key_evidence tag, falling back to answer-only check.")

        return True

    except Exception as e:
        logger.warning(f"Verify exception: {e}. Failing open (accepting sample).")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Global state for multiprocessing
# ─────────────────────────────────────────────────────────────────────────────
client_config = None
client = None
target_client = None
k_list = None
k_weights = None
class_weights = None
target_model = None
guide_model = None


def preprocess_item(item):
    global client_config, client, k_list, k_weights, class_weights, target_model, guide_model, target_client

    class_type = sample_prob_distribution(class_weights)
    augmentation_attempted = class_type == 1
    downgrade_reason = None
    k = sample_modify_k(k_list, k_weights)
    assert client_config is not None

    caption, edited_caption, edit_info, original_output_caption, original_output_edit = None, None, None, None, None

    if class_type == 1:
        # ── Step 1: Generate correct caption ──
        messages = get_prompt_create_caption(**item)
        response = safe_chat_create(target_client, messages, target_model, **client_config)
        
        if not response:
            return {**item, "type": 0, "augmentation_attempted": augmentation_attempted,
                    "downgrade_reason": "api_fail_caption", "num_modify": k, 
                    "caption_correct": None, "caption_modified": None, "modify_info": None, 
                    "original_output": {"caption": None, "modify": None}}

        caption, info, original_output_caption = process_caption(response)

        if info != 0:
            # ── Step 2: Edit caption (improved prompt) ──
            edit_prompt = get_prompt_edit_caption(caption, k=k, **item)
            edit_response = safe_chat_create(client, edit_prompt, guide_model, **client_config)
            
            if not edit_response:
                return {**item, "type": 0, "augmentation_attempted": augmentation_attempted,
                        "downgrade_reason": "api_fail_edit", "num_modify": k, 
                        "caption_correct": caption, "caption_modified": None, "modify_info": None, 
                        "original_output": {"caption": original_output_caption, "modify": None}}

            edited_caption, edit_info, original_output_edit, k = process_edit_response(edit_response, k=k)

            if edited_caption is None or edit_info is None:
                # Edit parsing failed → downgrade to type 0
                logger.info("Edit parsing failed. Downgrading to type 0.")
                class_type = 0
                downgrade_reason = "edit_parse_failed"
                caption, edited_caption, edit_info = None, None, None
            else:
                # ── Step 3: Upgraded verification ──
                is_valid = verify_edited_caption(
                    question=item['question'],
                    client=client,
                    client_config=client_config,
                    edited_caption=edited_caption,
                    original_caption=caption,
                    modifications=edit_info
                )
                if not is_valid:
                    # Verification failed → downgrade to type 0 (normal sample)
                    logger.info("Verification failed. Downgrading to type 0.")
                    class_type = 0
                    downgrade_reason = "verification_failed"
                    edited_caption = None
                    edit_info = None
                    original_output_edit = None
                # If valid, keep class_type = 1, keep all fields
        else:
            class_type = info
            downgrade_reason = "caption_parse_failed"

    elif class_type == 2:
        # Correct caption sample
        messages = get_prompt_create_caption(**item)
        response = safe_chat_create(target_client, messages, target_model, **client_config)
        if not response:
            return {**item, "type": 0, "augmentation_attempted": augmentation_attempted,
                    "downgrade_reason": "api_fail_caption_correct", "num_modify": k, 
                    "caption_correct": None, "caption_modified": None, "modify_info": None, 
                    "original_output": {"caption": None, "modify": None}}
        caption, info, original_output_caption = process_caption(response)
        if info == 0:
            class_type = 0

    # ── Output: format identical to original data_preprocess.py ──
    res = {
        **item,
        'type': class_type,
        'augmentation_attempted': augmentation_attempted,
        'downgrade_reason': downgrade_reason,
        'num_modify': k,
        'caption_correct': caption,
        "caption_modified": edited_caption,
        'modify_info': edit_info,
        'original_output': {
            "caption": original_output_caption,
            "modify": original_output_edit
        }
    }
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Preserve the existing output format during dataset conversion.
# ─────────────────────────────────────────────────────────────────────────────
def transform_then_save_datasets(dataset_list, random_indices, args):
    """Save datasets.DataDict type dataset for verl post training."""
    try:
        dataset = Dataset.from_list(dataset_list)
    except Exception as e:
        for i, item in enumerate(dataset_list):
            try:
                test_data = [item]
                test_dataset = Dataset.from_list(test_data)
            except Exception as row_error:
                print(f"Invalid row {i}:")
                print(f"  Data: {item}")
                print(f"  Error: {row_error}")
                for key, value in item.items():
                    print(f"  {key}: {type(value)} = {value}")

    logger.info('Total: %s', len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # Keep both splits non-empty even when an emergency save contains only a
    # handful of processed samples.  (For the normal 21k-sample run this is
    # identical to the original 98/2 split.)
    if len(dataset) == 1:
        train_indices = [0]
        test_indices = [0]
    else:
        split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * (1 - args.test_size))))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

    train_dataset = dataset.select(train_indices).map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=8
    )
    # Assign the cast result and explicitly declare the image feature type.
    train_features = train_dataset.features
    train_features['images'] = [datasets.Image()]
    train_dataset = train_dataset.cast(train_features)

    test_dataset = dataset.select(test_indices).map(
        function=make_map_fn("test"),
        with_indices=True,
        num_proc=8
    )
    # Use the test split features to avoid null/string conflicts in small splits.
    # A small test split can contain only clean samples with null caption metadata.
    test_features = test_dataset.features
    test_features['images'] = [datasets.Image()]
    test_dataset = test_dataset.cast(test_features)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, args.experiment_name + f"_{args.target_model}_" + current_time + (
        "test" if args.test else ""))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")
    train_output = os.path.join(str(save_dir), "train.parquet")
    test_output = os.path.join(str(save_dir), "test.parquet")
    train_dataset.to_parquet(train_output)
    test_dataset.to_parquet(test_output)

    args.prompts = {
        "process_prompts": {
            "create": prompts_create_caption,
            "modify": prompts_edit_caption
        },
        "dataset_prompts": {
            "user": prompt_dataset_user,
            "system": prompt_dataset_system,
            "assistant": prompt_dataset_assistant
        }
    }
    # Never persist credentials in experiment metadata.
    metadata_args = vars(args).copy()
    metadata_args.pop('azure_api_key', None)
    metadata_args.pop('local_api_key', None)
    with open(os.path.join(str(save_dir), "metadata.yaml"), 'w') as f:
        yaml.dump(metadata_args, f, default_flow_style=False)
    import shutil
    try:
        shutil.copy2(LOCAL_LOG_PATH, os.path.join(str(save_dir), 'logs.log'))
    except OSError as exc:
        # The training parquet is still valid if this optional HDFS log copy
        # is unsupported; the complete local log remains available.
        print(f"[warning] Could not copy local log to output directory: {exc}")
    if random_indices is not None:
        with open(os.path.join(str(save_dir), 'random_indices.txt'), 'w') as f:
            for idx in random_indices:
                f.write(f"{idx}\n")

    logger.info(f"Saved results to {save_dir}.")

    if args.test:
        example_dir = os.path.join(save_dir, "examples")
        os.makedirs(example_dir, exist_ok=True)
        num_examples = min(5, len(train_dataset))
        logger.info(f"Creating {num_examples} example samples for debugging in {example_dir}")

        for i in range(num_examples):
            example_item = train_dataset[i]
            example_item_dir = os.path.join(example_dir, f"sample_{i:04d}")
            os.makedirs(example_item_dir, exist_ok=True)

            if example_item["images"] and len(example_item["images"]) > 0:
                image = example_item["images"][0]
                if isinstance(image, Image.Image):
                    image_path = os.path.join(example_item_dir, "image.png")
                    image.save(image_path)

            json_data = {}
            for key, value in example_item.items():
                if key == "images":
                    continue
                elif key == "prompt" and isinstance(value, list):
                    json_data[key] = [dict(item) for item in value]
                elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    json_data[key] = value
                else:
                    json_data[key] = str(value)

            json_data["image_info"] = {
                "has_image": example_item["images"] and len(example_item["images"]) > 0,
                "image_type": str(type(example_item["images"][0])) if example_item["images"] and len(
                    example_item["images"]) > 0 else None
            }
            json_path = os.path.join(example_item_dir, "data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                import json as json_mod
                json_mod.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Debug examples created in {example_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Keep the mapping function compatible with the training prompt format.
# ─────────────────────────────────────────────────────────────────────────────
def make_map_fn(split):
    def construct_prompt_for_training(
            prompt_dataset_system,
            prompt_dataset_user,
            prompt_dataset_assistant,
            problem,
            image_placeholder,
            type,
            caption_correct,
            caption_modified,
    ):
        messages = []
        if prompt_dataset_system is not None:
            messages.append({
                "role": "system",
                "content": prompt_dataset_system,
            })
        messages.append({
            "role": "user",
            "content": prompt_dataset_user.format(problem=problem)
        })
        if type == 1 or type == 2:
            caption_use = caption_modified if type == 1 else caption_correct
            assert caption_use is not None
            if "<modified>" in caption_use or "<before>" in caption_use:
                print(f"Unexpected editing markup in the selected caption: {caption_use}")
            messages.append({
                "role": "assistant",
                "content": prompt_dataset_assistant.format(caption=caption_use)
            })
        return messages

    def has_chinese(text):
        pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(pattern.search(text))

    def process_fn(example, idx):
        problem = example.pop("question")
        answer = example.pop("answer")
        images = _as_image_list(example.pop("images", None))
        normalised_images = []
        for image in images:
            if isinstance(image, dict):
                image = Image.open(io.BytesIO(image['bytes']))
            assert isinstance(image, Image.Image), "Expect PIL.Image, but got {}".format(type(image))
            normalised_images.append(image)
        images = normalised_images
        _type = example.pop("type")
        if has_chinese(problem):
            prompt_dataset_assistant_single = prompt_dataset_assistant_chinese[
                random.randint(0, len(prompt_dataset_assistant) - 1)]
        else:
            prompt_dataset_assistant_single = prompt_dataset_assistant[
                random.randint(0, len(prompt_dataset_assistant) - 1)]

        prompt = construct_prompt_for_training(
            prompt_dataset_system,
            prompt_dataset_user,
            prompt_dataset_assistant_single,
            problem,
            "",
            _type,
            example["caption_correct"],
            example["caption_modified"]
        )
        if images is None:
            logger.error(f"image {idx} is None")
        return {
            "data_source": "hiyouga/geometry3k",
            "prompt": prompt,
            "images": images,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "type": _type,
                "split": split,
                "idx": idx,  # Batch-local sequence number, used for diagnostics only.
                "source_idx": example.pop('source_idx', -1),  # Original dataset index, which identifies samples across batches.
                "answer": answer,
                "question": problem,
                "caption_correct": example.pop('caption_correct'),
                "caption_modified": example.pop('caption_modified'),
                "modify_info": example.pop('modify_info'),
                "num_modify": example.pop('num_modify'),
                "original_output": example.pop("original_output")
            },
        }

    return process_fn


def _load_from_parquet(parquet_path, is_test, exclude_row_indices=None):
    """Restore question, answer, and image fields from prepared Parquet data.

    Use the restored rows to regenerate captions and edits. Excluded row indices
    can be supplied from the corrupt-image checker output."""
    import io as _io
    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    if is_test:
        raw = raw.select(list(range(min(10, len(raw)))))

    # Exclude rows with corrupt images.
    if exclude_row_indices:
        keep = [i for i in range(len(raw)) if i not in exclude_row_indices]
        skipped = len(raw) - len(keep)
        print(f"  Excluding {skipped} corrupt-image rows from parquet ({len(keep)} remaining).")
        raw = raw.select(keep)

    def restore(example):
        extra = example.get('extra_info', {})
        if isinstance(extra, str):
            try:
                import json as _json
                extra = _json.loads(extra)
            except Exception:
                extra = {}
        question = extra.get('question', '')
        answer   = extra.get('answer', '')

        # The image field can contain PIL images or serialized byte dictionaries.
        imgs = example.get('images', [])
        if imgs and isinstance(imgs[0], dict) and 'bytes' in imgs[0]:
            imgs = [Image.open(_io.BytesIO(imgs[0]['bytes']))]
        elif imgs and hasattr(imgs[0], 'save'):
            imgs = [imgs[0]]
        else:
            imgs = []

        src_idx = extra.get('source_idx', extra.get('idx', -1))
        return {
            'question': question,
            'answer': str(answer),
            'images': imgs,
            'source_idx': src_idx,
        }

    restored = raw.map(restore, remove_columns=raw.column_names)
    # Restored rows already contain source indices; use sequential selection indices.
    selected_indices = list(range(len(restored)))
    return restored, None, selected_indices


def _load_dataset(path, is_test, is_part, local_cache_dir=None):
    try:
        dataset = load_dataset(path)['train']
    except Exception as e:
        dataset = load_from_disk(path)
        print(dataset)
    dataset = dataset.rename_column("problem", "question")
    if is_test:
        selected_indices = list(range(0,1))
        print(f"Selected source indices: {selected_indices}")

        dataset = dataset.select(selected_indices)
        random_indices = None
    elif is_part:
        exclude_indices = []
        with open(
                os.environ['PREPROCESS_EXCLUDE_INDICES'],
                'r') as f:
            for line in f:
                num = line.strip()
                if num:
                    exclude_indices.append(int(num))
        available_indices = [i for i in range(len(dataset)) if i not in exclude_indices]
        random_indices = random.sample(available_indices, 21000)
        selected_indices = random_indices
        dataset = dataset.select(random_indices)
    else:
        selected_indices = list(range(len(dataset)))
        random_indices = None

    # Store the original source index for deduplication across batches.
    # Each selected index identifies the corresponding row in the source dataset.
    def add_source_idx(example, i):
        example['source_idx'] = selected_indices[i]
        return example
    # A Dataset loaded from an HDFS-mounted save_to_disk directory otherwise
    # writes its map cache beside the source Arrow files, where temporary-file
    # creation/rename is not supported reliably.  Force this derived cache to
    # a local disk instead.
    cache_file_name = None
    if local_cache_dir:
        cache_root = Path(local_cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file_name = str(cache_root / f"add_source_idx_{os.getpid()}.arrow")
    dataset = dataset.map(
        add_source_idx,
        with_indices=True,
        load_from_cache_file=False,
        cache_file_name=cache_file_name,
    )

    return dataset, random_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--parallel_num", type=int, default=int(16))
    parser.add_argument('--azure-api-key', type=str, default=os.getenv('AZURE_OPENAI_API_KEY'),
                        help='Azure API key; defaults to $AZURE_OPENAI_API_KEY (never stored in metadata).')
    parser.add_argument('--azure-endpoint', type=str,
                        default=os.getenv('AZURE_OPENAI_ENDPOINT'),
                        help='Azure OpenAI deployment endpoint; defaults to $AZURE_OPENAI_ENDPOINT.')
    parser.add_argument('--azure-api-version', type=str, default='2024-02-01')
    parser.add_argument('--qpm', type=int, default=120,
                        help='Aggregate Azure request limit across all workers, including retries (must be <= 120).')
    parser.add_argument('--local-api-url', type=str, default='http://localhost:18903/v1',
                        help='OpenAI-compatible endpoint of the local Qwen3-VL-8B caption server.')
    parser.add_argument('--local-api-key', type=str, default='EMPTY',
                        help='API key accepted by the local Qwen3-VL server.')
    parser.add_argument('--test_size', type=float, default=0.02)
    parser.add_argument("--save_dir", type=str, default='./preprocessed_dataset')
    parser.add_argument("--experiment_name", type=str, default="dataset_mixed_22k")
    parser.add_argument("--part", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        default="/tmp/verl_preprocess_cache",
        help="Local disk directory for HuggingFace map caches; do not place this on HDFS.",
    )
    parser.add_argument("--description", type=str, default="test dataset.")
    parser.add_argument('--model', type=str, default='gpt-5.4-2026-03-05',
                        help='Azure deployment/model used for caption editing and verification.')
    parser.add_argument("--target_model", type=str, default='qwen3-vl-8b',
                        help='Model name served by --local-api-url for image-to-caption generation.')
    parser.add_argument("--guide_model", type=str, default=None)
    parser.add_argument("--from_parquet", action="store_true",
                        help="Regenerate captions from prepared Parquet data, discarding the existing captions.")
    parser.add_argument("--exclude_indices_file", type=str, default=None,
                        help="Corrupt-image row indices from the image checker; used only with --from_parquet.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a saved training Parquet file and skip previously processed samples.")

    args = parser.parse_args()
    if not args.azure_endpoint:
        parser.error('Missing Azure endpoint: set AZURE_OPENAI_ENDPOINT or pass --azure-endpoint.')
    if not args.azure_api_key:
        parser.error('Missing Azure API key: set AZURE_OPENAI_API_KEY or pass --azure-api-key.')
    if not 0 < args.qpm <= 120:
        parser.error('--qpm must be in [1, 120].')
    args.guide_model = args.guide_model or args.model
    # Favor single-detail edits for more targeted perturbations.
    args.k_list = [1, 2, 3]
    args.k_weights = [0.4, 0.4, 0.2]
    # Attempt misleading-caption generation; failed filtering falls back to clean samples.
    args.class_weights = {"a": 0.499, "b": 0.5, "c": 0.001}

    k_list = args.k_list
    k_weights = args.k_weights
    class_weights = args.class_weights
    client_config = {key: getattr(args, key) for key in ["seed", "temperature", "max_tokens", "top_p"]}
    target_model = args.target_model
    guide_model = args.guide_model
    azure_client_kwargs = {
        'api_key': args.azure_api_key,
        'api_version': args.azure_api_version,
        'azure_endpoint': args.azure_endpoint,
    }
    local_client_kwargs = {
        'api_key': args.local_api_key,
        'base_url': args.local_api_url,
    }

    # One Manager-backed schedule is shared by every worker, so --parallel_num
    # increases concurrency only up to the fixed aggregate 120-QPM ceiling.
    rate_limit_manager = Manager()
    rate_limit_lock_obj = rate_limit_manager.Lock()
    rate_limit_next_slot_obj = rate_limit_manager.Value('d', 0.0)
    init_client(**azure_client_kwargs)
    target_init_client(**local_client_kwargs)
    init_rate_limiter(rate_limit_lock_obj, rate_limit_next_slot_obj, args.qpm)

    if args.from_parquet:
        # Load a prepared Parquet file from the dataset-path argument.
        exclude_row_indices = set()
        if args.exclude_indices_file and os.path.exists(args.exclude_indices_file):
            with open(args.exclude_indices_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('row_idx'):  # Skip the header.
                        continue
                    parts = line.split('\t')
                    try:
                        exclude_row_indices.add(int(parts[0]))
                    except ValueError:
                        pass
            print(f"Loaded {len(exclude_row_indices)} corrupt-image indices from {args.exclude_indices_file}")
        dataset, random_indices, _ = _load_from_parquet(args.dataset_path, args.test,
                                                         exclude_row_indices=exclude_row_indices or None)
    else:
        dataset, random_indices = _load_dataset(
            args.dataset_path, args.test, args.part, local_cache_dir=args.local_cache_dir
        )

    # Resume by excluding source indices already present in the checkpoint.
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResume checkpoint detected: {args.resume_from}")
        try:
            checkpoint_ds = load_dataset("parquet", data_files=args.resume_from, split="train")
            completed_source_indices = set()
            import json as _json
            for extra in checkpoint_ds['extra_info']:
                if isinstance(extra, str):
                    try:
                        ex_dict = _json.loads(extra)
                        completed_source_indices.add(ex_dict.get("source_idx", -1))
                    except:
                        pass
                elif isinstance(extra, dict):
                    completed_source_indices.add(extra.get("source_idx", -1))
                    
            original_len = len(dataset)
            dataset = dataset.filter(lambda x: x.get('source_idx', -1) not in completed_source_indices)
            print(f"Resume ready: skipped {original_len - len(dataset)} processed samples; {len(dataset)} remain.")
        except Exception as e:
            print(f"Could not load the resume checkpoint: {e}; restarting from the beginning.")

    dataset_list = []
    type_count = {0: 0, 1: 0, 2: 0}
    valid_augmentations = 0
    failed_augmentations = 0
    try:
        if args.test:
            pbar = tqdm(dataset, total=len(dataset))
            for example in pbar:
                res = preprocess_item(example)
                dataset_list.append(res)
                type_count[res.get('type', 0)] += 1
                valid_augmentations += int(res.get('type') == 1)
                failed_augmentations += int(bool(
                    res.get('augmentation_attempted') and res.get('downgrade_reason')
                ))
                pbar.set_postfix_str(f"valid_aug:{valid_augmentations} failed_aug:{failed_augmentations}")
        else:
            with Pool(processes=args.parallel_num, initializer=init_worker,
                      initargs=(azure_client_kwargs, local_client_kwargs, rate_limit_lock_obj,
                                rate_limit_next_slot_obj, args.qpm)) as pool:
                pbar = tqdm(pool.imap(preprocess_item, dataset), total=len(dataset))
                for res in pbar:
                    dataset_list.append(res)
                
                    # Update counters.
                    t_type = res.get('type', 0)
                    if t_type in type_count:
                        type_count[t_type] += 1
                    valid_augmentations += int(t_type == 1)
                    failed_augmentations += int(bool(
                        res.get('augmentation_attempted') and res.get('downgrade_reason')
                    ))
                        
                    total_processed = len(dataset_list)
                    r_0 = type_count[0] / total_processed
                    r_1 = type_count[1] / total_processed
                    r_2 = type_count[2] / total_processed
                    pbar.set_postfix_str(
                        f"valid_aug:{valid_augmentations} failed_aug:{failed_augmentations} "
                        f"Type0:{r_0:.1%} Type1:{r_1:.1%} Type2:{r_2:.1%}"
                    )
    except BaseException as e:
        print(f"\nProcessing interrupted or failed: {e}")
        print(f"Saving the {len(dataset_list)} completed samples before exiting.")

    if len(dataset_list) > 0:
        print(f"Augmentation statistics: successful augmented samples (type=1)={valid_augmentations}, "
              f"failed augmentations converted to clean samples={failed_augmentations}, type distribution={type_count}")
        print(f"Saving processed data")
        transform_then_save_datasets(dataset_list=dataset_list, random_indices=random_indices or [], args=args)
    else:
        print("No completed samples to save.")
