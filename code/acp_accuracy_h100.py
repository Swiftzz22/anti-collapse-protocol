#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACP (Anti-Collapse Pipeline) — ACCURACY edition, single H100 (80 GB)
=====================================================================

What this does
--------------
For every STUDENT model (Mistral-7B, Gemma-2-9B, Qwen2.5-7B, Llama-3.1-8B) and every TASK
(GSM8K, ARC-Challenge, CommonsenseQA, BoolQ) it runs the recursive self-training loop from the
notebook for 8 rounds and measures TEST ACCURACY at step 0 (base model) and after every round
(steps 1..8).

Each round:
  1. vLLM phase   : load student (+ current LoRA) -> evaluate on the test set (greedy)
                    -> generate synthetic solutions for QUOTA train questions over 5 routes
                    -> unload.
  2. Train phase  : load student in bf16 (+ LoRA) ->
                      acp   : correctness gate (cross-route answer self-consistency, NO gold labels)
                              -> T1 entropy band (frozen base = adapters disabled)
                              -> T2 novelty / degeneration
                              -> T3 semantic dedup + topic-cluster balance
                              -> counterfactual probes, rotating Red-Queen critics, route balance
                      naive : accept every generation that has an extractable answer
                              (standard self-consuming loop = collapse baseline)
                    -> route-aware LoRA training on the cumulative accepted pool -> save adapter -> unload.

Gold labels are NEVER used for filtering or training; they are only used to compute accuracy
and the diagnostic "label precision" of the accepted data.

Every phase runs in a fresh Python subprocess, so the GPU is guaranteed to be fully released
between model loads (vLLM does not always free memory in-process). Inside each worker the model
is also explicitly unloaded (del + gc + empty_cache + vLLM engine shutdown).

H100 speed features: vLLM (paged attention, CUDA graphs, prefix caching, batched multi-LoRA),
bf16 everywhere (no 4-bit), TF32 matmuls, FlashAttention-2 for training if installed, fused AdamW,
length-grouped micro-batches, no gradient checkpointing unless an OOM forces it, shared step-0
work between conditions, and full resume support (just re-run the script after a crash).

Install (fresh env, CUDA 12.x):
    pip install -U vllm "transformers>=4.46" "peft>=0.13" "sentence-transformers>=3.0" \
                   "datasets>=3.0" scikit-learn matplotlib hf_transfer
    pip install flash-attn --no-build-isolation      # optional but recommended

Run:
    python acp_accuracy_h100.py                       # everything
    python acp_accuracy_h100.py --models qwen2.5-7b --tasks gsm8k boolq
    python acp_accuracy_h100.py --summary-only        # just rebuild tables/plots

You must accept the licences on Hugging Face for meta-llama/Llama-3.1-8B-Instruct,
google/gemma-2-9b-it and mistralai/Mistral-7B-Instruct-v0.3 with the account behind HF_TOKEN.
"""

import os
import sys
import json
import gc
import re
import math
import time
import random
import zlib
import argparse
import subprocess
import shutil
from collections import Counter, defaultdict

import numpy as np

# =====================================================================================
#                                   USER CONFIG
# =====================================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "PASTE_YOUR_HF_TOKEN_HERE")   # <-- export HF_TOKEN (preferred) or paste it here

MODELS = {                                      # key -> Hugging Face repo
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-9b":   "google/gemma-2-9b-it",
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
TASKS = ["gsm8k", "arc_challenge", "commonsense_qa", "boolq"]
CONDITIONS = ["acp"]   # "acp" = full anti-collapse pipeline; "naive" = no filtering baseline.
                                # Remove "naive" to halve the runtime.

OUT_DIR = os.path.abspath("./acp_accuracy_results")
ROUNDS = 8                      # accuracy is reported at step 0 (base) and after rounds 1..8
QUOTA = 400                     # train questions per round (x 5 routes = 2000 candidates/round)
SEED = 42

ROUTES = ["direct", "paraphrase", "expand", "contrast", "socratic"]
SAMPLES_PER_ROUTE = 1
EVAL_MAX_EXAMPLES = None        # None = full test set; set e.g. 500 for a quick run

# ---- vLLM (generation + evaluation) ----
VLLM_GPU_UTIL = 0.90
VLLM_MAX_MODEL_LEN = 4096
VLLM_MAX_NUM_SEQS = 512
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.95

# ---- ACP gates ----
CORRECTNESS_MIN_ROUTES = 3      # >= 3 of 5 routes must agree on the final answer
ENTROPY_PCT_LOW, ENTROPY_PCT_HIGH = 10, 90
T2_MIN_NOVELTY = 0.10           # fraction of output 3-grams not copied from the seed questions
T2_MIN_DISTINCT = 0.20          # distinct-token ratio (degeneration guard)
NUM_CLUSTERS = 8
T3_MAX_KL = 1.5
DEDUP_THRESHOLD = 0.94
DEDUP_MEMORY = 4000
MAX_ROUTE_SHARE = 0.45
CRITIC_NAMES = ["template", "style", "entropy_shape", "lexical_diversity"]
CRITIC_REJECT_VOTES_NEEDED = 2
CRITIC_ROTATION_EVERY = 3
COUNTERFACTUAL_RATE = 0.15
TRIPWIRE_MIN_ACCEPT_RATE = 0.05
TRIPWIRE_MIN_ROUTE_ENTROPY = 1.0
ENTROPY_TOKEN_BUDGET = 12288    # tokens per forward pass when scoring reference entropy

# ---- LoRA training (bf16, no quantization) ----
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN_LR = 1e-4
TRAIN_MICRO_BS = 8
TRAIN_GRAD_ACCUM = 2         # effective batch 16 (same as the notebook)
TRAIN_MAX_STEPS_PER_ROUND = 64
TRAIN_EPOCHS_PER_ROUND = 1
TRAIN_MAX_LEN = 1024
ROUTE_UPWEIGHT_FACTOR = 2.0
GRAD_CHECKPOINTING = False      # auto-enabled if a step OOMs
MODEL_TRAIN_OVERRIDES = {
    "gemma2-9b": {"TRAIN_MICRO_BS": 4, "TRAIN_GRAD_ACCUM": 4},
}
KEEP_ALL_ADAPTERS = False       # False = keep only the latest adapter of each run (saves disk)

EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

# =====================================================================================
#                                   TASKS
# =====================================================================================
TASK_SPECS = {
    "gsm8k": {"kind": "math", "max_tokens": 512,
              "contrast": "Solve it, then verify the result with a different method or a quick check."},
    "arc_challenge": {"kind": "mcq", "max_tokens": 384,
                      "contrast": "Consider each option in turn and explain why the incorrect ones are wrong."},
    "commonsense_qa": {"kind": "mcq", "max_tokens": 384,
                       "contrast": "Consider each option in turn and explain why the incorrect ones are wrong."},
    "boolq": {"kind": "yesno", "max_tokens": 256,
              "contrast": "Weigh the evidence in the passage for both 'yes' and 'no' before deciding."},
}
ROUTE_SUFFIX = {
    "direct": "",
    "paraphrase": "Start by restating the question in your own words, then answer it.",
    "expand": "Work through the problem carefully, step by step, before answering.",
    "contrast": None,  # task specific, see TASK_SPECS
    "socratic": "Break the problem into smaller sub-questions, answer each one, then combine them.",
}
FORMAT_REMINDER = "Finish with the final line in the required 'Answer:' format."


def route_prompt(task, prompt, route):
    suffix = TASK_SPECS[task]["contrast"] if route == "contrast" else ROUTE_SUFFIX[route]
    if not suffix:
        return prompt
    return f"{prompt}\n\n{suffix} {FORMAT_REMINDER}"


def prompt_math(q):
    return ("Solve the following math word problem. Reason step by step, then write the final numeric "
            "answer on the last line in the form 'Answer: <number>'.\n\nProblem: " + q.strip())


def prompt_mcq(q, labels, texts):
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return ("Answer the following multiple-choice question. Explain your reasoning briefly, then write the "
            "final answer on the last line in the form 'Answer: <letter>'.\n\nQuestion: " + q.strip() +
            "\n\nOptions:\n" + opts)


def prompt_yesno(passage, q):
    q = q.strip()
    q = (q[:1].upper() + q[1:]).rstrip("?") + "?"
    return ("Read the passage and answer the question with yes or no. Explain your reasoning briefly, then "
            "write the final answer on the last line in the form 'Answer: yes' or 'Answer: no'.\n\n"
            "Passage: " + passage.strip() + "\n\nQuestion: " + q)


# ---------------- answer extraction ----------------
_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_ANS_NUM = re.compile(r"(?i:answer)\s*(?i:is)?\s*[:：]?\s*\**\s*\$?\s*(" + _NUM + ")")
_BOXED = re.compile(r"\\boxed\{\s*\$?(" + _NUM + ")")
_ANY_NUM = re.compile(_NUM)
_ANS_LET = re.compile(r"(?i:answer)\s*(?i:is)?\s*[:：]?\s*\**\s*\(?([A-E])\)?(?![A-Za-z])")
_LONE_LET = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")
_ANS_YN = re.compile(r"answer\s*(?:is)?\s*[:：]?\s*\**\s*(yes|no)\b", re.I)


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isinf(v) or math.isnan(v):
        return None
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def extract_math(text):
    for rx in (_ANS_NUM, _BOXED):
        m = rx.findall(text)
        if m:
            return norm_num(m[-1])
    m = _ANY_NUM.findall(text)
    return norm_num(m[-1]) if m else None


def extract_mcq(text, valid):
    m = [x for x in _ANS_LET.findall(text) if x in valid]
    if m:
        return m[-1]
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if lines:
        m = [x for x in _LONE_LET.findall(lines[-1]) if x in valid]
        if m:
            return m[-1]
    return None


def extract_yesno(text):
    m = _ANS_YN.findall(text)
    if m:
        return m[-1].lower()
    t = text.strip().lower()
    if not t:
        return None
    m = re.findall(r"\b(yes|no)\b", t.splitlines()[-1])
    if m:
        return m[-1]
    m = re.match(r"\W*(yes|no)\b", t)
    return m.group(1) if m else None


def extract_answer(task, text, valid=None):
    kind = TASK_SPECS[task]["kind"]
    if kind == "math":
        return extract_math(text)
    if kind == "mcq":
        return extract_mcq(text, valid or list("ABCDE"))
    return extract_yesno(text)


# =====================================================================================
#                                   SMALL UTILITIES
# =====================================================================================
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def task_dir(task):
    d = os.path.join(OUT_DIR, "tasks", task)
    os.makedirs(d, exist_ok=True)
    return d


def run_dir(model_key, task, cond):
    d = os.path.join(OUT_DIR, "runs", model_key, task, cond)
    os.makedirs(d, exist_ok=True)
    return d


def route_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values() if c))


def stable_seed(*parts):
    return (SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 31)


# =====================================================================================
#                          DATASET PREPARATION (CPU, main process)
# =====================================================================================
def prepare_tasks(tasks):
    from datasets import load_dataset
    for task in tasks:
        d = task_dir(task)
        tr, te = os.path.join(d, "train.jsonl"), os.path.join(d, "test.jsonl")
        if os.path.exists(tr) and os.path.exists(te):
            continue
        print(f"[data] preparing {task} ...", flush=True)
        train, test = [], []
        if task == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main")
            for split, out in (("train", train), ("test", test)):
                for i, x in enumerate(ds[split]):
                    out.append({"id": f"gsm8k-{split}-{i}", "prompt": prompt_math(x["question"]),
                                "gold": norm_num(x["answer"].split("####")[-1]), "valid": None})
        elif task == "arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
            remap = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            for split, out in (("train", train), ("validation", train), ("test", test)):
                for x in ds[split]:
                    labels = [remap.get(l, l) for l in x["choices"]["label"]]
                    gold = remap.get(x["answerKey"], x["answerKey"])
                    if gold not in labels:
                        continue
                    out.append({"id": f"arc-{x['id']}", "prompt": prompt_mcq(x["question"], labels, x["choices"]["text"]),
                                "gold": gold, "valid": labels})
        elif task == "commonsense_qa":
            ds = load_dataset("tau/commonsense_qa")
            # the official test split has no labels -> validation is the standard eval split
            for split, out in (("train", train), ("validation", test)):
                for x in ds[split]:
                    labels = list(x["choices"]["label"])
                    if x["answerKey"] not in labels:
                        continue
                    out.append({"id": f"csqa-{x['id']}", "prompt": prompt_mcq(x["question"], labels, x["choices"]["text"]),
                                "gold": x["answerKey"], "valid": labels})
        elif task == "boolq":
            ds = load_dataset("google/boolq")
            for split, out in (("train", train), ("validation", test)):
                for i, x in enumerate(ds[split]):
                    out.append({"id": f"boolq-{split}-{i}", "prompt": prompt_yesno(x["passage"], x["question"]),
                                "gold": "yes" if x["answer"] else "no", "valid": None})
        else:
            raise ValueError(task)
        write_jsonl(tr, train)
        write_jsonl(te, test)
        print(f"[data] {task}: train={len(train)} test={len(test)}", flush=True)


# =====================================================================================
#                               GPU HELPERS (workers only)
# =====================================================================================
def setup_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN attention is unstable here; use flash/mem-efficient SDPA
    except Exception:
        pass
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def free_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_report(tag=""):
    import torch
    if torch.cuda.is_available():
        print(f"  [mem {tag}] alloc={torch.cuda.memory_allocated() / 1e9:.1f}G "
              f"reserved={torch.cuda.memory_reserved() / 1e9:.1f}G", flush=True)


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_prompt_ids(tok, user_text):
    text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]   # template already contains BOS


def encode_sft(tok, prompt, output, max_len):
    """(input_ids, labels) with loss only on the assistant turn (incl. its end-of-turn token)."""
    p_ids = chat_prompt_ids(tok, prompt)
    f_text = tok.apply_chat_template([{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": output}], tokenize=False)
    f_ids = tok(f_text, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:   # template not prefix-stable -> safe fallback
        f_ids = p_ids + tok(output, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) >= max_len - 16:
        return None
    f_ids = f_ids[:max_len]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels


def pick_attn_impl(model_path):
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "eager" if "gemma" in model_path.lower() else "sdpa"


class VLLMSession:
    """Load vLLM on enter, fully unload on exit."""

    def __init__(self, model_path, use_lora, seed):
        self.model_path, self.use_lora, self.seed, self.llm = model_path, use_lora, seed, None

    def __enter__(self):
        from vllm import LLM
        kw = dict(model=self.model_path, tokenizer=self.model_path, dtype="bfloat16",
                  gpu_memory_utilization=VLLM_GPU_UTIL, max_model_len=VLLM_MAX_MODEL_LEN,
                  max_num_seqs=VLLM_MAX_NUM_SEQS, seed=self.seed, enable_prefix_caching=True)
        if self.use_lora:
            kw.update(enable_lora=True, max_lora_rank=LORA_R, max_loras=1)
        t0 = time.time()
        try:
            self.llm = LLM(**kw)
        except Exception as e:
            print(f"  [vllm] init failed ({type(e).__name__}: {e}); retrying without prefix caching", flush=True)
            free_memory()
            kw["enable_prefix_caching"] = False
            self.llm = LLM(**kw)
        print(f"  [vllm] loaded in {time.time() - t0:.0f}s (lora={self.use_lora})", flush=True)
        return self

    def generate(self, token_id_lists, sampling_params, lora_request=None):
        prompts = [{"prompt_token_ids": ids} for ids in token_id_lists]
        return self.llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=True)

    def __exit__(self, *exc):
        llm, self.llm = self.llm, None
        try:
            eng = getattr(llm, "llm_engine", None)
            core = getattr(eng, "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        del llm
        free_memory()
        print("  [vllm] unloaded", flush=True)
        return False


# =====================================================================================
#                  WORKER 1: vLLM  -> evaluate accuracy + generate candidates
# =====================================================================================
def worker_vllm(a):
    import torch  # noqa: F401
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    setup_torch()
    task, spec = a["task"], TASK_SPECS[a["task"]]
    tok = load_tokenizer(a["model_path"])
    adapter = a.get("adapter")

    with VLLMSession(a["model_path"], use_lora=adapter is not None, seed=a["seed"]) as sess:
        lora = LoRARequest("student", 1, adapter) if adapter else None

        # ---------- evaluation (greedy) ----------
        if a.get("eval_out"):
            test = read_jsonl(os.path.join(task_dir(task), "test.jsonl"))
            if EVAL_MAX_EXAMPLES:
                test = test[:EVAL_MAX_EXAMPLES]
            t0 = time.time()
            sp = SamplingParams(temperature=0.0, max_tokens=spec["max_tokens"])
            outs = sess.generate([chat_prompt_ids(tok, ex["prompt"]) for ex in test], sp, lora)
            correct, formatted, samples = 0, 0, []
            for ex, o in zip(test, outs):
                text = o.outputs[0].text
                pred = extract_answer(task, text, ex["valid"])
                formatted += pred is not None
                correct += (pred is not None and pred == ex["gold"])
                if len(samples) < 25:
                    samples.append({"id": ex["id"], "gold": ex["gold"], "pred": pred, "output": text})
            res = {"step": a["step"], "accuracy": correct / max(1, len(test)), "n": len(test),
                   "correct": correct, "format_rate": formatted / max(1, len(test)),
                   "eval_seconds": time.time() - t0, "samples": samples}
            save_json(a["eval_out"], res)
            print(f"  [eval] {a['model_key']} {task} step {a['step']}: acc={res['accuracy']:.4f} "
                  f"format={res['format_rate']:.3f} ({res['eval_seconds']:.0f}s)", flush=True)

        # ---------- synthetic generation (self-consuming loop) ----------
        if a.get("gen_out"):
            train = read_jsonl(os.path.join(task_dir(task), "train.jsonl"))
            items = [(i, train[i]) for i in a["gen_indices"]]
            reqs, meta = [], []
            for pid, ex in items:
                for route in ROUTES:
                    reqs.append(chat_prompt_ids(tok, route_prompt(task, ex["prompt"], route)))
                    meta.append((pid, ex, route))
            t0 = time.time()
            sp = SamplingParams(n=SAMPLES_PER_ROUTE, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
                                max_tokens=spec["max_tokens"])
            outs = sess.generate(reqs, sp, lora)
            rows = []
            for (pid, ex, route), o in zip(meta, outs):
                for k, c in enumerate(o.outputs):
                    text = c.text.strip()
                    if not text:
                        continue
                    rows.append({"prompt_id": pid, "sample": k, "prompt": ex["prompt"], "route": route,
                                 "output": text, "pred": extract_answer(task, text, ex["valid"]),
                                 "gold": ex["gold"], "truncated": c.finish_reason == "length",
                                 "round_id": a["gen_round"]})
            write_jsonl(a["gen_out"], rows)
            print(f"  [gen] round {a['gen_round']}: {len(rows)} candidates from {len(items)} questions "
                  f"({time.time() - t0:.0f}s)", flush=True)


# =====================================================================================
#                      ACP COMPONENTS (critics, counterfactuals, gates)
# =====================================================================================
class Cand:
    __slots__ = ("prompt_id", "prompt", "route", "output", "pred", "gold", "truncated", "round_id",
                 "entropy", "embedding", "cluster_id", "critic_votes", "is_cf", "accepted")

    def __init__(self, d):
        for k in ("prompt_id", "prompt", "route", "output", "pred", "gold", "truncated", "round_id"):
            setattr(self, k, d.get(k))
        self.entropy = self.embedding = self.cluster_id = None
        self.critic_votes, self.is_cf, self.accepted = [], False, False

    def clone_with(self, output):
        c = Cand({k: getattr(self, k) for k in ("prompt_id", "prompt", "route", "pred", "gold",
                                                 "truncated", "round_id")})
        c.output, c.is_cf = output, True
        return c

    def to_pool(self):
        return {"prompt": self.prompt, "output": self.output, "route": self.route, "round_id": self.round_id,
                "pred": self.pred, "gold": self.gold, "entropy": self.entropy}


# ---------------- correctness gate (task version of cross-route self-consistency) ----------------
def correctness_gate(cands, min_routes):
    by_prompt = defaultdict(list)
    for c in cands:
        if c.pred is not None and not c.truncated:
            by_prompt[c.prompt_id].append(c)
    kept = []
    for group in by_prompt.values():
        votes = Counter(c.pred for c in group)
        ranked = votes.most_common()
        top, top_n = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == top_n:
            continue                                   # tie -> no consensus
        if len({c.route for c in group if c.pred == top}) < min_routes:
            continue
        kept.extend(c for c in group if c.pred == top)
    return kept


# ---------------- counterfactual probes (measured only, never accepted) ----------------
NEGATION_TRIGGERS = [" is ", " was ", " are ", " were ", " can ", " does "]


def _shuffle_sentences(text, rng):
    s = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(s) < 3:
        return text
    rng.shuffle(s)
    return " ".join(s)


def _flip_negation(text, rng):
    for trig in NEGATION_TRIGGERS:
        idx = text.find(trig)
        if idx != -1:
            return text[:idx] + " " + trig.strip() + " not " + text[idx + len(trig):]
    return text


def make_counterfactuals(cands, seed):
    rng = random.Random(seed)
    cfs = []
    for c in cands:
        if rng.random() < COUNTERFACTUAL_RATE:
            new = _shuffle_sentences(c.output, rng) if rng.random() < 0.5 else _flip_negation(c.output, rng)
            if new != c.output:
                cfs.append(c.clone_with(new))
    return cfs


# ---------------- Red-Queen critics ----------------
TEMPLATE_PATTERNS = [r"^as an ai\b", r"^i cannot\b", r"^i'm sorry\b", r"^in conclusion,?\s*$",
                     r"^certainly!?\s*$", r"^sure,?\s*here", r"^title:\s*"]


def critic_template(c, **kw):
    text = c.output.strip().lower()
    if any(re.search(p, text) for p in TEMPLATE_PATTERNS):
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len(lines) >= 3 and len(set(lines)) <= max(1, len(lines) // 3)


def critic_style(c, **kw):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", c.output.strip()) if s]
    if len(sents) < 5:
        return False
    lengths = [len(s.split()) for s in sents]
    return float(np.std(lengths)) / (float(np.mean(lengths)) + 1e-8) < 0.08


def critic_entropy_shape(c, **kw):
    return c.entropy is not None and c.entropy < 0.15


def critic_lexical_diversity(c, tokenizer=None, **kw):
    toks = tokenizer.encode(c.output, add_special_tokens=False) if tokenizer is not None else c.output.split()
    if len(toks) < 5:
        return False
    return len(set(toks)) / len(toks) < 0.30


CRITIC_REGISTRY = {"template": critic_template, "style": critic_style,
                   "entropy_shape": critic_entropy_shape, "lexical_diversity": critic_lexical_diversity}


def evaluate_critics(c, active, tokenizer):
    votes = []
    for name in active:
        try:
            if CRITIC_REGISTRY[name](c, tokenizer=tokenizer):
                votes.append(name)
        except Exception:
            pass
    return votes


def active_critics_for(round_id, forced):
    if forced and forced.get("round") == round_id:
        return list(forced["critics"])
    n = len(CRITIC_NAMES)
    if n <= 2:
        return list(CRITIC_NAMES)
    block = (round_id - 1) // max(1, CRITIC_ROTATION_EVERY)
    return random.Random(block).sample(CRITIC_NAMES, max(2, n - 1))


def interleave_by_route(cands, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for c in cands:
        buckets[c.route].append(c)
    keys = sorted(buckets)
    for k in keys:
        rng.shuffle(buckets[k])
    out = []
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop())
    return out


# ---------------- T1 / T2 / T3 gates with persistent state ----------------
class ACPGates:
    def __init__(self, tok, centers, target_dist, seed_3grams, prev_state_path):
        self.tok, self.centers, self.target_dist, self.seed_3grams = tok, centers, target_dist, seed_3grams
        self.low, self.high = 0.0, 1e9
        st = np.load(prev_state_path) if prev_state_path and os.path.exists(prev_state_path) else None
        self.committed = st["committed"].astype(np.float32) if st is not None else np.zeros((0, 384), np.float32)
        self.cluster_counts = st["cluster_counts"].astype(np.float64) if st is not None else np.zeros(len(centers))
        self.committed = self.committed[-DEDUP_MEMORY:]

    def save(self, path):
        np.savez(path, committed=self.committed[-DEDUP_MEMORY:], cluster_counts=self.cluster_counts)

    def t2(self, text):
        tk = self.tok.encode(text, add_special_tokens=False)
        if len(tk) < 3:
            return False
        grams = [tuple(tk[i:i + 3]) for i in range(len(tk) - 2)]
        novelty = sum(1 for g in grams if g not in self.seed_3grams) / len(grams)
        distinct = len(set(tk)) / len(tk)
        return novelty >= T2_MIN_NOVELTY and distinct >= T2_MIN_DISTINCT

    def is_dup(self, emb):
        return self.committed.shape[0] > 0 and float((self.committed @ emb).max()) >= DEDUP_THRESHOLD

    def t3(self, cid):
        total = self.cluster_counts.sum()
        if total < 50:
            return True
        k = len(self.cluster_counts)
        cur = (self.cluster_counts + 1) / (total + k)
        new = cur.copy()
        new[cid] += 1 / (total + 1)
        kl = float(np.sum(new * np.log((new + 1e-8) / (self.target_dist + 1e-8))))
        return kl <= T3_MAX_KL

    def check(self, c):
        if c.entropy is None or not (self.low <= c.entropy <= self.high):
            return False, False, False
        if not self.t2(c.output) or self.is_dup(c.embedding):
            return True, False, False
        return True, True, self.t3(c.cluster_id)

    def commit(self, c):
        if self.is_dup(c.embedding):
            return False, "dup"
        if not self.t3(c.cluster_id):
            return False, "t3"
        self.committed = np.vstack([self.committed, c.embedding[None]])
        self.cluster_counts[c.cluster_id] += 1
        return True, "ok"


def t4_filter(cands, round_id, gates, prepare_fn, active, tok, existing_route_counts):
    stats = Counter()
    stats["input_candidates"] = len(cands)
    stats["active_critics"] = ",".join(active)
    if not cands:
        return [], dict(stats)

    # a. counterfactual probes
    cfs = make_counterfactuals(cands, seed=round_id)
    if cfs:
        prepare_fn(cfs)
        caught = 0
        for c in cfs:
            ok1, ok2, ok3 = gates.check(c)
            votes = evaluate_critics(c, active, tok)
            caught += (not (ok1 and ok2 and ok3)) or len(votes) >= CRITIC_REJECT_VOTES_NEEDED
        stats["cf_probes"], stats["cf_caught"], stats["cf_catch_rate"] = len(cfs), caught, caught / len(cfs)

    # b. T1/T2/T3
    stage = []
    for c in cands:
        ok1, ok2, ok3 = gates.check(c)
        if not ok1:
            stats["reject_t1"] += 1
        elif not ok2:
            stats["reject_t2"] += 1
        elif not ok3:
            stats["reject_t3"] += 1
        else:
            stage.append(c)
    stats["pass_t1t2t3"] = len(stage)

    # c. per-round route diversity
    stage_c, running = [], Counter()
    for c in interleave_by_route(stage, seed=round_id):
        n = sum(running.values())
        if n >= 20 and (running[c.route] + 1) / (n + 1) > MAX_ROUTE_SHARE:
            stats["reject_route_diversity"] += 1
            continue
        running[c.route] += 1
        stage_c.append(c)
    stats["pass_route_diversity"] = len(stage_c)

    # d. critics
    stage_d = []
    for c in stage_c:
        c.critic_votes = evaluate_critics(c, active, tok)
        if len(c.critic_votes) >= CRITIC_REJECT_VOTES_NEEDED:
            stats["reject_critics"] += 1
            continue
        stage_d.append(c)
    stats["pass_critics"] = len(stage_d)

    # e. global route balance + commit
    existing = Counter(existing_route_counts)
    final = []
    for c in stage_d:
        total = sum(existing.values()) + 1
        if total >= 50 and (existing[c.route] + 1) / total > MAX_ROUTE_SHARE + 0.15:
            stats["reject_global_route_balance"] += 1
            continue
        ok, why = gates.commit(c)
        if not ok:
            stats[f"reject_commit_{why}"] += 1
            continue
        existing[c.route] += 1
        c.accepted = True
        final.append(c)
    stats["final_accepted"] = len(final)
    stats["final_route_entropy"] = route_entropy(Counter(c.route for c in final))
    stats["accept_rate"] = len(final) / max(1, stats["input_candidates"])
    return final, dict(stats)


# =====================================================================================
#                 WORKER 2: filter (ACP / naive) + route-aware LoRA training
# =====================================================================================
def _collate(examples, pad_id):
    import torch
    L = max(len(x[0]) for x in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return {"input_ids": ids.pin_memory(), "attention_mask": att.pin_memory(), "labels": lab.pin_memory()}


def reference_bits_per_token(model, tok, cands):
    """T1: mean NLL (bits/token) of the assistant response under the FROZEN base model (adapters off)."""
    import torch
    import torch.nn.functional as F
    enc = []
    for i, c in enumerate(cands):
        e = encode_sft(tok, c.prompt, c.output, TRAIN_MAX_LEN)
        enc.append((i, e))
    order = sorted([x for x in enc if x[1] is not None], key=lambda x: len(x[1][0]))
    res = [None] * len(cands)
    batches, cur, cur_max = [], [], 0
    for item in order:
        L = len(item[1][0])
        if cur and max(cur_max, L) * (len(cur) + 1) > ENTROPY_TOKEN_BUDGET:
            batches.append(cur); cur, cur_max = [], 0
        cur.append(item); cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    model.eval()
    with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batches:
            batch = {k: v.cuda(non_blocking=True) for k, v in _collate([x[1] for x in b], tok.pad_token_id).items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            sl, lb = logits[:, :-1, :], batch["labels"][:, 1:]
            loss = F.cross_entropy(sl.float().reshape(-1, sl.size(-1)), lb.reshape(-1),
                                   reduction="none", ignore_index=-100).view(lb.shape)
            m = (lb != -100).float()
            bits = ((loss * m).sum(1) / m.sum(1).clamp(min=1) / math.log(2)).tolist()
            for (i, _), v in zip(b, bits):
                res[i] = float(v)
            del logits, sl, loss, batch
    return res


def ensure_clusters(task, embed_fn):
    path = os.path.join(task_dir(task), "clusters.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["centers"], z["target_dist"]
    from sklearn.cluster import MiniBatchKMeans
    train = read_jsonl(os.path.join(task_dir(task), "train.jsonl"))
    rng = random.Random(SEED)
    sample = rng.sample(train, min(5000, len(train)))
    emb = embed_fn([x["prompt"] for x in sample])
    km = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=256, random_state=SEED, n_init=10).fit(emb)
    counts = np.bincount(km.labels_, minlength=NUM_CLUSTERS)
    target = (counts + 1) / (counts.sum() + NUM_CLUSTERS)
    np.savez(path, centers=km.cluster_centers_.astype(np.float32), target_dist=target)
    return km.cluster_centers_.astype(np.float32), target


def nearest_cluster(emb, centers):
    d = (emb ** 2).sum(1, keepdims=True) - 2 * emb @ centers.T + (centers ** 2).sum(1)[None]
    return d.argmin(1)


def _train_once(model, examples, pad_id, mbs, accum, use_ckpt, seed):
    import torch
    from transformers import get_linear_schedule_with_warmup
    if use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    rng = random.Random(seed)
    order = list(range(len(examples)))
    micro, chunk = [], mbs * accum * 8
    for s in range(0, len(order), chunk):            # length-grouped micro-batches (less padding)
        block = sorted(order[s:s + chunk], key=lambda i: len(examples[i][0]), reverse=True)
        micro += [block[k:k + mbs] for k in range(0, len(block), mbs)]
    rng.shuffle(micro)
    steps = max(1, math.ceil(len(micro) / accum))
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        opt = torch.optim.AdamW(params, lr=TRAIN_LR, weight_decay=0.0, fused=True)
    except Exception:
        opt = torch.optim.AdamW(params, lr=TRAIN_LR, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, max(1, steps // 10), steps)
    model.train()
    model.config.use_cache = False
    losses, t0 = [], time.time()
    for s in range(steps):
        group = micro[s * accum:(s + 1) * accum]
        if not group:
            break
        step_loss = torch.zeros((), device="cuda")
        for mb in group:
            batch = {k: v.cuda(non_blocking=True) for k, v in _collate([examples[i] for i in mb], pad_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / len(group)).backward()
            step_loss += loss.detach() / len(group)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(step_loss)
        if (s + 1) % 16 == 0 or s + 1 == steps:
            print(f"    [train] step {s + 1}/{steps} loss={float(step_loss):.4f}", flush=True)
    model.eval()
    vals = [float(x) for x in losses]
    return {"train_steps": len(vals), "train_loss_mean": float(np.mean(vals)) if vals else None,
            "train_loss_last": vals[-1] if vals else None, "train_seconds": time.time() - t0,
            "micro_bs": mbs, "grad_accum": accum, "grad_ckpt": use_ckpt}


def train_lora_round(model, tok, pool, cond, round_id, model_key):
    import torch
    ov = MODEL_TRAIN_OVERRIDES.get(model_key, {})
    mbs, accum = ov.get("TRAIN_MICRO_BS", TRAIN_MICRO_BS), ov.get("TRAIN_GRAD_ACCUM", TRAIN_GRAD_ACCUM)
    eff = mbs * accum
    n = min(len(pool) * TRAIN_EPOCHS_PER_ROUND, TRAIN_MAX_STEPS_PER_ROUND * eff)
    if cond == "acp":                                    # route-aware up-weighting of rare routes
        counts = Counter(p["route"] for p in pool)
        mx = max(counts.values())
        w = np.array([1.0 + (ROUTE_UPWEIGHT_FACTOR - 1.0) * (1.0 - counts[p["route"]] / mx) for p in pool])
    else:
        w = np.ones(len(pool))
    rng = np.random.default_rng(stable_seed("train", round_id, cond, model_key))
    idx = rng.choice(len(pool), size=n, replace=True, p=w / w.sum())
    examples = [e for e in (encode_sft(tok, pool[i]["prompt"], pool[i]["output"], TRAIN_MAX_LEN) for i in idx)
                if e is not None]
    if not examples:
        return {"train_steps": 0}
    snap = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    use_ckpt = GRAD_CHECKPOINTING
    for attempt in range(4):
        oom = False
        try:
            return _train_once(model, examples, tok.pad_token_id, mbs, accum, use_ckpt,
                               seed=stable_seed("order", round_id, cond, attempt))
        except torch.cuda.OutOfMemoryError:
            oom = True
        # cleanup happens OUTSIDE the except block, so the failed step's tensors are really released
        if oom:
            print(f"    [train] OOM (micro_bs={mbs}, ckpt={use_ckpt}) -> restoring weights and retrying smaller",
                  flush=True)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for k, v in model.named_parameters():
                    if k in snap:
                        v.copy_(snap[k])
            free_memory()
            if not use_ckpt:
                use_ckpt = True
            elif mbs > 1:
                mbs, accum = mbs // 2, accum * 2
    raise RuntimeError("Training kept running out of memory.")


def worker_train(a):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    setup_torch()
    task, cond, r, model_key = a["task"], a["cond"], a["round"], a["model_key"]
    rd = a["run_dir"]
    t_start = time.time()
    tok = load_tokenizer(a["model_path"])
    tok.padding_side = "right"

    # ---------------- load student (bf16) + LoRA ----------------
    attn = pick_attn_impl(a["model_path"])
    base = AutoModelForCausalLM.from_pretrained(a["model_path"], torch_dtype=torch.bfloat16,
                                                attn_implementation=attn, device_map={"": 0},
                                                low_cpu_mem_usage=True)
    base.config.use_cache = False
    if a.get("prev_adapter"):
        model = PeftModel.from_pretrained(base, a["prev_adapter"], is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():                         # fp32 master copy of the LoRA weights
        if p.requires_grad:
            p.data = p.data.float()
    model.eval()
    print(f"  [train-worker] {model_key} loaded (attn={attn}, prev_adapter={bool(a.get('prev_adapter'))})", flush=True)
    gpu_report("student loaded")

    raw = read_jsonl(a["cand_file"])
    cands = [Cand(d) for d in raw]
    stats = {"round": r, "condition": cond, "num_generated": len(cands),
             "num_with_answer": sum(c.pred is not None for c in cands),
             "generated_label_precision": float(np.mean([c.pred == c.gold for c in cands])) if cands else 0.0}

    prev_stats = load_json(os.path.join(rd, f"stats_round_{r - 1}.json"), {}) if r > 1 else {}
    prev_pool = []
    for k in range(1, r):
        p = os.path.join(rd, f"pool_round_{k}.jsonl")
        if os.path.exists(p):
            prev_pool += read_jsonl(p)

    # ---------------- filtering ----------------
    if cond == "naive":
        accepted = [c for c in cands if c.pred is not None]
        stats.update({"num_after_correctness": len(accepted), "accept_rate": len(accepted) / max(1, len(cands)),
                      "final_route_entropy": route_entropy(Counter(c.route for c in accepted))})
    else:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDER_ID, device="cuda")

        def embed(texts):
            return embedder.encode(texts, batch_size=256, show_progress_bar=False,
                                   normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        centers, target = ensure_clusters(task, embed)
        train = read_jsonl(os.path.join(task_dir(task), "train.jsonl"))
        seed_3grams = set()
        for ex in random.Random(SEED).sample(train, min(500, len(train))):
            tk = tok.encode(ex["prompt"], add_special_tokens=False)
            seed_3grams.update(tuple(tk[i:i + 3]) for i in range(len(tk) - 2))
        prev_gate = os.path.join(rd, f"gate_state_round_{r - 1}.npz") if r > 1 else None
        gates = ACPGates(tok, centers, target, seed_3grams, prev_gate)
        prompt_cluster_cache = {}

        def prepare(cs):
            if not cs:
                return cs
            ents = reference_bits_per_token(model, tok, cs)
            embs = embed([c.output for c in cs])
            new_prompts = list({c.prompt for c in cs if c.prompt not in prompt_cluster_cache})
            if new_prompts:
                for p, k in zip(new_prompts, nearest_cluster(embed(new_prompts), centers)):
                    prompt_cluster_cache[p] = int(k)
            for c, e, v in zip(cs, ents, embs):
                c.entropy, c.embedding, c.cluster_id = e, v, prompt_cluster_cache[c.prompt]
            return cs

        t0 = time.time()
        kept = correctness_gate(cands, CORRECTNESS_MIN_ROUTES)
        stats["num_after_correctness"] = len(kept)
        prepare(kept)
        valid = [c.entropy for c in kept if c.entropy is not None and np.isfinite(c.entropy)]
        if len(valid) > 10:
            gates.low = float(np.percentile(valid, ENTROPY_PCT_LOW))
            gates.high = float(np.percentile(valid, ENTROPY_PCT_HIGH))
        stats["entropy_band"] = [gates.low, gates.high]
        active = active_critics_for(r, prev_stats.get("forced_critics"))
        accepted, fstats = t4_filter(kept, r, gates, prepare, active, tok, Counter(p["route"] for p in prev_pool))
        stats.update(fstats)
        stats["accept_rate"] = len(accepted) / max(1, len(cands))   # relative to all generated
        stats["filter_seconds"] = time.time() - t0
        gates.save(os.path.join(rd, f"gate_state_round_{r}.npz"))
        # tripwires -> force critic rotation next round
        if stats["accept_rate"] < TRIPWIRE_MIN_ACCEPT_RATE or stats.get("final_route_entropy", 99) < TRIPWIRE_MIN_ROUTE_ENTROPY:
            others = [c for c in CRITIC_NAMES if c not in ("template", "style")]
            stats["forced_critics"] = {"round": r + 1, "critics": ["template", "style"] + others}
            print("  [tripwire] forcing critic rotation next round", flush=True)
        del embedder
        free_memory()

    stats["num_accepted"] = len(accepted)
    stats["accepted_label_precision"] = float(np.mean([c.pred == c.gold for c in accepted])) if accepted else 0.0
    stats["accepted_route_dist"] = dict(Counter(c.route for c in accepted))
    print(f"  [filter] {cond}: generated={len(cands)} accepted={len(accepted)} "
          f"label_precision(accepted)={stats['accepted_label_precision']:.3f}", flush=True)

    # ---------------- route-aware training on the cumulative pool ----------------
    this_pool = [c.to_pool() for c in accepted]
    pool = prev_pool + this_pool
    stats["pool_size"] = len(pool)
    if pool:
        stats.update(train_lora_round(model, tok, pool, cond, r, model_key))
    else:
        print("  [train] empty pool -> adapter unchanged", flush=True)
    os.makedirs(a["out_adapter"], exist_ok=True)
    model.save_pretrained(a["out_adapter"])
    write_jsonl(os.path.join(rd, f"pool_round_{r}.jsonl"), this_pool)
    stats["phase_seconds"] = time.time() - t_start
    save_json(os.path.join(rd, f"stats_round_{r}.json"), stats)

    # ---------------- explicit unload ----------------
    del model, base
    free_memory()
    gpu_report("student unloaded")


# =====================================================================================
#                                ORCHESTRATOR (main process)
# =====================================================================================
def run_worker(kind, args, tag):
    os.makedirs(os.path.join(OUT_DIR, "_worker_args"), exist_ok=True)
    path = os.path.join(OUT_DIR, "_worker_args", f"{tag}.json")
    save_json(path, args)
    env = dict(os.environ)
    env.update(HF_TOKEN=HF_TOKEN, HUGGING_FACE_HUB_TOKEN=HF_TOKEN, TOKENIZERS_PARALLELISM="false",
               VLLM_WORKER_MULTIPROC_METHOD="spawn")
    if kind == "train":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", kind, "--args", path]
    for attempt in range(2):
        t0 = time.time()
        rc = subprocess.run(cmd, env=env).returncode
        if rc == 0:
            return time.time() - t0
        print(f"[orchestrator] worker {kind} failed with code {rc} (attempt {attempt + 1}/2)", flush=True)
        time.sleep(10)
    raise RuntimeError(f"Worker {kind} failed twice: {tag}")


def download_model(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, token=HF_TOKEN,
                             ignore_patterns=["*.pth", "*.pt", "original/*", "consolidated*", "*.gguf", "*.msgpack",
                                              "*.h5", "*.onnx"])


def round_indices(task, round_id):
    with open(os.path.join(task_dir(task), "train.jsonl")) as f:
        n = sum(1 for _ in f)
    perm = np.random.default_rng(stable_seed("prompts", task)).permutation(n)   # same for every model/condition
    start = ((round_id - 1) * QUOTA) % n
    return [int(perm[(start + k) % n]) for k in range(min(QUOTA, n))]


def run_one(model_key, model_path, task, cond):
    rd = run_dir(model_key, task, cond)
    base_dir = run_dir(model_key, task, "_base")      # step-0 eval + round-1 generations shared by conditions
    state_path = os.path.join(rd, "state.json")
    state = load_json(state_path, {"accuracy": {}, "format_rate": {}, "trained": 0, "rounds": {}})
    print(f"\n{'=' * 90}\nRUN  model={model_key}  task={task}  condition={cond}\n{'=' * 90}", flush=True)

    def adapter_dir(step):
        return os.path.join(rd, "adapters", f"round_{step}")

    for step in range(ROUNDS + 1):
        t_round = time.time()
        adapter = adapter_dir(step) if step > 0 else None
        src = base_dir if step == 0 else rd
        eval_path = os.path.join(src, f"eval_step_{step}.json")
        gen_path = os.path.join(src, f"gen_round_{step + 1}.jsonl") if step < ROUNDS else None
        need_eval = not os.path.exists(eval_path)
        need_gen = gen_path is not None and not os.path.exists(gen_path) and state["trained"] < step + 1

        # ---- phase A: vLLM eval (step) + generation (round step+1) ----
        if need_eval or need_gen:
            run_worker("vllm", {
                "model_key": model_key, "model_path": model_path, "task": task, "adapter": adapter,
                "step": step, "seed": stable_seed("vllm", model_key, task, cond if step else "base", step),
                "eval_out": eval_path if need_eval else None,
                "gen_out": gen_path if need_gen else None, "gen_round": step + 1,
                "gen_indices": round_indices(task, step + 1) if need_gen else None,
            }, tag=f"{model_key}_{task}_{cond}_vllm_{step}")
        ev = load_json(eval_path)
        state["accuracy"][str(step)] = ev["accuracy"]
        state["format_rate"][str(step)] = ev["format_rate"]
        print(f"[orchestrator] {model_key} | {task} | {cond} | STEP {step}: accuracy = {ev['accuracy']:.4f}",
              flush=True)

        # ---- phase B: filter + train (round step+1) ----
        if step < ROUNDS and state["trained"] < step + 1:
            run_worker("train", {
                "model_key": model_key, "model_path": model_path, "task": task, "cond": cond,
                "round": step + 1, "run_dir": rd, "cand_file": gen_path,
                "prev_adapter": adapter, "out_adapter": adapter_dir(step + 1),
            }, tag=f"{model_key}_{task}_{cond}_train_{step + 1}")
            st = load_json(os.path.join(rd, f"stats_round_{step + 1}.json"), {})
            state["rounds"][str(step + 1)] = {k: v for k, v in st.items() if not isinstance(v, (dict, list))}
            state["trained"] = step + 1
            if step > 0 and not KEEP_ALL_ADAPTERS:
                shutil.rmtree(adapter_dir(step), ignore_errors=True)
            print(f"[orchestrator] round {step + 1} done in {(time.time() - t_round) / 60:.1f} min "
                  f"(accepted={st.get('num_accepted')}, label_precision={st.get('accepted_label_precision', 0):.3f})",
                  flush=True)
        save_json(state_path, state)
        save_json(os.path.join(rd, "results.json"),
                  {"model": model_key, "task": task, "condition": cond, "accuracy": state["accuracy"],
                   "format_rate": state["format_rate"], "rounds": state["rounds"]})


# ---------------- summary: CSV, markdown table, plots ----------------
def build_summary():
    rows = []
    root = os.path.join(OUT_DIR, "runs")
    if not os.path.isdir(root):
        return
    for mk in sorted(os.listdir(root)):
        for task in sorted(os.listdir(os.path.join(root, mk))):
            for cond in sorted(os.listdir(os.path.join(root, mk, task))):
                res = load_json(os.path.join(root, mk, task, cond, "results.json"))
                if res:
                    rows.append(res)
    if not rows:
        return
    steps = list(range(ROUNDS + 1))
    csv_path = os.path.join(OUT_DIR, "accuracy_summary.csv")
    with open(csv_path, "w") as f:
        f.write("model,task,condition," + ",".join(f"step_{s}" for s in steps) + ",delta_final_vs_base\n")
        for r in rows:
            acc = [r["accuracy"].get(str(s)) for s in steps]
            delta = (acc[-1] - acc[0]) if acc[-1] is not None and acc[0] is not None else ""
            f.write(f"{r['model']},{r['task']},{r['condition']}," +
                    ",".join("" if a is None else f"{a:.4f}" for a in acc) + f",{delta}\n")
    md_path = os.path.join(OUT_DIR, "accuracy_table.md")
    with open(md_path, "w") as f:
        f.write("| Model | Task | Cond | " + " | ".join(f"S{s}" for s in steps) + " | Δ(S8−S0) |\n")
        f.write("|" + "---|" * (len(steps) + 4) + "\n")
        for r in rows:
            acc = [r["accuracy"].get(str(s)) for s in steps]
            cells = ["—" if a is None else f"{100 * a:.1f}" for a in acc]
            d = f"{100 * (acc[-1] - acc[0]):+.1f}" if acc[-1] is not None and acc[0] is not None else "—"
            f.write(f"| {r['model']} | {r['task']} | {r['condition']} | " + " | ".join(cells) + f" | {d} |\n")
    print("\n" + open(md_path).read())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]
    models = sorted({r["model"] for r in rows})
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(models)}
    ncol = 2
    nrow = max(1, math.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.8 * nrow), squeeze=False)
    for ax, task in zip(axes.flatten(), tasks):
        for r in rows:
            if r["task"] != task:
                continue
            xs = [s for s in steps if str(s) in r["accuracy"]]
            ys = [100 * r["accuracy"][str(s)] for s in xs]
            ls, mk = ("-", "o") if r["condition"] == "acp" else ("--", "x")
            ax.plot(xs, ys, ls, marker=mk, color=colors[r["model"]], label=f"{r['model']} ({r['condition']})")
        ax.set_title(task); ax.set_xlabel("Self-training step (0 = base)"); ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(steps); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    for ax in axes.flatten()[len(tasks):]:
        ax.axis("off")
    plt.suptitle("Accuracy under recursive self-training: ACP (solid) vs naive (dashed)", fontweight="bold")
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "accuracy_curves.png")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[summary] {csv_path}\n[summary] {md_path}\n[summary] {png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["vllm", "train"])
    ap.add_argument("--args")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    if a.worker:
        os.environ["HF_TOKEN"] = HF_TOKEN
        wargs = load_json(a.args)
        (worker_vllm if a.worker == "vllm" else worker_train)(wargs)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    if a.summary_only:
        build_summary()
        return
    if not HF_TOKEN.startswith("hf_") or "PASTE" in HF_TOKEN:
        raise RuntimeError("Put your real Hugging Face token in HF_TOKEN at the top of this file.")
    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except Exception:
        pass
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)

    prepare_tasks(a.tasks)
    t_all = time.time()
    for mk in a.models:
        print(f"\n[orchestrator] downloading / locating {MODELS[mk]} ...", flush=True)
        path = download_model(MODELS[mk])
        for task in a.tasks:
            for cond in a.conditions:
                t0 = time.time()
                run_one(mk, path, task, cond)
                print(f"[orchestrator] run {mk}/{task}/{cond} finished in {(time.time() - t0) / 60:.1f} min "
                      f"(total elapsed {(time.time() - t_all) / 3600:.2f} h)", flush=True)
                build_summary()
    build_summary()
    print(f"\nALL DONE in {(time.time() - t_all) / 3600:.2f} h. Results in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
