# Learning From Itself Without Losing Itself: A Label-Free Defense Against Model Collapse in LLMs

> **Anonymous code and data release for double-blind review.**

**Anti-Collapse Protocol (ACP)** is a provenance-agnostic, label-free recipe for repeated synthetic fine-tuning. It decides which self-generated samples a model may train on using:

- a **consensus gate**: admit an answer only when ≥ *m* of *K* independent reasoning routes agree
- **T1 Entropy Banding**: keep mid-surprisal (learnable) samples under the frozen base model
- **T2 Tail-Mass & Novelty Floors**: preserve rare-token support, with a near-duplicate cap
- **T3 Domain Isotropy**: prevent topical lock-in
- **T4 Red-Queen Critics**: keep multiple reasoning routes (mechanisms) alive

These run as a closed loop, **Synthesize → Screen → Train → Measure → Adapt**, with tripwires. No gate ever reads a gold label or a provenance tag.

📄 Paper: [`paper/paper.pdf`](paper/paper.pdf) · 🌐 Project page: [`docs/index.html`](docs/index.html)

## Key results

| | |
|---|---|
| Accuracy within noise of base after 8 self-consuming generations | **14 / 16** settings (mean −0.72 pp) |
| Rounds where admitted-label precision > generated precision | **128 / 128** |
| Label noise in training pool | **20.0% → 11.0%** |
| Naive loop vs. ACP at equal GSM8K accuracy (G4) | **48.1%** vs **8.5%** wrong labels |
| Open-web rare-token mass (Regime II) | **+13–31%**, ≥ 97.6% teacher–route diversity kept |

## Repository layout

| Path | Content |
|---|---|
| `code/acp_accuracy_h100.py` | **Regime I**: self-consuming loop (4 students × 4 tasks × 8 generations) on a single H100, using vLLM + LoRA |
| `code/acp_openweb_regime2.ipynb` | **Regime II**: two-teacher lattice on FineWeb, 4-bit QLoRA, written for 2× T4 |
| `analysis/` | Statistics and figure scripts, plus their derived JSON outputs |
| `raw_logs/runs/` | Regime I per-round stats (`stats_round_*.json`) and `results.json` for every model/task |
| `raw_logs/openweb_*/` | Regime II admitted samples, metrics and diagnostics per backbone |
| `raw_logs/accuracy_summary.csv` | Accuracy after every generation, all 17 runs (16 ACP + 1 naive) |
| `figures/` | All data figures in the paper (PDF + PNG) |
| `paper/paper.pdf` | Anonymized submission |
| `docs/` | Project website (static, no build step) |

## Reproduce every number and figure from the logs

```bash
pip install -r requirements.txt
cd analysis
python expB.py              # Regime I: accuracy deltas, significance, precision lift
python expA.py              # Regime II: tail mass, route lattice entropy
python consensus_bound.py   # consensus-precision bound and common-mode ρ̂
python repetition_audit.py  # size-matched repetition audit (downloads a tokenizer)
python make_figures_acl.py  # regenerates figures/main_* and figures/app_*
```

On Windows, run with `python -X utf8 ...` so the console can print Unicode symbols.

## Re-running the experiments

**Regime I** (needs one 80 GB GPU and about 1 GPU-day per model):

```bash
pip install -U vllm "transformers>=4.46" "peft>=0.13" "sentence-transformers>=3.0" \
               "datasets>=3.0" scikit-learn matplotlib hf_transfer
export HF_TOKEN=...   # your Hugging Face token (never commit it)
python code/acp_accuracy_h100.py                                   # everything
python code/acp_accuracy_h100.py --models qwen2.5-7b --tasks gsm8k # a subset
python code/acp_accuracy_h100.py --summary-only                    # rebuild tables/plots
```

You must accept the Hugging Face licences for Llama-3.1-8B-Instruct, Gemma-2-9B-it and Mistral-7B-Instruct-v0.3.

**Regime II**: open `code/acp_openweb_regime2.ipynb` on a 2× T4 machine and follow the instructions in its first cell.

## Notes

- Gold labels are used **only** for evaluation and for the diagnostic "label precision". They are never used for filtering or training.
- In Regime II logs, the Qwen teacher is stored as `teacher_primary`.
- `raw_logs/openweb_*/accepted.jsonl` contains text from the public FineWeb corpus and is subject to its licence.

## License

Code is released under the MIT License (see `LICENSE`). Third-party models and datasets keep their own licences.
