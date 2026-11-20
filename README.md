# Second Opinion: do LLMs know when they're wrong about pet health?

A free, open-source benchmark that tests how well open-weight language models judge their own uncertainty on veterinary multiple-choice questions. It runs on Google Colab's free GPU — no API keys, no cost.

For each question and model it measures:

1. **Correctness**: did the model pick the right answer?
2. **Token confidence**: the probability the model put on the answer it chose, read straight from its output probabilities.
3. **Stated confidence**: we then show the model its own answer and ask *"How confident are you, 0–100?"*

The goal is to see how the three relate: when a model sounds sure, is it actually right, and do its probabilities agree with what it says?

## Quick start (Google Colab)

1. Open `llm_uncertainty_colab_drive.ipynb` in Colab and set **Runtime → Change runtime type → T4 GPU**.
2. Put `llm-uncertainty.zip` and your questions CSV in a folder in your Google Drive.
3. Point the notebook at that folder: `CODE_ZIP_ON_DRIVE` in cell 2 and `DRIVE_FOLDER` in **Settings** (cell 4). Then choose the model(s), set `DRIVE_FILE` to your CSV, and pick a new `RUN_NAME`. Use `FIRST_N_QUESTIONS = 5` for a quick test.
4. **Runtime → Run all.** Results are saved to Drive after every question; re-run with the same `RUN_NAME` to resume after a disconnect.

Run from the command line instead:

```bash
pip install -r requirements-hf.txt
python -m llmtest run -c config.colab.toml -q questions.csv --name my-run --out results
```

## Question format

A CSV with one row per question:

| Column | Notes |
|---|---|
| `question_id` (or `id`) | Unique ID |
| `question` | Question text |
| `option_a` … `option_e` (or `A` … `E`) | Two or more options |
| `correct_option` (or `answer`) | Letter of the correct option |
| anything else | e.g. `species`, `clinical_risk`; kept and used for breakdowns |

Option E can be a "not sure / insufficient information" choice. Hide it for a run with `--not-sure hide`.

## Models

Configured in `config.colab.toml`; any Hugging Face causal LM can be added.

| Model | Notes on a free T4 |
|---|---|
| Gemma 4 E4B | 4-bit, fast |
| Gemma 4 12B | 4-bit, fast |
| gpt-oss-20b | Slow with reasoning (~20–30 s/question). For speed, answer without reasoning: set `max_tokens = 8` and `assistant_prefill = "<\|channel\|>final<\|message\|>"` |

On a T4, 4-bit models load their unquantised layers in `float16` (`dtype = "float16"`); `float32` does not fit Gemma 4.

## Outputs

Each run folder contains:

- `summary.csv`: one row per question × model. Key columns: `is_correct`, `p_answer` (token confidence), `stated_conf` (0–1), `stated_conf_raw`, `option_probs` (JSON), `abstained`, `status`.
- `report.md`: accuracy, calibration (ECE, Brier), AUROC for token and stated confidence, their rank correlation, and breakdowns by species, risk and domain.
- `raw.jsonl`: every model response in full.
- `run_meta.json`: settings and run history.

Use rows with `status == "ok"`. `stated_conf` is blank when the model replied in words rather than a number.

## Settings worth knowing

| Setting (`[run]` in config) | Default | What it does |
|---|---|---|
| `ask_confidence` | `true` | Ask the 0–100 follow-up question |
| `confidence_prompt` | see config | Wording of that question |
| `offer_not_sure` / `--not-sure` | offer | Show or hide the "not sure" option |
| `repeats` | 1 | Ask each question several times |
| `temperature` | 0.0 | Greedy decoding |

## Limitations

- One answer per question by default; small question sets give noisy estimates.
- Models run 4-bit quantised, which can shift their probabilities.
- Stated confidence depends on prompt wording; small models often give the same number every time.
- Results are a research measurement, not veterinary advice.

## License

Code: Apache License 2.0 (see `LICENSE`). Model weights are covered by their own licenses. The question set may be licensed separately (e.g. CC BY 4.0).
