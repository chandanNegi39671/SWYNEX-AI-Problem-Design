# SWYNEX-AI-Problem-Design
### Banking Support Query Router: a confidence-gated intent classifier (BANKING77)

> **Task 1: AI Problem Design** | SWYNEX Technologies Internship
> Narrow use case: **single-label text classification** with a human-in-the-loop fallback.

---

## 1. Problem statement
Customers of a digital bank send free-text queries ("my card hasn't arrived", "why was I charged twice?"). A support team must tag each query with one of **77 intents** and route it to the right queue. Manual triage is slow and inconsistent.

**Goal:** automatically route the queries the model is sure about, and send everything else (including off-topic input) to a human, with measurable targets for routing quality.

## 2. User and pain point
| | |
|---|---|
| **Primary user** | Tier-1 support agent / support-ops lead at a fintech |
| **Pain** | Manual tagging of each ticket, inconsistent labels, slow first response |
| **Value** | Most tickets are routed instantly and correctly; agents spend time only on the ambiguous ones |

## 3. Why this is a *narrow* problem
One input (a short text), one output (one of 77 intents plus a confidence score and an AUTO/HUMAN decision). No generation, no dialogue, no free-form answers.

## 4. Data
| Item | Detail |
|---|---|
| Source | **BANKING77** (PolyAI, Casanueva et al., ACL 2020 NLP4ConvAI) |
| Location | `github.com/PolyAI-LDN/task-specific-datasets` (`banking_data/`) |
| Licence | **CC BY 4.0**, confirmed from the `LICENSE` file of the original PolyAI repository (copy in `data/LICENSE-CC-BY-4.0.txt`) |
| Size | 10,003 train + 3,080 test, 77 balanced-ish classes (35 to 187 examples per class in train) |
| Changes | None to text or labels; only the train/val split below is made by this project |
| Text | Real customer-style queries, mean about 12 words (2 to 79) |
| Split used | Train 8,002 / **Val 2,001** (stratified 80/20 of the official train) / **Test 3,080** (official test, touched once) |

Files are in `data/`.

## 5. Constraints
- Free / open-source tools only, CPU only (scikit-learn)
- Latency target: **< 50 ms per query** on CPU
- Model must be small and explainable enough to audit
- Wrong auto-routes cost more than a human review, so the system must be able to **abstain**

## 6. Evaluation approach
1. **Protocol:** model selection, calibration and threshold choice use *validation only*. The test set is evaluated exactly once at the end.
2. **Metrics:** accuracy, macro-F1, calibration (ECE), and for the gated system: *coverage* (share auto-routed) and *precision on auto-routed*.
3. **Baselines:** majority class (floor), word TF-IDF + LogReg / LinearSVC.
4. **Analysis:** per-class report, top confusion pairs, misclassified examples, learning curve, typo-noise robustness, out-of-scope (off-topic) rejection.

## 7. Success criteria (defined *before* training) and results

| Criterion | Target | Result (test) | Met |
|---|---|---|---|
| Accuracy (all queries) | >= 88% | **90.91%** | Yes |
| Macro-F1 | >= 0.88 | **0.909** | Yes |
| Precision on auto-routed | >= 95% | **96.66%** | Yes |
| Coverage (auto-routed share) | >= 80% | **84.48%** | Yes |
| Calibration ECE | <= 0.05 | **0.023** | Yes |
| Latency / query | < 50 ms | **1.45 ms** | Yes |
| Off-topic sent to human | >= 80% | **88%** (22/25) | Yes |

Majority-class floor on validation is **1.9%**, so the model is far above chance.

### Model selection (validation)
| Model | Val acc | Val macro-F1 |
|---|---|---|
| Majority baseline | 0.019 | 0.001 |
| Word TF-IDF + LogReg | 0.881 | 0.883 |
| Word TF-IDF + LinearSVC | 0.883 | 0.884 |
| **Word+char TF-IDF + LinearSVC** (selected) | **0.9075** | **0.907** |

Char n-grams give about +2.4 points and make the model more tolerant to typos.

### Calibration and gating
LinearSVC scores are not probabilities (ECE 0.807 raw). **Temperature scaling** fitted on validation (T = 0.1865) brings ECE to 0.024 on val and 0.023 on test. The confidence threshold **0.82** was chosen on validation as the lowest value giving at least 97% precision. On test: 84.5% auto-routed at 96.7% precision, 15.5% sent to humans.

### Learning curve (val accuracy)
10% data to 100%: accuracy keeps rising up to the full 8,002 examples (25% = 0.808, 50% = 0.870, 75% = 0.894, 100% = 0.9075), so more labelled data would likely help.

### Robustness to typos (test accuracy)
| Noise rate | 0% | 3% | 6% | 10% |
|---|---|---|---|---|
| Accuracy | 0.909 | 0.894 | 0.877 | 0.860 |

### Error analysis
Most errors are between semantically adjacent intents, not random:
`top_up_failed -> top_up_reverted` (5), `fiat_currency_support -> exchange_via_app` (4), `why_verify_identity -> unable_to_verify_identity` (4), `change_pin -> get_physical_card` (4). Weakest intents by F1: `top_up_failed` (0.75), `balance_not_updated_after_bank_transfer` (0.775), `pending_transfer` (0.78). Several "errors" are label-ambiguous (e.g. "My transfer is pending" labelled as another transfer intent). Full lists: `results/per_class_report.csv`, `results/misclassified_test.csv`.

## 8. Limitations and risks (honest)
- Off-topic detection uses only a confidence threshold: **3 of 25** off-topic prompts still got auto-routed. A dedicated out-of-scope class or a larger OOS set would be needed in production.
- The OOS set is small (25 hand-written sentences), so the 88% figure is indicative, not precise.
- Banking77 is single-turn English; real tickets are noisier, multilingual, and may contain several intents.
- Intent taxonomy is fixed at 77; new intents require relabelling and retraining.
- The label ambiguity in the dataset puts a ceiling on achievable accuracy.

## 9. Future work
Fine-tune a small transformer (e.g. DistilBERT) and compare against this baseline; add an explicit out-of-scope class; active learning on low-confidence tickets; monitor drift in production.

## 10. Reproduce
`results/model.joblib` is intentionally not committed (it is in `.gitignore`). On a fresh clone, **run `python src/train_eval.py` first** (about 35 s on CPU, verified on a fresh venv); prediction and tests need the model it generates. Training is seeded (`SEED = 42`), so metrics are reproducible.

```bash
pip install -r requirements.txt
python src/train_eval.py      # STEP 1 (required): trains, calibrates, evaluates, writes results/ incl. model.joblib
python src/predict.py "my card has not arrived yet"   # needs step 1
pytest -q tests               # needs step 1 (5 tests)
```
Example output:
```
python src/predict.py "my card has not arrived yet"
{'decision': 'AUTO_ROUTE', 'intent': 'card_arrival', 'confidence': 0.907, 'top_k': [('card_arrival', 0.907), ('transfer_not_received_by_recipient', 0.057), ('card_delivery_estimate', 0.007)]}
```

## 11. Repo structure
```
data/        train.csv, test.csv (BANKING77), ATTRIBUTION.md, LICENSE-CC-BY-4.0.txt
src/         train_eval.py, predict.py
tests/       test_pipeline.py
results/     metrics.json, per_class_report.csv, misclassified_test.csv, threshold_curve_val.csv
             (model.joblib is generated by train_eval.py and git-ignored)
```

## 12. Attribution and citation
This project uses the **BANKING77** dataset created by PolyAI (Casanueva, Temčinas, Gerz, Henderson, Vulić), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/PolyAI-LDN/task-specific-datasets. No changes were made to the data itself. Full details and BibTeX: [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).

> Casanueva et al. (2020). *Efficient Intent Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020. https://arxiv.org/abs/2003.04807

The licence applies to the dataset. The code and write-up in this repository are the author's own work.

---
Author: **Chandan Singh** | SWYNEX Technologies Internship, Task 1
