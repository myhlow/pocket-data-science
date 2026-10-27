# Part X: Mobile GBDT Machine Learning for LMSYS Preference

**Track 6 &middot; Module 11** &nbsp;|&nbsp; **Kaggle Team:** Pocket Data Lab &nbsp;|&nbsp; **Live Article:** [Pocket Data Science X: Mobile GBDT for LLM Preferences](https://malcolmlow.com/2026/10/26/pocket-data-science-10-kaggle-llm-classification-gbdt-android-termux/)

This module advances from surface-level heuristics (Part IX) to full supervised machine learning on mobile ARM64 hardware (Android Termux PRoot). We engineer **58 vectorized linguistic, structural, readability, and prompt-overlap features** across all 57,477 pairwise battles and train calibrated linear and gradient-boosted tree architectures.

---

## 🏆 Full 4-Experiment Kaggle Benchmark Scorecard

| Experiment & Architecture | Model Hyperparameters | 5-Fold Local CV Loss | Kaggle Public LB Score | Submission Ref |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Calibrated Logistic Regression** | $L_2$ regularized multinomial logistic regression (`C=0.1`) | `1.05613` | **`1.06173`** | `57667634` |
| **Exp 2: Standalone LightGBM GBDT** | `max_depth=6`, `num_leaves=31`, `lr=0.05`, 160 estimators | `1.03601` | **`1.04081`** | `57667635` |
| **Exp 3: Standalone CatBoost Classifier** | Oblivious symmetric trees, `depth=5`, `lr=0.06`, 300 iterations | `1.03594` | **`1.04455`** | `57667637` |
| **Exp 4: Dual GBDT Blended Ensemble** | SLSQP-optimized convex blend (50.6% LightGBM + 49.4% CatBoost) | **`1.03493`** | **`1.04158`** | `57657442` |

*Official Leaderboard Standing: **Rank #170 out of 290 teams** (+52 positions gained over Part IX).*

---

## 🚀 Standalone Kaggle Notebook Kernels
Each experiment is available as an independent, self-contained Kaggle Code competition notebook:
* **Exp 1:** [`myhlow/lmsys-part-x-exp1-logistic-regression`](https://www.kaggle.com/code/myhlow/lmsys-part-x-exp1-logistic-regression)
* **Exp 2:** [`myhlow/lmsys-part-x-exp2-lightgbm`](https://www.kaggle.com/code/myhlow/lmsys-part-x-exp2-lightgbm)
* **Exp 3:** [`myhlow/lmsys-part-x-exp3-catboost`](https://www.kaggle.com/code/myhlow/lmsys-part-x-exp3-catboost)
* **Exp 4 (Dual Blend):** [`myhlow/llm-preference-mobile-gbdt-pocket-data-lab`](https://www.kaggle.com/code/myhlow/llm-preference-mobile-gbdt-pocket-data-lab)

---

## 📂 Source Code Structure
* `main.py`: Full stand-alone Python pipeline executing feature extraction, model fitting, and test inference.
* `main.ipynb`: Self-contained Jupyter notebook formatted for Kaggle competition runner.
* `run_ml_experiments.py`: Local 5-fold cross-validation benchmarking harness with SLSQP blend optimizer.
* `test_feature_extraction.py`: Unit test verifying matrix generation runtime (14.36s for 57,477 rows).
* `kernel-metadata.json`: Kaggle API configuration metadata.

---

## 📊 Complete 58-Dimensional Feature Extraction Dictionary

All 58 dense features are engineered via a single-pass vectorized function `extract_features(df)` executing in **14.36s** across 57,477 rows on mobile CPU (Termux ARM64):

### Mathematical Notation & Operators Legend
* `p`, `r_A`, `r_B`: Raw text strings for prompt, Response A, and Response B.
* `|s|`: Character length of text string `s` (i.e. `len(s)`).
* `words(s)`: Total whitespace-tokenized word count of string `s`.
* `W_p`, `W_a`, `W_b`: Sets of unique lowercased vocabulary tokens: `W_p = set(p.lower().split())`, `W_a = set(r_A.lower().split())`, `W_b = set(r_B.lower().split())`.
* `|W|`: Cardinality (number of unique tokens) of word set `W`.
* `∩`, `∪`: Set intersection (shared overlap) and set union (combined vocabulary).
* `1[·]`: Iverson bracket indicator function (returns `1.0` if true, `0.0` otherwise).
* `count(s, sub)`: Frequency of non-overlapping substring occurrences.
* `count_regex(s, pat)` / `match(s, pat)`: Regular expression frequency and boolean matching.

### 1. Surface Length & Proportions (8 Features)

| Feature Variable | Extraction / Math Formula | Behavioral & Linguistic Signal |
| :--- | :--- | :--- |
| `len_p` | `|p|` | Character length of user prompt text string `p`. |
| `len_a` | `|r_A|` | Character length of Response A text string `r_A`. |
| `len_b` | `|r_B|` | Character length of Response B text string `r_B`. |
| `len_diff` | `len_a - len_b` | Absolute character length disparity between Response A and Response B. |
| `len_ratio` | `(len_a - len_b) / (len_a + len_b + 1)` | Normalized length differential bounded in (-1.0, +1.0); denominator `+1` avoids division by zero. |
| `len_ratio_p_a` | `len_a / (len_p + 1)` | Expansion ratio of Response A relative to prompt length. |
| `len_ratio_p_b` | `len_b / (len_p + 1)` | Expansion ratio of Response B relative to prompt length. |
| `len_ratio_p_diff` | `len_ratio_p,a - len_ratio_p,b` | Differential prompt expansion ratio between responses. |

### 2. Lexical Density & Readability (8 Features)

| Feature Variable | Extraction / Math Formula | Behavioral & Linguistic Signal |
| :--- | :--- | :--- |
| `words_p` | `words(p)` | Total whitespace-split word token count of prompt `p`. |
| `words_a` | `words(r_A)` | Total whitespace-split word token count of Response A `r_A`. |
| `words_b` | `words(r_B)` | Total whitespace-split word token count of Response B `r_B`. |
| `words_diff` | `words_a - words_b` | Absolute difference in word counts between Response A and Response B. |
| `words_ratio` | `(words_a - words_b) / (words_a + words_b + 1)` | Normalized word count disparity ratio bounded in (-1.0, +1.0). |
| `mean_word_len_a` | `len_a / (words_a + 1)` | Mean characters per word in Response A (information density proxy; `+1` avoids zero-division). |
| `mean_word_len_b` | `len_b / (words_b + 1)` | Mean characters per word in Response B (information density proxy; `+1` avoids zero-division). |
| `mean_word_len_diff` | `mean_word_len_a - mean_word_len_b` | Disparity in mean word length (vocabulary maturity and morphological complexity differential). |

### 3. Structural & Markdown Formatting (24 Features)

| Feature Variable | Extraction / Math Formula | Behavioral & Linguistic Signal |
| :--- | :--- | :--- |
| `has_code_a` | `1['```' ∈ r_A]` | Binary indicator (0 or 1): Response A contains markdown code block fences `````. |
| `has_code_b` | `1['```' ∈ r_B]` | Binary indicator (0 or 1): Response B contains markdown code block fences `````. |
| `code_diff` | `has_code_a - has_code_b` | Differential code block presence between Response A and Response B. |
| `count_code_a` | `count(r_A, '```')` | Total markdown code fence delimiter occurrences in Response A. |
| `count_code_b` | `count(r_B, '```')` | Total markdown code fence delimiter occurrences in Response B. |
| `count_code_diff` | `count_code_a - count_code_b` | Differential code block fence delimiter count. |
| `list_a` | `count_regex(r_A, ^\s*[-*•]\s)` | Total unordered bullet list item lines in Response A. |
| `list_b` | `count_regex(r_B, ^\s*[-*•]\s)` | Total unordered bullet list item lines in Response B. |
| `list_diff` | `list_a - list_b` | Differential unordered bullet list item count. |
| `has_list_a` | `1[list_a > 0]` | Binary indicator (0 or 1): Response A incorporates unordered bullet lists. |
| `has_list_b` | `1[list_b > 0]` | Binary indicator (0 or 1): Response B incorporates unordered bullet lists. |
| `has_list_diff` | `has_list_a - has_list_b` | Differential bullet list usage flag. |
| `num_list_a` | `count_regex(r_A, ^\s*\d+\.\s)` | Total numbered/ordered list item lines in Response A. |
| `num_list_b` | `count_regex(r_B, ^\s*\d+\.\s)` | Total numbered/ordered list item lines in Response B. |
| `num_list_diff` | `num_list_a - num_list_b` | Differential numbered list item count. |
| `header_a` | `count_regex(r_A, ^#{1,4}\s)` | Markdown section header lines (`#` to `####`) in Response A. |
| `header_b` | `count_regex(r_B, ^#{1,4}\s)` | Markdown section header lines (`#` to `####`) in Response B. |
| `header_diff` | `header_a - header_b` | Differential markdown section header count between responses. |
| `bold_a` | `count_regex(r_A, \*\*[^*]+\*\*)` | Bold markdown emphasis spans (`**...**`) in Response A. |
| `bold_b` | `count_regex(r_B, \*\*[^*]+\*\*)` | Bold markdown emphasis spans (`**...**`) in Response B. |
| `bold_diff` | `bold_a - bold_b` | Differential bold emphasis span frequency. |
| `newline_a` | `count(r_A, '\n')` | Total newline count in Response A (proxy for vertical paragraph structure). |
| `newline_b` | `count(r_B, '\n')` | Total newline count in Response B (proxy for vertical paragraph structure). |
| `newline_diff` | `newline_a - newline_b` | Differential newline count / vertical spacing. |

### 4. Stylistic & Politeness Markers (12 Features)

| Feature Variable | Extraction / Math Formula | Behavioral & Linguistic Signal |
| :--- | :--- | :--- |
| `excl_a` | `count(r_A, '!')` | Total exclamation marks in Response A (enthusiasm/intensity signal). |
| `excl_b` | `count(r_B, '!')` | Total exclamation marks in Response B (enthusiasm/intensity signal). |
| `excl_diff` | `excl_a - excl_b` | Differential exclamation frequency / emotional intensity. |
| `quest_a` | `count(r_A, '?')` | Total question marks in Response A (inquiry/follow-up prompts). |
| `quest_b` | `count(r_B, '?')` | Total question marks in Response B (inquiry/follow-up prompts). |
| `quest_diff` | `quest_a - quest_b` | Differential question mark frequency. |
| `sorry_a` | `1[match(r_A, sorry|apologize)]` | Binary flag (0 or 1): Response A exhibits apologetic hedging expressions. |
| `sorry_b` | `1[match(r_B, sorry|apologize)]` | Binary flag (0 or 1): Response B exhibits apologetic hedging expressions. |
| `sorry_diff` | `sorry_a - sorry_b` | Differential apologetic tone flag between responses. |
| `sure_a` | `1[match(r_A, sure|certainly|gladly)]` | Binary flag (0 or 1): Response A uses affirmative conversational opening. |
| `sure_b` | `1[match(r_B, sure|certainly|gladly)]` | Binary flag (0 or 1): Response B uses affirmative conversational opening. |
| `sure_diff` | `sure_a - sure_b` | Differential affirmative opening flag. |

### 5. Prompt-Response Semantic Alignment (6 Features)

| Feature Variable | Extraction / Math Formula | Behavioral & Linguistic Signal |
| :--- | :--- | :--- |
| `jaccard_a` | `|W_p ∩ W_a| / |W_p ∪ W_a|` | Lexical Jaccard similarity between unique prompt words `W_p = set(p.lower().split())` and Response A words `W_a = set(r_A.lower().split())`. |
| `jaccard_b` | `|W_p ∩ W_b| / |W_p ∪ W_b|` | Lexical Jaccard similarity between unique prompt words `W_p` and Response B words `W_b = set(r_B.lower().split())`. |
| `jaccard_diff` | `jaccard_a - jaccard_b` | Differential lexical alignment toward prompt vocabulary: positive values indicate Response A shares a higher proportion of prompt query terms than Response B. |
| `cov_a` | `|W_p ∩ W_a| / |W_p|` | Prompt keyword recall/coverage: fraction of unique prompt words in `W_p` addressed by Response A `W_a`. |
| `cov_b` | `|W_p ∩ W_b| / |W_p|` | Prompt keyword recall/coverage: fraction of unique prompt words in `W_p` addressed by Response B `W_b`. |
| `cov_diff` | `cov_a - cov_b` | Differential prompt keyword coverage ratio: positive values indicate Response A directly covered more user prompt terms. |
