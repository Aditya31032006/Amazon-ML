# 🏢 Multilingual Business Entity Resolution Pipeline
> **ML Challenge 2026** | Macro $F_{0.5}$-Optimized Scalable Entity Resolution System

---

## 📌 Problem Overview
In large-scale commercial platforms, business identity data arrives from multiple independent sources with noisy, partial, and language-varying fragments. The challenge is to match records from **Source 2** and **Source 3** to a deduplicated reference **Source 1**:
* **Source 1**: Deduplicated reference entity list.
* **Source 2 & Source 3**: Non-unique, noisy records with naming variations, typos, transliterated Indic scripts, French names, and missing components.
* **Singletons**: Source 1 entities that have **zero matches** in Source 2 or 3. They must be assigned an empty match list (`""`).
* **Evaluation Metric**: **Macro $F_{0.5}$** (Precision-heavy: false merges penalize twice as much as missed links).

---

## 🏗️ System Architecture

Comparing every record directly would require over **8.8 Trillion** pairwise comparisons. This system implements a high-throughput, precision-focused **2-stage architecture**:

```
[Raw TSV Datasets: S1, S2, S3]
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Text Cleaning, Normalization & Transliteration      │
│  • ftfy mojibake repair (e.g., CafÃ© -> cafe)               │
│  • unidecode Indic script transliteration (Tamil/Hindi)     │
│  • Unicode NFKD normalization, accent removal, lowercasing  │
│  • Legal suffix canonicalization (cleaned_name vs core_name)│
│  • Country-aware address standardization (TN in US vs India)│
│  • Missingness binary indicator flags                       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Multi-Key Candidate Generation / Blocking          │
│  • Hard Partitioning by Country (US, INDIA, FRANCE)         │
│  • Union of 5 Complementary Blocking Keys:                  │
│    - Key 1: Core Name Prefix (3-4 characters)               │
│    - Key 2: Soundex Phonetic Code (Robotics == Robotix)     │
│    - Key 3: Address Anchor (Street # + Street Name)         │
│    - Key 4: Distinctive Name Tokens (Inverted Index)        │
│    - Key 5: Name Bigram / First-Two-Words Key               │
│  • Target: >= 95% Ground-Truth Recall Ceiling               │
│  • Outputs audited file: output/candidate_pairs.tsv         │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Pairwise Feature Engineering                       │
│  • Fuzzy string metrics (Levenshtein, Jaro-Winkler via C++) │
│  • Character 3-gram Jaccard & TF-IDF Cosine Similarities    │
│  • Street number match / mismatch indicators                │
│  • Candidate Rank & Gap Features (resolves look-alikes)     │
│  • Explicit NaN missingness preservation for tree models    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Matching Classifier (LightGBM / XGBoost)           │
│  • Trained on Ground Truth positives + Hard Negatives       │
│  • Leak-free GroupKFold cross-validation by S1 entity       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Threshold Optimization & Singleton Handling        │
│  • Sweep probability cutoff to maximize Macro F_0.5         │
│  • High-confidence cutoff defaults singletons to empty list │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: Validation & Submission Packaging                  │
│  • Validated using official utils/validate_submission.py    │
│  • Generates output/matching_results.tsv                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # Centralized path registry & hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                # TSV reader & high-speed file storage
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py          # Core NLP, regex, transliteration & suffix logic
│   │   └── normalizer.py            # Batch DataFrame normalizer service
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── keys.py                  # Soundex, Address Anchors, and Prefix extractors
│   │   ├── indexer.py               # Inverted Index hash buckets & block pruning
│   │   └── candidate_generator.py   # Multi-source candidate coordinator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── blocking_evaluator.py    # Ground truth recall evaluation service
│   └── pipelines/
│       ├── __init__.py
│       ├── run_phase1_cleaning.py   # Phase 1: Data Cleaning runner
│       └── run_phase2_blocking.py   # Phase 2: Candidate Blocking runner
├── tests/
│   ├── test_phase1_cleaning.py      # Phase 1 unit test suite
│   └── test_phase2_blocking.py      # Phase 2 unit test suite
├── output/                          # Target directory for competition submissions
│   ├── candidate_pairs.tsv          # Audited candidate pairs
│   └── matching_results.tsv         # Final predicted entity matches
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Ignores large datasets, virtualenvs, cache
└── README.md
```

---

## 🚀 Step-by-Step Execution Commands

### 1. Environment Setup
Activate your Python virtual environment and ensure dependencies are installed:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 2. Phase 1: Text Cleaning & Normalization
Cleans raw text, repairs mojibake, transliterates non-Latin scripts, strips corporate noise, canonicalizes addresses, and extracts `core_name`.

* **Quick Sanity Check (5,000 records per file)**:
  ```powershell
  python src/pipelines/run_phase1_cleaning.py --sample 5000
  ```

* **Full Dataset Run (All 6 Train & Test Files)**:
  ```powershell
  python src/pipelines/run_phase1_cleaning.py
  ```
  *Output saved to:* `data/processed/*_cleaned.tsv`

* **Run Phase 1 Unit Tests**:
  ```powershell
  python tests/test_phase1_cleaning.py
  ```

---

### 3. Phase 2: Multi-Key Candidate Generation & Blocking
Partitions records by country and indexes them using Soundex, name prefixes, address anchors, and rare tokens.

* **Run Phase 2 Unit Tests**:
  ```powershell
  python tests/test_phase2_blocking.py
  ```

* **Evaluate Blocking Recall on Training Data (Sample Mode)**:
  ```powershell
  python src/pipelines/run_phase2_blocking.py --mode train --sample 5000
  ```

* **Full Training Evaluation (Verifies Ground-Truth Recall $\ge 95\%$)**:
  ```powershell
  python src/pipelines/run_phase2_blocking.py --mode train
  ```

* **Generate Test Candidate Pairs (`output/candidate_pairs.tsv`)**:
  ```powershell
  python src/pipelines/run_phase2_blocking.py --mode test
  ```

---

### 4. Official Submission Verification
Before uploading or packaging, validate the output format using the official validator:
```powershell
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

---

## 🎯 Evaluation Metric Formula

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

* **Precision is weighted 2× over Recall**: False merges severely degrade the score.
* **Singletons**: Correctly identifying an entity with no matches earns a score of **1.0**. Incorrectly predicting any match for a singleton earns a **0.0**.
