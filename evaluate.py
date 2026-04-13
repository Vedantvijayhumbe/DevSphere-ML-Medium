#!/usr/bin/env python3
"""
Digit Doctor -- Automated Evaluation Script

This script is INTERNAL to the CI/CD pipeline.
Participants do NOT have access to this file.
It runs from the base repository during PR evaluation.

Scoring (cumulative, max 40 pts):
  TC-2: Accuracy >= 75%        -> +11 pts
  TC-3: Accuracy >= 85%        -> +9 pts
  TC-4: Accuracy >= 90%        -> +7 pts
  TC-5: Unseen Data >= 80%     -> +9 pts
  TC-6: No Overfitting         -> +4 pts
"""

import os
import sys
import json
import warnings
import traceback

# Suppress TensorFlow noise for clean CI output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore')

import numpy as np


# -------------------------------------------------------------------
#  CONFIGURATION
# -------------------------------------------------------------------

MODEL_PATH = "model.h5"
NOTEBOOK_PATH = "notebook.ipynb"
RESULTS_JSON = "grading_results.json"
RESULTS_MD = "grading_summary.md"

UNSEEN_SEED = 42
UNSEEN_FRACTION = 0.15  # 15% of test set held out as "unseen"
MAX_SCORE = 40


# -------------------------------------------------------------------
#  TEST RESULT DATA CLASS
# -------------------------------------------------------------------

class TestResult:
    """Stores the outcome of a single test case."""

    def __init__(self, tc_id, name, passed, points_earned, points_possible, detail=""):
        self.tc_id = tc_id
        self.name = name
        self.passed = passed
        self.points_earned = points_earned
        self.points_possible = points_possible
        self.detail = detail

    def to_dict(self):
        return {
            "id": self.tc_id,
            "name": self.name,
            "passed": self.passed,
            "points_earned": self.points_earned,
            "points_possible": self.points_possible,
            "detail": self.detail,
        }


# -------------------------------------------------------------------
#  INTEGRITY CHECKS
# -------------------------------------------------------------------

def check_integrity():
    """TC-0: Verify notebook exists, uses MNIST, no pretrained models."""
    if not os.path.exists(NOTEBOOK_PATH):
        return TestResult("TC-0", "Notebook Integrity", False, 0, 0,
                          "Notebook file not found")

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        content = f.read().lower()

    if 'mnist' not in content:
        return TestResult("TC-0", "Notebook Integrity", False, 0, 0,
                          "Dataset changed -- must use MNIST")

    forbidden = [
        'applications.vgg', 'applications.resnet', 'applications.mobilenet',
        'applications.inception', 'applications.efficientnet',
        'applications.densenet', 'applications.xception',
        'hub.load', 'tfhub', 'torch.hub',
        'pretrained=true',
    ]
    for kw in forbidden:
        if kw in content:
            return TestResult("TC-0", "Notebook Integrity", False, 0, 0,
                              f"Pretrained model usage detected: {kw}")

    return TestResult("TC-0", "Notebook Integrity", True, 0, 0, "All checks passed")


# -------------------------------------------------------------------
#  NOTEBOOK EXECUTION
# -------------------------------------------------------------------

def execute_notebook():
    """Execute the notebook end-to-end to generate model.h5."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})


# -------------------------------------------------------------------
#  MODEL LOADING
# -------------------------------------------------------------------

def load_model():
    """TC-1: Load the participant's saved model."""
    from tensorflow.keras.models import load_model as keras_load_model

    if not os.path.exists(MODEL_PATH):
        return None, TestResult("TC-1", "Model Loads Successfully", False, 0, 0,
                                f"'{MODEL_PATH}' not found -- did your notebook save the model?")
    try:
        model = keras_load_model(MODEL_PATH)
        return model, TestResult("TC-1", "Model Loads Successfully", True, 0, 0,
                                 "Model loaded and architecture valid")
    except Exception as e:
        return None, TestResult("TC-1", "Model Loads Successfully", False, 0, 0,
                                f"Load error: {str(e)[:120]}")


# -------------------------------------------------------------------
#  DATA PREPARATION
# -------------------------------------------------------------------

def prepare_data():
    """
    Load MNIST and split test set into:
      - test_eval: standard evaluation (85%)
      - unseen:    held-out generalization check (15%)
    Both are normalized to [0, 1] and reshaped to (N, 28, 28, 1).
    """
    from tensorflow.keras.datasets import mnist

    (_, _), (X_test, y_test) = mnist.load_data()

    # Deterministic split using fixed seed
    rng = np.random.RandomState(UNSEEN_SEED)
    n_test = len(X_test)
    n_unseen = int(n_test * UNSEEN_FRACTION)
    indices = rng.permutation(n_test)

    unseen_idx = indices[:n_unseen]
    test_idx = indices[n_unseen:]

    X_test_eval = X_test[test_idx].astype('float32') / 255.0
    y_test_eval = y_test[test_idx]

    X_unseen = X_test[unseen_idx].astype('float32') / 255.0
    y_unseen = y_test[unseen_idx]

    X_test_eval = X_test_eval.reshape(-1, 28, 28, 1)
    X_unseen = X_unseen.reshape(-1, 28, 28, 1)

    return X_test_eval, y_test_eval, X_unseen, y_unseen


# -------------------------------------------------------------------
#  MODEL EVALUATION
# -------------------------------------------------------------------

def evaluate_accuracy(model, X, y):
    """Run inference and compute accuracy."""
    predictions = model.predict(X, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    # Support both integer and one-hot encoded labels
    true_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y

    return float(np.mean(pred_labels == true_labels))


# -------------------------------------------------------------------
#  TEST CASE RUNNER
# -------------------------------------------------------------------

def run_accuracy_tests(test_acc, unseen_acc):
    """Run all accuracy-based test cases (cumulative scoring)."""
    results = []

    # TC-2: Accuracy >= 75%
    passed = test_acc >= 0.75
    results.append(TestResult(
        "TC-2", "Accuracy >= 75%  (Pipeline Fixed)", passed,
        11 if passed else 0, 11,
        f"Achieved: {test_acc*100:.2f}%"
    ))

    # TC-3: Accuracy >= 85%
    passed = test_acc >= 0.85
    results.append(TestResult(
        "TC-3", "Accuracy >= 85%  (Good Model)", passed,
        9 if passed else 0, 9,
        f"Achieved: {test_acc*100:.2f}%"
    ))

    # TC-4: Accuracy >= 90%
    passed = test_acc >= 0.90
    results.append(TestResult(
        "TC-4", "Accuracy >= 90%  (Strong Model)", passed,
        7 if passed else 0, 7,
        f"Achieved: {test_acc*100:.2f}%"
    ))

    # TC-5: Unseen data >= 80%
    passed = unseen_acc >= 0.80
    results.append(TestResult(
        "TC-5", "Unseen Data >= 80% (Generalization)", passed,
        9 if passed else 0, 9,
        f"Achieved: {unseen_acc*100:.2f}%"
    ))

    # TC-6: No overfitting (gap <= 10%)
    gap = test_acc - unseen_acc
    passed = gap <= 0.10
    results.append(TestResult(
        "TC-6", "No Overfitting  (Gap <= 10%)", passed,
        4 if passed else 0, 4,
        f"Gap: {gap*100:.2f}% ({'OK' if passed else 'OVERFITTING DETECTED'})"
    ))

    return results


# -------------------------------------------------------------------
#  OUTPUT FORMATTING -- CONSOLE
# -------------------------------------------------------------------

def format_console(all_results, test_acc, unseen_acc, total_score):
    """Print a clean, tabular report to the console (visible in CI logs)."""

    print()
    print("================================================================")
    print("             DIGIT DOCTOR -- EVALUATION REPORT                  ")
    print("================================================================")
    print()

    # Table header
    print("+------+----------------------------------------+--------+---------+")
    print("|  ID  | Test Case                              | Status | Points  |")
    print("+------+----------------------------------------+--------+---------+")

    for r in all_results:
        status = "  PASS" if r.passed else "  FAIL"
        if r.points_possible == 0:
            pts_str = "  --"
        elif r.points_earned > 0:
            pts_str = f" +{r.points_earned}"
        else:
            pts_str = "  +0"

        print(f"| {r.tc_id} | {r.name:<38} | {status} | {pts_str:>7} |")

    print("+------+----------------------------------------+--------+---------+")
    print()

    # Metrics
    gap = test_acc - unseen_acc
    print(f"  Test Accuracy:    {test_acc*100:.2f}%")
    print(f"  Unseen Accuracy:  {unseen_acc*100:.2f}%")
    print(f"  Overfit Gap:      {gap*100:.2f}%")
    print()

    passed_count = sum(1 for r in all_results if r.passed)
    total_count = len(all_results)
    print(f"  Tests Passed: {passed_count}/{total_count}")
    print()

    # Final score
    print("================================================================")
    print(f"  FINAL SCORE: {total_score} / {MAX_SCORE}")
    print("================================================================")

    if total_score >= 36:
        print("  Status: OUTSTANDING -- You are a true Digit Doctor!")
    elif total_score >= 27:
        print("  Status: PASS -- Great work! Strong model.")
    elif total_score >= 11:
        print("  Status: PASS -- Good start, room to improve!")
    else:
        print("  Status: NEEDS IMPROVEMENT -- Keep debugging!")
    print()


# -------------------------------------------------------------------
#  OUTPUT FORMATTING -- MARKDOWN (for PR comment)
# -------------------------------------------------------------------

def generate_markdown(all_results, test_acc, unseen_acc, total_score):
    """Generate a markdown summary to be posted as a PR comment."""

    passed_count = sum(1 for r in all_results if r.passed)
    total_count = len(all_results)
    gap = test_acc - unseen_acc

    if total_score >= 36:
        status = "Outstanding"
    elif total_score >= 27:
        status = "Pass -- Great Work"
    elif total_score >= 11:
        status = "Pass -- Good Start"
    else:
        status = "Needs Improvement"

    md = f"""## Digit Doctor -- Evaluation Report

### Test Results ({passed_count}/{total_count} passed)

| # | Test Case | Status | Points |
|---|-----------|--------|--------|
"""
    for r in all_results:
        s = "Pass" if r.passed else "Fail"
        if r.points_possible == 0:
            pts = "--"
        elif r.points_earned > 0:
            pts = f"+{r.points_earned}"
        else:
            pts = "+0"
        md += f"| {r.tc_id} | {r.name} | {s} | {pts} |\n"

    md += f"""
### Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | **{test_acc*100:.2f}%** |
| Unseen Accuracy | **{unseen_acc*100:.2f}%** |
| Overfit Gap | {gap*100:.2f}% |
| Tests Passed | {passed_count}/{total_count} |

### Score: **{total_score} / {MAX_SCORE}**

**Status: {status}**

"""
    if total_score >= MAX_SCORE:
        md += "---\n*Perfect score! You have mastered the Digit Doctor challenge!*\n"
    elif total_score >= 11:
        md += "---\n*Keep optimizing to climb the leaderboard!*\n"
    else:
        md += "---\n*The model needs more work. Check the hints in the notebook!*\n"

    return md


# -------------------------------------------------------------------
#  SAVE RESULTS
# -------------------------------------------------------------------

def save_results(all_results, test_acc, unseen_acc, total_score, md_summary):
    """Persist results as JSON and Markdown files."""
    results = {
        "test_accuracy": round(test_acc, 4),
        "unseen_accuracy": round(unseen_acc, 4),
        "total_score": total_score,
        "max_score": MAX_SCORE,
        "status": "pass" if total_score >= 11 else "fail",
        "test_cases": [r.to_dict() for r in all_results],
    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_MD, 'w') as f:
        f.write(md_summary)

    print(f"  Results saved to {RESULTS_JSON}")
    print(f"  Summary saved to {RESULTS_MD}")


# -------------------------------------------------------------------
#  EARLY EXIT HELPER
# -------------------------------------------------------------------

def abort(all_results, message):
    """Handle early termination with proper output."""
    print(f"\n  ERROR: {message}")
    format_console(all_results, 0.0, 0.0, 0)
    md = generate_markdown(all_results, 0.0, 0.0, 0)
    save_results(all_results, 0.0, 0.0, 0, md)
    sys.exit(1)


# -------------------------------------------------------------------
#  MAIN PIPELINE
# -------------------------------------------------------------------

def main():
    all_results = []
    test_acc = 0.0
    unseen_acc = 0.0

    print("\n" + "=" * 63)
    print("  DIGIT DOCTOR -- AUTOMATED EVALUATION PIPELINE")
    print("=" * 63)

    # -- Step 1: Integrity Check -----------------------------------
    print("\nStep 1/5: Integrity Check")
    integrity = check_integrity()
    all_results.append(integrity)
    if not integrity.passed:
        abort(all_results, f"Integrity check failed: {integrity.detail}")
    print(f"  {integrity.detail}")

    # -- Step 2: Execute Notebook ----------------------------------
    print("\nStep 2/5: Execute Notebook")
    print("  Running notebook end-to-end (this may take a few minutes)...")
    try:
        execute_notebook()
        print("  Notebook executed successfully")
    except Exception as e:
        err_msg = str(e)[:200]
        print(f"  Notebook execution failed:\n     {err_msg}")
        all_results.append(TestResult("TC-1", "Model Loads Successfully", False, 0, 0,
                                      "Notebook execution error"))
        abort(all_results, "Notebook must run without errors. Fix your code and try again.")

    # -- Step 3: Load Model ----------------------------------------
    print("\nStep 3/5: Load Model")
    model, load_result = load_model()
    all_results.append(load_result)
    if model is None:
        abort(all_results, f"Model load failed: {load_result.detail}")
    print(f"  {load_result.detail}")

    # -- Step 4: Evaluate ------------------------------------------
    print("\nStep 4/5: Evaluate Model")
    try:
        X_test, y_test, X_unseen, y_unseen = prepare_data()
        print(f"  Test samples:   {len(X_test)}")
        print(f"  Unseen samples: {len(X_unseen)}")

        test_acc = evaluate_accuracy(model, X_test, y_test)
        unseen_acc = evaluate_accuracy(model, X_unseen, y_unseen)

        print(f"  Test Accuracy:   {test_acc*100:.2f}%")
        print(f"  Unseen Accuracy: {unseen_acc*100:.2f}%")
    except Exception as e:
        print(f"  Evaluation error: {str(e)[:200]}")
        abort(all_results, "Model evaluation failed. Check your model architecture.")

    # Run test cases
    accuracy_results = run_accuracy_tests(test_acc, unseen_acc)
    all_results.extend(accuracy_results)

    # -- Step 5: Score and Report ----------------------------------
    print("\nStep 5/5: Calculate Score")
    total_score = sum(r.points_earned for r in all_results)
    total_score = max(0, min(total_score, MAX_SCORE))

    # Generate outputs
    format_console(all_results, test_acc, unseen_acc, total_score)
    md = generate_markdown(all_results, test_acc, unseen_acc, total_score)
    save_results(all_results, test_acc, unseen_acc, total_score, md)

    # Exit code: 0 if minimum threshold met, 1 otherwise
    sys.exit(0 if total_score >= 11 else 1)


if __name__ == "__main__":
    main()
