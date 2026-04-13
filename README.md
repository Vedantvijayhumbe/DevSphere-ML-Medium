# Digit Doctor -- The Broken CNN (Medium Difficulty)

> *"The previous developer built a CNN for digit recognition. It compiles, it trains... but it is basically guessing randomly. Can you fix it?"*

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)

---

## The Story

A junior developer was tasked with building a **Convolutional Neural Network (CNN)** to classify handwritten digits using the **MNIST dataset**. They wrote the full pipeline -- data loading, preprocessing, model building, training, and evaluation.

The code **runs without any errors**. It compiles fine. It trains.

But the accuracy? Suspiciously low. That is essentially random guessing across 10 digit classes (0-9).

Something is fundamentally wrong with the pipeline.

**Your mission:** Diagnose the issues, fix the bugs, and optimize the model to achieve maximum accuracy.

---

## Objective

1. **Read and understand** the existing code in `notebook.ipynb`
2. **Find and fix** the bugs that make the model perform like a random guesser
3. **Optimize** the model to achieve the highest accuracy possible
4. **Save** your model and submit via Pull Request

### Scoring Checkpoints

Your submission is automatically evaluated when you create a Pull Request:

| Test Case | Threshold | Points | Description |
|---|---|---|---|
| TC-2 | Test Accuracy >= 75% | **+25 pts** | Pipeline is fixed and working |
| TC-3 | Test Accuracy >= 85% | **+20 pts** | Model architecture is solid |
| TC-4 | Test Accuracy >= 90% | **+15 pts** | Well-optimized model |
| TC-5 | Unseen Data >= 80% | **+20 pts** | Model generalizes well |
| TC-6 | No Overfitting | **+10 pts** | Train-test gap <= 10% |

> **Maximum score: 90 points** (cumulative -- earn points for each threshold you clear)

### Overfitting Check

Your model is evaluated on a **hidden, unseen portion** of the MNIST dataset. If the gap between your test accuracy and unseen accuracy exceeds **10%**, you will **not** receive the No Overfitting bonus.

---

## What You Get

```
DevSphere-ML-Medium/
├── notebook.ipynb            <-- THE BROKEN NOTEBOOK (fix this!)
├── requirements.txt          <-- Python dependencies
└── README.md                 <-- You are here
```

> The evaluation is handled automatically -- you do not need to run any test scripts.

---

## Rules

### You MAY:
- Fix bugs in the notebook
- Change the model architecture (layers, filters, activations)
- Tune hyperparameters (learning rate, epochs, batch size, optimizer)
- Modify preprocessing (normalization, reshaping)
- Add regularization (Dropout, BatchNormalization, etc.)

### You may NOT:
- Change the dataset -- must use `keras.datasets.mnist`
- Use pretrained or external models
- Change the model save filename -- must save as `model.h5`
- Hardcode predictions or cheat the evaluation

---

## How to Submit

### Step 1: Fork this repository
Click the **Fork** button on GitHub to create your own copy.

### Step 2: Clone your fork
```bash
git clone https://github.com/YOUR-USERNAME/DevSphere-ML-Medium.git
cd DevSphere-ML-Medium
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Open and fix the notebook
```bash
jupyter notebook notebook.ipynb
```

Read the code carefully -- look for the HINT and BUG comments. Fix the bugs, optimize the model, and make sure it saves as `model.h5`.

Alternatively, open the notebook directly in **Google Colab**.

### Step 5: Commit your changes
```bash
git add .
git commit -m "Fixed and optimized the CNN model"
git push origin main
```

### Step 6: Create a Pull Request
Go to the **original repository** and create a Pull Request from your fork.

### Step 7: Get your score
The evaluation runs **automatically** on your PR. Your score and detailed feedback will appear as a **comment on your Pull Request** within a few minutes.

> **Tip:** Every time you push a new commit to your PR branch, the evaluation re-runs automatically. Iterate until you are satisfied with your score.

---

## Hints

The notebook contains `BUG` and `HINT` comments at suspicious locations. Read them carefully.

<details>
<summary>Stuck? Click for general guidance</summary>

Think about:
1. **Data preprocessing** -- What range should pixel values be in?
2. **Architecture** -- What makes a CNN different from a plain neural network?
3. **Activation functions** -- Can a network learn complex patterns without non-linearity?
4. **Output layer** -- How does a model output probabilities for classification?
5. **Loss function** -- Is the loss function designed for classification?
6. **Optimizer** -- Some optimizers converge much faster than others
7. **Training** -- Is the model training long enough?

</details>

---

## Sample Output (what success looks like)

When your PR is evaluated, you will see a comment like this:

```
Digit Doctor -- Evaluation Report

Test Results (6/7 passed)

| TC-2 | Accuracy >= 75%  | Pass | +25 |
| TC-3 | Accuracy >= 85%  | Pass | +20 |
| TC-4 | Accuracy >= 90%  | Pass | +15 |
| TC-5 | Unseen >= 80%    | Pass | +20 |
| TC-6 | No Overfitting   | Pass | +10 |

Score: 90 / 90
Status: Outstanding -- You are a true Digit Doctor!
```

---

## Leaderboard Criteria

Participants are ranked by:
1. **Total Score** (primary)
2. **Test Accuracy** (tiebreaker)
3. **Submission Time** (earlier is better)

---

*Built for DevSphere by GDG IIIT Lucknow*
