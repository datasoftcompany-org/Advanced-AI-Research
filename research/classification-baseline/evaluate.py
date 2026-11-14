"""Evaluate a keyword classifier against a majority-class baseline."""

from collections import Counter
import re


TRAIN_LABELS = ("positive", "positive", "positive", "negative", "negative")
EVALUATION = (
    ("A good start", "positive"),
    ("Excellent result", "positive"),
    ("A bad outcome", "negative"),
    ("Terrible experience", "negative"),
    ("Not good", "negative"),
    ("Acceptable overall", "positive"),
)
KEYWORDS = {
    "positive": {"good", "excellent"},
    "negative": {"bad", "terrible"},
}


def classify(text, default):
    words = set(re.findall(r"[a-z]+", text.lower()))
    scores = {label: len(words & keywords) for label, keywords in KEYWORDS.items()}
    if scores["positive"] == scores["negative"]:
        return default
    return max(scores, key=scores.__getitem__)


def main():
    majority = Counter(TRAIN_LABELS).most_common(1)[0][0]
    baseline_correct = 0
    keyword_correct = 0
    print("Expected | Predicted | Example")
    for text, label in EVALUATION:
        prediction = classify(text, majority)
        baseline_correct += majority == label
        keyword_correct += prediction == label
        print(f"{label} | {prediction} | {text}")

    total = len(EVALUATION)
    for name, correct in (("Majority", baseline_correct), ("Keywords", keyword_correct)):
        print(f"{name} accuracy: {correct}/{total} ({correct / total:.1%})")


if __name__ == "__main__":
    main()
