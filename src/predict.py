from transformers import pipeline

_clf = None


def _get_model():
    global _clf
    if _clf is None:
        _clf = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,          # return scores for all 3 classes
        )
    return _clf


def predict(text: str):
    """Returns (label, confidence, all_scores) for the given text."""
    results = _get_model()(text[:512])
    # Depending on the version, results may be nested one level deeper
    scores = results[0] if isinstance(results[0], list) else results
    scores = sorted(scores, key=lambda s: s["score"], reverse=True)

    top = scores[0]
    label = top["label"].capitalize()            # Negative / Neutral / Positive
    all_scores = {s["label"].capitalize(): float(s["score"]) for s in scores}
    return label, float(top["score"]), all_scores