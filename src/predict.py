"""Inference with confidence-gated routing.
Usage: python src/predict.py "my card has not arrived yet"
"""
import sys, joblib
from scipy.special import softmax

def load(path="results/model.joblib"):
    return joblib.load(path)

def route(text, bundle=None, top_k=3):
    b = bundle or load()
    p = softmax(b["model"].decision_function([text]) / b["T"], axis=1)[0]
    idx = p.argsort()[::-1][:top_k]
    top = [(b["classes"][i], float(p[i])) for i in idx]
    decision = "AUTO_ROUTE" if top[0][1] >= b["threshold"] else "HUMAN_REVIEW"
    return {"decision": decision, "intent": top[0][0], "confidence": round(top[0][1], 3),
            "top_k": [(c, round(s, 3)) for c, s in top]}

if __name__ == "__main__":
    print(route(" ".join(sys.argv[1:]) or "my card has not arrived yet"))
