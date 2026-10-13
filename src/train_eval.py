"""Train + evaluate the banking-query intent router (BANKING77).
Protocol: test set is touched exactly once, at the very end.
Run: python src/train_eval.py
"""
import json, random, time, joblib
import numpy as np, pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED)
train_full = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
train, val = train_test_split(train_full, test_size=0.2, stratify=train_full.category, random_state=SEED)
classes = sorted(train_full.category.unique())
res = {"n_train": len(train), "n_val": len(val), "n_test": len(test), "n_classes": len(classes)}

def feats():
    return make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
                      TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))
def word_only():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

cands = {
    "majority_baseline": make_pipeline(word_only(), DummyClassifier(strategy="most_frequent")),
    "tfidf_word+LogReg": make_pipeline(word_only(), LogisticRegression(C=20, max_iter=2000)),
    "tfidf_word+LinearSVC": make_pipeline(word_only(), LinearSVC(C=1)),
    "tfidf_word+char+LinearSVC": make_pipeline(feats(), LinearSVC(C=1)),
}
sel = {}
for n, m in cands.items():
    t = time.time(); m.fit(train.text, train.category)
    p = m.predict(val.text)
    sel[n] = {"val_acc": round(accuracy_score(val.category, p), 4),
              "val_macro_f1": round(f1_score(val.category, p, average="macro"), 4),
              "fit_sec": round(time.time() - t, 1)}
    print(n, sel[n])
res["model_selection_val"] = sel
best = max((k for k in sel if k != "majority_baseline"), key=lambda k: sel[k]["val_acc"])
res["selected_model"] = best
model = cands[best]

# ---- calibration: temperature scaling on validation ----
Zv = model.decision_function(val.text); yv = np.array([classes.index(c) for c in val.category])
def nll(T): return -np.log(softmax(Zv / T, axis=1)[np.arange(len(yv)), yv] + 1e-12).mean()
T = minimize_scalar(nll, bounds=(0.01, 5), method="bounded").x
def proba(texts): return softmax(model.decision_function(texts) / T, axis=1)
def ece(P, y, bins=10):
    conf, pred = P.max(1), P.argmax(1); e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        mk = (conf > lo) & (conf <= lo + 1 / bins)
        if mk.any(): e += mk.mean() * abs((pred[mk] == y[mk]).mean() - conf[mk].mean())
    return e
Pv = proba(val.text)
res["calibration"] = {"temperature": round(float(T), 4), "ece_before": round(ece(softmax(Zv, axis=1), yv), 4),
                      "ece_after": round(ece(Pv, yv), 4)}

# ---- threshold: highest coverage with precision >= 97% on VAL ----
conf, pred = Pv.max(1), Pv.argmax(1); ok = pred == yv
thr_curve, thr = [], 1.0
for t in np.round(np.arange(0.05, 1.0, 0.01), 2):
    mk = conf >= t
    if mk.sum() == 0: continue
    thr_curve.append({"threshold": float(t), "coverage": round(mk.mean(), 4), "precision": round(ok[mk].mean(), 4)})
for r in thr_curve:
    if r["precision"] >= 0.97: thr = r["threshold"]; break
res["threshold_chosen_on_val"] = thr
pd.DataFrame(thr_curve).to_csv("results/threshold_curve_val.csv", index=False)

# ---- learning curve (val) ----
lc = []
for f in [0.1, 0.25, 0.5, 0.75, 1.0]:
    sub = train.sample(frac=f, random_state=SEED)
    m = make_pipeline(feats(), LinearSVC(C=1)) if "char" in best else make_pipeline(word_only(), LinearSVC(C=1))
    m.fit(sub.text, sub.category); lc.append({"train_frac": f, "n": len(sub), "val_acc": round(accuracy_score(val.category, m.predict(val.text)), 4)})
res["learning_curve"] = lc

# ---- FINAL TEST (once) ----
yt = np.array([classes.index(c) for c in test.category]); Pt = proba(test.text)
pt = Pt.argmax(1); ct = Pt.max(1); mk = ct >= thr
res["test"] = {"accuracy": round((pt == yt).mean(), 4),
               "macro_f1": round(f1_score(yt, pt, average="macro"), 4),
               "ece": round(ece(Pt, yt), 4),
               "auto_route_coverage": round(mk.mean(), 4),
               "auto_route_precision": round((pt[mk] == yt[mk]).mean(), 4),
               "human_review_rate": round(1 - mk.mean(), 4)}
t0 = time.time(); [model.decision_function([x]) for x in test.text[:300]]
res["test"]["latency_ms_per_query"] = round((time.time() - t0) / 300 * 1000, 2)

rep = classification_report(yt, pt, target_names=classes, output_dict=True, zero_division=0)
pc = pd.DataFrame(rep).T.iloc[:-3].sort_values("f1-score"); pc.to_csv("results/per_class_report.csv")
res["worst_5_intents"] = pc.head(5)[["precision", "recall", "f1-score"]].round(3).to_dict("index")

# confusion pairs + misclassified examples
from collections import Counter
pairs = Counter((classes[a], classes[b]) for a, b in zip(yt, pt) if a != b).most_common(10)
res["top_confusions"] = [{"true": a, "pred": b, "count": c} for (a, b), c in pairs]
err = test.assign(pred=[classes[i] for i in pt], conf=ct.round(3))[pt != yt].sort_values("conf", ascending=False)
err.to_csv("results/misclassified_test.csv", index=False)

# ---- robustness: typo noise ----
def noise(s, rate):
    out = []
    for ch in s:
        r = random.random()
        if ch.isalpha() and r < rate / 3: continue
        if ch.isalpha() and r < 2 * rate / 3: out += [ch, ch]; continue
        out.append(ch)
    return "".join(out)
rob = {}
for rate in [0, 0.03, 0.06, 0.1]:
    nt = [noise(s, rate) for s in test.text]; rob[str(rate)] = round(accuracy_score(test.category, model.predict(nt)), 4)
res["robustness_typo_acc"] = rob

# ---- out-of-scope gate ----
oos = ["What's the weather like in Delhi tomorrow?", "Write me a poem about the ocean", "Who won the cricket match yesterday?",
 "Recommend a good pizza place nearby", "How do I fix a flat bicycle tire?", "Tell me a joke", "What is the capital of Australia?",
 "Translate good morning into French", "How many calories are in a banana?", "Best movies released this year?",
 "Can you help me debug my Python code?", "What time does the sun set today?", "How do I bake chocolate chip cookies?",
 "Explain how photosynthesis works", "Play some relaxing music", "What is the tallest mountain in the world?",
 "How do I learn to play guitar?", "Is it going to rain this weekend?", "Suggest a name for my puppy", "Who is the president of France?",
 "How far is the moon from earth?", "Give me a workout plan for beginners", "What's a good book to read?", "Set an alarm for 6 am",
 "How do airplanes stay in the air?"]
Po = proba(oos); rej = (Po.max(1) < thr).mean()
res["out_of_scope"] = {"n": len(oos), "rejected_to_human_rate": round(float(rej), 4)}

json.dump(res, open("results/metrics.json", "w"), indent=2)
joblib.dump({"model": model, "classes": classes, "T": float(T), "threshold": thr}, "results/model.joblib")
print(json.dumps(res, indent=2))
