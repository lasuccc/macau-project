"""Step 2: majority-class reference and logistic regression (C=1, max_iter=1000)."""
import json
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from baseline_common import BASE, SEED, evaluate, load_scaled, out, save_predictions

tr, va = load_scaled()
maj = int(np.bincount(tr["y"], minlength=6).argmax())
save_predictions("majority", va, np.full(len(va["y"]), maj))
print("majority class (training):", maj, "| validation macro F1:",
      round(evaluate(va["y"], np.full(len(va["y"]), maj), va["users"])[0]["macro_f1_pooled"], 4))

cfg = BASE["logistic"]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"]).fit(tr["X"], tr["y"])
conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
assert clf.classes_.tolist() == list(range(6))
pred_val = clf.predict(va["X"])
save_predictions("logistic", va, pred_val)
info = {"C": cfg["C"], "max_iter": cfg["max_iter"], "n_iter": int(np.max(clf.n_iter_)),
        "converged": not conv, "convergence_warnings": conv,
        "train_macro_f1": evaluate(tr["y"], clf.predict(tr["X"]), tr["users"])[0]["macro_f1_pooled"],
        "val_macro_f1": evaluate(va["y"], pred_val, va["users"])[0]["macro_f1_pooled"]}
joblib.dump(clf, out("logistic_model.joblib"))
out("logistic_info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
if conv:
    print("NOTE: it did not converge. Check the scaling and the features before raising max_iter.")
