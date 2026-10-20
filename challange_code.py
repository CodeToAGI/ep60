"""EP60 - Monitoring: deploy a digit classifier, send it two kinds of drifting traffic, and measure what a monitor sees and what it misses.
Run:  python ep60_monitor.py     # ~3-6 min; writes monitor/ + ep60_results.json
pip install torch scikit-learn numpy pandas scipy matplotlib "evidently>=0.7"
# a missing tool is skipped and reported, not faked
"""
import importlib.util, json, warnings
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.datasets import load_digits
from torch import nn
warnings.filterwarnings("ignore")

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEED = 0
EPOCHS = 40
HIDDEN = 64
WEEKS = 8
BATCH = 250
NOISE = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # camera noise per week
SWAP = (3, 8)                                        # concept drift classes
SWAP_FROM = 4
ALPHA = 0.05
DRIFT_SHARE = 0.3                                    # alert threshold
FLOOR = 0.9                                          # retrain trigger
CLEAN = 20
MON = Path(__file__).with_name("monitor")

# ── 2. DATA + MODEL ───────────────────────────────────────────────────────
def load():
    x, y = load_digits(return_X_y=True)
    p = np.random.RandomState(0).permutation(len(x))
    x, y = x[p] / 16.0, y[p]
    return x[:700], y[:700], x[700:1000], y[700:1000], x[1000:1300], y[1000:1300], x[1300:], y[1300:]

def train(x, y):
    torch.manual_seed(SEED); torch.set_num_threads(1)
    net = nn.Sequential(nn.Linear(64, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x))
        for i in range(0, len(x), 64):
            b = order[i:i + 64]
            loss = nn.functional.cross_entropy(net(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()

def predict(net, x):
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(x, dtype=torch.float32)), 1).numpy()
    return p.argmax(1), p.max(1)

def corrupt(x, sigma, rng):
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1) if sigma else x

def live_batch(xm, ym, week, scenario, rng):
    i = rng.choice(len(xm), BATCH, replace=False)
    x, y = xm[i], ym[i].copy()
    if scenario == "sensor":
        x = corrupt(x, NOISE[week - 1], rng)
    elif week >= SWAP_FROM:
        a, b = SWAP
        ma, mb = y == a, y == b
        y[ma], y[mb] = b, a
    return x, y

# ── 3. MONITOR ────────────────────────────────────────────────────────────
def ks_share(ref, cur):
    return float(np.mean([ks_2samp(ref[:, j], cur[:, j]).pvalue < ALPHA for j in range(ref.shape[1])]))

EV_ERR = []
def ev_share(ref, cur, save=None):
    if EV_ERR or importlib.util.find_spec("evidently") is None:
        return None
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
        cols = [f"p{j}" for j in range(ref.shape[1])]
        dd = DataDefinition(numerical_columns=cols)
        ds = lambda a: Dataset.from_pandas(pd.DataFrame(a, columns=cols), data_definition=dd)
        snap = Report([DataDriftPreset()]).run(ds(cur), ds(ref))
        if save:
            snap.save_html(str(save))
        return float(next(m["value"]["share"] for m in snap.dict()["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")))
    except Exception as e:
        EV_ERR.append(f"{type(e).__name__}: {str(e)[:120]}")
        return None

def week_row(net, ref, x, y, week, save=None):
    pred, conf = predict(net, x)
    return dict(week=week, acc=float((pred == y).mean()), conf=float(conf.mean()),
                ks=ks_share(ref, x), ev=ev_share(ref, x, save))

def first(rows, f):
    return next((r["week"] for r in rows if f(r)), None)

# ── 4. RETRAIN ────────────────────────────────────────────────────────────
def retrain(net, tr, rt, xm, ym, week):
    sigma = NOISE[week - 1]
    rng = np.random.default_rng(SEED + 1)
    new = train(np.vstack([tr[0], corrupt(rt[0], sigma, rng)]), np.concatenate([tr[1], rt[1]]))
    tests = {"clean": xm, "at trigger": corrupt(xm, sigma, rng), "latest week": corrupt(xm, NOISE[-1], rng)}
    acc = lambda m: {k: float((predict(m, v)[0] == ym).mean()) for k, v in tests.items()}
    return dict(week=week, sigma=sigma, old=acc(net), new=acc(new))

# ── 5. DASHBOARD ──────────────────────────────────────────────────────────
def dashboard(R, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#050815", "axes.facecolor": "#0d1117",
                         "axes.edgecolor": "#6272a4", "text.color": "#f8f8f2",
                         "axes.labelcolor": "#f8f8f2", "xtick.color": "#f8f8f2",
                         "ytick.color": "#f8f8f2", "axes.grid": True, "grid.alpha": 0.15})
    S, C, wk = R["sensor"], R["concept"], list(range(1, WEEKS + 1))
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.4))
    def line(a, rows, key, label, color, **kw):
        a.plot(wk, [r[key] for r in rows], marker="o", color=color, label=label, **kw)
    a = ax[0, 0]
    line(a, S, "acc", "sensor noise", "#8be9fd"); line(a, C, "acc", "labels change", "#ffb86c")
    a.axhline(FLOOR, color="#ff5555", ls="--", label=f"floor {FLOOR:.0%} = retrain trigger")
    a.set_title("Accuracy (needs labels)"); a.legend(fontsize=8)
    a = ax[0, 1]
    line(a, S, "ks", "sensor noise: KS share", "#8be9fd"); line(a, C, "ks", "labels change: KS share", "#ffb86c")
    if S[0]["ev"] is not None:
        a.plot(wk, [r["ev"] for r in S], "x", color="#ff79c6", ms=9, label="sensor noise: Evidently")
    a.axhline(DRIFT_SHARE, color="#ff5555", ls="--", label=f"alert at {DRIFT_SHARE:.0%} of columns")
    a.set_title("Input drift: share of drifted pixel columns"); a.legend(fontsize=8)
    a = ax[1, 0]
    line(a, S, "conf", "sensor noise", "#8be9fd"); line(a, C, "conf", "labels change", "#ffb86c")
    a.axhline(R["conf_threshold"], color="#ff5555", ls="--", label="alert below (mean - 3 sd of clean weeks)")
    a.set_title("Mean prediction confidence"); a.legend(fontsize=8)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("week")
    a = ax[1, 1]
    rt = R["retrain"]
    if rt.get("status") == "ok":
        keys = list(rt["old"])
        xs = np.arange(len(keys))
        a.bar(xs - 0.2, [rt["old"][k] for k in keys], 0.4, color="#ff5555", label="old model")
        a.bar(xs + 0.2, [rt["new"][k] for k in keys], 0.4, color="#50fa7b", label=f"retrained (week {rt['week']})")
        a.set_xticks(xs, keys); a.set_ylim(0, 1.3)
        a.set_title("Accuracy on live images the new model never saw"); a.legend(fontsize=8, loc="upper right", ncol=2)
    else:
        a.text(0.5, 0.5, str(rt.get("status")), ha="center", va="center", wrap=True)
        a.set_title("Retraining")
    fig.suptitle("A deployed digit classifier, 8 weeks of traffic: what the monitor saw", fontsize=14, color="#8be9fd")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

# ── 6. RUN ────────────────────────────────────────────────────────────────
def run():
    MON.mkdir(exist_ok=True)
    xtr, ytr, xref, yref, xrt, yrt, xm, ym = load()
    net = train(xtr, ytr)
    pred, conf = predict(net, xref)
    rng = np.random.default_rng(SEED)
    cal = [week_row(net, xref, *live_batch(xm, ym, 1, "sensor", rng), 0) for _ in range(CLEAN)]
    thr = float(np.mean([r["conf"] for r in cal]) - 3 * np.std([r["conf"] for r in cal]))
    fa = [week_row(net, xref, *live_batch(xm, ym, 1, "sensor", rng), 0) for _ in range(CLEAN)]
    R = dict(
        ver={p: version(p) for p in ("torch", "numpy", "pandas", "scipy", "matplotlib", "evidently") if importlib.util.find_spec(p)},
        weeks=WEEKS, batch=BATCH, floor=FLOOR, drift_share=DRIFT_SHARE, alpha=ALPHA,
        ref_acc=float((pred == yref).mean()), ref_conf=float(conf.mean()), conf_threshold=thr,
        false_alarms=dict(n=CLEAN, drift=sum(r["ks"] >= DRIFT_SHARE for r in fa),
                          conf=sum(r["conf"] < thr for r in fa), max_ks=max(r["ks"] for r in fa))
    )
    for sc in ("sensor", "concept"):
        rng = np.random.default_rng(SEED + 7)
        R[sc] = [week_row(net, xref, *live_batch(xm, ym, w, sc, rng), w,
                          MON / "drift_report.html" if sc == "sensor" and w == WEEKS else None)
                 for w in range(1, WEEKS + 1)]
    R["alerts"] = {sc: dict(
        drift=first(R[sc], lambda r: r["ks"] >= DRIFT_SHARE),
        conf=first(R[sc], lambda r: r["conf"] < thr),
        acc=first(R[sc], lambda r: r["acc"] < FLOOR)
    ) for sc in ("sensor", "concept")}
    gap = [abs(r["ev"] - r["ks"]) for r in R["sensor"] + R["concept"] if r["ev"] is not None]
    R["evidently"] = dict(
        status="ok" if gap and not EV_ERR else (f"failed: {EV_ERR[0]}" if EV_ERR else "skipped: evidently is not installed (pip install evidently)"),
        max_gap=max(gap) if gap else None
    )
    t = R["alerts"]["sensor"]["acc"]
    try:
        R["retrain"] = dict(status="ok", **retrain(net, (xtr, ytr), (xrt, yrt), xm, ym, t)) if t else dict(status="not triggered: accuracy never fell below the floor")
    except Exception as e:
        R["retrain"] = dict(status=f"failed: {type(e).__name__}: {str(e)[:140]}")
    dashboard(R, MON / "dashboard.png")
    Path(__file__).with_name("ep60_results.json").write_text(json.dumps(R))
    return R

if __name__ == "__main__":
    R = run()
    fa = R["false_alarms"]
    print(f"deployed model: {R['ref_acc']:.3f} accuracy, {R['ref_conf']:.3f} mean confidence on held-out reference data")
    print(f"clean traffic, {fa['n']} batches: drift alerts {fa['drift']}, confidence alerts {fa['conf']} (largest drift share {fa['max_ks']:.2f})")
    for sc in ("sensor", "concept"):
        print(f"scenario {sc}: first alert by input drift week {R['alerts'][sc]['drift']}, by confidence week {R['alerts'][sc]['conf']}, by accuracy week {R['alerts'][sc]['acc']}")
        for r in R[sc]:
            print(f"  week {r['week']}  accuracy {r['acc']:.3f}  confidence {r['conf']:.3f}  drifted columns {r['ks']:.0%}" +
                  (f"  evidently {r['ev']:.0%}" if r["ev"] is not None else ""))
    print("evidently:", R["evidently"]["status"])
    rt = R["retrain"]
    print("retrain:", rt["status"] if rt["status"] != "ok" else f"week {rt['week']} old {rt['old']} new {rt['new']}")
