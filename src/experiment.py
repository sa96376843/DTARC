"""Prequential delayed-feedback experiments for DTARC and matched baselines.

All predictions are made before model updates. For an h-day target, a sample is
eligible for training only after its h-day forward return is observable.
"""
from pathlib import Path
import argparse, copy, json, math, time, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"
OUT.mkdir(exist_ok=True)


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, 32), nn.ReLU(), nn.Dropout(0.10),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class ADWIN:
    """Bernoulli-error ADWIN with Hoeffding cuts and a bounded window."""
    def __init__(self, delta=0.01, min_window=16, max_window=200):
        self.delta, self.min_window, self.max_window = delta, min_window, max_window
        self.w = []
    def update(self, x):
        self.w.append(float(x))
        if len(self.w) > self.max_window: self.w.pop(0)
        n = len(self.w)
        if n < 2*self.min_window: return None
        a = np.asarray(self.w)
        for cut in range(self.min_window, n-self.min_window+1):
            old, new = a[:cut], a[cut:]
            m = 1/len(old) + 1/len(new)
            eps = math.sqrt(0.5*m*math.log(4*n/self.delta))
            if abs(old.mean()-new.mean()) > eps:
                pooled = (old.var()+new.var())/2
                severity = abs(old.mean()-new.mean()) / math.sqrt(pooled+0.02)
                self.w = list(new)
                return {"severity": float(severity), "old_mean": float(old.mean()), "new_mean": float(new.mean())}
        return None


class PageHinkley:
    def __init__(self, delta=0.005, threshold=5.0):
        self.delta, self.threshold = threshold, threshold
        self.delta = delta; self.n = 0; self.mean = 0.; self.cum = 0.; self.min_cum = 0.
    def update(self, x):
        self.n += 1; self.mean += (x-self.mean)/self.n
        self.cum += x-self.mean-self.delta; self.min_cum = min(self.min_cum, self.cum)
        if self.n >= 20 and self.cum-self.min_cum > self.threshold:
            severity = (self.cum-self.min_cum)/(math.sqrt(self.n)+1e-9)
            self.n = 0; self.mean = self.cum = self.min_cum = 0.
            return {"severity": float(severity), "old_mean": np.nan, "new_mean": np.nan}
        return None


def make_detector(name, cfg):
    return ADWIN(cfg["adwin_delta"]) if name == "adwin" else PageHinkley(threshold=cfg["ph_threshold"])


def purge_label_boundary(train, h):
    """Remove training rows whose h-day labels mature after the split boundary."""
    train = train.sort_values(["code", "date"]).copy()
    keep = train.groupby("code").cumcount() < (train.groupby("code")["code"].transform("size") - h)
    return train.loc[keep].copy()


def initial_fit(train, features, h, seed, epochs=120, purge_extra=0, purge_boundary=True):
    seed_all(seed)
    if purge_boundary:
        train = purge_label_boundary(train, h + purge_extra)
    med = train[features].median().fillna(0)
    X0 = train[features].fillna(med).fillna(0).values
    scaler = StandardScaler().fit(X0)
    X = scaler.transform(X0); y = train[f"y_{h}"].values.astype(np.float32)
    model = MLP(len(features))
    pos_weight = torch.tensor([(y == 0).sum()/max((y == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for idx in perm.split(128):
            opt.zero_grad(); loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    return {"model": model, "scaler": scaler, "median": med, "pos_weight": pos_weight}


def transform(d, base, features):
    X = d[features].fillna(base["median"]).fillna(0).values
    return base["scaler"].transform(X).astype(np.float32)


def predict(model, X, batch=4096):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i+batch]))).numpy())
    return np.concatenate(out)


def farthest_select(X, candidates, n, rng):
    candidates = list(dict.fromkeys(map(int, candidates)))
    if len(candidates) <= n: return candidates
    if not candidates: return []
    chosen = [int(rng.choice(candidates))]
    dist = ((X[candidates]-X[chosen[0]])**2).sum(axis=1)
    cmap = {c:i for i,c in enumerate(candidates)}
    while len(chosen) < n:
        remaining = [c for c in candidates if c not in chosen]
        if not remaining: break
        vals = np.array([dist[cmap[c]] for c in remaining])
        nxt = remaining[int(np.argmax(vals))]; chosen.append(nxt)
        nd = ((X[candidates]-X[nxt])**2).sum(axis=1)
        dist = np.minimum(dist, nd)
    return chosen


def stratified(candidates, labels, n, pos_rate, rng):
    candidates = np.array(list(dict.fromkeys(map(int, candidates))), dtype=int)
    if n <= 0 or len(candidates) == 0: return []
    npos = int(round(n*pos_rate)); chosen = []
    for val, k in [(1, npos), (0, n-npos)]:
        pool = candidates[labels[candidates] == val]
        if len(pool) and k: chosen.extend(rng.choice(pool, size=k, replace=len(pool) < k).tolist())
    if len(chosen) < n:
        extra = rng.choice(candidates, size=n-len(chosen), replace=len(candidates) < n-len(chosen))
        chosen.extend(extra.tolist())
    return chosen[:n]


def dtarc_sample(memory, anchor, X, y, latest_y, n, severity, threshold, rng, variant="full"):
    if not memory or n == 0: return []
    abrupt = severity >= threshold
    quotas = {"recent": .50, "diverse": .20, "anchor": .20, "uniform": .10} if abrupt else {"recent": .20, "diverse": .50, "anchor": .20, "uniform": .10}
    if variant == "no_typing":
        quotas = {"recent": .35, "diverse": .35, "anchor": .20, "uniform": .10}
    if variant == "no_provenance":
        quotas = {"recent": 0.0, "diverse": 0.0, "anchor": 0.0, "uniform": 1.0}
    removed = {"no_anchor": "anchor", "no_diversity": "diverse"}.get(variant)
    if removed:
        # Remove one mechanism and redistribute its quota proportionally over
        # the remaining mechanisms so every ablation uses the same batch size.
        mass = quotas.pop(removed)
        total = sum(quotas.values())
        quotas = {k: v + mass*v/total for k, v in quotas.items()}
    mem = np.array(memory); selected = []
    pools = {"recent": mem[-max(32, len(mem)//5):].tolist(),
             "anchor": [i for i in memory if i in anchor], "uniform": memory,
             "diverse": memory}
    pos_rate = float(np.mean(latest_y)) if len(latest_y) else float(np.mean(y[mem]))
    for kind, frac in quotas.items():
        k = int(round(n*frac))
        if kind == "diverse":
            ids = farthest_select(X, pools[kind], min(k*3, len(pools[kind])), rng)
            ids = stratified(ids, y, k, pos_rate, rng)
        else:
            ids = stratified(pools[kind], y, k, pos_rate, rng)
        selected.extend(ids)
    selected = list(dict.fromkeys(selected))
    if len(selected) < n:
        fill = stratified([i for i in memory if i not in selected], y, n-len(selected), pos_rate, rng)
        selected.extend(fill)
    return selected[:n]


def online_update(model, X, y, latest_ids, replay_ids, base_state, pos_weight, lr, prox, epochs=2):
    if not len(latest_ids): return
    ids = np.array(list(latest_ids)+list(replay_ids), dtype=int)
    if not len(ids): return
    model.train(); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt, yt = torch.tensor(X[ids]), torch.tensor(y[ids], dtype=torch.float32)
    old = [p.detach().clone() for p in base_state]
    for _ in range(epochs):
        opt.zero_grad(); task = loss_fn(model(Xt), yt)
        penalty = sum(((p-o)**2).sum() for p,o in zip(model.parameters(), old))
        loss = task + prox*penalty; loss.backward(); opt.step()


def run_online(method, d, base, features, h, seed, cfg, split, variant="full", allow_premature=False, delay_batches=None):
    seed_all(seed+1000)
    rng = np.random.default_rng(seed+1000)
    model = copy.deepcopy(base["model"])
    work = d[d.split.isin(["train", split])].copy().sort_values(["date", "code"]).reset_index(drop=True)
    work = work.dropna(subset=[f"y_{h}"])
    X = transform(work, base, features); y = work[f"y_{h}"].values.astype(np.int64)
    train_ids = np.where(work.split.values == "train")[0]
    test_ids = np.where(work.split.values == split)[0]
    init_n = min(cfg["buffer"], len(train_ids))
    memory = rng.choice(train_ids, size=init_n, replace=False).tolist()
    memory.sort(); anchor = set(memory[:min(128, len(memory)//4)])
    detector_name = "adwin" if "adwin" in method else ("ph" if method.startswith("ph") else None)
    detector = make_detector(detector_name, cfg) if detector_name else None
    test_dates = np.sort(work.loc[test_ids, "date"].unique())
    delay = max(1, int(math.ceil(h/5))) if delay_batches is None else int(delay_batches)
    batches = [test_dates[i:i+5] for i in range(0, len(test_dates), 5)]
    pending, records, events = [], [], []
    start = time.perf_counter(); update_count = 0
    for step, dates in enumerate(batches):
        ids = np.where(work.date.isin(dates).values & (work.split.values == split))[0]
        if not len(ids): continue
        probs = predict(model, X[ids])
        for i,p in zip(ids, probs):
            records.append((work.at[i,"date"], work.at[i,"code"], int(y[i]), float(p), float(work.at[i,f"fwd_ret_{h}"])))
        if allow_premature:
            # Deliberately invalid protocol-degradation diagnostic. Never use
            # this branch for model selection or reported primary performance.
            matured = ids.tolist()
        else:
            matured = pending[0] if len(pending) >= delay else []
        if len(pending) >= delay: pending = pending[1:]
        drift = None
        if len(matured):
            pm = predict(model, X[matured]); errors = (pm >= .5).astype(int) != y[matured]
            if detector:
                for e in errors:
                    hit = detector.update(float(e))
                    if hit is not None: drift = hit
        should_update = len(matured) > 0 and (method in ["periodic", "periodicER"] or drift is not None)
        replay_ids = []
        if should_update:
            if method in ["periodic", "static"]: replay_ids = []
            elif method in ["periodicER", "adwinER", "phER"]:
                replay_ids = rng.choice(memory, size=min(cfg["replay"], len(memory)), replace=False).tolist()
            elif method == "adwinERstrat":
                replay_ids = stratified(memory, y, min(cfg["replay"], len(memory)), float(np.mean(y[matured])), rng)
            elif "DTARC" in method:
                replay_ids = dtarc_sample(memory, anchor, X, y, y[matured], min(cfg["replay"], len(memory)),
                                          drift["severity"], cfg["abrupt_threshold"], rng, variant)
            prox = 0.0 if variant == "no_proximal" else cfg["prox"]
            if "DTARC" in method and drift:
                prox = prox/(1+drift["severity"]) if drift["severity"] >= cfg["abrupt_threshold"] else prox*(1+.5*drift["severity"])
            before = [p.detach().clone() for p in model.parameters()]
            online_update(model, X, y, matured, replay_ids, before, base["pos_weight"], cfg["online_lr"], prox)
            update_count += 1
            if drift:
                events.append({"date": str(pd.Timestamp(dates[-1])), "severity": drift["severity"],
                               "type": "abrupt" if drift["severity"] >= cfg["abrupt_threshold"] else "gradual",
                               "detector": detector_name, "update": update_count})
        if len(matured):
            for i in matured:
                if len(memory) >= cfg["buffer"]:
                    removable = [j for j in range(len(memory)) if memory[j] not in anchor]
                    if removable: memory.pop(int(rng.choice(removable)))
                memory.append(int(i))
        pending.append(ids.tolist())
    pred = pd.DataFrame(records, columns=["date","code","y","p","fwd_ret"])
    pred["method"] = method; pred["market"] = d.market.iloc[0]; pred["seed"] = seed
    pred["split"] = split; pred["h"] = h
    return pred, pd.DataFrame(events), {"updates":update_count,"seconds":time.perf_counter()-start,"final_memory":len(memory)}


def static_predictions(method, d, base, features, h, seed, split):
    test = d[(d.split == split)].dropna(subset=[f"y_{h}"]).copy().sort_values(["date","code"])
    X = transform(test, base, features); p = predict(base["model"], X)
    out = test[["date","code",f"y_{h}",f"fwd_ret_{h}"]].rename(columns={f"y_{h}":"y",f"fwd_ret_{h}":"fwd_ret"})
    out["p"] = p; out["method"] = method; out["market"] = d.market.iloc[0]; out["seed"] = seed; out["split"] = split; out["h"] = h
    return out, pd.DataFrame(), {"updates":0,"seconds":0.,"final_memory":0}


def logistic_predictions(d, features, h, seed, split):
    train = purge_label_boundary(d[d.split == "train"].dropna(subset=[f"y_{h}"]), h)
    test = d[d.split == split].dropna(subset=[f"y_{h}"]).copy()
    med = train[features].median().fillna(0); scaler = StandardScaler().fit(train[features].fillna(med).fillna(0))
    X = scaler.transform(train[features].fillna(med).fillna(0)); Xt = scaler.transform(test[features].fillna(med).fillna(0))
    clf = LogisticRegression(C=.2, max_iter=2000, class_weight="balanced", random_state=seed).fit(X, train[f"y_{h}"])
    out = test[["date","code",f"y_{h}",f"fwd_ret_{h}"]].rename(columns={f"y_{h}":"y",f"fwd_ret_{h}":"fwd_ret"})
    out["p"] = clf.predict_proba(Xt)[:,1]; out["method"] = "logistic"; out["market"] = d.market.iloc[0]; out["seed"] = seed; out["split"] = split; out["h"] = h
    return out


def evaluate(pred):
    rows = []
    for keys,g in pred.groupby(["market","split","h","method","seed"]):
        y,p = g.y.values,g.p.values; yh = (p >= .5).astype(int)
        rows.append(dict(zip(["market","split","h","method","seed"],keys)) | {
            "n":len(g),"accuracy":accuracy_score(y,yh),"macro_f1":f1_score(y,yh,average="macro"),
            "mcc":matthews_corrcoef(y,yh),"auc":roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
            "brier":brier_score_loss(y,p)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode",choices=["tune","final"],required=True)
    ap.add_argument("--horizon",type=int,default=5); ap.add_argument("--seeds",default="11,22,33,44,55")
    args = ap.parse_args(); h=args.horizon
    info = json.loads((DATA/"dataset_summary.json").read_text()); features=info["features"]
    data = {m:pd.read_csv(DATA/f"{m.lower()}_processed.csv",parse_dates=["date"]) for m in ["CN","TH"]}
    default = {"adwin_delta":.01,"ph_threshold":5.,"buffer":1024,"replay":64,"prox":.01,"online_lr":3e-4,"abrupt_threshold":.5}
    if args.mode == "tune":
        all_preds=[]; configs=[]
        base=initial_fit(data["CN"][data["CN"].split=="train"].dropna(subset=[f"y_{h}"]),features,h,42)
        for delta in [.001,.05,.20,.50,.80]:
            for buf in [512,1024]:
                for prox in [.001,.01]:
                    for online_lr in [.0001,.0003]:
                        cfg=default|{"adwin_delta":delta,"buffer":buf,"prox":prox,"online_lr":online_lr}
                        p,e,r=run_online("adwinDTARC",data["CN"],base,features,h,42,cfg,"validation")
                        p["config"]=json.dumps(cfg,sort_keys=True); all_preds.append(p); configs.append(cfg)
                        print("tune",cfg,"events",len(e))
        pd.concat(all_preds).to_csv(OUT/f"tune_predictions_h{h}.csv",index=False)
        metrics=evaluate(pd.concat(all_preds).assign(config=pd.concat(all_preds).config))
        # Direct configuration-level scoring avoids relying on aggregated runs.
        scores=[]
        for cfg,p in zip(configs,all_preds):
            y,pr=p.y.values,p.p.values; yh=(pr>=.5).astype(int)
            scores.append(cfg|{"mcc":matthews_corrcoef(y,yh),"brier":brier_score_loss(y,pr),
                               "score":matthews_corrcoef(y,yh)-.1*brier_score_loss(y,pr)})
        scores=sorted(scores,key=lambda x:x["score"],reverse=True); best={k:scores[0][k] for k in default}
        (OUT/"best_config.json").write_text(json.dumps({"best":best,"validation_scores":scores},indent=2))
        print("BEST",best)
        return
    cfg=default|json.loads((OUT/"best_config.json").read_text())["best"]
    seeds=list(map(int,args.seeds.split(","))); all_preds=[]; all_events=[]; runtime=[]
    for market,d in data.items():
        train=d[d.split=="train"].dropna(subset=[f"y_{h}"])
        for seed in seeds:
            base=initial_fit(train,features,h,seed)
            for method in ["static","periodic","periodicER","adwinER","phER","adwinDTARC","phDTARC"]:
                if method=="static": p,e,r=static_predictions(method,d,base,features,h,seed,"test")
                else: p,e,r=run_online(method,d,base,features,h,seed,cfg,"test")
                all_preds.append(p)
                if len(e): all_events.append(e.assign(market=market,seed=seed,method=method,h=h))
                runtime.append(r|{"market":market,"seed":seed,"method":method,"h":h})
            all_preds.append(logistic_predictions(d,features,h,seed,"test"))
            print("finished",market,seed,"h",h)
    pred=pd.concat(all_preds,ignore_index=True)
    pred.to_csv(OUT/f"predictions_h{h}.csv",index=False)
    if all_events: pd.concat(all_events,ignore_index=True).to_csv(OUT/f"drift_events_h{h}.csv",index=False)
    pd.DataFrame(runtime).to_csv(OUT/f"runtime_h{h}.csv",index=False)
    evaluate(pred).to_csv(OUT/f"metrics_h{h}.csv",index=False)
    print(evaluate(pred).groupby(["market","method"])[["accuracy","macro_f1","mcc","auc","brier"]].mean().round(4))

if __name__ == "__main__": main()
