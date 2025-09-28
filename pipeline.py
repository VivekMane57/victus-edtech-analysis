# pipeline.py
import os, glob, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pandas import merge_asof
from scipy.signal import spectrogram
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import shap

# ---------- helpers ----------
def _find_one(base, patterns):
    """Find first file matching any of the patterns inside base."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(base, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No file found in {base} for patterns: {patterns}")

def _read_csv_smart(path, **kw):
    # quiet down dtype warnings on large EEG files
    kw.setdefault("low_memory", False)
    return pd.read_csv(path, **kw)

def _tcol(df):
    # choose a reasonable time-like column
    prefer = ["UnixTime", "TimeStamp", "Timestamp", "routineStamp", "time", "Time", "Unix Time", "Unix_Timestamp"]
    for c in prefer:
        if c in df.columns:
            return c
    # fallback: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise KeyError("No time-like column found")

def _merge_nearest(a, b, tol=0.5):
    if a is None: return b
    if b is None: return a
    return merge_asof(
        a.sort_values("Time"), b.sort_values("Time"),
        on="Time", direction="nearest", tolerance=tol
    )

def _ensure_dirs(models_dir, figs_dir):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

# ---------- STEP 01: preprocess & merge ----------
def _step01_preprocess(base_path, models_dir, figs_dir):
    # accept any prefix: "<id>_EYE.csv", etc.
    eye = _read_csv_smart(_find_one(base_path, ["*_EYE.csv"]))
    eeg = _read_csv_smart(_find_one(base_path, ["*_EEG.csv"]))
    gsr = _read_csv_smart(_find_one(base_path, ["*_GSR.csv"]))
    tiva = _read_csv_smart(_find_one(base_path, ["*_TIVA.csv"]))
    try:
        psy = _read_csv_smart(_find_one(base_path, ["*_PSY.csv", "*_ENG.csv"]))
    except FileNotFoundError:
        psy = None  # optional

    # EYE → PupilDiameter
    eye_t = _tcol(eye)
    pupil_cols = [c for c in ["ET_PupilLeft", "ET_PupilRight", "PupilLeft", "PupilRight", "PupilDiameter"] if c in eye.columns]
    if not pupil_cols:
        raise KeyError("No pupil columns found in *_EYE.csv")
    eye_small = eye[[eye_t] + pupil_cols].copy()
    eye_small["PupilDiameter"] = eye_small[pupil_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    eye_small = eye_small.rename(columns={eye_t: "Time"})[["Time", "PupilDiameter"]]

    # EEG → BetaPower (fallback to other bands)
    eeg_t = _tcol(eeg)
    eeg_num = eeg.apply(pd.to_numeric, errors="ignore")
    beta_cols = [c for c in eeg_num.columns if c.lower().startswith("beta_") or c.lower() == "beta"]
    if not beta_cols:
        alt = ["delta_", "theta_", "alpha_", "gamma_"]
        beta_cols = [c for c in eeg_num.columns if any(c.lower().startswith(p) for p in alt)]
    eeg_small = eeg_num[[eeg_t] + (beta_cols or [])].copy()
    if beta_cols:
        eeg_small["BetaPower"] = eeg_small[beta_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        eeg_small["BetaPower"] = np.nan
    eeg_small = eeg_small.rename(columns={eeg_t: "Time"})[["Time", "BetaPower"]]

    # GSR → conductance
    gsr_t = _tcol(gsr)
    gsr_num = gsr.apply(pd.to_numeric, errors="ignore")
    gsr_candidates = [c for c in gsr_num.columns if (("gsr" in c.lower()) or ("eda" in c.lower())) and any(k in c.lower() for k in ["conduct", "micro", "skin"])]
    if not gsr_candidates:
        gsr_candidates = [c for c in gsr_num.columns if pd.api.types.is_numeric_dtype(gsr_num[c]) and c != gsr_t]
    use_gsr = gsr_candidates[0]
    gsr_small = gsr_num[[gsr_t, use_gsr]].copy().rename(columns={gsr_t: "Time", use_gsr: "GSR"})

    # TIVA → EmotionAvg (+ Valence/Arousal/Blink if present)
    tiva_t = _tcol(tiva)
    cols = tiva.columns
    val   = next((c for c in cols if "valence" in c.lower() or c.lower() == "val"), None)
    aro   = next((c for c in cols if "arousal" in c.lower() or c.lower() == "aro"), None)
    blink = next((c for c in cols if "blink"   in c.lower()), None)
    emo_words = ["joy", "anger", "sad", "fear", "disgust", "surprise", "neutral", "happy", "contempt"]
    emo_cols  = [c for c in cols if any(w in c.lower() for w in emo_words)]
    keep = [tiva_t] + ([val] if val else []) + ([aro] if aro else []) + ([blink] if blink else []) + emo_cols[:10]
    tiva_small = tiva[keep].copy().rename(columns={tiva_t: "Time"})
    if emo_cols:
        tiva_small["EmotionAvg"] = tiva_small[emo_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    elif val and aro:
        v = pd.to_numeric(tiva_small[val], errors="coerce")
        a = pd.to_numeric(tiva_small[aro], errors="coerce")
        v01 = (v - v.min()) / (v.max() - v.min() + 1e-9)
        a01 = (a - a.min()) / (a.max() - a.min() + 1e-9)
        tiva_small["EmotionAvg"] = (v01 + a01) / 2
    if val:   tiva_small = tiva_small.rename(columns={val: "Valence"})
    if aro:   tiva_small = tiva_small.rename(columns={aro: "Arousal"})
    if blink: tiva_small = tiva_small.rename(columns={blink: "BlinkRate"})

    # Merge streams (nearest)
    data = None
    for df in [eye_small, eeg_small, gsr_small, tiva_small]:
        data = _merge_nearest(data, df, tol=0.5)

    # Engagement proxy
    if "Engagement" not in data.columns:
        if "BlinkRate" in data.columns:
            br = pd.to_numeric(data["BlinkRate"], errors="coerce")
            data["Engagement"] = -(br - br.mean()) / (br.std() + 1e-6)
        elif "GSR" in data.columns:
            g = pd.to_numeric(data["GSR"], errors="coerce")
            data["Engagement"] = 1 - (g - g.min()) / (g.max() - g.min() + 1e-9)
        else:
            data["Engagement"] = np.nan

    out_csv = os.path.join(base_path, "processed_merged.csv")
    data.to_csv(out_csv, index=False)

    # a few quick plots
    t = pd.to_numeric(data["Time"], errors="coerce").values
    if "PupilDiameter" in data:
        plt.figure(figsize=(10, 4)); plt.plot(t, data["PupilDiameter"]); plt.title("Pupil")
        plt.savefig(os.path.join(figs_dir, "01_pupil.png")); plt.close()
    if "BetaPower" in data:
        plt.figure(figsize=(10, 4)); plt.plot(t, data["BetaPower"]); plt.title("Beta")
        plt.savefig(os.path.join(figs_dir, "01_beta.png")); plt.close()
        ok = np.isfinite(t)
        dt = np.median(np.diff(t[ok])) if ok.sum() > 1 else None
        fs = 1.0 / dt if (dt and dt > 0) else 128.0
        f, ts, Sxx = spectrogram(np.nan_to_num(pd.to_numeric(data["BetaPower"], errors="coerce").values), fs)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(ts, f, 10*np.log10(Sxx + 1e-12), shading="gouraud")
        plt.title("Beta Spectrogram"); plt.ylabel("Hz"); plt.xlabel("s")
        plt.savefig(os.path.join(figs_dir, "01_beta_spectrogram.png")); plt.close()

    cols_corr = [c for c in ["PupilDiameter", "EmotionAvg", "Engagement", "Valence", "BetaPower", "GSR"] if c in data.columns]
    if len(cols_corr) >= 2:
        C = data[cols_corr].apply(pd.to_numeric, errors="coerce").corr()
        plt.figure(figsize=(5, 4)); plt.imshow(C, cmap="viridis")
        plt.xticks(range(len(cols_corr)), cols_corr, rotation=45, ha='right'); plt.yticks(range(len(cols_corr)), cols_corr)
        for i in range(len(cols_corr)):
            for j in range(len(cols_corr)):
                plt.text(j, i, f"{C.iloc[i, j]:.2f}", ha='center', va='center', color='w')
        plt.title("Correlation")
        plt.savefig(os.path.join(figs_dir, "01_correlation.png")); plt.close()

    return out_csv

# ---------- STEP 02: clean / scale / prune ----------
def _step02_clean(base_path):
    df = pd.read_csv(os.path.join(base_path, "processed_merged.csv"))
    feats = [c for c in df.columns if c != "Time"]
    X = df[feats].apply(pd.to_numeric, errors="coerce").values

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imp.fit_transform(X)
    X_std = scaler.fit_transform(X_imp)

    # low variance
    vt = VarianceThreshold(1e-5)
    X_lv = vt.fit_transform(X_std)
    kept = np.array(feats)[vt.get_support()]

    # high-correlation prune
    Xd = pd.DataFrame(X_lv, columns=kept)
    corr = Xd.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X_clean = Xd.drop(columns=drop)

    out = os.path.join(base_path, "processed_clean.csv")
    X_clean.to_csv(out, index=False)
    return out

# ---------- STEP 03: PCA ----------
def _step03_pca(base_path, figs_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    X = pd.read_csv(os.path.join(base_path, "processed_clean.csv")).apply(pd.to_numeric, errors="coerce").fillna(0).values
    pca = PCA().fit(X)
    cum = pca.explained_variance_ratio_.cumsum()
    k90 = int(np.argmax(cum >= 0.90) + 1) if len(cum) else 0

    plt.figure(figsize=(8, 4))
    plt.plot(cum); plt.title("PCA cumulative variance"); plt.xlabel("components"); plt.ylabel("cum var")
    plt.savefig(os.path.join(figs_dir, "03_pca_cumvar.png")); plt.close()

    import pickle
    with open(os.path.join(models_dir, "pca_components.pkl"), "wb") as f:
        pickle.dump({"components": pca.components_, "explained": pca.explained_variance_ratio_}, f)
    return k90

# ---------- STEP 04: LDA ----------
def _step04_lda(base_path, models_dir):
    Xdf = pd.read_csv(os.path.join(base_path, "processed_clean.csv")).apply(pd.to_numeric, errors="coerce").fillna(0)
    raw = pd.read_csv(os.path.join(base_path, "processed_merged.csv"))

    # labels
    if "Correct" in raw.columns and raw["Correct"].notna().any():
        y = (raw["Correct"].fillna(0) > 0).astype(int).values
    elif "Valence" in raw.columns and raw["Valence"].notna().any():
        med = raw["Valence"].median()
        y = (raw["Valence"] > med).astype(int).values
    else:
        med = raw["Engagement"].median()
        y = (raw["Engagement"] > med).astype(int).values

    lda = LinearDiscriminantAnalysis()
    try:
        scores = cross_val_score(lda, Xdf.values, y, cv=5)
        lda.fit(Xdf.values, y)
        coef = pd.Series(lda.coef_[0], index=Xdf.columns).sort_values(key=lambda s: s.abs(), ascending=False)
        coef.head(20).to_csv(os.path.join(base_path, "lda_top_features.csv"), index=True)
        import pickle; os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "lda_model.pkl"), "wb") as f: pickle.dump(lda, f)
        return float(scores.mean())
    except Exception:
        # if labels are still degenerate, skip
        return float("nan")

# ---------- STEP 05: Autoencoder ----------
def _step05_autoencoder(base_path, models_dir):
    import torch, torch.nn as nn, torch.optim as optim
    X = pd.read_csv(os.path.join(base_path, "processed_clean.csv")).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype("float32")
    import torch as _t
    if X.ndim != 2 or X.shape[1] == 0:
        return float("nan")
    X = _t.tensor(X); d_in = X.shape[1]

    class AE(nn.Module):
        def __init__(self, d_in, d_lat=8):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_in,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,d_lat))
            self.dec = nn.Sequential(nn.Linear(d_lat,64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,d_in))
        def forward(self,x): z=self.enc(x); return self.dec(z), z

    model = AE(d_in); opt = optim.Adam(model.parameters(), lr=1e-3); lossf = nn.MSELoss()
    # short training; tune if needed
    for _ in range(50):
        opt.zero_grad(); xh, z = model(X); loss = lossf(xh, X); loss.backward(); opt.step()
    _t.save(model.state_dict(), os.path.join(models_dir, "autoencoder_model.pt"))
    with _t.no_grad():
        _, Z = model(X)
        np.savetxt(os.path.join(base_path, "autoencoder_latent.csv"), Z.numpy(), delimiter=",")
    return float(loss.detach().cpu().item())

# ---------- STEP 06: XGBoost + SHAP ----------
def _step06_xgb_shap(base_path, figs_dir):
    Xdf = pd.read_csv(os.path.join(base_path, "processed_clean.csv")).apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0)
    raw = pd.read_csv(os.path.join(base_path, "processed_merged.csv"))

    # labels: Correct -> Valence -> Engagement
    if "Correct" in raw.columns and raw["Correct"].notna().any():
        y = (raw["Correct"].fillna(0) > 0).astype(int)
    elif "Valence" in raw.columns and raw["Valence"].notna().any():
        med = raw["Valence"].median()
        y = (raw["Valence"] > med).astype(int)
    else:
        med = raw["Engagement"].median()
        y = (raw["Engagement"] > med).astype(int)

    n = min(len(Xdf), len(y))
    X = Xdf.iloc[:n].reset_index(drop=True)
    y = y.iloc[:n].astype(int).reset_index(drop=True)

    # widen split if needed; if still one-class, SKIP instead of crash
    if y.nunique() < 2:
        if "Engagement" in raw.columns:
            lo, hi = raw["Engagement"].quantile(0.4), raw["Engagement"].quantile(0.6)
            mask = (raw["Engagement"] <= lo) | (raw["Engagement"] >= hi)
            X = X.loc[mask.values].reset_index(drop=True)
            y = (raw.loc[mask, "Engagement"] > raw["Engagement"].median()).astype(int).reset_index(drop=True)
    if y.nunique() < 2:
        with open(os.path.join(base_path, "xgb_skipped.txt"), "w") as f:
            f.write("XGB skipped due to single-class labels.")
        return float("nan")

    clf = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1, eval_metric="logloss"
    )
    scores = cross_val_score(clf, X.values, y.values, cv=3)
    clf.fit(X.values, y.values)

    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(base_path, "xgb_feature_importance.csv"))

    try:
        expl = shap.Explainer(clf, X); sv = expl(X); vals = sv.values
    except Exception:
        expl = shap.TreeExplainer(clf); vals = expl.shap_values(X)
    plt.figure(); shap.summary_plot(vals, X, show=False); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "06_shap_summary.png"), dpi=150); plt.close()

    return float(scores.mean())

# ---------- PUBLIC: run one subject ----------
def run_subject(base_path, models_dir="./models", figs_dir="./notebooks/figures"):
    """Run the full pipeline for one subject folder, e.g., r'D:\\IITB\\STData\\1'."""
    print(f"=== SUBJECT: {base_path} ===")
    _ensure_dirs(models_dir, figs_dir)
    _step01_preprocess(base_path, models_dir, figs_dir)
    _step02_clean(base_path)
    k90     = _step03_pca(base_path, figs_dir, models_dir)
    lda_acc = _step04_lda(base_path, models_dir)
    ae_loss = _step05_autoencoder(base_path, models_dir)
    xgb_acc = _step06_xgb_shap(base_path, figs_dir)
    return {
        "subject": os.path.basename(base_path),
        "k90": int(k90) if k90==k90 else None,
        "lda_acc": None if np.isnan(lda_acc) else round(float(lda_acc), 3),
        "ae_loss": None if np.isnan(ae_loss) else round(float(ae_loss), 6),
        "xgb_acc": None if np.isnan(xgb_acc) else round(float(xgb_acc), 3),
    }

# convenient CLI
if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if not sub:
        print("Usage: python pipeline.py <subject_folder>")
        sys.exit(1)
    print(run_subject(sub))
