import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # évite les crash OpenMP sous Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import io
import numpy as np
from scipy.linalg import solveh_banded
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Torch pour le modèle DL
import torch
import torch.nn as nn

# UTILS & SIGNALS

def zscore_per_sample(x: np.ndarray):
    x = x.astype(np.float32)
    mu = x.mean()
    sigma = x.std() + 1e-8
    return (x - mu) / sigma, mu, sigma

def un_zscore(xz: np.ndarray, mu: float, sigma: float):
    return xz * sigma + mu

def generate_signal(L=1600, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, L)
    # composante utile : sinusoïdes + impulsions
    signal = (
        1.0 * np.sin(2 * np.pi * 7 * t)
        + 0.6 * np.sin(2 * np.pi * 13 * t + rng.uniform(0, 2 * np.pi))
    )
    for _ in range(rng.integers(3, 8)):
        c = rng.uniform(0, 1)
        w = rng.uniform(0.003, 0.02)
        a = rng.uniform(0.3, 1.0)
        signal += a * np.exp(-0.5 * ((t - c) / w) ** 2)

    # baseline lente : poly + sin basse fréquence
    baseline = (
        rng.uniform(-0.5, 0.5)
        + rng.uniform(-1, 1) * (t - 0.5)
        + rng.uniform(-0.5, 0.5) * (t - 0.5) ** 2
        + 0.3 * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t)
    )

    noise = rng.normal(0, 0.1, L)
    s = signal + baseline + noise
    return t, signal, baseline, s

# Métriques
def mse(a, b): return float(np.mean((a - b) ** 2))
def mae(a, b): return float(np.mean(np.abs(a - b)))
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


#   MÉTHODES CLASSIQUES
from scipy.signal import savgol_filter
from scipy.linalg import solveh_banded

def remove_baseline_savgol(signal: np.ndarray, win=101, poly=3):
    win = int(win)
    if win % 2 == 0:  # doit être impair
        win += 1
    base = savgol_filter(signal, window_length=win, polyorder=poly, mode="interp")
    corrected = signal - base
    return corrected, base

def whittaker_asls_safe(y, lam=1e5, p=0.01, niter=10, w_min=1e-3, eps=1e-8):
    y = np.asarray(y, dtype=np.float64)
    n = y.size

    # D^T D pour différences secondes (pentadiagonale)
    d0 = np.zeros(n, dtype=np.float64)
    d1 = np.zeros(n-1, dtype=np.float64)
    d2 = np.zeros(n-2, dtype=np.float64)

    # bords "naturels"
    d0[:] = 6.0
    d0[0] = d0[-1] = 1.0
    d0[1] = d0[-2] = 5.0
    d1[:] = -4.0
    d1[0] = d1[-1] = -2.0
    d2[:] = 1.0

    d0 *= lam; d1 *= lam; d2 *= lam

    # initialisation
    z = y.copy()
    w = np.ones(n, dtype=np.float64)

    for _ in range(niter):
        # poids asymétriques bornés
        w = p * (y > z) + (1.0 - p) * (y < z)
        w = np.clip(w, w_min, 1.0)

        # matrice bande 'ab' pour solveh_banded (upper=True)
        ab = np.zeros((3, n), dtype=np.float64)
        ab[0, 2:] = d2                      # 2e sur-diagonale
        ab[1, 1:] = d1                      # 1ère sur-diagonale
        ab[2, :]  = d0 + w + eps            # diagonale principale + jitter

        rhs = w * y

        # solve SPD banded
        z = solveh_banded(ab, rhs, lower=False, overwrite_ab=True, overwrite_b=True, check_finite=False)

    return z

def remove_baseline_asls(signal, lam=1e5, p=0.01, niter=10, w_min=1e-3, eps=1e-8):
    try:
        base = whittaker_asls_safe(signal, lam=lam, p=p, niter=niter, w_min=w_min, eps=eps)
        return signal - base, base
    except Exception as e:
        try:
            base = whittaker_asls_safe(signal, lam=max(lam, 1e6), p=p, niter=niter, w_min=max(w_min, 5e-3), eps=max(eps, 1e-6))
            return signal - base, base
        except Exception as e2:
            raise RuntimeError(f"AsLS failed even after stabilization: {e2}")

#     MODÈLE ResUNet-1D

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, k, padding=p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool1d(2), ConvBlock(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diff = skip.size(-1) - x.size(-1)
        if diff > 0:
            x = nn.functional.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :skip.size(-1)]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ResUNet1D(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = ConvBlock(1, base)          # L
        self.enc2 = Down(base, base*2)          # L/2
        self.enc3 = Down(base*2, base*4)        # L/4
        self.bottleneck = ConvBlock(base*4, base*8)
        self.up_e2 = Up(base*8 + base*2, base*4)  # L/4 -> L/2 + skip e2
        self.up_e1 = Up(base*4 + base,   base*2)  # L/2 -> L   + skip e1
        self.out_conv = nn.Conv1d(base*2, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)       # L
        e2 = self.enc2(e1)      # L/2
        e3 = self.enc3(e2)      # L/4
        b  = self.bottleneck(e3)
        u2 = self.up_e2(b, e2)  # L/2
        u1 = self.up_e1(u2, e1) # L
        return self.out_conv(u1)

def predict_resunet(x_1d: np.ndarray, ckpt_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResUNet1D().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    xz, mu, sigma = zscore_per_sample(x_1d)
    with torch.no_grad():
        y = model(torch.from_numpy(xz[None, None, :])).to(device).cpu().numpy()[0,0]
    # On reste en échelle normalisée pour comparer SG/AsLS aussi z-score ou je remets en raw :
    # return un_zscore(y, mu, sigma)
    return y


#          UI
st.set_page_config(page_title="Baseline Removal 1D", page_icon="🧠", layout="wide")
st.title("Baseline Removal (1D) — ResUNet-1D vs Classical")

with st.sidebar:
    st.header("Entrée du signal")
    source = st.radio("Source", ["Charger un CSV", "Générer un signal synthétique"])
    if source == "Charger un CSV":
        up = st.file_uploader("CSV (1 colonne)", type=["csv"])
    else:
        L = st.number_input("Longueur L", min_value=256, max_value=10000, value=1600, step=64)
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
    st.markdown("---")

    st.header("Méthode")
    method = st.selectbox("Choisir une méthode", ["ResUNet-1D (DL)", "Savitzky–Golay", "AsLS (safe)"])
    if method == "ResUNet-1D (DL)":
        ckpt_path = st.text_input("Checkpoint .pt", value="best_resunet.pt")
        st.caption("Dépose ton fichier .pt dans le dossier de l'app (ou donne le chemin complet).")
    elif method == "Savitzky–Golay":
        win = st.slider("Fenêtre (impair)", min_value=31, max_value=1201, value=601, step=10)
        poly = st.slider("Ordre polynôme", min_value=2, max_value=5, value=2, step=1)
    else:
        lam = st.select_slider("λ (Whittaker)", options=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6], value=1e5)
        p   = st.select_slider("p (asymmetry)", options=[0.001, 0.01, 0.05], value=0.05)
        niter = st.slider("Itérations", min_value=5, max_value=30, value=10, step=1)

    st.markdown("---")
    st.header("Affichage & métriques")
    scale = st.radio("Échelle d'affichage / métriques", ["RAW", "z-score"], index=0)
    export_metrics = st.checkbox("Exporter les métriques (si ground-truth)")

    run = st.button("Corriger le signal", type="primary")

# Charger / générer le signal
if source == "Charger un CSV":
    if up is not None:
        arr = np.loadtxt(io.StringIO(up.getvalue().decode("utf-8")), delimiter=",")
        if arr.ndim > 1:
            arr = arr[:,0]
        sig = arr.astype(np.float32)
        t = np.arange(len(sig))
    else:
        st.info("Charge un CSV pour continuer ou génère un signal synthétique.")
        sig = None
        t = None
else:
    t, _, _, s = generate_signal(L=L, seed=seed)
    sig = s.astype(np.float32)

if sig is not None:
    st.subheader("Signal d'entrée")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(t if t is not None else np.arange(len(sig)), sig, lw=1)
    ax.set_title("Signal mesuré")
    st.pyplot(fig, clear_figure=True)

if run:
    try:
        # CAS 1 : CSV importé (pas de ground-truth)
        if source == "Charger un CSV":
            xz, mu_x, sigma_x = zscore_per_sample(sig)

            if method == "ResUNet-1D (DL)":
                if not os.path.isfile(ckpt_path):
                    st.error(f"Checkpoint introuvable: {ckpt_path}")
                    st.stop()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = ResUNet1D().to(device)
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    y_pred_z = model(torch.from_numpy(xz[None, None, :]).to(device)).cpu().numpy()[0,0]
                y_pred_raw = un_zscore(y_pred_z, mu_x, sigma_x)
                base_est = None
                method_name = "ResUNet-1D"
            elif method == "Savitzky–Golay":
                y_pred_raw, base_est = remove_baseline_savgol(sig, win=win, poly=poly)
                method_name = f"SavGol (win={win}, poly={poly})"
            else:
                y_pred_raw, base_est = remove_baseline_asls(sig, lam=lam, p=p, niter=niter)
                method_name = f"AsLS (λ={lam:.1e}, p={p})"

            # Sélection d'échelle pour l'affichage
            if scale == "RAW":
                x_disp = sig
                y_disp = y_pred_raw
                base_disp = base_est
                title_scale = "RAW"
            else:
                # en z-score par rapport à X (entrée)
                x_disp, muX, stdX = zscore_per_sample(sig)
                y_disp = (y_pred_raw - muX) / (stdX + 1e-8)
                base_disp = None if base_est is None else (base_est - muX) / (stdX + 1e-8)
                title_scale = "z-score"

            st.subheader(f"Comparaison (sans ground-truth) — {title_scale}")
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(x_disp, label=f"Entrée ({title_scale})", alpha=0.7)
            ax.plot(y_disp, label=f"Sortie ({method_name})", lw=2)
            if base_disp is not None:
                ax.plot(base_disp, label="Baseline estimée", alpha=0.6)
            ax.legend(ncol=2); ax.set_title("Correction sur signal importé")
            st.pyplot(fig, clear_figure=True)

            # Export : uniquement la sortie (pas de métriques sans ground-truth)
            if scale == "RAW":
                out_to_save = y_pred_raw
                fname = "signal_corrige_raw.csv"
            else:
                out_to_save = y_disp
                fname = "signal_corrige_z.csv"
            st.download_button("⬇️ Télécharger le signal corrigé (CSV)",
                               data="\n".join(map(str, out_to_save.astype(np.float32).tolist())).encode("utf-8"),
                               file_name=fname, mime="text/csv")

        # CAS 2 : Signal synthétique (ground-truth dispo)
        else:
            t, x_true_raw, b_true_raw, s_raw = generate_signal(L=L, seed=seed)

            # Normalisations cohérentes avec le training
            xz_in,  mu_xin,  sigma_xin  = zscore_per_sample(s_raw)      # entrée modèle
            xz_tgt, mu_ytgt, sigma_ytgt = zscore_per_sample(x_true_raw) # cible (utile pour remontée en raw côté DL)

            base_est_raw = None
            if method == "ResUNet-1D (DL)":
                if not os.path.isfile(ckpt_path):
                    st.error(f"Checkpoint introuvable: {ckpt_path}")
                    st.stop()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = ResUNet1D().to(device)
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    y_pred_z = model(torch.from_numpy(xz_in[None, None, :]).to(device)).cpu().numpy()[0,0]
                # Remontée à l'échelle de la VÉRITÉ terrain (Y)
                y_pred_raw = un_zscore(y_pred_z, mu_ytgt, sigma_ytgt)
                method_name = "ResUNet-1D"
            elif method == "Savitzky–Golay":
                y_pred_raw, base_est_raw = remove_baseline_savgol(s_raw, win=win, poly=poly)
                method_name = f"SavGol (win={win}, poly={poly})"
            else:
                y_pred_raw, base_est_raw = remove_baseline_asls(s_raw, lam=lam, p=p, niter=niter)
                method_name = f"AsLS (λ={lam:.1e}, p={p})"

            # Échelle choisie pour affichage + métriques
            if scale == "RAW":
                x_ref = x_true_raw
                s_disp = s_raw
                y_disp = y_pred_raw
                base_disp = base_est_raw
                title_scale = "RAW"
            else:
                # tout en z-score par rapport à la VÉRITÉ terrain x_true (référence)
                x_ref, muY, stdY = zscore_per_sample(x_true_raw)
                s_disp = (s_raw - muY) / (stdY + 1e-8)
                y_disp = (y_pred_raw - muY) / (stdY + 1e-8)
                base_disp = None if base_est_raw is None else (base_est_raw - muY) / (stdY + 1e-8)
                title_scale = "z-score"

            # Métriques sur l'échelle choisie
            m_mse = mse(x_ref, y_disp)
            m_mae = mae(x_ref, y_disp)
            m_r2  = r2_score(x_ref, y_disp)

            st.subheader(f"Métriques — {title_scale}")
            c1, c2, c3 = st.columns(3)
            c1.metric("MSE", f"{m_mse:.4f}")
            c2.metric("MAE", f"{m_mae:.4f}")
            c3.metric("R²",  f"{m_r2:.4f}")

            # Tracés
            st.subheader(f"Comparaison avec vérité terrain — {title_scale}")
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(s_disp, label=f"Entrée s(t) ({title_scale})", alpha=0.6)
            ax.plot(x_ref, label=f"Cible x(t) ({title_scale})", lw=2)
            ax.plot(y_disp, label=f"Sortie ({method_name})", lw=2)
            if base_disp is not None:
                ax.plot(base_disp, label="Baseline estimée", alpha=0.6)
            ax.legend(ncol=2); ax.set_title("Avant/Après & Ground Truth")
            st.pyplot(fig, clear_figure=True)

            with st.expander("Afficher la baseline vraie/estimée"):
                fig2, ax2 = plt.subplots(figsize=(10,3))
                if scale == "RAW":
                    ax2.plot(b_true_raw, label="Baseline vraie b(t)")
                else:
                    b_true_z, _, _ = zscore_per_sample(b_true_raw)
                    ax2.plot(b_true_z, label="Baseline vraie b(t) (z-score)")
                if base_disp is not None:
                    ax2.plot(base_disp, label="Baseline estimée", alpha=0.7)
                ax2.legend(); ax2.set_title("Baseline (vraie vs estimée)")
                st.pyplot(fig2, clear_figure=True)

            # Exports
            # Signal corrigé (échelle choisie)
            out_arr = y_disp.astype(np.float32)
            st.download_button(f"⬇️ Télécharger le signal corrigé ({title_scale}, CSV)",
                               data="\n".join(map(str, out_arr.tolist())).encode("utf-8"),
                               file_name=f"signal_corrige_{'raw' if scale=='RAW' else 'z'}.csv",
                               mime="text/csv")

            # Export des métriques si demandé
            if export_metrics:
                import json, pandas as pd
                metrics = {
                    "method": method_name,
                    "scale": title_scale,
                    "MSE": m_mse,
                    "MAE": m_mae,
                    "R2": m_r2,
                    "L": int(len(s_raw)),
                    "seed": int(seed),
                    "params": {
                        "win": int(win) if method.startswith("SavGol") else None,
                        "poly": int(poly) if method.startswith("SavGol") else None,
                        "lam": float(lam) if method.startswith("AsLS") else None,
                        "p": float(p) if method.startswith("AsLS") else None,
                        "niter": int(niter) if method.startswith("AsLS") else None,
                    }
                }
                # JSON
                st.download_button("Export métriques (JSON)",
                    data=json.dumps(metrics, indent=2).encode("utf-8"),
                    file_name="metrics.json", mime="application/json")
                # CSV (1 ligne)
                dfm = pd.DataFrame([metrics]).drop(columns=["params"])
                st.download_button("Export métriques (CSV)",
                    data=dfm.to_csv(index=False).encode("utf-8"),
                    file_name="metrics.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erreur: {e}")
        st.stop()


