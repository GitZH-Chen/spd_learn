"""
.. _liebn-tsmnet:

LieBN with TSMNet on Hinss2021 EEG Dataset
===========================================

This tutorial reproduces the TSMNet experiments from Chen et al., "A Lie
Group Approach to Riemannian Batch Normalization", ICLR 2024
:cite:p:`chen2024liebn`, evaluating LieBN on the Hinss2021 mental workload
EEG dataset.

We compare batch normalization strategies under two evaluation protocols:

- **Inter-session**: Leave-one-session-out within each subject (with UDA)
- **Inter-subject**: Leave-one-subject-out across subjects (with UDA)

Models compared:

- **TSMNet**: No batch normalization
- **TSMNet+SPDDSMBN**: Domain-specific SPD batch normalization
- **TSMNet+LieBN-AIM**: LieBN under the Affine-Invariant Metric
- **TSMNet+LieBN-LEM**: LieBN under the Log-Euclidean Metric
- **TSMNet+LieBN-LCM**: LieBN under the Log-Cholesky Metric

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Setup and Imports
# -----------------
#

import json
import random
import time
import warnings

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from moabb.datasets import Hinss2021
from moabb.paradigms import RestingStateToP300Adapter
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from spd_learn.modules import (
    BiMap,
    CovLayer,
    SPDBatchNormLie,
    LogEig,
    ReEig,
    SPDBatchNormMeanVar,
)


warnings.filterwarnings("ignore")


def set_reproducibility(seed=42):
    """Set random seeds and enable deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


SEED = 42
set_reproducibility(SEED)
RESULTS_PATH = Path("examples/applied_examples/liebn_tsmnet_results.json")


######################################################################
# LieBN Implementation
# --------------------
#
# The reusable LieBN implementation now lives in ``spd_learn.modules`` and is
# imported above to keep this example focused on the TSMNet experiment logic.


######################################################################
# TSMNet with Configurable Batch Normalization
# ---------------------------------------------
#
# We build a TSMNet model that supports no BN, SPDBatchNormMeanVar, or
# SPDBatchNormLie, matching the reference architecture:
#
# ``Conv_temporal -> Conv_spatial -> CovLayer -> BiMap -> ReEig ->
# [BN] -> LogEig -> Linear``
#
# Architecture: temporal_filters=4, spatial_filters=40,
# subspace_dims=20, temp_kernel=25.
#


def make_bn_layer(n, bn_type, bn_kwargs):
    """Create the appropriate batch normalization layer."""
    if bn_type == "SPDBN":
        return SPDBatchNormMeanVar(n, momentum=bn_kwargs.get("momentum", 0.1))
    elif bn_type == "LieBN":
        return SPDBatchNormLie(n, **bn_kwargs)
    else:
        raise ValueError(f"Unknown bn_type: {bn_type}")


class TSMNetLieBN(nn.Module):
    """TSMNet with configurable SPD batch normalization.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_classes : int
        Number of output classes.
    n_temp_filters : int
        Temporal convolution filters.
    n_spatial_filters : int
        Spatial convolution filters.
    n_subspace : int
        BiMap output dimension.
    temp_kernel : int
        Temporal kernel length.
    bn_type : str or None
        None, 'SPDBN', or 'LieBN'.
    bn_kwargs : dict or None
        Keyword arguments for the BN layer.
    """

    def __init__(
        self,
        n_chans,
        n_classes,
        n_temp_filters=4,
        n_spatial_filters=40,
        n_subspace=20,
        temp_kernel=25,
        bn_type=None,
        bn_kwargs=None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.bn_type = bn_type

        n_tangent = int(n_subspace * (n_subspace + 1) / 2)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1,
                n_temp_filters,
                kernel_size=(1, temp_kernel),
                padding="same",
                padding_mode="reflect",
            ),
            nn.Conv2d(n_temp_filters, n_spatial_filters, (n_chans, 1)),
            nn.Flatten(start_dim=2),
        )
        self.covpool = CovLayer()
        self.spdnet = nn.Sequential(
            BiMap(n_spatial_filters, n_subspace),
            ReEig(threshold=1e-4),
        )

        self.spdbn = None
        if bn_type is not None:
            self.spdbn = make_bn_layer(n_subspace, bn_type, bn_kwargs or {})

        self.logeig = nn.Sequential(
            LogEig(upper=True, flatten=True),
        )
        self.classifier = nn.Linear(n_tangent, n_classes)

    def forward(self, x):
        # x: (batch, n_chans, n_times)
        h = self.cnn(x[:, None, ...])  # add channel dim for Conv2d
        C = self.covpool(h)
        S = self.spdnet(C)
        if self.spdbn is not None:
            S = self.spdbn(S)
        z = self.logeig(S)
        return self.classifier(z)


######################################################################
# Dataset Loading: Hinss2021
# --------------------------
#
# The Hinss2021 dataset contains EEG recordings of 15 subjects
# performing mental workload tasks at 3 difficulty levels (easy,
# medium, difficult) across 2 sessions each.
#
# - **15 subjects**, **2 sessions** per subject
# - **3 classes**: easy, medium, difficult
# - **30 EEG channels** (frontal + parietal selection)
# - **Bandpass**: 4--36 Hz
# - **Epoch**: 0--2 seconds post-cue
#
# Data is downloaded automatically via MOABB.
#

CHANNELS = [
    "Fp1",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "F2",
    "F4",
    "F6",
    "F8",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C3",
    "C4",
    "CPz",
    "PO3",
    "PO4",
    "POz",
    "Oz",
    "Fz",
]

print("Loading Hinss2021 dataset via MOABB...")
print("(First run will download ~2GB of data)")

dataset = Hinss2021()
paradigm = RestingStateToP300Adapter(
    fmin=4,
    fmax=36,
    events=["easy", "medium", "diff"],
    tmin=0,
    tmax=2,
    channels=CHANNELS,
    resample=250,
)

le = LabelEncoder()

# Load data for all subjects
all_data = {}
for subj in dataset.subject_list:
    X_subj, labels_subj, meta_subj = paradigm.get_data(dataset=dataset, subjects=[subj])
    y_subj = le.fit_transform(labels_subj)
    sessions = meta_subj["session"].values
    all_data[subj] = {
        "X": torch.tensor(X_subj, dtype=torch.float32),
        "y": torch.tensor(y_subj, dtype=torch.long),
        "sessions": sessions,
    }
    print(
        f"  Subject {subj:2d}: {X_subj.shape[0]} trials, "
        f"shape={X_subj.shape[1:]}, sessions={sorted(set(sessions))}"
    )

n_chans = all_data[1]["X"].shape[1]
n_classes = len(le.classes_)
print(f"\nn_chans={n_chans}, n_classes={n_classes}, classes={le.classes_}")


######################################################################
# Training & Evaluation Utilities
# --------------------------------
#
# We match the reference protocol:
#
# - **Optimizer**: ``geoopt.RiemannianAdam`` (amsgrad, lr=1e-3, wd=1e-4)
# - **Epochs**: 50
# - **Batch size**: 50
# - **Score**: balanced accuracy
# - **UDA**: Forward pass on target domain to refit BN running stats
#


def train_model(model, train_loader, optimizer, criterion, epochs=50):
    """Train the model and return epoch times."""
    epoch_times = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
        epoch_times.append(time.time() - t0)
    return epoch_times


def adapt_bn(model, X_target, batch_size=50):
    """Unsupervised domain adaptation: refit BN stats on target data.

    Passes the target domain data through the model with BN in train
    mode (updating running stats), then sets it back to eval. This
    matches the reference REFIT adaptation strategy.
    """
    # Find BN layers
    bn_layers = []
    for module in model.modules():
        if isinstance(module, (SPDBatchNormMeanVar, SPDBatchNormLie)):
            bn_layers.append(module)

    if not bn_layers:
        return

    model.eval()
    # Reset running stats and put BN in train mode
    for layer in bn_layers:
        if isinstance(layer, SPDBatchNormMeanVar):
            layer.reset_running_stats()
        elif isinstance(layer, SPDBatchNormLie):
            if layer.metric == "AIM":
                layer.running_mean.copy_(torch.eye(layer.n).unsqueeze(0))
            else:
                layer.running_mean.zero_()
            layer.running_var.fill_(1.0)
        layer.train()

    # Forward pass to compute target-specific stats
    loader = DataLoader(
        TensorDataset(X_target, torch.zeros(len(X_target), dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for X_batch, _ in loader:
            _ = model(X_batch)

    # Set back to eval
    for layer in bn_layers:
        layer.eval()
    model.eval()


def evaluate(model, X, y, batch_size=50):
    """Compute balanced accuracy on the given data."""
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    return balanced_accuracy_score(y_true, y_pred)


######################################################################
# Experiment Runner
# -----------------
#
# Runs a single experiment configuration across all folds of a given
# evaluation protocol (inter-session or inter-subject).
#


def run_tsmnet_experiment(
    all_data,
    n_chans,
    n_classes,
    protocol="inter-session",
    bn_type=None,
    bn_kwargs=None,
    epochs=50,
    batch_size=50,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    verbose=True,
):
    """Run TSMNet experiment under the specified evaluation protocol.

    Parameters
    ----------
    all_data : dict
        Per-subject data: {subj: {X, y, sessions}}.
    protocol : str
        'inter-session' or 'inter-subject'.
    bn_type : str or None
        None, 'SPDBN', or 'LieBN'.
    bn_kwargs : dict or None
        BN layer configuration.

    Returns
    -------
    dict
        Results with mean, std, max, scores, fit_time.
    """
    import geoopt

    scores = []
    fit_times = []
    subjects = sorted(all_data.keys())

    if protocol == "inter-session":
        # For each subject, train on one session, adapt + test on the other
        for subj in subjects:
            data = all_data[subj]
            X, y = data["X"], data["y"]
            sessions = data["sessions"]
            unique_sessions = sorted(set(sessions))

            for test_session in unique_sessions:
                test_mask = sessions == test_session
                train_mask = ~test_mask

                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]

                run_seed = seed + subj * 100 + int(test_session)
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)

                model = TSMNetLieBN(
                    n_chans,
                    n_classes,
                    bn_type=bn_type,
                    bn_kwargs=bn_kwargs,
                )
                optimizer = geoopt.optim.RiemannianAdam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    amsgrad=True,
                )
                criterion = nn.CrossEntropyLoss()

                train_loader = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=batch_size,
                    shuffle=True,
                )

                epoch_times = train_model(
                    model, train_loader, optimizer, criterion, epochs
                )

                # UDA: adapt BN stats to target session
                if bn_type is not None:
                    adapt_bn(model, X_test, batch_size)

                score = evaluate(model, X_test, y_test, batch_size)
                scores.append(score)
                fit_time = np.mean(epoch_times[-10:])
                fit_times.append(fit_time)

                if verbose:
                    print(
                        f"  S{subj:02d} session={test_session}: "
                        f"bacc={score:.4f}, fit_time={fit_time:.2f}s"
                    )

    elif protocol == "inter-subject":
        # Leave-one-subject-out
        for test_subj in subjects:
            train_subjects = [s for s in subjects if s != test_subj]

            X_train = torch.cat([all_data[s]["X"] for s in train_subjects])
            y_train = torch.cat([all_data[s]["y"] for s in train_subjects])
            X_test = all_data[test_subj]["X"]
            y_test = all_data[test_subj]["y"]

            run_seed = seed + test_subj
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)

            model = TSMNetLieBN(
                n_chans,
                n_classes,
                bn_type=bn_type,
                bn_kwargs=bn_kwargs,
            )
            optimizer = geoopt.optim.RiemannianAdam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=True,
            )
            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
            )

            epoch_times = train_model(model, train_loader, optimizer, criterion, epochs)

            # UDA: adapt BN stats to target subject
            if bn_type is not None:
                adapt_bn(model, X_test, batch_size)

            score = evaluate(model, X_test, y_test, batch_size)
            scores.append(score)
            fit_time = np.mean(epoch_times[-10:])
            fit_times.append(fit_time)

            if verbose:
                print(
                    f"  Leave-out S{test_subj:02d}: "
                    f"bacc={score:.4f}, fit_time={fit_time:.2f}s"
                )

    mean_score = np.mean(scores) * 100
    std_score = np.std(scores) * 100
    max_score = np.max(scores) * 100
    mean_fit_time = np.mean(fit_times)

    if verbose:
        print(
            f"  => {mean_score:.2f} +/- {std_score:.2f} "
            f"(max={max_score:.2f}, fit_time={mean_fit_time:.2f}s)"
        )

    return {
        "mean": mean_score,
        "std": std_score,
        "max": max_score,
        "scores": [s * 100 for s in scores],
        "fit_time": mean_fit_time,
    }


######################################################################
# Model Configurations
# --------------------
#
# We test the same configurations as the reference experiments:
#
# - TSMNet (no BN)
# - TSMNet + SPDDSMBN
# - TSMNet + LieBN-AIM (theta=1)
# - TSMNet + LieBN-LEM (theta=1)
# - TSMNet + LieBN-LCM (theta=1)
#
# Additional deformed metrics for specific protocols:
#
# - TSMNet + LieBN-LCM (theta=0.5) for inter-session
# - TSMNet + LieBN-AIM (theta=-0.5) for inter-subject
#

configs = {
    "TSMNet": {"bn_type": None, "bn_kwargs": None},
    "SPDDSMBN": {"bn_type": "SPDBN", "bn_kwargs": {"momentum": 0.1}},
    "AIM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LEM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LEM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
    "LCM-(1)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 1.0,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
}

# Additional configs per protocol (from experiments_Hinss21.sh)
inter_session_extra = {
    "LCM-(0.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "LCM",
            "theta": 0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
}

inter_subject_extra = {
    "AIM-(-0.5)": {
        "bn_type": "LieBN",
        "bn_kwargs": {
            "metric": "AIM",
            "theta": -0.5,
            "alpha": 1.0,
            "beta": 0.0,
            "momentum": 0.1,
        },
    },
}


######################################################################
# Inter-Session Experiments
# -------------------------
#
# Leave-one-session-out within each subject, with UDA adaptation of
# BN statistics on the target session.
#

print("\n" + "=" * 60)
print("INTER-SESSION EVALUATION (with UDA)")
print("=" * 60)

inter_session_configs = {**configs, **inter_session_extra}
inter_session_results = {}

for name, cfg in inter_session_configs.items():
    print(f"\n--- {name} ---")
    inter_session_results[name] = run_tsmnet_experiment(
        all_data,
        n_chans,
        n_classes,
        protocol="inter-session",
        bn_type=cfg["bn_type"],
        bn_kwargs=cfg["bn_kwargs"],
        epochs=50,
        batch_size=50,
        lr=1e-3,
    )


######################################################################
# Inter-Subject Experiments
# -------------------------
#
# Leave-one-subject-out across all subjects, with UDA adaptation of
# BN statistics on the target subject.
#

print("\n" + "=" * 60)
print("INTER-SUBJECT EVALUATION (with UDA)")
print("=" * 60)

inter_subject_configs = {**configs, **inter_subject_extra}
inter_subject_results = {}

for name, cfg in inter_subject_configs.items():
    print(f"\n--- {name} ---")
    inter_subject_results[name] = run_tsmnet_experiment(
        all_data,
        n_chans,
        n_classes,
        protocol="inter-subject",
        bn_type=cfg["bn_type"],
        bn_kwargs=cfg["bn_kwargs"],
        epochs=50,
        batch_size=50,
        lr=1e-3,
    )


######################################################################
# Save Results
# ------------
#

saved = {
    "inter_session": {
        name: {k: v for k, v in res.items() if k != "scores"}
        for name, res in inter_session_results.items()
    },
    "inter_subject": {
        name: {k: v for k, v in res.items() if k != "scores"}
        for name, res in inter_subject_results.items()
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(saved, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")


######################################################################
# Results Table
# -------------
#


def _print_results(title, results):
    """Print a results comparison table."""
    methods = list(results.keys())
    hdr = f"{'Method':<14} | {'Fit Time':>8} | {'Mean+-STD':>14} {'Max':>8}"
    sep = "=" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print(sep)
    for m in methods:
        r = results[m]
        ft = f"{r['fit_time']:.2f}"
        m_str = f"{r['mean']:.2f}+-{r['std']:.2f}"
        m_max = f"{r['max']:.2f}"
        print(f"{m:<14} | {ft:>8} | {m_str:>14} {m_max:>8}")
    print(sep)


_print_results("Inter-Session (balanced accuracy %)", inter_session_results)
_print_results("Inter-Subject (balanced accuracy %)", inter_subject_results)


######################################################################
# Visualization
# -------------
#

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, (title, results) in zip(
    axes,
    [
        ("Inter-Session", inter_session_results),
        ("Inter-Subject", inter_subject_results),
    ],
):
    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    stds = [results[m]["std"] for m in methods]
    x_pos = np.arange(len(methods))

    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=3,
        color="#3498db",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(
        y=100.0 / n_classes,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Chance ({100.0 / n_classes:.0f}%)",
    )
    ax.legend(loc="lower right")
    ymin = min(means) - max(stds) - 5
    ymax = max(means) + max(stds) + 5
    ax.set_ylim(max(0, ymin), min(100, ymax))

plt.suptitle(
    "LieBN + TSMNet on Hinss2021: Inter-Session vs Inter-Subject",
    fontweight="bold",
)
plt.tight_layout()
plt.show()


######################################################################
# Notes
# -----
#
# **Protocol details:**
#
# - Inter-session: For each subject, leave-one-session-out CV (2 folds
#   per subject, 30 folds total). UDA refits BN on target session.
# - Inter-subject: Leave-one-subject-out CV (15 folds). UDA refits
#   BN on target subject.
# - Score: balanced accuracy (sklearn ``balanced_accuracy_score``).
#
# **Differences from the reference implementation:**
#
# - **Domain-specific BN**: The reference uses per-domain running
#   statistics (separate stats per session/subject). Our simplified
#   version uses global running stats during training, then refits on
#   the target domain during UDA.
# - **Channels**: We use 30 channels matching the reference selection
#   (with ``Fz`` replacing unavailable ``FPz``).
# - **Data loader**: We use standard PyTorch data loading rather than
#   the reference's ``StratifiedDomainDataLoader``.
# - **Momentum scheduling**: The reference uses
#   ``MomentumBatchNormScheduler`` to decay BN momentum during
#   training. We use fixed momentum.
#
