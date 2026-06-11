"""FootyNet — a recurrent late-fusion 1X2 classifier (the deep-learning variant).

Architecture (see ``docs/DEEP_LEARNING_DESIGN.md``, grounded in paper11/paper7/paper3):

* two **shared-weight LSTM encoders** over the home and away teams' last-K match
  sequences (``src.sequence_data``) — many-to-one (paper11);
* a **static MLP branch** over the existing engineered per-fixture features
  (market logits, Elo, Dixon-Coles, understat, lineup) — paper11 future-work / paper1;
* **late fusion** ``concat(h_home, h_away, |h_home - h_away|, h_static)`` → Dense → softmax(3)
  (paper3: late fusion > early fusion).

Training: categorical cross-entropy (= multinomial log loss, the project's primary
metric), early stopping on validation log loss, optional class weights / label
smoothing for the draw imbalance (paper7), and post-hoc **temperature scaling**
(Guo et al. 2017) — the same calibration philosophy as the rest of the pipeline.

Notes/deviations: PyTorch ``nn.LSTM`` exposes only *inter-layer* dropout (Keras-style
recurrent dropout from paper11 is not built in), so recurrent dropout is approximated
by inter-layer dropout (when >1 layer) plus dropout on the sequence embedding.
Sequences are front-padded; the final LSTM hidden state therefore ends on a real
match, and a no-history team is gated to a zero embedding via its mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # torch is an optional (DL-only) dependency; import lazily-friendly.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - surfaced only without torch
    raise ImportError(
        "FootyNet requires PyTorch. Install it with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from src.sequence_data import SEQ_FEATURE_DIM

SEQUENCE_KEYS = ("seq_home", "seq_away", "mask_home", "mask_away")


class FootyNet(nn.Module):
    """Recurrent late-fusion classifier returning 3-class logits."""

    def __init__(
        self,
        static_dim: int,
        seq_dim: int = SEQ_FEATURE_DIM,
        hidden: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        static_hidden: int = 32,
        fusion_hidden: int = 64,
        cell: str = "lstm",
    ):
        super().__init__()
        self.cell = cell
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        # attribute kept named ``lstm`` so saved checkpoints stay compatible across cell types
        self.lstm = rnn_cls(
            seq_dim, hidden, num_layers=lstm_layers, batch_first=True,
            dropout=(dropout if lstm_layers > 1 else 0.0),
        )
        self.seq_dropout = nn.Dropout(dropout)
        self.static = nn.Sequential(nn.Linear(static_dim, static_hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 3 + static_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 3),
        )

    def _encode(self, seq: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        _, hidden = self.lstm(seq)                # LSTM: (h_n, c_n); GRU: h_n
        h_n = hidden[0] if self.cell == "lstm" else hidden
        emb = self.seq_dropout(h_n[-1])           # last layer hidden state -> [N, H]
        has_history = mask.amax(dim=1, keepdim=True)  # [N, 1]: 0 if all padding
        return emb * has_history                  # no-history team -> zero embedding

    def forward(self, seq_home, seq_away, mask_home, mask_away, static):
        h = self._encode(seq_home, mask_home)
        a = self._encode(seq_away, mask_away)
        s = self.static(static)
        fused = torch.cat([h, a, (h - a).abs(), s], dim=1)
        return self.head(fused)                   # logits [N, 3]


@dataclass
class FootyNetResult:
    """Trained FootyNet plus the calibration temperature and the training history."""
    model: FootyNet
    temperature: float
    best_val_logloss: float
    history: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


def _to_tensor(data: dict, device: str) -> dict:
    """Convert a dict of numpy arrays (seq/mask/static[/y]) to float/long tensors."""
    out = {k: torch.as_tensor(np.asarray(data[k]), dtype=torch.float32, device=device)
           for k in (*SEQUENCE_KEYS, "static")}
    if "y" in data:
        out["y"] = torch.as_tensor(np.asarray(data["y"]), dtype=torch.long, device=device)
    return out


def _forward(model: FootyNet, t: dict) -> "torch.Tensor":
    return model(t["seq_home"], t["seq_away"], t["mask_home"], t["mask_away"], t["static"])


def fit_temperature(logits: "torch.Tensor", y: "torch.Tensor") -> float:
    """Single-parameter temperature scaling fit on validation logits (Guo et al. 2017)."""
    T = nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=60)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / T.clamp(min=1e-2), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp(min=1e-2).item())


def predict_proba(model: FootyNet, data: dict, temperature: float = 1.0, device: str = "cpu") -> np.ndarray:
    """Temperature-scaled softmax probabilities [N, 3] for the given inputs."""
    model.eval()
    with torch.no_grad():
        logits = _forward(model, _to_tensor(data, device))
        probs = torch.softmax(logits / max(temperature, 1e-2), dim=1)
    return probs.cpu().numpy()


def train_footynet(
    train_data: dict,
    val_data: dict,
    *,
    hidden: int = 32,
    lstm_layers: int = 1,
    dropout: float = 0.3,
    cell: str = "lstm",
    optimizer: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
    class_weights=None,
    label_smoothing: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> FootyNetResult:
    """Train FootyNet with early stopping on validation log loss; calibrate at the end.

    ``train_data``/``val_data`` are dicts of numpy arrays with keys
    ``seq_home``/``seq_away`` ``[N, K, F]``, ``mask_home``/``mask_away`` ``[N, K]``,
    ``static`` ``[N, D]`` and ``y`` ``[N]`` (0=home, 1=draw, 2=away). Returns the best
    model (lowest val log loss), the fitted temperature, and the per-epoch history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = _to_tensor(train_data, device)
    va = _to_tensor(val_data, device)
    static_dim = tr["static"].shape[1]

    model = FootyNet(static_dim, hidden=hidden, lstm_layers=lstm_layers, dropout=dropout, cell=cell).to(device)
    weight_t = (torch.as_tensor(np.asarray(class_weights), dtype=torch.float32, device=device)
                if class_weights is not None else None)
    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()  # unweighted/unsmoothed = true log loss
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        TensorDataset(tr["seq_home"], tr["seq_away"], tr["mask_home"], tr["mask_away"], tr["static"], tr["y"]),
        batch_size=batch_size, shuffle=True,
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        for sh, sa, mh, ma, st, yy in loader:
            opt.zero_grad()
            logits = model(sh, sa, mh, ma, st)
            loss = criterion(logits, yy)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = _forward(model, va)
            val_loss = float(val_criterion(val_logits, va["y"]).item())
        history.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = _forward(model, va)
    temperature = fit_temperature(val_logits, va["y"])

    return FootyNetResult(
        model=model,
        temperature=temperature,
        best_val_logloss=best_val,
        history=history,
        config={
            "hidden": hidden, "lstm_layers": lstm_layers, "dropout": dropout, "cell": cell,
            "optimizer": optimizer, "lr": lr, "weight_decay": weight_decay,
            "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
            "label_smoothing": label_smoothing, "seed": seed, "static_dim": static_dim,
        },
    )
