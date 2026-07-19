# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Confidence calibration for nearest-prototype assignment.

The raw nearest-prototype signal (peak of ``softmax(-distance)``) is a fine
*ordinal* score — bigger means more confident — but it is not a probability:
softmax over manifold distances is arbitrarily peaked depending on the
distance scale, so "0.9" from one run does not mean the same thing as "0.9"
from another, and it certainly does not mean "90% of such assignments are
correct." In a regulated / financial setting where a misfiled document has
real consequences, an assignment needs a *calibrated* score and a principled
*abstain* decision.

This module provides three standard, dependency-light pieces:

- **Temperature scaling** — a single scalar ``T`` fit by minimizing negative
  log-likelihood of ``softmax(logits / T)`` against held-out labels. The
  cheapest, most robust multiclass post-hoc calibrator (Guo et al. 2017).
- **Platt scaling** — a 1-D logistic map ``sigmoid(a·s + b)`` fit on the top-1
  score ``s`` versus assignment correctness. Recalibrates the reported
  confidence number directly.
- **Conformal abstention** — a distribution-free split-conformal threshold on
  the nonconformity score ``1 - p(true)``. Given a target ``coverage`` it
  yields ``q̂`` such that routing every document with ``1 - p_top1 > q̂`` to a
  human review queue gives a finite-sample coverage guarantee on the kept set,
  with no distributional assumptions.

Everything here is fit on **labeled** data. Only the incremental few-shot /
supervised scenarios (S3 / S5) carry labels (their per-DocSet support
members), so :func:`support_loo_logits` builds an honest *leave-one-out*
calibration set from the support prototypes. S1 / S2 have no labels; they keep
the ordinal signal and can still abstain via a plain confidence floor.

No LLM dependency lives here — this is pure numpy / scipy / torch, so it stays
inside the Apache-2.0 framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from clustering.manifolds.base import ManifoldHead

CalibrationMethod = Literal["none", "temperature", "platt"]

# Bounds for the temperature search. T -> 0 sharpens toward argmax; T large
# flattens toward uniform. The optimum for well-behaved logits is near 1.
_T_MIN = 1e-2
_T_MAX = 1e2


def _softmax_np(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Row-wise numerically-stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast("np.ndarray[Any, Any]", exp / exp.sum(axis=-1, keepdims=True))


def _nll(temperature: float, logits: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Mean negative log-likelihood of ``softmax(logits / T)`` under ``labels``."""
    probs = _softmax_np(logits / max(temperature, _T_MIN))
    n = logits.shape[0]
    true = probs[np.arange(n), labels]
    return float(-np.log(np.clip(true, 1e-12, 1.0)).mean())


def _fit_temperature(logits: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Fit the temperature that minimizes NLL via bounded scalar search."""
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:  # pragma: no cover — scipy is a core dep
        return 1.0
    res = minimize_scalar(_nll, args=(logits, labels), bounds=(_T_MIN, _T_MAX), method="bounded")
    t = float(getattr(res, "x", 1.0))
    return float(np.clip(t, _T_MIN, _T_MAX))


def _fit_platt(scores: np.ndarray[Any, Any], correct: np.ndarray[Any, Any]) -> tuple[float, float]:
    """Fit ``sigmoid(a·score + b) ≈ P(correct)`` via 1-D logistic regression.

    Degenerate calibration sets — all-correct or all-wrong — have no gradient
    for a logistic fit, so we fall back to the identity map ``(1, 0)`` and let
    temperature / conformal carry the signal instead.
    """
    if len({int(c) for c in correct.tolist()}) < 2:
        return 1.0, 0.0
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:  # pragma: no cover — scikit-learn is a core dep
        return 1.0, 0.0
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(scores.reshape(-1, 1), correct.astype(np.int64))
    a = float(model.coef_[0][0])
    b = float(model.intercept_[0])
    return a, b


def _conformal_threshold(nonconformity: np.ndarray[Any, Any], coverage: float) -> float:
    """Split-conformal quantile ``q̂`` for the target ``coverage`` in (0, 1).

    Keeping every point whose nonconformity ``≤ q̂`` gives ``≥ coverage``
    marginal coverage in finite samples (Vovk et al.). We use the standard
    conservative rank ``⌈(n + 1)·coverage⌉ / n``.
    """
    n = int(nonconformity.shape[0])
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(nonconformity, level, method="higher"))


@dataclass(frozen=True)
class Calibrator:
    """A fitted confidence calibrator + optional conformal abstain gate.

    Immutable and cheap to carry on :class:`~clustering.scenarios.base.ScenarioResult`
    metadata (see :meth:`as_dict`). Apply it to a batch of assignment logits
    (``-distance`` to each prototype) with :meth:`apply`.
    """

    method: CalibrationMethod = "none"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    conformal_threshold: float | None = None
    coverage: float | None = None
    abstain_threshold: float | None = None
    n_calibration: int = 0

    def _tempered_softmax(self, logits_np: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        t = self.temperature if self.method == "temperature" else 1.0
        return _softmax_np(logits_np / max(t, _T_MIN))

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(calibrated_confidence, abstain)`` for a ``[N, K]`` logit batch.

        ``calibrated_confidence`` is in ``[0, 1]``; ``abstain`` is a boolean
        ``[N]`` tensor flagging documents whose confidence is below the
        conformal / floor gate and should be routed to human review. An empty
        batch returns empty tensors.
        """
        n = int(logits.shape[0])
        if n == 0:
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

        logits_np = (
            logits.detach().cpu().float().numpy()
            if hasattr(logits, "detach")
            else np.asarray(logits, dtype=np.float64)
        )
        probs = self._tempered_softmax(logits_np)
        top1 = probs.max(axis=-1)

        if self.method == "platt":
            cal = 1.0 / (1.0 + np.exp(-(self.platt_a * top1 + self.platt_b)))
        else:
            cal = top1
        cal = np.clip(cal, 0.0, 1.0)

        abstain = np.zeros(n, dtype=bool)
        if self.conformal_threshold is not None:
            abstain |= (1.0 - top1) > self.conformal_threshold
        if self.abstain_threshold is not None:
            abstain |= cal < self.abstain_threshold

        return (
            torch.as_tensor(cal, dtype=torch.float32),
            torch.as_tensor(abstain, dtype=torch.bool),
        )

    def as_dict(self) -> dict[str, Any]:
        """Provenance dict for :class:`ScenarioResult` metadata / attestation."""
        return {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "conformal_threshold": self.conformal_threshold,
            "coverage": self.coverage,
            "abstain_threshold": self.abstain_threshold,
            "n_calibration": self.n_calibration,
        }


def fit_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    method: CalibrationMethod = "temperature",
    coverage: float | None = None,
    abstain_threshold: float | None = None,
) -> Calibrator:
    """Fit a :class:`Calibrator` from labeled assignment logits.

    Args:
        logits: ``[M, K]`` assignment logits (``-distance`` to each of the
            ``K`` prototypes) for ``M`` labeled calibration documents.
        labels: ``[M]`` integer class indices in ``[0, K)``.
        method: ``"temperature"`` (default), ``"platt"``, or ``"none"``.
        coverage: If set in ``(0, 1)``, also fit a split-conformal abstain
            threshold guaranteeing (marginally) that fraction of coverage on
            the kept set. ``None`` disables conformal abstention.
        abstain_threshold: Optional absolute calibrated-confidence floor;
            documents below it abstain regardless of conformal coverage.

    Returns:
        A fitted :class:`Calibrator`. With fewer than two calibration points,
        or an unknown method, returns an identity (``method="none"``)
        calibrator so callers degrade to the ordinal signal rather than fail.
    """
    logits_np = (
        logits.detach().cpu().float().numpy()
        if hasattr(logits, "detach")
        else np.asarray(logits, dtype=np.float64)
    )
    labels_np = (
        labels.detach().cpu().numpy().astype(np.int64)
        if hasattr(labels, "detach")
        else np.asarray(labels, dtype=np.int64)
    )
    m = int(logits_np.shape[0])
    if m < 2 or method not in ("none", "temperature", "platt"):
        return Calibrator(method="none", n_calibration=m)

    temperature = _fit_temperature(logits_np, labels_np) if method == "temperature" else 1.0

    platt_a, platt_b = 1.0, 0.0
    if method == "platt":
        probs = _softmax_np(logits_np)
        top1 = probs.max(axis=-1)
        pred = probs.argmax(axis=-1)
        correct = (pred == labels_np).astype(np.int64)
        platt_a, platt_b = _fit_platt(top1, correct)

    conformal_threshold: float | None = None
    if coverage is not None:
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1); got {coverage!r}.")
        t = temperature if method == "temperature" else 1.0
        cal_probs = _softmax_np(logits_np / max(t, _T_MIN))
        true_prob = cal_probs[np.arange(m), labels_np]
        conformal_threshold = _conformal_threshold(1.0 - true_prob, coverage)

    return Calibrator(
        method=method,
        temperature=temperature,
        platt_a=platt_a,
        platt_b=platt_b,
        conformal_threshold=conformal_threshold,
        coverage=coverage,
        abstain_threshold=abstain_threshold,
        n_calibration=m,
    )


def support_loo_logits(
    support_embeddings: torch.Tensor,
    support_labels: list[str | None],
    categories: list[str],
    manifold: ManifoldHead,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build leave-one-out calibration logits from a labeled support set.

    For each support document we recompute its own category's prototype with
    that document *excluded* (the manifold-mean of the remaining same-class
    samples), then take ``-distance`` to every category prototype. This is an
    honest, non-optimistic calibration set: the document never sees itself in
    its own prototype, so the fitted temperature / conformal threshold reflect
    generalization rather than memorization.

    Categories with only a single support sample contribute no LOO row (there
    is nothing left to form the leave-one-out prototype), and are simply
    skipped. Returns ``None`` when fewer than two usable rows remain — too
    little to calibrate — so the caller keeps the ordinal signal.

    Args:
        support_embeddings: ``[M, D]`` on-manifold support embeddings.
        support_labels: Length-``M`` labels aligned with the embeddings.
        categories: Ordered category names; column ``j`` of the returned
            logits corresponds to ``categories[j]``.
        manifold: Active manifold head (``expmap0`` + ``pairwise_dist``).

    Returns:
        ``(logits[M', K], labels[M'])`` or ``None``.
    """
    cat_index = {c: j for j, c in enumerate(categories)}
    by_cat: dict[str, list[int]] = {c: [] for c in categories}
    for i, lbl in enumerate(support_labels):
        if lbl in by_cat:
            by_cat[lbl].append(i)

    rows: list[torch.Tensor] = []
    row_labels: list[int] = []
    for cat, idxs in by_cat.items():
        if len(idxs) < 2:
            # No leave-one-out prototype possible for a singleton class.
            continue
        for held_out in idxs:
            protos: list[torch.Tensor] = []
            for other in categories:
                members = [k for k in by_cat[other] if not (other == cat and k == held_out)]
                if not members:
                    # The held-out class collapsed (shouldn't happen: len>=2),
                    # or an empty class — fall back to any available member.
                    members = by_cat[other] or [held_out]
                ambient_mean = support_embeddings[torch.tensor(members)].mean(dim=0)
                protos.append(manifold.expmap0(ambient_mean.unsqueeze(0)).squeeze(0))
            proto_stack = torch.stack(protos, dim=0)  # [K, D]
            query = support_embeddings[held_out].unsqueeze(0)  # [1, D]
            dist = manifold.pairwise_dist(query, proto_stack).squeeze(0)  # [K]
            rows.append(-dist)
            row_labels.append(cat_index[cat])

    if len(rows) < 2:
        return None
    return torch.stack(rows, dim=0), torch.tensor(row_labels, dtype=torch.long)


def fit_support_calibrator(
    support_embeddings: torch.Tensor,
    support_labels: list[str | None],
    categories: list[str],
    manifold: ManifoldHead,
    *,
    method: CalibrationMethod,
    coverage: float | None,
    abstain_threshold: float | None,
) -> Calibrator | None:
    """Fit a calibrator from a labeled support set, or ``None`` if not possible.

    Convenience wrapper the labeled scenarios (S3 / S5) call: skips fitting
    entirely when no parametric calibration and no conformal coverage is
    requested, and returns ``None`` when the support set is too small to build
    a leave-one-out calibration set — in both cases the caller keeps the
    ordinal confidence (and may still apply ``abstain_threshold`` as a plain
    floor downstream).
    """
    if method == "none" and coverage is None:
        return None
    loo = support_loo_logits(support_embeddings, support_labels, categories, manifold)
    if loo is None:
        return None
    logits, labels = loo
    return fit_calibrator(
        logits,
        labels,
        method=method,
        coverage=coverage,
        abstain_threshold=abstain_threshold,
    )
