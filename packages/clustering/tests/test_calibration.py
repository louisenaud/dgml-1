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

"""Tests for confidence calibration (§4.4): temperature / Platt / conformal."""

from __future__ import annotations

import torch
from clustering.calibration import (
    Calibrator,
    fit_calibrator,
    support_loo_logits,
)
from clustering.config.schema import ManifoldConfig
from clustering.manifolds import build_manifold


def _separable_logits(
    n_per: int = 30, k: int = 3, sep: float = 6.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confident, mostly-correct logits: each class sits ``sep`` above the rest."""
    g = torch.Generator().manual_seed(0)
    rows = []
    labels = []
    for c in range(k):
        base = torch.zeros(n_per, k)
        base[:, c] = sep
        base = base + 0.5 * torch.randn(n_per, k, generator=g)
        rows.append(base)
        labels.extend([c] * n_per)
    return torch.cat(rows, dim=0), torch.tensor(labels, dtype=torch.long)


def test_identity_calibrator_returns_ordinal_peak() -> None:
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.9]])
    cal_conf, abstain = Calibrator().apply(logits)
    # method="none": calibrated == softmax peak, no abstention.
    expected = torch.softmax(logits, dim=-1).amax(dim=-1)
    assert torch.allclose(cal_conf, expected, atol=1e-6)
    assert not bool(abstain.any())


def test_temperature_fit_is_positive_and_in_range() -> None:
    logits, labels = _separable_logits()
    cal = fit_calibrator(logits, labels, method="temperature")
    assert cal.method == "temperature"
    assert 1e-2 <= cal.temperature <= 1e2
    assert cal.n_calibration == logits.shape[0]


def test_calibrated_confidence_in_unit_interval() -> None:
    logits, labels = _separable_logits()
    for method in ("temperature", "platt"):
        cal = fit_calibrator(logits, labels, method=method)
        conf, _ = cal.apply(logits)
        assert float(conf.min()) >= 0.0
        assert float(conf.max()) <= 1.0


def test_conformal_gate_abstains_on_uncertain_rows() -> None:
    logits, labels = _separable_logits()
    cal = fit_calibrator(logits, labels, method="temperature", coverage=0.9)
    assert cal.conformal_threshold is not None
    # A confident row (clear winner) is kept; a near-uniform row abstains.
    confident = torch.tensor([[8.0, 0.0, 0.0]])
    uncertain = torch.tensor([[0.1, 0.0, 0.05]])
    _, keep = cal.apply(confident)
    _, drop = cal.apply(uncertain)
    assert not bool(keep.any())
    assert bool(drop.any())


def test_abstain_threshold_floor() -> None:
    # No parametric calibration, just an absolute floor on the ordinal peak.
    cal = fit_calibrator(*_separable_logits(), method="none", abstain_threshold=0.99)
    # method falls back to "none" (identity), but the floor still applies.
    low = torch.tensor([[0.2, 0.1, 0.0]])  # peak well under 0.99
    _, abstain = Calibrator(abstain_threshold=0.99).apply(low)
    assert bool(abstain.any())
    # The fitted calibrator carries the floor through fit_calibrator too.
    assert cal.method == "none"


def test_empty_batch_returns_empty() -> None:
    conf, abstain = Calibrator(method="temperature", temperature=2.0).apply(torch.zeros((0, 3)))
    assert conf.shape == (0,)
    assert abstain.shape == (0,)


def test_too_few_points_degrades_to_none() -> None:
    cal = fit_calibrator(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), method="temperature")
    assert cal.method == "none"


def test_support_loo_logits_shapes() -> None:
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=4, curvature=0.0))
    g = torch.Generator().manual_seed(1)
    cats = ["A", "B"]
    # 3 samples per class, well separated.
    emb = torch.cat(
        [
            torch.zeros(3, 4) + 0.01 * torch.randn(3, 4, generator=g),
            torch.ones(3, 4) * 5 + 0.01 * torch.randn(3, 4, generator=g),
        ],
        dim=0,
    )
    labels: list[str | None] = ["A", "A", "A", "B", "B", "B"]
    out = support_loo_logits(emb, labels, cats, manifold)
    assert out is not None
    logits, loo_labels = out
    assert logits.shape == (6, 2)  # one LOO row per support doc, K=2 columns
    assert loo_labels.shape == (6,)
    # LOO calibration then fits without error and yields a real temperature.
    cal = fit_calibrator(logits, loo_labels, method="temperature", coverage=0.8)
    assert cal.method == "temperature"
    assert cal.conformal_threshold is not None


def test_support_loo_singletons_return_none() -> None:
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=4, curvature=0.0))
    # One sample per class ⇒ no leave-one-out prototype possible.
    emb = torch.tensor([[0.0, 0, 0, 0], [5.0, 5, 5, 5]])
    assert support_loo_logits(emb, ["A", "B"], ["A", "B"], manifold) is None
