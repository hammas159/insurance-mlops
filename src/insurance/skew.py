"""Training/serving skew detection.

Distinct from drift, and the distinction matters because the fixes are different.

**Drift** is the world changing. The model was right and reality moved; you retrain.
**Skew** is the two code paths disagreeing. Training computed `age` in years and
serving computes it in months, or training saw a cleaned column and serving sees the
raw one. The model was never right in production and retraining will not help.

Skew is the more common failure and the harder one to see, because every individual
component passes its own tests. It only appears when the two pipelines are compared
directly — which is what this module does.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SkewFinding:
    feature: str
    kind: str
    detail: str
    severity: str  # "critical" | "warning"


def _numeric(values: Sequence) -> list[float]:
    return [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]


def compare_feature(
    name: str, training: Sequence, serving: Sequence, *, tolerance: float = 0.25
) -> list[SkewFinding]:
    findings: list[SkewFinding] = []

    train_missing = sum(1 for v in training if v is None) / len(training) if training else 0.0
    serve_missing = sum(1 for v in serving if v is None) / len(serving) if serving else 0.0
    if abs(train_missing - serve_missing) > 0.1:
        # Usually a join that silently became an inner join on one side.
        findings.append(
            SkewFinding(
                name,
                "missingness",
                f"missing {train_missing:.1%} in training vs {serve_missing:.1%} in serving",
                "critical",
            )
        )

    train_values = _numeric(training)
    serve_values = _numeric(serving)

    train_types = {type(v).__name__ for v in training if v is not None}
    serve_types = {type(v).__name__ for v in serving if v is not None}
    if train_types and serve_types and train_types != serve_types:
        findings.append(
            SkewFinding(
                name,
                "type",
                f"training {sorted(train_types)} vs serving {sorted(serve_types)}",
                "critical",
            )
        )

    if not train_values or not serve_values:
        return findings

    train_mean = statistics.fmean(train_values)
    serve_mean = statistics.fmean(serve_values)

    # A unit change moves the mean by a constant factor while the shape is unchanged.
    # Worth naming separately because it is the single most common skew and the
    # easiest to fix once someone says the word "units".
    if train_mean and serve_mean:
        ratio = serve_mean / train_mean
        for factor, unit in (
            (12, "years vs months"),
            (1 / 12, "months vs years"),
            (1000, "units vs thousands"),
            (1 / 1000, "thousands vs units"),
            (100, "fraction vs percent"),
            (1 / 100, "percent vs fraction"),
        ):
            if math.isclose(ratio, factor, rel_tol=0.05):
                findings.append(
                    SkewFinding(
                        name,
                        "unit",
                        f"serving is {ratio:.3g}x training - looks like {unit}",
                        "critical",
                    )
                )
                break

    if train_mean and abs(serve_mean - train_mean) / abs(train_mean) > tolerance:
        findings.append(
            SkewFinding(
                name,
                "distribution",
                f"mean {train_mean:.4g} in training vs {serve_mean:.4g} in serving",
                "warning",
            )
        )

    train_range = (min(train_values), max(train_values))
    out_of_range = sum(1 for v in serve_values if not train_range[0] <= v <= train_range[1])
    if out_of_range / len(serve_values) > 0.05:
        # The model is extrapolating, which for tree models means predicting from a
        # leaf that was fitted on nothing like this.
        findings.append(
            SkewFinding(
                name,
                "range",
                f"{out_of_range / len(serve_values):.1%} of serving values fall outside the "
                f"training range {train_range[0]:.4g}-{train_range[1]:.4g}",
                "warning",
            )
        )

    return findings


def detect_skew(training: list[dict], serving: list[dict], *, tolerance: float = 0.25) -> dict:
    """Compare two batches of feature vectors, feature by feature."""
    train_features = {k for row in training for k in row}
    serve_features = {k for row in serving for k in row}

    findings: list[SkewFinding] = []

    for name in sorted(train_features - serve_features):
        findings.append(
            SkewFinding(
                name, "missing_feature", "present in training, absent in serving", "critical"
            )
        )
    for name in sorted(serve_features - train_features):
        findings.append(
            SkewFinding(name, "extra_feature", "present in serving, absent in training", "critical")
        )

    for name in sorted(train_features & serve_features):
        findings.extend(
            compare_feature(
                name,
                [row.get(name) for row in training],
                [row.get(name) for row in serving],
                tolerance=tolerance,
            )
        )

    critical = [f for f in findings if f.severity == "critical"]
    return {
        "findings": [f.__dict__ for f in findings],
        "critical": len(critical),
        "warnings": len(findings) - len(critical),
        # A blocking verdict, not a dashboard tile. Skew is not something to watch
        # trend upward - it means the two pipelines disagree today.
        "safe_to_serve": not critical,
    }
