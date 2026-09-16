"""Four candidate models at the release gate. One ships.

    python demo.py

Each candidate fails a different way a deployed model actually breaks:
an undocumented card, training/serving skew, weak discrimination, unfair
outcomes. No model training, no network, no dependencies.
"""
import sys

sys.path.insert(0, "src")

from insurance.governance import ModelCard, ReleaseGate
from insurance.skew import detect_skew


def card(**over) -> ModelCard:
    base = dict(
        name="motor-claims-severity",
        version="4.2.0",
        intended_use="Predict claim severity band at first notification of loss.",
        out_of_scope=["underwriting decisions", "fraud referral"],
        training_data="2019-2024 motor claims, UK book, 412k rows",
        limitations=["not validated on commercial fleet", "no telematics features"],
        owner="claims-analytics",
        reviewed_by="model-risk",
    )
    base.update(over)
    return ModelCard(**base)


# Training rows and serving rows for the skew check. The serving side has drifted:
# vehicle_age arrives in months where training used years.
TRAINING = [{"vehicle_age": 6.0, "claim_count": 1} for _ in range(200)]
SERVING = [{"vehicle_age": 72.0, "claim_count": 1} for _ in range(200)]
skew_clean = detect_skew(TRAINING, TRAINING)
skew_broken = detect_skew(TRAINING, SERVING)

CANDIDATES = [
    ("v4.2.0  documented, reviewed, calibrated", dict(
        card=card(), metrics={"gini": 0.45, "brier": 0.12},
        fairness={"worst_disparate_impact": 0.92}, skew=skew_clean)),
    ("v4.3.0  no owner recorded", dict(
        card=card(version="4.3.0", owner=""), metrics={"gini": 0.45, "brier": 0.12},
        fairness={"worst_disparate_impact": 0.92}, skew=skew_clean)),
    ("v4.4.0  units changed between train and serve", dict(
        card=card(version="4.4.0"), metrics={"gini": 0.45, "brier": 0.12},
        fairness={"worst_disparate_impact": 0.92}, skew=skew_broken)),
    ("v4.5.0  barely separates, and unevenly", dict(
        card=card(version="4.5.0"), metrics={"gini": 0.08, "brier": 0.31},
        fairness={"worst_disparate_impact": 0.61}, skew=skew_clean)),
]

print("INPUT")
for label, _ in CANDIDATES:
    print(f"   {label}")
print()

print("OUTPUT")
gate = ReleaseGate()
for label, kw in CANDIDATES:
    result = gate.evaluate(**kw)
    verdict = "APPROVED" if result["approved"] else "REFUSED"
    print(f"   {label[:44]:44} {verdict}")
    for f in result["failures"]:
        print(f"      - {f}")
print()
print("   The gate refuses. It does not warn and proceed, because a warning")
print("   on a release pipeline is a release.")
