"""Governance: model cards, consent, and the release gate.

Three obligations that are usually documentation and are treated here as code, because
documentation drifts from reality and code does not.

**Model cards** describe what a model is for and — more usefully — what it is *not*
for. The out-of-scope section is the one that prevents a claims-triage model being
quietly repurposed for pricing.

**Consent and purpose limitation.** Pakistan's PDPA, like GDPR, ties personal data to
the purpose it was collected for. A dataset gathered for claims processing is not
available for marketing because it happens to be in the same warehouse. Enforced here
at read time rather than promised in a policy document.

**The release gate** refuses rather than warns. A gate that emits a warning is a gate
that gets merged past on a Friday.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class GovernanceError(RuntimeError):
    pass


@dataclass
class ModelCard:
    name: str
    version: str
    intended_use: str = ""
    out_of_scope: list[str] = field(default_factory=list)
    training_data: str = ""
    evaluation: dict[str, float] = field(default_factory=dict)
    fairness: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    owner: str = ""
    reviewed_by: str = ""
    created_at: float = field(default_factory=time.time)

    # Fields without which the card is decoration.
    REQUIRED = ("intended_use", "out_of_scope", "training_data", "owner", "limitations")

    def missing_fields(self) -> list[str]:
        return [f for f in self.REQUIRED if not getattr(self, f)]

    def is_complete(self) -> bool:
        return not self.missing_fields()

    def to_markdown(self) -> str:
        lines = [
            f"# Model card: {self.name} v{self.version}",
            "",
            f"**Owner:** {self.owner or 'UNASSIGNED'}",
            f"**Reviewed by:** {self.reviewed_by or 'NOT REVIEWED'}",
            "",
            "## Intended use",
            self.intended_use or "_not stated_",
            "",
            "## Out of scope",
        ]
        lines += [f"- {item}" for item in self.out_of_scope] or ["- _not stated_"]
        lines += ["", "## Training data", self.training_data or "_not stated_", ""]

        if self.evaluation:
            lines += ["## Evaluation", ""]
            lines += [f"| metric | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(self.evaluation.items())]
            lines += [""]

        if self.fairness:
            lines += ["## Fairness", ""]
            lines += [f"| measure | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(self.fairness.items())]
            lines += [""]

        lines += ["## Limitations"]
        lines += [f"- {item}" for item in self.limitations] or ["- _not stated_"]
        return "\n".join(lines)


@dataclass
class ConsentRecord:
    subject: str
    purposes: frozenset[str]
    granted_at: float
    withdrawn_at: float | None = None

    def permits(self, purpose: str, *, at: float | None = None) -> bool:
        at = at if at is not None else time.time()
        if at < self.granted_at:
            return False
        if self.withdrawn_at is not None and at >= self.withdrawn_at:
            return False
        return purpose in self.purposes


@dataclass
class ConsentLedger:
    records: dict[str, ConsentRecord] = field(default_factory=dict)

    def grant(self, subject: str, purposes: set[str], at: float | None = None) -> ConsentRecord:
        record = ConsentRecord(
            subject=subject, purposes=frozenset(purposes),
            granted_at=at if at is not None else time.time(),
        )
        self.records[subject] = record
        return record

    def withdraw(self, subject: str, at: float | None = None) -> None:
        record = self.records.get(subject)
        if record is None:
            raise GovernanceError(f"no consent on record for {subject!r}")
        record.withdrawn_at = at if at is not None else time.time()

    def permitted(self, subjects: list[str], purpose: str, *, at: float | None = None) -> list[str]:
        """Filter a cohort to those who consented to this purpose.

        Absence of a record is absence of consent. Defaulting the other way is how a
        marketing model ends up trained on claims data.
        """
        return [
            s for s in subjects
            if (r := self.records.get(s)) is not None and r.permits(purpose, at=at)
        ]


@dataclass
class ReleaseGate:
    """Conditions a model must satisfy before it may serve."""

    min_gini: float = 0.30
    max_brier: float = 0.25
    max_disparate_impact_gap: float = 0.20  # worst ratio must be >= 0.80
    require_complete_card: bool = True
    require_review: bool = True
    require_no_skew: bool = True

    def evaluate(
        self, *, card: ModelCard, metrics: dict, fairness: dict | None = None,
        skew: dict | None = None,
    ) -> dict:
        failures: list[str] = []

        if self.require_complete_card and not card.is_complete():
            failures.append(f"model card incomplete: {', '.join(card.missing_fields())}")
        if self.require_review and not card.reviewed_by:
            failures.append("model card has not been reviewed")

        gini = metrics.get("gini")
        if gini is None:
            failures.append("no discrimination metric supplied")
        elif gini < self.min_gini:
            failures.append(f"gini {gini:.3f} below minimum {self.min_gini}")

        brier = metrics.get("brier")
        if brier is not None and brier > self.max_brier:
            # Calibration is checked separately from ranking: a model can rank well
            # and still be wrong about the level, and an insurer prices from the level.
            failures.append(f"brier {brier:.3f} above maximum {self.max_brier}")

        if fairness is not None:
            worst = fairness.get("worst_disparate_impact")
            if worst is not None and worst < 1 - self.max_disparate_impact_gap:
                failures.append(
                    f"disparate impact {worst:.3f} below "
                    f"{1 - self.max_disparate_impact_gap:.2f}"
                )

        if self.require_no_skew and skew is not None and not skew.get("safe_to_serve", True):
            failures.append(f"{skew.get('critical', 0)} critical training/serving skew findings")

        return {
            "approved": not failures,
            "failures": failures,
            # Named so the decision is attributable in an audit, not anonymous.
            "gate": "insurance-mlops/release/v1",
        }
