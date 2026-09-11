"""Point-in-time correct feature store.

The most expensive bug in production machine learning, and the one that almost never
shows up in a notebook:

    claim filed         2026-03-01
    fraud confirmed     2026-05-14
    feature `prior_claims_count` recomputed nightly, current value 4

Training on today's `prior_claims_count` teaches the model a number that **did not
exist** when the decision had to be made. Offline accuracy is excellent. Online
accuracy collapses, and the gap is blamed on drift.

A point-in-time join fixes it by asking a different question. Not *"what is this
customer's claim count?"* but *"what was it, as far as anyone knew, at 09:14 on the
first of March?"*

Every feature value here carries the moment it **became knowable**, which is not the
moment it became true. A claim filed on Monday and entered into the system on
Wednesday was not available to a model running on Tuesday, and pretending otherwise is
the same bug wearing a hat.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field


class FeatureStoreError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureValue:
    entity: str
    feature: str
    value: float | str | None
    # When the fact became true in the world.
    event_time: float
    # When the system could first have known it. Never earlier than event_time.
    available_at: float

    def __post_init__(self) -> None:
        if self.available_at < self.event_time:
            raise FeatureStoreError(
                f"{self.feature}: available_at precedes event_time - a value cannot be "
                "knowable before it happens"
            )


@dataclass
class FeatureStore:
    """Append-only history per (entity, feature), queried as of a moment."""

    _history: dict[tuple[str, str], list[FeatureValue]] = field(default_factory=dict)

    def write(
        self, entity: str, feature: str, value, event_time: float,
        available_at: float | None = None,
    ) -> FeatureValue:
        record = FeatureValue(
            entity=entity, feature=feature, value=value, event_time=event_time,
            available_at=event_time if available_at is None else available_at,
        )
        series = self._history.setdefault((entity, feature), [])
        series.append(record)
        # Sorted by availability, because availability is what a query filters on.
        series.sort(key=lambda r: r.available_at)
        return record

    def get_as_of(self, entity: str, feature: str, as_of: float) -> FeatureValue | None:
        """The most recent value that was *knowable* at `as_of`.

        Binary search on availability. Values whose `available_at` is later are
        invisible, which is the entire point - a value the system could not have read
        cannot be a feature.
        """
        series = self._history.get((entity, feature))
        if not series:
            return None
        times = [r.available_at for r in series]
        index = bisect.bisect_right(times, as_of)
        return series[index - 1] if index else None

    def vector_as_of(
        self, entity: str, features: Sequence[str], as_of: float
    ) -> dict[str, object]:
        return {f: (v.value if (v := self.get_as_of(entity, f, as_of)) else None)
                for f in features}

    def build_training_set(
        self, labels: Sequence[tuple[str, float, int]], features: Sequence[str]
    ) -> tuple[list[dict], list[int]]:
        """Assemble a training set from (entity, decision_time, label) triples.

        `decision_time` is when the model would have had to act — **not** when the
        label became known. Joining on the label time is the leak: it hands the model
        every fact that accumulated while the outcome was being determined.
        """
        rows, targets = [], []
        for entity, decision_time, label in labels:
            rows.append(self.vector_as_of(entity, features, decision_time))
            targets.append(label)
        return rows, targets

    def staleness(self, entity: str, feature: str, as_of: float) -> float | None:
        """How old the value in use is, in seconds.

        Reported because a point-in-time-correct feature can still be useless: a
        risk score last refreshed eight months ago is technically legitimate and
        practically fiction.
        """
        record = self.get_as_of(entity, feature, as_of)
        return None if record is None else as_of - record.available_at

    def purge_before(self, cutoff: float) -> int:
        """Delete values whose availability predates `cutoff`.

        Retention is a legal obligation, not housekeeping. Pakistan's PDPA requires
        personal data to be kept no longer than the purpose needs, so purging has to
        be a first-class operation rather than a script somebody remembers to run.
        """
        removed = 0
        for key, series in list(self._history.items()):
            keep = [r for r in series if r.available_at >= cutoff]
            removed += len(series) - len(keep)
            if keep:
                self._history[key] = keep
            else:
                del self._history[key]
        return removed

    def entities(self) -> set[str]:
        return {entity for entity, _ in self._history}
