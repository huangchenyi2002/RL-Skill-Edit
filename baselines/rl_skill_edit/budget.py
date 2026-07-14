from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


_STRUCTURAL_LIMITS = (
    "student_rollouts",
    "teacher_rollouts",
    "reference_rollouts",
    "editor_calls",
    "evaluator_calls",
)
_USAGE_LIMITS = ("input_tokens", "output_tokens", "wall_time_seconds")
_REQUIRED_LIMITS = _STRUCTURAL_LIMITS + _USAGE_LIMITS
_ROLES = frozenset(("student", "teacher", "reference"))


class BudgetExceeded(RuntimeError):
    """Raised before a reservation or usage record would exceed a limit."""


@dataclass(frozen=True)
class BudgetSnapshot:
    student_rollouts: int = 0
    teacher_rollouts: int = 0
    reference_rollouts: int = 0
    editor_calls: int = 0
    evaluator_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0
    cache_hits: int = 0
    cached_student_rollouts: int = 0
    cached_teacher_rollouts: int = 0
    cached_reference_rollouts: int = 0
    cached_editor_calls: int = 0
    cached_evaluator_calls: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    ledger_id: str
    kind: str
    cache_hit: bool
    role: str | None = None
    task_count: int = 0
    repetitions: int = 0


class BudgetLedger:
    """Thread-safe ledger with atomic structural reservations and usage records."""

    def __init__(self, limits: Mapping[str, Any]) -> None:
        if not isinstance(limits, Mapping):
            raise TypeError("limits must be a mapping")
        missing = set(_REQUIRED_LIMITS) - set(limits)
        if missing:
            raise ValueError(f"budget limits are missing: {sorted(missing)}")
        unknown = set(limits) - set(_REQUIRED_LIMITS)
        if unknown:
            raise ValueError(f"unknown budget limits: {sorted(unknown)}")

        parsed: dict[str, int | Decimal] = {}
        for name in _STRUCTURAL_LIMITS + ("input_tokens", "output_tokens"):
            parsed[name] = _nonnegative_integer(name, limits[name])
        parsed["wall_time_seconds"] = _nonnegative_decimal(
            "wall_time_seconds", limits["wall_time_seconds"]
        )
        self._limits = parsed
        self._counts: dict[str, int | Decimal] = {
            name: 0 for name in _STRUCTURAL_LIMITS + ("input_tokens", "output_tokens")
        }
        self._counts["wall_time_seconds"] = Decimal(0)
        self._counts["cache_hits"] = 0
        for name in (
            "cached_student_rollouts",
            "cached_teacher_rollouts",
            "cached_reference_rollouts",
            "cached_editor_calls",
            "cached_evaluator_calls",
        ):
            self._counts[name] = 0
        self._ledger_id = uuid.uuid4().hex
        self._reservations: dict[str, BudgetReservation] = {}
        self._recorded_reservations: set[str] = set()
        self._lock = threading.RLock()

    def reserve_evaluation(
        self,
        role: str,
        *,
        task_count: int,
        repetitions: int = 1,
        cache_hit: bool = False,
    ) -> BudgetReservation:
        normalized_role = _role(role)
        tasks = _positive_integer("task_count", task_count)
        repeats = _positive_integer("repetitions", repetitions)
        cached = _boolean("cache_hit", cache_hit)
        reservation = BudgetReservation(
            reservation_id=uuid.uuid4().hex,
            ledger_id=self._ledger_id,
            kind="evaluation",
            cache_hit=cached,
            role=normalized_role,
            task_count=tasks,
            repetitions=repeats,
        )
        with self._lock:
            deltas = {
                f"{normalized_role}_rollouts": tasks * repeats,
                "evaluator_calls": 1,
            }
            self._check_deltas(deltas)
            self._apply_deltas(deltas)
            if cached:
                self._counts["cache_hits"] += 1
                self._counts[f"cached_{normalized_role}_rollouts"] += tasks * repeats
                self._counts["cached_evaluator_calls"] += 1
            self._reservations[reservation.reservation_id] = reservation
        return reservation

    def record_evaluation(
        self,
        reservation: BudgetReservation,
        *,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
    ) -> None:
        self._record(
            reservation,
            expected_kind="evaluation",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed_seconds,
        )

    def reserve_editor(self, *, cache_hit: bool = False) -> BudgetReservation:
        cached = _boolean("cache_hit", cache_hit)
        reservation = BudgetReservation(
            reservation_id=uuid.uuid4().hex,
            ledger_id=self._ledger_id,
            kind="editor",
            cache_hit=cached,
        )
        with self._lock:
            self._check_deltas({"editor_calls": 1})
            self._apply_deltas({"editor_calls": 1})
            if cached:
                self._counts["cache_hits"] += 1
                self._counts["cached_editor_calls"] += 1
            self._reservations[reservation.reservation_id] = reservation
        return reservation

    def record_editor(
        self,
        reservation: BudgetReservation,
        *,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
    ) -> None:
        self._record(
            reservation,
            expected_kind="editor",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed_seconds,
        )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                student_rollouts=int(self._counts["student_rollouts"]),
                teacher_rollouts=int(self._counts["teacher_rollouts"]),
                reference_rollouts=int(self._counts["reference_rollouts"]),
                editor_calls=int(self._counts["editor_calls"]),
                evaluator_calls=int(self._counts["evaluator_calls"]),
                input_tokens=int(self._counts["input_tokens"]),
                output_tokens=int(self._counts["output_tokens"]),
                wall_time_seconds=float(self._counts["wall_time_seconds"]),
                cache_hits=int(self._counts["cache_hits"]),
                cached_student_rollouts=int(self._counts["cached_student_rollouts"]),
                cached_teacher_rollouts=int(self._counts["cached_teacher_rollouts"]),
                cached_reference_rollouts=int(
                    self._counts["cached_reference_rollouts"]
                ),
                cached_editor_calls=int(self._counts["cached_editor_calls"]),
                cached_evaluator_calls=int(self._counts["cached_evaluator_calls"]),
            )

    def remaining_fraction(self) -> float:
        with self._lock:
            fractions: list[float] = []
            for name in _REQUIRED_LIMITS:
                limit = Decimal(self._limits[name])
                used = Decimal(self._counts[name])
                if limit == 0:
                    fractions.append(1.0 if used == 0 else 0.0)
                else:
                    fractions.append(float(max(Decimal(0), 1 - used / limit)))
            return min(fractions)

    def _record(
        self,
        reservation: BudgetReservation,
        *,
        expected_kind: str,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
    ) -> None:
        token_input = _nonnegative_integer("input_tokens", input_tokens)
        token_output = _nonnegative_integer("output_tokens", output_tokens)
        elapsed = _nonnegative_decimal("elapsed_seconds", elapsed_seconds)
        with self._lock:
            stored = self._validate_reservation(reservation, expected_kind)
            if stored.cache_hit:
                if token_input != 0 or token_output != 0 or elapsed != 0.0:
                    raise ValueError("cache-hit reservations must record zero usage")
                self._finish_reservation(stored)
                return
            deltas: dict[str, int | Decimal] = {
                "input_tokens": token_input,
                "output_tokens": token_output,
                "wall_time_seconds": elapsed,
            }
            self._check_deltas(deltas)
            self._apply_deltas(deltas)
            self._finish_reservation(stored)

    def _validate_reservation(
        self,
        reservation: BudgetReservation,
        expected_kind: str,
    ) -> BudgetReservation:
        if not isinstance(reservation, BudgetReservation):
            raise TypeError("reservation must be a BudgetReservation")
        if reservation.ledger_id != self._ledger_id:
            raise ValueError("reservation belongs to a different ledger")
        if reservation.reservation_id in self._recorded_reservations:
            raise ValueError("reservation has already been recorded")
        stored = self._reservations.get(reservation.reservation_id)
        if stored is None or stored != reservation:
            raise ValueError("reservation is unknown or has been modified")
        if stored.kind != expected_kind:
            raise ValueError(
                f"expected a {expected_kind} reservation, got {stored.kind}"
            )
        return stored

    def _finish_reservation(self, reservation: BudgetReservation) -> None:
        self._reservations.pop(reservation.reservation_id)
        self._recorded_reservations.add(reservation.reservation_id)

    def _check_deltas(self, deltas: Mapping[str, int | Decimal]) -> None:
        exceeded: list[str] = []
        for name, delta in deltas.items():
            projected = self._counts[name] + delta
            if projected > self._limits[name]:
                exceeded.append(f"{name}={projected} exceeds {self._limits[name]}")
        if exceeded:
            raise BudgetExceeded("; ".join(exceeded))

    def _apply_deltas(self, deltas: Mapping[str, int | Decimal]) -> None:
        for name, delta in deltas.items():
            self._counts[name] += delta


def _role(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("role must be a string")
    normalized = value.strip().lower()
    if normalized not in _ROLES:
        raise ValueError(f"unknown evaluation role: {value!r}")
    return normalized


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    result = _integer(name, value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _nonnegative_decimal(name: str, value: Any) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetReservation",
    "BudgetSnapshot",
]
