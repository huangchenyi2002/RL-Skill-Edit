from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

import numpy as np

from .types import Action, SkillModule


class EditOperator(str, Enum):
    ADD_RULE = "ADD_RULE"
    REWRITE_RULE = "REWRITE_RULE"
    DELETE_RULE = "DELETE_RULE"
    ADD_EXAMPLE = "ADD_EXAMPLE"
    REWRITE_EXAMPLE = "REWRITE_EXAMPLE"
    MERGE_REDUNDANT_RULES = "MERGE_REDUNDANT_RULES"
    REORDER_CONTENT = "REORDER_CONTENT"
    STOP = "STOP"


@dataclass(frozen=True)
class ActionSpace:
    max_modules: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_modules, bool)
            or not isinstance(self.max_modules, int)
            or self.max_modules < 1
        ):
            raise ValueError("max_modules must be a positive integer")

    @property
    def size(self) -> int:
        return self.max_modules * len(EditOperator)

    def encode(self, slot: int, operator: EditOperator) -> int:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        if not 0 <= slot < self.max_modules:
            raise ValueError(f"slot must be in [0, {self.max_modules})")
        if not isinstance(operator, EditOperator):
            raise TypeError("operator must be an EditOperator")
        return slot * len(EditOperator) + list(EditOperator).index(operator)

    def _module_by_slot(self, modules: Sequence[SkillModule]) -> dict[int, SkillModule]:
        by_slot: dict[int, SkillModule] = {}
        for module in modules:
            if not 0 <= module.slot < self.max_modules:
                raise ValueError(f"module slot {module.slot} exceeds max_modules")
            if module.slot in by_slot:
                raise ValueError(f"duplicate module slot: {module.slot}")
            by_slot[module.slot] = module
        if not by_slot or 0 not in by_slot or by_slot[0].module_id != "global":
            raise ValueError("slot 0 must contain the global module")
        expected_slots = set(range(len(by_slot)))
        if set(by_slot) != expected_slots:
            raise ValueError("module slots must be contiguous from zero")
        return by_slot

    def mask(self, modules: Sequence[SkillModule]) -> np.ndarray:
        by_slot = self._module_by_slot(modules)
        mask = np.zeros(self.size, dtype=np.bool_)
        for slot in by_slot:
            for operator in EditOperator:
                if operator is not EditOperator.STOP:
                    mask[self.encode(slot, operator)] = True
        mask[self.encode(0, EditOperator.STOP)] = True
        return mask

    def decode(self, index: int, modules: Sequence[SkillModule]) -> Action:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("action index must be an integer")
        numeric_index = int(index)
        if not 0 <= numeric_index < self.size:
            raise ValueError(f"action index must be in [0, {self.size})")
        by_slot = self._module_by_slot(modules)
        if not bool(self.mask(modules)[numeric_index]):
            raise ValueError(
                f"action index {numeric_index} is not valid for the current modules"
            )
        slot, operator_offset = divmod(numeric_index, len(EditOperator))
        operator = tuple(EditOperator)[operator_offset]
        return Action(
            module_id=by_slot[slot].module_id,
            operator=operator,
            index=numeric_index,
        )


__all__ = ["ActionSpace", "EditOperator"]
