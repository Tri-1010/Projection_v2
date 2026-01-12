from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import config


@dataclass(frozen=True)
class DataSchema:
    id_col: str
    mob_col: str
    state_col: str
    ead_col: str
    date_col: str
    segment_cols: Sequence[str]
    cohort_col: str | None = None

    @property
    def required_columns(self) -> List[str]:
        cols = [
            self.id_col,
            self.mob_col,
            self.state_col,
            self.ead_col,
            self.date_col,
            *self.segment_cols,
        ]
        if self.cohort_col:
            cols.append(self.cohort_col)
        return cols

    def rename_map(self) -> dict:
        return {
            "id": self.id_col,
            "mob": self.mob_col,
            "state": self.state_col,
            "ead": self.ead_col,
            "date": self.date_col,
        }


def default_schema() -> DataSchema:
    return DataSchema(
        id_col=config.SCHEMA["id_col"],
        mob_col=config.SCHEMA["mob_col"],
        state_col=config.SCHEMA["state_col"],
        ead_col=config.SCHEMA["ead_col"],
        date_col=config.SCHEMA["date_col"],
        segment_cols=config.SCHEMA["segment_cols"],
        cohort_col=config.SCHEMA.get("cohort_col"),
    )
