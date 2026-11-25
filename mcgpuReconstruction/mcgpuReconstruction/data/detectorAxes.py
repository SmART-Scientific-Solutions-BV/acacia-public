import logging
from dataclasses import dataclass

__all__ = ["DetectorAxes"]


@dataclass
class DetectorAxes:
    first:  str
    second: str

    def __post_init__(self) -> None:
        validAxes: set[str] = {"u", "v"}
        if self.first.lower() not in validAxes or self.second.lower() not in validAxes:
            logging.error(f"Detector axes must be 'u' and 'v', got '{self.first}' and '{self.second}'.")
        if self.first.lower() == self.second.lower():
            logging.error(f"Detector axes must be different, got '{self.first}' and '{self.second}'.")
        self.first          = self.first.lower()
        self.second         = self.second.lower()

    @property
    def uAxisIndex(self) -> int:
        return 0 if self.first == "u" else 1

    @property
    def vAxisIndex(self) -> int:
        return 0 if self.first == "v" else 1


