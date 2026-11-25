from dataclasses import dataclass

__all__ = [
    "FilterChoice", "RamLak", "Hann", "Cosine", "Hamming", "SheppLogan",
    "filterChoiceFromString", "filterChoiceToString",
]


@dataclass(frozen=True)
class RamLak:
    pass


@dataclass(frozen=True)
class Hann:
    cutFrequency: float = 1.0


@dataclass(frozen=True)
class Cosine:
    cutFrequency: float = 1.0


@dataclass(frozen=True)
class Hamming:
    cutFrequency: float = 1.0


@dataclass(frozen=True)
class SheppLogan:
    cutFrequency: float = 1.0


FilterChoice = RamLak | Hann | Cosine | Hamming | SheppLogan

_STRING_TO_TYPE = {
    "ram-lak": RamLak, "hann": Hann, "cosine": Cosine, "hamming": Hamming, "shepp-logan": SheppLogan,
}
_TYPE_TO_STRING = {value: key for key, value in _STRING_TO_TYPE.items()}


def filterChoiceFromString(filterName: str) -> FilterChoice | None:
    filterType = _STRING_TO_TYPE.get(filterName.lower())
    return filterType() if filterType is not None else None


def filterChoiceToString(choice: FilterChoice) -> str:
    return _TYPE_TO_STRING[type(choice)]
