import numpy as np

__all__ = [
    "DEFAULT_AIR_PATCH_COORDINATES", "DEFAULT_LINEARIZE_EPSILON",
    "estimateI0FromAirPatch", "linearize",
]

DEFAULT_AIR_PATCH_COORDINATES: tuple[float, float, float, float] = (0.05, 0.25, 0.10, 0.90)
DEFAULT_LINEARIZE_EPSILON: float = 1e-6


def estimateI0FromAirPatch(
        raw: np.ndarray, airPatchCoordinates: tuple[float, float, float, float] = DEFAULT_AIR_PATCH_COORDINATES,
) -> float:
    axis0Count, axis1Count = raw.shape
    f0Start, f0End, f1Start, f1End = airPatchCoordinates
    axis0Start, axis0End = int(axis0Count * f0Start), int(axis0Count * f0End)
    axis1Start, axis1End = int(axis1Count * f1Start), int(axis1Count * f1End)
    return float(raw[axis0Start:axis0End, axis1Start:axis1End].mean())


def linearize(raw: np.ndarray, i0: float, epsilon: float = DEFAULT_LINEARIZE_EPSILON) -> np.ndarray:
    ratio = np.clip(raw / i0, a_min=epsilon, a_max=None)
    return -np.log(ratio)
