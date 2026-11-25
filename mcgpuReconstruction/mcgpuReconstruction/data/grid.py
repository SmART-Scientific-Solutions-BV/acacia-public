import numpy as np
from dataclasses import dataclass


__all__ = [
    "Grid"
]


@dataclass
class Grid:
    size: list[int]
    spacing: list[float]
    origin: list[float]
    SPATIAL_UNIT: str = 'millimeter'

    iPositions: list[float] = None
    jPositions: list[float] = None
    kPositions: list[float] = None

    def __post_init__(self) -> None:
        self.__verifyInputs__()
        self.__constructGrid__()

    def __verifyInputs__(self) -> None:
        if isinstance(self.spacing, float) or isinstance(self.spacing, int):
            self.spacing = [float(self.spacing)] * len(self.size)

        if isinstance(self.origin, float) or isinstance(self.origin, int):
            self.origin = [float(self.origin)] * len(self.size)

    def __constructGrid__(self) -> None:
        # Create empty list for spatial volume sizes
        self.spatialVolumeSize = []

        # Create i, j and k position vectors
        positions = ['iPositions', 'jPositions', 'kPositions']

        # Loop over positions
        for i, position in enumerate(positions):
            # Create positions vector for corresponding dimension
            positionsVector = np.linspace(
                start=self.origin[i],
                stop=self.origin[i] + (self.size[i] * self.spacing[i]),
                num=self.size[i],
                endpoint=False,
                dtype=float
            )

            # Set attribute
            setattr(self, position, positionsVector)

            # Determine volume size
            self.spatialVolumeSize.append(max(positionsVector) - min(positionsVector))

    def updateGrid(self, size: list[int], spacing: list[float], origin: list[float]) -> None:
        # Update values and make new grid from these
        self.size, self.spacing, self.origin = size, spacing, origin
        self.__constructGrid__()

    def moveOriginToVolumeCenter(self) -> None:
        self.origin = [-(i * j) / 2 for i, j in zip(self.size, self.spacing)]
        self.__constructGrid__()
