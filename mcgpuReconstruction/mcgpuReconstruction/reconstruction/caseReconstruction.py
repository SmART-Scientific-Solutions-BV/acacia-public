import logging
import time
from pathlib import Path

import numpy as np

from mcgpuReconstruction.data.ct import Ct
from mcgpuReconstruction.data.detector import Detector
from mcgpuReconstruction.data.projection import Projection
from mcgpuReconstruction.data.projectionGeometry import ProjectionGeometry
from mcgpuReconstruction.data.projectionStack import ProjectionStack
from mcgpuReconstruction.inputOutput.projectionReader import readBinaryProjection, readHeader
from mcgpuReconstruction.reconstruction.fdkReconstruction import FdkReconstruction
from mcgpuReconstruction.reconstruction.filterChoice import FilterChoice, RamLak
from mcgpuReconstruction.reconstruction.linearization import (
    DEFAULT_AIR_PATCH_COORDINATES, estimateI0FromAirPatch, linearize,
)

__all__ = [
    "DEFAULT_VOLUME_AXIS_ORDER", "rawToLinearizedProjections", "reconstructFromLinearizedProjections",
    "reconstructRawCase",
]

DEFAULT_VOLUME_AXIS_ORDER: tuple[int, int, int] = (1, 2, 0)


def logLoopProgress(startTime: float, index: int, total: int, label: str, logInterval: int) -> None:
    if not logInterval or (index + 1) % logInterval != 0:
        return
    elapsed = time.monotonic() - startTime
    perProjection = elapsed / (index + 1)
    remaining = perProjection * (total - (index + 1))
    logging.info(f"{label}: {index + 1}/{total} projection(s) done "
                 f"({perProjection:.2f} s/projection, ~{remaining:.0f}s remaining)")


def rawToLinearizedProjections(
        projectionPaths: list[Path],
        datasetIndex: int = 0,
        airPatchCoordinates: tuple[float, float, float, float] = DEFAULT_AIR_PATCH_COORDINATES,
        progressLogInterval: int = 30,
) -> list[tuple[np.ndarray, dict]]:
    logging.info(f"Loading {len(projectionPaths)} raw projection(s), dataset index {datasetIndex}")

    linearizedProjections = []
    startTime = time.monotonic()
    for index, path in enumerate(projectionPaths):
        header = readHeader(path)
        if header is None:
            continue

        raw = readBinaryProjection(path=path, header=header, datasetIndex=datasetIndex)

        i0 = estimateI0FromAirPatch(raw, airPatchCoordinates=airPatchCoordinates)
        linearized = linearize(raw, i0)

        linearizedProjections.append((linearized, header))
        logLoopProgress(startTime, index, len(projectionPaths), "Loading", progressLogInterval)

    return linearizedProjections


def reconstructFromLinearizedProjections(
        linearizedProjections: list[tuple[np.ndarray, dict]],
        volumeAxisOrder: tuple[int, int, int] = DEFAULT_VOLUME_AXIS_ORDER,
        filterChoice: FilterChoice = RamLak(),
) -> Ct:
    if not linearizedProjections:
        raise ValueError("No usable projections were provided for reconstruction")

    projections = []
    for linearized, header in linearizedProjections:
        detector = Detector.fromHeader(header)
        geometry = ProjectionGeometry.fromHeader(header)
        projections.append(Projection(data=linearized, detector=detector, geometry=geometry))

    projectionStack = ProjectionStack(projections=projections)

    firstGeometry = projections[0].geometry
    volumeSize = [firstGeometry.phantomSize[i] for i in volumeAxisOrder]
    volumeSpacing = [firstGeometry.phantomPitch[i] for i in volumeAxisOrder]
    volumeOrigin = [firstGeometry.phantomPosition[i] for i in volumeAxisOrder]

    reconstruction = FdkReconstruction(projectionStack=projectionStack, verbose=False)
    reconstruction.volumeSize = volumeSize
    reconstruction.volumeSpacing = volumeSpacing
    reconstruction.volumeOrigin = volumeOrigin
    reconstruction.filterChoice = filterChoice

    extentMm = [size * spacing for size, spacing in zip(volumeSize, volumeSpacing)]
    logging.info(f"volumeAxisOrder={volumeAxisOrder}: size(X,Y,Z)={volumeSize} voxels, "
                 f"spacing={volumeSpacing} mm, origin={volumeOrigin} mm, extent={extentMm} mm, "
                 f"filter={filterChoice}")

    return reconstruction.reconstruct()


def reconstructRawCase(
        projectionPaths: list[Path],
        datasetIndex: int = 0,
        volumeAxisOrder: tuple[int, int, int] = DEFAULT_VOLUME_AXIS_ORDER,
        airPatchCoordinates: tuple[float, float, float, float] = DEFAULT_AIR_PATCH_COORDINATES,
        progressLogInterval: int = 30,
        filterChoice: FilterChoice = RamLak(),
) -> Ct:
    linearizedProjections = rawToLinearizedProjections(
        projectionPaths=projectionPaths, datasetIndex=datasetIndex,
        airPatchCoordinates=airPatchCoordinates, progressLogInterval=progressLogInterval,
    )
    return reconstructFromLinearizedProjections(
        linearizedProjections, volumeAxisOrder=volumeAxisOrder, filterChoice=filterChoice,
    )
