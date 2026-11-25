import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

from mcgpuReconstruction.data.grid import Grid

__all__ = ["Ct"]


@dataclass
class Ct:
    data:            np.ndarray
    grid:            Grid
    isReconstructed: bool = False

    def __post_init__(self) -> None:
        if list(self.data.shape) != list(self.grid.size):
            logging.warning(f"Ct data shape {self.data.shape} does not match grid size {self.grid.size}.")

    def permute(self, dimensions: tuple[int, int, int]) -> None:
        if len(dimensions) != len(self.data.shape):
            logging.error(f"Expected dimensions of length {len(self.data.shape)}, got {len(dimensions)}.")
            return

        oldShape: tuple[int, ...] = self.data.shape

        self.data: np.ndarray = np.transpose(a=self.data, axes=dimensions)
        self.grid: Grid       = Grid(
            size    = [self.grid.size[d] for d in dimensions],
            spacing = [self.grid.spacing[d] for d in dimensions],
            origin  = [self.grid.origin[d] for d in dimensions],
        )
        logging.info(f"CT permuted with dimensions {dimensions}. Old shape: {oldShape}. New shape: {self.data.shape}.")

    def flip(self, axis: int | tuple[int, ...]) -> None:
        """
        Flip the data along one or more axes, reflecting voxel-centre positions
        through the world origin.
        """
        self.data = np.ascontiguousarray(a=np.flip(m=self.data, axis=axis))
        logging.info(f"CT flipped along axis {axis}. Current shape: {self.data.shape}.")

    def visualize(
        self,
        vMin: float | None = None,
        vMax: float | None = None,
        title: str = "",
        save: bool = False,
        outputFolder: Path | None = None,
    ) -> None:
        displayMin: float = vMin if vMin is not None else float(np.percentile(a=self.data, q=1))
        displayMax: float = vMax if vMax is not None else float(np.percentile(a=self.data, q=99))

        sliceZ: int = self.data.shape[0] // 2
        sliceY: int = self.data.shape[1] // 2
        sliceX: int = self.data.shape[2] // 2

        sliceZPositionMm: float = float(self.grid.iPositions[sliceZ])
        sliceYPositionMm: float = float(self.grid.jPositions[sliceY])
        sliceXPositionMm: float = float(self.grid.kPositions[sliceX])

        volumeType: str = "Reconstructed CT" if self.isReconstructed else "CT"
        plotTitle: str  = title if title else volumeType

        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        figure.suptitle(t=plotTitle)

        axes[0].imshow(
            X=self.data[sliceZ, :, :],
            cmap="bone",
            origin="lower",
            vmin=displayMin,
            vmax=displayMax,
            extent=[self.grid.kPositions[0], self.grid.kPositions[-1], self.grid.jPositions[0], self.grid.jPositions[-1]],
            aspect="equal",
        )
        axes[0].set_title(label=f"Coronal (Z = {sliceZPositionMm:.2f} mm, slice {sliceZ}/{self.data.shape[0]})")
        axes[0].set_xlabel(xlabel="X [mm]")
        axes[0].set_ylabel(ylabel="Y [mm]")

        axes[1].imshow(
            X=self.data[:, sliceY, :],
            cmap="bone",
            origin="lower",
            vmin=displayMin,
            vmax=displayMax,
            extent=[self.grid.kPositions[0], self.grid.kPositions[-1], self.grid.iPositions[0], self.grid.iPositions[-1]],
            aspect="equal",
        )
        axes[1].set_title(label=f"Axial (Y = {sliceYPositionMm:.2f} mm, slice {sliceY}/{self.data.shape[1]})")
        axes[1].set_xlabel(xlabel="X [mm]")
        axes[1].set_ylabel(ylabel="Z [mm]")

        axes[2].imshow(
            X=self.data[:, :, sliceX],
            cmap="bone",
            origin="lower",
            vmin=displayMin,
            vmax=displayMax,
            extent=[self.grid.jPositions[0], self.grid.jPositions[-1], self.grid.iPositions[0], self.grid.iPositions[-1]],
            aspect="equal",
        )
        axes[2].set_title(label=f"Sagittal (X = {sliceXPositionMm:.2f} mm, slice {sliceX}/{self.data.shape[2]})")
        axes[2].set_xlabel(xlabel="Y [mm]")
        axes[2].set_ylabel(ylabel="Z [mm]")

        if save:
            if outputFolder is None:
                logging.error(msg="outputFolder must be provided when save=True.")
                plt.close(fig=figure)
                return
            outputFolder.mkdir(parents=True, exist_ok=True)
            fileName: str = "reconstructedCt.png" if self.isReconstructed else "originalCt.png"
            plt.savefig(outputFolder / fileName, dpi=100)
            logging.info(msg=f"Saved CT visualization to {outputFolder / fileName}.")
            plt.close(fig=figure)
        else:
            plt.show()

    def _normalize(self) -> np.ndarray:
        lowerValue: float = float(np.percentile(a=self.data, q=1))
        upperValue: float = float(np.percentile(a=self.data, q=99))

        if upperValue == lowerValue:
            return np.zeros_like(a=self.data, dtype=np.float32)
        return np.clip(a=(self.data - lowerValue) / (upperValue - lowerValue), a_min=0, a_max=1).astype(np.float32)

    @property
    def normalized(self) -> np.ndarray:
        return self._normalize()
