import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from mcgpuReconstruction.data.detector import Detector
from mcgpuReconstruction.data.detectorAxes import DetectorAxes
from mcgpuReconstruction.data.projection import Projection

__all__ = ["ProjectionStack"]


@dataclass
class ProjectionStack:
    projections: list[Projection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.projections) > 1:
            referenceAxes: DetectorAxes = self.projections[0].detector.axes
            for projection in self.projections[1:]:
                if projection.detector.axes.first != referenceAxes.first or \
                   projection.detector.axes.second != referenceAxes.second:
                    logging.error(
                        f"All projections must share the same detector axis order. "
                        f"Expected [{referenceAxes.first}, {referenceAxes.second}], "
                        f"got [{projection.detector.axes.first}, {projection.detector.axes.second}]."
                    )

    @property
    def angles(self) -> list[float]:
        return [projection.angle for projection in self.projections]

    @property
    def nProjections(self) -> int:
        return len(self.projections)

    @property
    def detector(self) -> Detector | None:
        return self.projections[0].detector if self.projections else None

    @property
    def data(self) -> np.ndarray:
        return np.stack(arrays=[projection.data for projection in self.projections], axis=0)

    @property
    def firstAxisLabel(self) -> str:
        return self.projections[0].firstAxisLabel if self.projections else ""

    @property
    def secondAxisLabel(self) -> str:
        return self.projections[0].secondAxisLabel if self.projections else ""

    def visualizeSinogram(self, vMin=None, vMax=None, title="Sinogram", outputFolder=None) -> None:
        if not self.projections:
            logging.error("Cannot visualise sinogram: projection stack is empty.")
            return

        centralIndex: int            = self.projections[0].data.shape[0] // 2
        sinogram: np.ndarray         = np.stack(arrays=[p.data[centralIndex, :] for p in self.projections], axis=0)
        sampleProjection: np.ndarray = self.projections[self.nProjections // 2].data

        sinogramMin: float   = vMin if vMin is not None else float(np.percentile(a=sinogram, q=1))
        sinogramMax: float   = vMax if vMax is not None else float(np.percentile(a=sinogram, q=99))
        projectionMin: float = vMin if vMin is not None else float(np.percentile(a=sampleProjection, q=1))
        projectionMax: float = vMax if vMax is not None else float(np.percentile(a=sampleProjection, q=99))

        figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        figure.suptitle(t=title)

        image0 = axes[0].imshow(
            X=sinogram, map="bone", origin="upper", aspect="auto", vmin=sinogramMin, vmax=sinogramMax
        )
        axes[0].set_title(title=f"Sinogram (central {self.firstAxisLabel} row)")
        axes[0].set_xlabel(xlabel=self.secondAxisLabel)
        axes[0].set_ylabel(ylabel="Projection index")
        plt.colorbar(image0, ax=axes[0], label="Intensity")

        image1 = axes[1].imshow(
            X=sampleProjection, cmap="bone", origin="upper", aspect="auto", vmin=projectionMin, vmax=projectionMax
        )
        axes[1].set_title(title=f"Projection at {self.projections[self.nProjections // 2].angle:.1f} degrees")
        axes[1].set_xlabel(xlabel=self.secondAxisLabel)
        axes[1].set_ylabel(ylabel=self.firstAxisLabel)
        plt.colorbar(image1, ax=axes[1], label="Intensity")

        plt.tight_layout()

        if outputFolder is not None:
            outputFolder.mkdir(parents=True, exist_ok=True)
            plt.savefig(outputFolder / "sinogram.png", dpi=100)
            plt.close(figure)
        else:
            plt.show()


