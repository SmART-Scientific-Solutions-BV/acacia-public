import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from mcgpuReconstruction.data.detector import Detector
from mcgpuReconstruction.data.projectionGeometry import ProjectionGeometry

__all__ = ["Projection"]


@dataclass
class Projection:
    data:     np.ndarray
    detector: Detector
    geometry: ProjectionGeometry

    @property
    def firstAxisLabel(self) -> str:
        return self.detector.axes.first.upper()

    @property
    def secondAxisLabel(self) -> str:
        return self.detector.axes.second.upper()

    @property
    def angle(self) -> float:
        return self.geometry.gantryAngle

    def visualize(self, vMin=None, vMax=None, title="", outputFolder=None) -> None:
        displayMin: float = vMin if vMin is not None else float(np.percentile(self.data, q=1))
        displayMax: float = vMax if vMax is not None else float(np.percentile(self.data, q=99))

        figure, axes = plt.subplots(figsize=(8, 6))
        axes.imshow(X=self.data, cmap="bone", origin="upper", vmin=displayMin, vmax=displayMax)
        axes.set_xlabel(xlabel=self.secondAxisLabel)
        axes.set_ylabel(ylabel=self.firstAxisLabel)
        axes.set_title(label=title if title else f"Projection at {self.angle:.1f} degrees")
        plt.tight_layout()

        if outputFolder is not None:
            outputFolder.mkdir(parents=True, exist_ok=True)
            plt.savefig(outputFolder / f"projection_{self.angle:.1f}deg.png", dpi=100)
            plt.close(fig=figure)
        else:
            plt.show()


