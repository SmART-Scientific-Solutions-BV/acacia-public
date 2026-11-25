from dataclasses import dataclass

from mcgpuReconstruction.data.detectorAxes import DetectorAxes

__all__ = ["Detector"]


@dataclass
class Detector:
    axes:                DetectorAxes
    nPixelsU:            int
    nPixelsV:            int
    uSpacingMm:          float
    vSpacingMm:          float
    uSizeMm:             float
    vSizeMm:             float
    distanceToIsocenter: float
    uVector:             list[float]
    vVector:             list[float]
    wVector:             list[float]
    piercingPointOffset: list[float]
    useDetectorResponse: bool
    responseEnergy:      list[float]
    responseProbability: list[float]

    @property
    def nPixelsFirstAxis(self) -> int:
        return self.nPixelsU if self.axes.first == "u" else self.nPixelsV

    @property
    def nPixelsSecondAxis(self) -> int:
        return self.nPixelsU if self.axes.second == "u" else self.nPixelsV

    @property
    def firstAxisSpacingMm(self) -> float:
        return self.uSpacingMm if self.axes.first == "u" else self.vSpacingMm

    @property
    def secondAxisSpacingMm(self) -> float:
        return self.uSpacingMm if self.axes.second == "u" else self.vSpacingMm

    @classmethod
    def fromHeader(cls, header: dict) -> "Detector":
        detectorHeader: dict = header["detector"]
        geometryHeader: dict = header["geometry"]

        axes: DetectorAxes      = DetectorAxes(first=detectorHeader["axes"][0], second=detectorHeader["axes"][1])
        axisOrder: list[str]    = [axes.first, axes.second]

        nPixelsPerAxis: dict[str, int]   = {
            axisOrder[0]: int(detectorHeader["nPixels"][0]),
            axisOrder[1]: int(detectorHeader["nPixels"][1])
        }
        spacingPerAxis: dict[str, float] = {
            axisOrder[0]: float(detectorHeader["spacing"][0]),
            axisOrder[1]: float(detectorHeader["spacing"][1])
        }
        sizePerAxis: dict[str, float] = {
            axisOrder[0]: float(detectorHeader["size"][0]),
            axisOrder[1]: float(detectorHeader["size"][1])
        }

        return cls(
            axes                = axes,
            nPixelsU            = nPixelsPerAxis["u"],
            nPixelsV            = nPixelsPerAxis["v"],
            uSpacingMm          = spacingPerAxis["u"],
            vSpacingMm          = spacingPerAxis["v"],
            uSizeMm             = sizePerAxis["u"],
            vSizeMm             = sizePerAxis["v"],
            distanceToIsocenter = float(detectorHeader["distanceToIsocenter"]),
            uVector             = list(map(float, geometryHeader["uVector"])),
            vVector             = list(map(float, geometryHeader["vVector"])),
            wVector             = list(map(float, geometryHeader["wVector"])),
            piercingPointOffset = list(map(float, geometryHeader["piercingPointOffset"])),
            useDetectorResponse = bool(detectorHeader["useDetectorResponse"]),
            responseEnergy      = list(map(float, detectorHeader["responseEnergy"])),
            responseProbability = list(map(float, detectorHeader["responseProbability"])),
        )


