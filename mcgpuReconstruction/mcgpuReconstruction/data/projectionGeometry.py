from dataclasses import dataclass

__all__ = ["ProjectionGeometry"]


@dataclass
class ProjectionGeometry:
    sourceIsocenterDistance: float
    sourceDetectorDistance:  float
    gantryAngle:             float
    focusVector:             list[float]
    centerVector:            list[float]
    isocenterOffset:         list[float]
    piercingPointOffset:     list[float]
    phantomPosition:         list[float]
    phantomPitch:            list[float]
    phiRotation:             bool
    thetaRotation:           bool
    phantomSize: list[int]

    @classmethod
    def fromHeader(cls, header: dict) -> "ProjectionGeometry":
        simulationHeader: dict = header["simulation"]
        tubeHeader: dict       = simulationHeader["tube"]
        geometryHeader: dict   = header["geometry"]

        return cls(
            sourceIsocenterDistance = float(tubeHeader["sourceIsocenterDistance"]),
            sourceDetectorDistance  = float(tubeHeader["sourceDetectorDistance"]),
            gantryAngle             = float(simulationHeader["currentAngle"]),
            focusVector             = list(map(float, geometryHeader["focusVector"])),
            centerVector            = list(map(float, geometryHeader["centerVector"])),
            isocenterOffset         = list(map(float, geometryHeader["isocenterOffset"])),
            piercingPointOffset     = list(map(float, geometryHeader["piercingPointOffset"])),
            phantomPosition         = list(map(float, simulationHeader["phantomPosition"])),
            phantomPitch            = list(map(float, simulationHeader["phantomPitch"])),
            phiRotation             = bool(simulationHeader["phiRotation"]),
            thetaRotation           = bool(simulationHeader["thetaRotation"]),
            phantomSize             = list(map(int, simulationHeader["phantomSize"])),
        )


