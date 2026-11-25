import logging

logging.getLogger("itk").setLevel(logging.WARNING)

import itk
import numpy as np

from mcgpuReconstruction.data.grid import Grid
from mcgpuReconstruction.data.ct import Ct
from mcgpuReconstruction.data.detector import Detector
from mcgpuReconstruction.data.projectionGeometry import ProjectionGeometry
from mcgpuReconstruction.data.projectionStack import ProjectionStack
from mcgpuReconstruction.reconstruction.filterChoice import (
    FilterChoice, RamLak, Hann, Cosine, Hamming, SheppLogan, filterChoiceFromString, filterChoiceToString,
)

__all__ = ["FdkReconstruction", "IMAGE_TYPE", "ItkImage", "HAS_CUDA", "CUDA_IMAGE_TYPE"]

IMAGE_TYPE: type = itk.Image[itk.F, 3]
ItkImage:   type = type(IMAGE_TYPE.New())

HAS_CUDA: bool = hasattr(itk.RTK, "CudaFDKConeBeamReconstructionFilter")
if HAS_CUDA:
    CUDA_IMAGE_TYPE: type = itk.RTK.CudaImage[itk.F, 3]
else:
    CUDA_IMAGE_TYPE: type = None


class FdkReconstruction:
    """
    FDK cone-beam reconstruction using RTK.

    Parameters
    ----------
    projectionStack : ProjectionStack
        The projection data to reconstruct.
    verbose : bool, optional
        If True, logs diagnostic information during reconstruction.
    """

    def __init__(self, projectionStack: ProjectionStack, verbose: bool = False) -> None:
        self._projectionStack:  ProjectionStack                                 = projectionStack
        self._geometry:         itk.RTK.ThreeDCircularProjectionGeometry | None = None
        self._filterChoice:     FilterChoice                                    = RamLak()
        self._volumeSize:       list[int] | None                                = None
        self._volumeSpacing:    list[float] | None                              = None
        self._volumeOrigin:     list[float] | None                              = None
        self.verbose:           bool                                            = verbose
        self._nDigitsToRoundTo: int                                             = 2

        logging.info("Created instance of FdkReconstruction.")

    @property
    def sourceIsocenterDistance(self) -> float:
        return self._projectionStack.projections[0].geometry.sourceIsocenterDistance

    @property
    def sourceDetectorDistance(self) -> float:
        return self._projectionStack.projections[0].geometry.sourceDetectorDistance

    @property
    def nProjections(self) -> int:
        return self._projectionStack.nProjections

    @property
    def angles(self) -> list[float]:
        return self._projectionStack.angles

    @property
    def detector(self):
        return self._projectionStack.detector

    @property
    def volumeSize(self) -> list[int] | None:
        return self._volumeSize

    @volumeSize.setter
    def volumeSize(self, size: list[int]) -> None:
        if len(size) != 3:
            logging.error(f"volumeSize must have exactly 3 elements, got {len(size)}.")
            return
        if any(v <= 0 for v in size):
            logging.error(f"All volumeSize values must be positive integers, got {size}.")
            return
        self._volumeSize = size
        logging.info(f"Volume size set to {size} [nPixelsX, nPixelsY, nPixelsZ].")

    @property
    def volumeSpacing(self) -> list[float] | None:
        return self._volumeSpacing

    @volumeSpacing.setter
    def volumeSpacing(self, spacing: list[float]) -> None:
        if len(spacing) != 3:
            logging.error(f"volumeSpacing must have exactly 3 elements, got {len(spacing)}.")
            return
        if any(v <= 0 for v in spacing):
            logging.error(f"All volumeSpacing values must be positive, got {spacing}.")
            return
        self._volumeSpacing = spacing
        logging.info(f"Volume spacing set to {spacing} mm [xSpacing, ySpacing, zSpacing].")

    @property
    def volumeOrigin(self) -> list[float] | None:
        if self._volumeOrigin is not None:
            return self._volumeOrigin
        geometry: ProjectionGeometry = self._projectionStack.projections[0].geometry
        return list(geometry.phantomPosition)

    @volumeOrigin.setter
    def volumeOrigin(self, origin: list[float]) -> None:
        if len(origin) != 3:
            logging.error(f"volumeOrigin must have exactly 3 elements, got {len(origin)}.")
            return
        self._volumeOrigin = list(origin)
        logging.info(f"Volume origin set to {origin} mm [xOrigin, yOrigin, zOrigin].")

    @property
    def filter(self) -> str:
        return filterChoiceToString(self._filterChoice)

    @filter.setter
    def filter(self, filterName: str) -> None:
        choice = filterChoiceFromString(filterName)
        if choice is None:
            supported: list[str] = ["ram-lak", "hann", "cosine", "hamming", "shepp-logan"]
            logging.error(f"Unsupported filter '{filterName}'. Choose from: {supported}.")
            return
        self._filterChoice = choice
        logging.info(f"Ramp filter set to '{filterName}' (default parameters).")

    @property
    def filterChoice(self) -> FilterChoice:
        return self._filterChoice

    @filterChoice.setter
    def filterChoice(self, choice: FilterChoice) -> None:
        self._filterChoice = choice
        logging.info(f"Ramp filter set to {choice!r}.")

    @property
    def geometry(self) -> itk.RTK.ThreeDCircularProjectionGeometry:
        if self._geometry is None:
            self._geometry = self._buildGeometry()
        return self._geometry

    def _buildGeometry(self) -> itk.RTK.ThreeDCircularProjectionGeometry:
        geometryObject: itk.RTK.ThreeDCircularProjectionGeometry = itk.RTK.ThreeDCircularProjectionGeometry.New()

        for projection in self._projectionStack.projections:
            geometry: ProjectionGeometry = projection.geometry
            detector: Detector           = projection.detector

            focusVector: np.ndarray = np.array(geometry.focusVector, dtype=np.float64)
            sid: float              = float(
                                        round(
                                            np.linalg.norm(x=focusVector, ord=2),
                                            self._nDigitsToRoundTo
                                        ))

            centerVector: np.ndarray = np.array(geometry.centerVector, dtype=np.float64)
            sdd: float               = float(
                                        round(
                                            np.linalg.norm(x=focusVector - centerVector, ord=2),
                                            self._nDigitsToRoundTo
                                        ))

            gantryAngle: float = round(
                                    float(
                                        np.degrees(
                                            np.arctan2(
                                                centerVector[1],
                                                centerVector[0]
                                            )
                                        )
                                    ) % 360.0,
                                    self._nDigitsToRoundTo
                                )

            uVector: np.ndarray = np.array(detector.uVector, dtype=np.float64)
            vVector: np.ndarray = np.array(detector.vVector, dtype=np.float64)
            wVector: np.ndarray = np.array(detector.wVector, dtype=np.float64)

            projectionOffsetX: float = 0.0
            projectionOffsetY: float = 0.0

            outOfPlaneAngle: float = round(
                                        float(
                                            np.degrees(
                                                np.arcsin(
                                                    np.clip(
                                                        vVector[2],
                                                        -1.0,
                                                        1.0
                                                    )
                                                )
                                            )
                                        ),
                                        self._nDigitsToRoundTo
                                        )

            inPlaneAngle: float = round(
                                        float(
                                            np.degrees(
                                                np.arctan2(
                                                    uVector[1],
                                                    -uVector[2]
                                                )
                                            )
                                        ),
                                        self._nDigitsToRoundTo
                                        )

            idealSourcePosition: np.ndarray = -sid * wVector
            sourceDeviation: np.ndarray     = focusVector - idealSourcePosition
            sourceOffsetX: float            = round(float(np.dot(sourceDeviation, uVector)), self._nDigitsToRoundTo)
            sourceOffsetY: float            = round(float(np.dot(sourceDeviation, vVector)), self._nDigitsToRoundTo)

            geometryObject.AddProjection(
                            sid,
                            sdd,
                            gantryAngle,
                            projectionOffsetX,
                            projectionOffsetY,
                            outOfPlaneAngle,
                            inPlaneAngle,
                            sourceOffsetX,
                            sourceOffsetY
                        )

        logging.info(f"Geometry built: {self.nProjections} projections, SID={self.sourceIsocenterDistance} mm, SDD={self.sourceDetectorDistance} mm.")
        return geometryObject

    def _stackProjections(self) -> ItkImage:
        """
        Stack all projections into a 3D ITK image for the RTK FDK filter.

        ITK reads image origins as the world position of the centre of voxel 0,
        not the outer edge. For an n-pixel detector centred on the piercing
        point, the centre of pixel 0 sits at -(n - 1) / 2 * spacing, not
        -n / 2 * spacing. Using the latter introduces a half-pixel detector
        offset which, after back-projection, becomes a sub-pixel shift in the
        reconstructed volume.
        """
        detector: Detector = self.detector

        sampleData: np.ndarray = self._projectionStack.projections[0].data
        nRows: int             = sampleData.shape[0]
        nColumns: int          = sampleData.shape[1]

        if detector.axes.first == "u":
            rowSpacing: float    = detector.uSpacingMm
            columnSpacing: float = detector.vSpacingMm
        else:
            rowSpacing: float    = detector.vSpacingMm
            columnSpacing: float = detector.uSpacingMm

        cleaned: list[np.ndarray]      = [np.nan_to_num(x=p.data, nan=0.0) for p in self._projectionStack.projections]
        stackedProjections: np.ndarray = np.stack(arrays=cleaned, axis=0).astype(dtype=np.float32)

        itkProjections: ItkImage    = itk.image_from_array(stackedProjections)
        detectorSpacing: itk.Vector = itk.Vector[itk.F, 3]()
        detectorOrigin: itk.Point   = itk.Point[itk.F, 3]()

        detectorSpacing[0]              = columnSpacing
        detectorSpacing[1]              = rowSpacing
        detectorSpacing[2]              = 1.0

        detectorOrigin[0] = -self._projectionStack.projections[0].geometry.piercingPointOffset[1]
        detectorOrigin[1] = -self._projectionStack.projections[0].geometry.piercingPointOffset[0]
        detectorOrigin[2] = 0.0

        itkProjections.SetSpacing(detectorSpacing)
        itkProjections.SetOrigin(detectorOrigin)

        if self.verbose:
            logging.debug(
                f"Stacked projections shape: {stackedProjections.shape} "
                f"(nProj={self.nProjections}, nRows={nRows}, nCols={nColumns}), "
                f"origin: [{detectorOrigin[0]:.4f}, {detectorOrigin[1]:.4f}]."
            )

        return itkProjections

    def _buildVolumeTemplate(self, imageType: type = IMAGE_TYPE) -> ItkImage | None:
        """
        Allocate a zeroed ITK image to serve as the reconstruction volume template.

        """
        if self._volumeSize is None or self._volumeSpacing is None:
            logging.error("volumeSize and volumeSpacing must be set before reconstructing.")
            return None

        volumeTemplate              = imageType.New()
        itkSize:    itk.Size        = itk.Size[3]()
        itkSpacing: itk.Vector      = itk.Vector[itk.F, 3]()
        itkOrigin:  itk.Point       = itk.Point[itk.F, 3]()
        itkRegion:  itk.ImageRegion = itk.ImageRegion[3]()

        # Have to for-loop over this
        for i in range(3):
            itkSize[i]    = self._volumeSize[i]
            itkSpacing[i] = self._volumeSpacing[i]
            itkOrigin[i]  = self.volumeOrigin[i]

        itkRegion.SetSize(itkSize)
        volumeTemplate.SetRegions(itkRegion)
        volumeTemplate.SetSpacing(itkSpacing)
        volumeTemplate.SetOrigin(itkOrigin)
        volumeTemplate.Allocate()
        volumeTemplate.FillBuffer(0)

        if self.verbose:
            logging.debug(
                f"Volume template built: size={list(self._volumeSize)}, "
                f"spacing={list(self._volumeSpacing)}, origin={self.volumeOrigin}."
            )

        return volumeTemplate

    @staticmethod
    def _applyFilter(rampFilter, choice: FilterChoice) -> None:
        match choice:
            case RamLak():
                rampFilter.SetHannCutFrequency(0.0)
            case Hann(cutFrequency=cutFrequency):
                rampFilter.SetHannCutFrequency(cutFrequency)
            case Cosine(cutFrequency=cutFrequency):
                rampFilter.SetCosineCutFrequency(cutFrequency)
            case Hamming(cutFrequency=cutFrequency):
                rampFilter.SetHammingFrequency(cutFrequency)
            case SheppLogan(cutFrequency=cutFrequency):
                rampFilter.SetSheppLoganCutFrequency(cutFrequency)

    def _reconstructOnCpu(self, projections: ItkImage) -> ItkImage:
        fdkFilter = itk.RTK.FDKConeBeamReconstructionFilter[IMAGE_TYPE].New()
        fdkFilter.SetInput(0, self._buildVolumeTemplate())
        fdkFilter.SetInput(1, projections)
        fdkFilter.SetGeometry(self.geometry)
        self._applyFilter(fdkFilter.GetRampFilter(), self._filterChoice)
        fdkFilter.Update()
        return fdkFilter.GetOutput()

    def _reconstructOnGpu(self, projections: ItkImage) -> ItkImage:
        fdkFilter = itk.RTK.CudaFDKConeBeamReconstructionFilter.New()
        fdkFilter.SetInput(0, self._buildVolumeTemplate(imageType=CUDA_IMAGE_TYPE))
        fdkFilter.SetInput(1, self._cpuToCuda(projections))
        fdkFilter.SetGeometry(self.geometry)
        self._applyFilter(fdkFilter.GetRampFilter(), self._filterChoice)
        fdkFilter.Update()
        cudaOutput = fdkFilter.GetOutput()
        cudaOutput.Update()
        return self._cudaToCpu(cudaImage=cudaOutput, cpuTemplate=self._buildVolumeTemplate())

    def reconstruct(self) -> Ct:
        projections: ItkImage = self._stackProjections()

        if HAS_CUDA:
            logging.info("Reconstructing on GPU...")
            itkVolume: ItkImage = self._reconstructOnGpu(projections)
        else:
            logging.info("Reconstructing on CPU...")
            itkVolume: ItkImage = self._reconstructOnCpu(projections)

        volumeArray: np.ndarray = itk.array_from_image(itkVolume)
        nPixelsZ, nPixelsY, nPixelsX = volumeArray.shape
        xSpacing, ySpacing, zSpacing = self._volumeSpacing

        itkOriginXYZ: list[float] = [float(x) for x in itkVolume.GetOrigin()]
        gridOrigin:   list[float] = [itkOriginXYZ[2], itkOriginXYZ[1], itkOriginXYZ[0]]

        grid: Grid = Grid(
            size    = [nPixelsZ, nPixelsY, nPixelsX],
            spacing = [zSpacing, ySpacing, xSpacing],
            origin  = gridOrigin,
        )

        logging.info(f"Reconstruction complete: shape={volumeArray.shape}, range=[{volumeArray.min():.4f}, {volumeArray.max():.4f}], mean={volumeArray.mean():.4f}.")

        ct = Ct(data=volumeArray, grid=grid)
        ct.permute(dimensions=(1, 2, 0))
        ct.flip(axis=(0, 1, 2))
        ct.isReconstructed = True
        return ct

    def computeRmse(self, referenceCt: Ct, testCt: Ct) -> float:
        return float(np.sqrt(np.mean((referenceCt.data - testCt.data) ** 2)))

    @staticmethod
    def _cpuToCuda(cpuImage: ItkImage) -> "CudaImage":
        cudaImage = CUDA_IMAGE_TYPE.New()
        cudaImage.SetPixelContainer(cpuImage.GetPixelContainer())
        cudaImage.CopyInformation(cpuImage)
        cudaImage.SetBufferedRegion(cpuImage.GetBufferedRegion())
        cudaImage.SetRequestedRegion(cpuImage.GetRequestedRegion())
        return cudaImage

    @staticmethod
    def _cudaToCpu(cudaImage: "CudaImage", cpuTemplate: ItkImage) -> ItkImage:
        cpuTemplate.Update()
        dataManager = cudaImage.GetCudaDataManager()
        dataManager.SetCPUBufferPointer(cpuTemplate.GetBufferPointer())
        dataManager.UpdateCPUBuffer()
        return cpuTemplate


