import logging
import itk
import numpy as np
from pathlib import Path

__all__ = ["MhaWriter"]


class MhaWriter:
    def write(
            self,
            data: np.ndarray,
            origin: tuple[float, float, float],
            spacing: tuple[float, float, float],
            path: str | Path,
            dtype: np.dtype = np.float32
            ) -> None:
        logging.info(f"Writing data to file: {path}")

        if not self._validateInputs(data=data, origin=origin, spacing=spacing, path=path, dtype=dtype):
            logging.error(f"MhaWriter failed to write data to file!")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            itkImage: itk.Image = itk.GetImageFromArray(arr=data.astype(dtype))
            itkImage.SetSpacing(spacing)
            itkImage.SetOrigin(origin)

            itk.imwrite(itkImage, filename=path, compression=True)
        except Exception as e:
            logging.error(f"MhaWriter failed to write data to file with error: {e}")

    @staticmethod
    def _validateInputs(
            data: np.ndarray,
            origin: tuple[float, float, float],
            spacing: tuple[float, float, float],
            path: str | Path,
            dtype: np.dtype
    ) -> bool:
        logging.info(f"Validating input arguments")

        if not isinstance(data, np.ndarray):
            logging.error(f"Input argument data for MhaWriter must be numpy array, got {type(data).__name__} instead")
            return False

        if data.ndim != 3:
            logging.error(f"Input argument data for MhaWriter must be 3D array, "
                          f"got {data.ndim}D with shape {data.shape} instead")
            return False

        if data.size == 0:
            logging.error(f"Input argument data for MhaWriter is empty!")
            return False

        if not np.all(np.isfinite(data)):
            logging.error(f"Input argument data for MhaWriter contains NaN or Inf values!")
            return False

        if len(origin) != 3:
            logging.error(f"Input argument origin for MhaWriter must have 3 elements, got {len(origin)} instead")
            return False

        if len(spacing) != 3:
            logging.error(f"Input argument spacing for MhaWriter must be 3 elements, got {len(spacing)} instead")
            return False

        if any(s <= 0 for s in spacing):
            logging.error(f"Input argument spacing must be strictly positive, got {spacing} instead")
            return False

        if not isinstance(dtype, type) or not issubclass(dtype, np.generic):
            logging.error(f"Input argument dtype for MhaWriter must be of type np.dtype (e.g. np.float32),"
                          f" got {type(dtype).__name__} instead")
            return False

        if not isinstance(path, (str, Path)):
            logging.error(f"Input argument path for MhaWriter must be of type str or Path, "
                          f"got {type(path).__name__} instead")
            return False

        if Path(path).suffix.lower() != ".mha":
            logging.error(f"Input argument path for MhaWriter must have extension '.mha', "
                          f"got {Path(path).suffix} instead")
            return False
        return True

