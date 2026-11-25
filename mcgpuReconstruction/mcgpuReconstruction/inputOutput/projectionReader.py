import json
import logging
from pathlib import Path

import numpy as np

__all__ = ["readHeader", "readBinaryProjection", "readProjectionAngle", "groupProjectionsByCase"]


def readHeader(path: Path) -> dict | None:
    headerPath = path.with_suffix(".json")
    if not headerPath.is_file():
        logging.error(f"No header found for {path}, skipping this projection.")
        return None
    with open(headerPath) as file:
        return json.load(file)


def _binaryDtype(rawString: str, endianness: str) -> np.dtype:
    byteOrder = "<" if endianness == "L" else ">"
    if rawString == "single":
        return np.dtype(np.float32).newbyteorder(byteOrder)
    if rawString == "double":
        return np.dtype(np.float64).newbyteorder(byteOrder)
    raise ValueError(f"Unsupported binary dtype string: {rawString!r}")


def readBinaryProjection(path: Path, header: dict, datasetIndex: int = 0) -> np.ndarray:
    size: list[int] = header["data"]["size"]
    nCols, nRows = size[0], size[1]
    nDatasets = size[2] if len(size) > 2 else 1
    typeList = header["data"]["type"]
    dtypeString = typeList[datasetIndex] if isinstance(typeList, list) else typeList
    endianness = header["data"].get("endianness", "L")
    dtype = _binaryDtype(rawString=dtypeString, endianness=endianness)

    rawData = np.frombuffer(path.read_bytes(), dtype=dtype)
    projection = rawData.reshape((nCols, nRows, nDatasets), order="F")
    return projection[:, :, datasetIndex].astype(np.float32)


def readProjectionAngle(path: Path) -> float:
    headerPath = path.with_suffix(".json")
    if headerPath.is_file():
        try:
            return float(json.loads(headerPath.read_text())["simulation"]["currentAngle"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            pass
    try:
        return float(path.stem.split("_projection_")[1].split("_degrees")[0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot determine the projection angle of {path}") from error


def groupProjectionsByCase(searchFolder: str | Path) -> dict[str, list[Path]]:
    searchFolder = Path(searchFolder)
    paths = sorted(searchFolder.rglob("*.bin"))
    if not paths:
        raise ValueError(f"No .bin files found under {searchFolder} (checked recursively).")

    groups: dict[str, list[Path]] = {}
    for path in paths:
        if "_projection_" not in path.stem:
            logging.warning(f"Skipping {path}: filename doesn't match the expected "
                            f"'<caseName>_projection_<angle>_degrees' pattern.")
            continue
        caseName = path.stem.split("_projection_")[0]
        groups.setdefault(caseName, []).append(path)

    if not groups:
        raise ValueError(f"Found files under {searchFolder}, but none matched the expected "
                         f"'<caseName>_projection_<angle>_degrees' filename pattern.")
    return groups
