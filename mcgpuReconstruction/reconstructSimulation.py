import logging

logging.getLogger("itk").setLevel(logging.WARNING)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from mcgpuReconstruction.data.ct import Ct
from mcgpuReconstruction.inputOutput.mhaWriter import MhaWriter
from mcgpuReconstruction.inputOutput.projectionReader import groupProjectionsByCase, readProjectionAngle
from mcgpuReconstruction.reconstruction.caseReconstruction import reconstructRawCase
from mcgpuReconstruction.reconstruction.filterChoice import FilterChoice, RamLak


def reconstructFolder(
        inputFolder:  Path,
        outputFolder: Path,
        datasetIndex: int          = 0,
        filterChoice: FilterChoice = RamLak(),
) -> dict[str, Ct]:
    cases: dict[str, list[Path]] = groupProjectionsByCase(inputFolder)
    logging.info(f"Found {len(cases)} case(s) in {inputFolder}: {list(cases)}")

    cts: dict[str, Ct] = {}
    for caseName, projectionPaths in cases.items():
        projectionPaths = sorted(projectionPaths, key=readProjectionAngle)
        logging.info(f"Reconstructing {caseName} from {len(projectionPaths)} projection(s)")
        ct: Ct = reconstructRawCase(projectionPaths, datasetIndex=datasetIndex, filterChoice=filterChoice)

        caseOutputFolder: Path = outputFolder / caseName
        MhaWriter().write(
            data=ct.data,
            origin=tuple(ct.grid.origin),
            spacing=tuple(ct.grid.spacing),
            path=caseOutputFolder / f"{caseName}.mha",
        )
        ct.visualize(title=caseName, save=True, outputFolder=caseOutputFolder)
        cts[caseName] = ct
    return cts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(asctime)s] - %(message)s")

    INPUT_FOLDER:  Path         = Path(r"path/to/simulation/folder")
    OUTPUT_FOLDER: Path         = INPUT_FOLDER / "reconstruction"
    DATASET_INDEX: int          = 0
    FILTER_CHOICE: FilterChoice = RamLak()

    reconstructFolder(INPUT_FOLDER, OUTPUT_FOLDER, DATASET_INDEX, FILTER_CHOICE)
