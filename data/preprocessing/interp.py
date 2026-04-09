import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.augmentation.augmentation import interpolate_timeseries

REQUIRED_COLUMNS = ["FILE_ID", "SITE_ID"]


def generate_interp(
    config: dict,
    df: pd.DataFrame
) -> int:

    nro_rois = config["N_ROIS"]
    raw_path = Path(config["RAW_PATH"])
    atlas = config["ATLAS"]
    tr_goal = config["TR"]
    output_path = Path(config["INTERP_PATH"])
    prefix = config["PREFIX"]
    min_timesteps = config["MIN_TIMESTEPS"]

    output_path.mkdir(parents=True, exist_ok=True)

    idx = 0
    for idx, row in df.iterrows():
        filename = row.FILE_ID + f"_rois_{atlas}.1D"

        arr: np.ndarray = np.loadtxt(raw_path / filename)

        if arr.shape[1] != nro_rois:
            raise ValueError(f"Se esperaba N_ROIS en la segunda dimension del array cargado, obtenido: {arr.shape[1]}")

        arr = arr.T.copy()

        arr_interp = interpolate_timeseries(arr, row.SITE_ID, tr_goal, min_timesteps)
        output_filename = prefix + filename
        np.savetxt(output_path / output_filename, arr_interp)

    return idx


def load_and_validate_df(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["CSV_PATH"])
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"El DataFrame no contiene las columnas requeridas: {missing}")
    return df


def main(args: argparse.Namespace) -> None:

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = load_and_validate_df(config)

    total = generate_interp(config, df)

    print(f"Series temporales interpoladas con exito. Total: {total} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate and crop ROI timeseries from fMRI atlas data."
    )
    parser.add_argument("--config", default="./config/config.yaml")
    args = parser.parse_args()
    main(args)