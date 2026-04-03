import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

PROBLEMATIC_SITES: dict[str, str] = {
    'PITT':      'Pitt',
    'MAX_MUN':   'MaxMun_a',
    'YALE':      'Yale',
    'TRINITY':   'Trinity',
    'STANFORD':  'Stanford',
    'CALTECH':   'Caltech',
    'OLIN':      'Olin',
    'LEUVEN_2':  'Leuven_2',
    'LEUVEN_1':  'Leuven_1',
}

KEEP_COLUMNS = ['SUB_ID', 'SITE_ID', 'FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'EYE_STATUS_AT_SCAN']


# ─────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────

def _build_file_id(row: pd.Series, site_prefix: str) -> str:
    """Construct a FILE_ID from a row's SUB_ID and its site prefix."""
    sub_id = str(row['SUB_ID']).zfill(7)
    return f"{site_prefix}_{sub_id}"


def fix_filenames(*, df: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing FILE_IDs (where FILE_ID == 'no_filename') for all sites."""
    df = df.copy()
    total_fixed = 0

    for site, prefix in PROBLEMATIC_SITES.items():
        mask = (df['SITE_ID'] == site) & (df['FILE_ID'] == 'no_filename')
        for idx in df.index[mask]:
            df.loc[idx, 'FILE_ID'] = _build_file_id(df.loc[idx], prefix)
            total_fixed += 1

    non_problematic_mask = (~df['SITE_ID'].isin(PROBLEMATIC_SITES)) & (df['FILE_ID'] == 'no_filename')
    for idx in df.index[non_problematic_mask]:
        df.loc[idx, 'FILE_ID'] = _build_file_id(df.loc[idx], df.loc[idx, 'SITE_ID'])
        total_fixed += 1

    print(f"  Fixed {total_fixed} missing FILE_IDs.")
    return df


def filter_rows_by_available_files(*, df: pd.DataFrame, raw_fmri_path: Path, atlas: str) -> pd.DataFrame:
    """Remove rows whose FILE_ID has no matching file on disk."""
    available_ids = {
        f.name.replace(f"_rois_{atlas}.1D", "")
        for f in raw_fmri_path.glob(f"*_rois_{atlas}.1D")
    }

    if not available_ids:
        raise FileNotFoundError(
            f"No files matching '*_rois_{atlas}.1D' found in {raw_fmri_path}.\n"
            "Check that RAW_PATH and ATLAS in config.yaml are correct."
        )

    missing_ids = set(df['FILE_ID']) - available_ids
    if missing_ids:
        print(f"  Removing {len(missing_ids)} rows with no matching file.")

    filtered_df = df[df['FILE_ID'].isin(available_ids)].reset_index(drop=True)

    assert len(filtered_df) == len(available_ids & set(df['FILE_ID'])), \
        "Mismatch between filtered rows and available file IDs."

    if filtered_df.empty:
        raise ValueError(
            "All rows were filtered out — FILE_IDs in the CSV do not match any file on disk.\n"
            f"  Sample disk IDs : {sorted(available_ids)[:5]}\n"
            f"  Sample CSV IDs  : {df['FILE_ID'].tolist()[:5]}\n"
            "Check fix_filenames logic or RAW_PATH/ATLAS in config.yaml."
        )

    print(f"  Kept {len(filtered_df)}/{len(df)} rows after filtering.")
    return filtered_df


def fix_dx_group(*, df: pd.DataFrame) -> pd.DataFrame:
    """Remap DX_GROUP: 2 (control) → 0, 1 (ASD) stays as 1."""
    df = df.copy()
    df['DX_GROUP'] = df['DX_GROUP'].replace({2: 0})

    unexpected = set(df['DX_GROUP'].unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"Unexpected values in DX_GROUP after remap: {unexpected}")

    print(f"  DX_GROUP distribution: {df['DX_GROUP'].value_counts().to_dict()}")
    return df


def add_n_timesteps_info(*, df: pd.DataFrame, raw_fmri_path: Path, atlas: str, n_rois: int, column_name: str) -> pd.DataFrame:
    """Add a column with the number of timesteps (T) for each subject's fMRI file."""
    df = df.copy()

    if df.empty:
        raise ValueError("DataFrame is empty before adding timesteps — check the filtering step.")

    for idx, row in df.iterrows():
        data_path = raw_fmri_path / (row['FILE_ID'] + f'_rois_{atlas}.1D')
        data = np.loadtxt(data_path)

        if idx == 0 and data.shape[1] != n_rois:
            raise ValueError(f"Expected data with {n_rois} ROIs, got shape {data.shape}.")

        df.loc[idx, column_name] = data.shape[0]

    df[column_name] = df[column_name].astype('int64')
    print(f"  Successfully added '{column_name}' column.")
    return df


def select_columns(*, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only the relevant columns."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[columns].copy()


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    raw_fmri_path: Path,
    atlas: str,
    n_rois: int,
    timesteps_column: str,
    *,
    step_fix_filenames: bool = True,
    step_filter_rows: bool = True,
    step_add_timesteps: bool = True,
    step_fix_dx_group: bool = True,
    step_select_columns: bool = True,
) -> pd.DataFrame:

    steps = [
        (step_fix_filenames,  "Fix missing FILE_IDs",          lambda d: fix_filenames(df=d)),
        (step_filter_rows,    "Filter rows by available files", lambda d: filter_rows_by_available_files(df=d, raw_fmri_path=raw_fmri_path, atlas=atlas)),
        (step_add_timesteps,  "Add number of timesteps",       lambda d: add_n_timesteps_info(df=d, raw_fmri_path=raw_fmri_path, atlas=atlas, n_rois=n_rois, column_name=timesteps_column)),
        (step_fix_dx_group,   "Fix DX_GROUP labels",           lambda d: fix_dx_group(df=d)),
        (step_select_columns, "Select columns",                lambda d: select_columns(df=d, columns=KEEP_COLUMNS + [timesteps_column])),
    ]

    for enabled, label, step_fn in steps:
        if enabled:
            print(f"[Step] {label}...")
            df = step_fn(df)

    return df


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    raw_fmri_path = Path(config["RAW_PATH"]).resolve()
    df = pd.read_csv(Path(args.input_csv).resolve())

    df = run_pipeline(
        df,
        raw_fmri_path    = raw_fmri_path,
        atlas            = config["ATLAS"],
        n_rois           = config["N_ROIS"],
        timesteps_column = args.timesteps_c_name,
        step_fix_filenames  = not args.skip_fix_filenames,
        step_filter_rows    = not args.skip_filter_rows,
        step_add_timesteps  = not args.skip_add_timesteps,
        step_fix_dx_group   = not args.skip_fix_dx_group,
        step_select_columns = not args.skip_select_columns,
    )

    out_path = Path(args.output_csv).resolve()
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline: raw CSV → data_train.csv"
    )
    parser.add_argument("--config",           default="./config/config.yaml")
    parser.add_argument("--input_csv",        default="./data/csv/Phenotypic_V1_0b_preprocessed1.csv")
    parser.add_argument("--output_csv",       default="./data/csv/data_train.csv")
    parser.add_argument("--timesteps_c_name", default="T")

    parser.add_argument("--skip_fix_filenames",   action="store_true")
    parser.add_argument("--skip_filter_rows",     action="store_true")
    parser.add_argument("--skip_add_timesteps",   action="store_true")
    parser.add_argument("--skip_fix_dx_group",    action="store_true")
    parser.add_argument("--skip_select_columns",  action="store_true")

    main(parser.parse_args())