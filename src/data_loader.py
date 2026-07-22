import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import pyreadstat
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Named Constants
DEFAULT_TIMEOUT_SEC: int = 30
CHUNK_SIZE_BYTES: int = 8192
DATA_ENCODING: str = "latin1"
MIN_ACCEPTABLE_ROW_RATIO: float = 0.40  # Warn if we keep less than 40% of demographic participants

# Dataset Configuration Dict
NHANES_DATASETS: Dict[str, Dict[str, str]] = {
    "demographics": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",
        "filename": "DEMO_L.xpt"
    },
    "body_measures": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt",
        "filename": "BMX_L.xpt"
    },
    "biochemistry": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.xpt",
        "filename": "BIOPRO_L.xpt"
    },
    "glycohemoglobin": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt",
        "filename": "GHB_L.xpt"
    },
    "cholesterol": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt",
        "filename": "TCHOL_L.xpt"
    },
    "smoking": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.xpt",
        "filename": "SMQ_L.xpt"
    },
    "complete_blood_count": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt",
        "filename": "CBC_L.xpt"
    },
    "physical_activity": {
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.xpt",
        "filename": "PAQ_L.xpt"
    }
}


def download_datasets(raw_dir: Path, datasets: Dict[str, Dict[str, str]]) -> None:
    """Downloads NHANES .XPT files from the CDC servers if they do not already exist.

    Args:
        raw_dir (Path): Path to directory where raw files should be saved.
        datasets (Dict[str, Dict[str, str]]): Dictionary containing urls and filenames.
    
    Raises:
        requests.RequestException: If a network request fails.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for name, info in datasets.items():
        dest_path = raw_dir / info["filename"]
        if not dest_path.exists():
            logger.info(f"Downloading {name} from {info['url']}...")
            try:
                response = requests.get(info["url"], stream=True, timeout=DEFAULT_TIMEOUT_SEC)
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        f.write(chunk)
                logger.info(f"Successfully downloaded {info['filename']}.")
            except requests.RequestException as e:
                logger.error(f"Failed to download {name} from {info['url']}: {e}")
                raise e
        else:
            logger.info(f"File {info['filename']} already exists, skipping download.")


def load_datasets(raw_dir: Path, datasets: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    """Loads downloaded .XPT files into pandas DataFrames.

    Args:
        raw_dir (Path): Directory where raw XPT files are stored.
        datasets (Dict[str, Dict[str, str]]): Dataset configurations.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset names to their DataFrames.
    
    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If loading fails due to file corruption or invalid formats.
    """
    dfs = {}
    for name, info in datasets.items():
        file_path = raw_dir / info["filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        logger.info(f"Loading {info['filename']} using pyreadstat...")
        try:
            df, meta = pyreadstat.read_xport(str(file_path), encoding=DATA_ENCODING)
            logger.info(f"Loaded {info['filename']} with shape {df.shape}")
            dfs[name] = df
        except Exception as e:
            logger.error(f"Error reading SAS xport file {file_path}: {e}")
            raise ValueError(f"Failed to load {file_path}: {e}") from e
    return dfs


def merge_datasets(dfs: Dict[str, pd.DataFrame], merge_on: str = "SEQN") -> pd.DataFrame:
    """Merges multiple DataFrames on a common column using an inner join.

    Args:
        dfs (Dict[str, pd.DataFrame]): Dictionary of DataFrames.
        merge_on (str): The column name to join on.

    Returns:
        pd.DataFrame: The merged DataFrame.
        
    Raises:
        KeyError: If the merge key is missing from any DataFrame.
        ValueError: If the input dict of DataFrames is empty.
    """
    if not dfs:
        raise ValueError("Cannot merge an empty dictionary of DataFrames.")
        
    keys = list(dfs.keys())
    merged_df = dfs[keys[0]]
    
    if merge_on not in merged_df.columns:
        raise KeyError(f"Merge key '{merge_on}' is missing from the '{keys[0]}' DataFrame.")
        
    logger.info(f"Merging datasets on '{merge_on}'...")
    for key in keys[1:]:
        df = dfs[key]
        if merge_on not in df.columns:
            raise KeyError(f"Merge key '{merge_on}' is missing from the '{key}' DataFrame.")
        
        # Identify overlapping columns (excluding the merge key)
        overlapping_cols = set(merged_df.columns).intersection(set(df.columns)) - {merge_on}
        if overlapping_cols:
            logger.info(f"Dropping overlapping columns from {key} to avoid merge conflicts: {overlapping_cols}")
            df = df.drop(columns=list(overlapping_cols))
            
        merged_df = pd.merge(merged_df, df, on=merge_on, how="inner")
        logger.info(f"Merged {key}, current shape: {merged_df.shape}")
        
    return merged_df


def validate_merge(merged_df: pd.DataFrame, initial_df: pd.DataFrame, merge_on: str = "SEQN") -> None:
    """Validates the merged dataset for row-count drops, missingness, and uniqueness.

    Args:
        merged_df (pd.DataFrame): The final merged DataFrame.
        initial_df (pd.DataFrame): The base demographics DataFrame before any merge.
        merge_on (str): The column used to join the tables.
    """
    logger.info("Validating merged dataset...")
    
    # 1. Uniqueness check
    if not merged_df[merge_on].is_unique:
        logger.warning(f"Merge key '{merge_on}' contains duplicate values in the merged dataset!")
    else:
        logger.info(f"Uniqueness check passed: '{merge_on}' is unique in the merged dataset.")
        
    # 2. Row count retention check
    initial_rows = len(initial_df)
    final_rows = len(merged_df)
    retention_ratio = final_rows / initial_rows if initial_rows > 0 else 0
    
    logger.info(f"Row retention: {final_rows} / {initial_rows} ({retention_ratio:.1%})")
    if retention_ratio < MIN_ACCEPTABLE_ROW_RATIO:
        logger.warning(
            f"Significant row-count drop! Kept only {retention_ratio:.1%} of demographic subjects. "
            f"Check if inner joins are too restrictive."
        )
        
    # 3. Missingness check
    total_cells = merged_df.size
    missing_cells = merged_df.isnull().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    logger.info(f"Overall missingness ratio in merged dataset: {missing_ratio:.1%}")
    
    # Identify high missingness features
    missing_by_col = merged_df.isnull().mean()
    high_missingness_cols = missing_by_col[missing_by_col > 0.50]
    if not high_missingness_cols.empty:
        logger.warning(
            f"There are {len(high_missingness_cols)} columns with >50% missing values. "
            f"Top missing columns:\n{high_missingness_cols.sort_values(ascending=False).head(10)}"
        )


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the merged DataFrame as a CSV file to the processed data directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (Path): Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully.")


def main() -> pd.DataFrame:
    """Main orchestration function to run the data collection pipeline.

    Returns:
        pd.DataFrame: The validated and merged NHANES DataFrame.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    raw_dir = project_root / "data" / "raw"
    processed_file = project_root / "data" / "processed" / "merged_data.csv"
    
    # 1. Download all 8 NHANES tables (skips any files already present on disk)
    download_datasets(raw_dir, NHANES_DATASETS)
    
    # 2. Load SAS tables
    dfs = load_datasets(raw_dir, NHANES_DATASETS)
    
    # 3. Merge SAS tables
    merged_df = merge_datasets(dfs, merge_on="SEQN")
    
    # 4. Perform pipeline validations
    validate_merge(merged_df, dfs["demographics"], merge_on="SEQN")
    
    # 5. Save the output
    save_dataset(merged_df, processed_file)
    
    return merged_df


if __name__ == "__main__":
    main()
