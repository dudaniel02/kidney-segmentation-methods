"""
Shared configuration loader.
Every script does:  from utils.config import load_config
"""

import os
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load YAML config and set nnU-Net environment variables."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Set nnU-Net env vars so CLI tools pick them up
    os.environ["nnUNet_raw"] = str(cfg["paths"]["nnunet_raw"])
    os.environ["nnUNet_preprocessed"] = str(cfg["paths"]["nnunet_preprocessed"])
    os.environ["nnUNet_results"] = str(cfg["paths"]["nnunet_results"])
    
    return cfg


def get_dataset_path(cfg: dict, stage: int) -> Path:
    """Return the nnUNet_raw path for stage 1 or 2."""
    key = f"stage{stage}_name"
    return Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"][key]
