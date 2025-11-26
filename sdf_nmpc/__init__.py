from importlib.resources import files, as_file
from platformdirs import user_cache_dir
from pathlib import Path


def default_config_dir() -> Path:
    cfg_pkg = files("sdf_nmpc") / "config"
    with as_file(cfg_pkg) as p:
        return Path(p)

def default_data_dir() -> Path:
    cfg_pkg = files("sdf_nmpc") / "data"
    with as_file(cfg_pkg) as p:
        return Path(p)

def cache_dir() -> Path:
    cache_dir = Path(user_cache_dir('sdf_nmpc'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir