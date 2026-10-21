"""Small helpers shared by every stage: project paths, the model config, logging, JSON output.

Usage::

    from src.utils.utils import load_config, setup_logger
    cfg = load_config()                 # configs/final_model.yaml
    cfg.features                        # the 111 model features, in the model's column order
    cfg.path('raw_dir')                 # absolute path
    log = setup_logger('train', 'artifacts/logs/train.log')
"""
import json
import logging
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'final_model.yaml'


class Config(dict):
    """The YAML config as a dict, with the feature list and paths at hand."""

    @property
    def features(self):
        return [c for table in self['features'].values() for c in table]

    @property
    def feature_sources(self):
        return {c: table for table, columns in self['features'].items() for c in columns}

    @property
    def model_tables(self):
        """The feature tables the model's features come from, in the configured order."""
        return [t for t in self['tables'] if t in self['features']]

    def path(self, name):
        p = Path(self['paths'][name])
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path=DEFAULT_CONFIG):
    cfg = Config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
    features = cfg.features
    if len(features) != len(set(features)):
        raise ValueError(f'{path}: a feature is listed twice')
    unknown = set(cfg['features']) - set(cfg['tables'])
    if unknown:
        raise ValueError(f'{path}: features listed for tables that are not built: {sorted(unknown)}')
    return cfg


def setup_logger(name, log_file=None, level=logging.INFO):
    """A logger writing to the console and, if given, to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handlers = [logging.StreamHandler()]
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def to_builtin(value):
    """numpy and pandas values to plain Python, for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return str(value)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=to_builtin), encoding='utf-8')
    return path
