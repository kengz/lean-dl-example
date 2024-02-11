from pathlib import Path

import hydra

DIR = Path(__file__).parent.parent


def get_cfg(config_dir: str = None, config_name: str = 'config.yaml', overrides: list = []):
    config_dir = config_dir or str(DIR / 'config')
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(config_name, overrides=overrides)
    return cfg


cfg = get_cfg()
