from sdf_nmpc import default_config_dir
from sdf_nmpc.utils.stability import get_r_tilde_max
from sdf_nmpc.utils.config import Config
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='params', help='Compute for given param file.')
    args = parser.parse_args()

    cfg_file = f'params_{args.params}.yaml'
    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))

    sol = get_r_tilde_max(cfg)
    print(f'maximum r_tilde value: {sol:.4f}')
