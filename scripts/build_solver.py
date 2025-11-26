import os
from sdf_nmpc import default_config_dir
from sdf_nmpc.ocp import build_solver
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='mode', help='Build for given param file.')
    args = parser.parse_args()

    cfg_file = f'{args.mode}.yaml'

    print(f'Building for {args.mode}, loading {cfg_file}')

    build_solver(os.path.join(default_config_dir(), cfg_file))
