import argparse
from base.utils.parse_utils import parse_ini, parse_value
from collections import namedtuple 
import importlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Pretrain')
    parser.add_argument('--config', default='./config.ini',
                        help='Config .ini file directory')
    parser.add_argument('--clean', action='store_true', help='Clean experiments')
    parser.add_argument('--background', action='store_true', help='Run experiment in the background')
    parser.add_argument('--override', default=None, help='Arguments for overriding config')
    parser.add_argument('--exp_name', default='normal_exp', help='Name of experiment to run')

    args = parser.parse_args()

    cfg = parse_ini(args.config)

    # Display config
    print("Display Config:")
    print(open(args.config, 'r').read())

    # Confirm config to prevent errors
    if not args.background:
        choice = input("If config is correct, press y: ", )
        if choice != 'y':
            print('Exit!')
            exit()

    if args.override is not None:
        equality_split = args.override.split('=')
        num_equality = len(equality_split)
        assert num_equality > 0
        if num_equality == 2:
            override_dict = {equality_split[0]: parse_value(equality_split[1])}
        else:
            keys = [equality_split[0]]  # First key
            keys += [equality.split(',')[-1] for equality in equality_split[1:-1]]  # Other keys
            values = [equality.replace(',' + key, '') for equality, key in zip(equality_split[1:-1], keys[1:])]  # Get values other than last field
            values.append(equality_split[-1])  # Get last value
            values = [value.replace('[', '').replace(']', '') for value in values]

            override_dict = {key: parse_value(value) for key, value in zip(keys, values)}

        cfg_dict = cfg._asdict()

        Config = namedtuple('Config', tuple(set(cfg._fields + tuple(override_dict.keys()))))
        
        cfg_dict.update(override_dict)

        cfg = Config(**cfg_dict)
        cfg = cfg._replace(name=cfg.name + "_" + args.override)

    importlib.import_module(f'exp_utils.{args.exp_name}').run_exp(cfg)


if __name__ == '__main__':
    main()
