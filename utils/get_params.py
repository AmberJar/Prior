import os
import argparse
import json


def get_hparams(config, init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=config,
                        help='JSON file for configuration')

    args = parser.parse_args()

    config_path = args.config
    if init:
        with open(config_path, "r") as f:
            data = f.read()

    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()