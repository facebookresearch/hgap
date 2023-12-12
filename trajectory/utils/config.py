# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import collections
import pickle
import torch 

class Config(collections.Mapping):

    def __init__(self, _class, verbose=True, savepath=None, **kwargs):
        self._class = _class
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        # only print and save on the main process
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0
        if rank == 0:
            if verbose:
                print(self)

            if savepath is not None:
                savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath
                pickle.dump(self, open(savepath, 'wb'))
                print(f'Saved config to: {savepath}\n')


    def __repr__(self):
        string = f'\nConfig: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __call__(self):
        return self.make()

    def __getattr__(self, attr):
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def make(self, **kwargs):
        if 'GPT' in str(self._class) or 'VAE' in str(self._class) or 'Trainer' in str(self._class) or 'Prior' in str(self._class) or 'Critic' in str(self._class):
            return self._class(self)
        else:
            return self._class(**self._dict, **kwargs)
