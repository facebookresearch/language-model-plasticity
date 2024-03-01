# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch

from . import BaseWrapperDataset


class RollDataset(BaseWrapperDataset):
    def __init__(self, dataset, shifts):
        super().__init__(dataset)
        self.shifts = shifts

    def __getitem__(self, index):
        item = self.dataset[index]
        return torch.roll(item, self.shifts)
