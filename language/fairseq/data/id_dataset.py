# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch

from . import FairseqDataset


class IdDataset(FairseqDataset):
    def __getitem__(self, index):
        return index

    def __len__(self):
        return 0

    def collater(self, samples):
        return torch.tensor(samples)
