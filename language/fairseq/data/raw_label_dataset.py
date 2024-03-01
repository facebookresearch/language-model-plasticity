# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch

from . import FairseqDataset


class RawLabelDataset(FairseqDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.tensor(samples)
