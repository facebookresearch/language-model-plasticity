# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from . import BaseWrapperDataset


class OffsetTokensDataset(BaseWrapperDataset):
    def __init__(self, dataset, offset):
        super().__init__(dataset)
        self.offset = offset

    def __getitem__(self, idx):
        return self.dataset[idx] + self.offset
