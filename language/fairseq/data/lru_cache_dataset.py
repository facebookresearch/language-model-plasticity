# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

from functools import lru_cache

from . import BaseWrapperDataset


class LRUCacheDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        return self.dataset[index]

    @lru_cache(maxsize=8)
    def collater(self, samples):
        return self.dataset.collater(samples)
