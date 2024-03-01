# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import importlib
import os

from .ema import EMA


def build_ema(model, cfg, device):
    return EMA(model, cfg, device)


# automatically import any Python files in the models/ema/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.models.ema." + file_name)
