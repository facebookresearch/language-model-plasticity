# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import importlib
import os


# automatically import any Python files in the criterions/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("fairseq.model_parallel.criterions." + module)
