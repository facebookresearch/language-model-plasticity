# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@register_tokenizer("space", dataclass=FairseqDataclass)
class SpaceTokenizer(object):
    def __init__(self, *unused):
        self.space_tok = re.compile(r"\s+")

    def encode(self, x: str) -> str:
        return self.space_tok.sub(" ", x)

    def decode(self, x: str) -> str:
        return x
