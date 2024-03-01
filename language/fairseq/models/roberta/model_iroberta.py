# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

"""
Inductivise Roberta so that it quickly adapts to new data/language
"""
import logging
import torch
from fairseq import distributed_utils
from fairseq.utils import safe_getattr
from fairseq.models import register_model, register_model_architecture


from .model import RobertaModel, base_architecture


logger = logging.getLogger(__name__)


@register_model("iroberta")
class IRobertaModel(RobertaModel):
    def __init__(self, args, encoder):
        super(IRobertaModel, self).__init__(args, encoder)
        self._num_updates = 0
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(IRobertaModel, IRobertaModel).add_args(parser)
        parser.add_argument(
            "--clear_embed_every_K_updates", type=int, default=0,
            help='Clear token embedding layer every K updates'
        )
        parser.add_argument(
            "--emb_reinit", type=str, default='roberta',
            help='How to reinitialise embedding every time we reset them'
        )
    
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates
    
    def reset_embeddings(self, embed_values=None):
        if embed_values == None:
            self.encoder.sentence_encoder.embed_tokens.reset_parameters()
        else:
            with torch.no_grad():
                self.encoder.sentence_encoder.embed_tokens.weight.copy_(embed_values)


# TODO: add ('iroberta', 'iroberta')

@register_model_architecture("iroberta", "iroberta_base")
def iroberta_base_architecture(args):
    args.emb_reinit = safe_getattr(args, "emb_reinit", 'roberta')
    args.clear_embed_every_K_updates = safe_getattr(args, "clear_embed_every_K_updates", 0)
    logger.warning("Clear Token Embedding Layer Every {} Updates".format(args.clear_embed_every_K_updates))
    base_architecture(args)