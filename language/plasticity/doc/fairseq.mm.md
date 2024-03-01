# Notes on fairseq code structure
- view the mindmap using markmap
# fairseq_cli
## train.py
### main
#### train
##### trainer.train_step
###### model.set_num_updates()
###### model.train()
# fairseq
## trainer.py
### Trainer
#### self._wrapped_model
## models
### fairseq_encoder.py
#### FairseqEncoder
- set_num_updates
### transformer
#### transformer_encoder.py
##### TransformerEncoder
###### TransformerEncoderBase
### roberta
#### model.py
##### RobertaModel <- FairseqEncoderModel ('roberta')
###### RobertaEncoder <- FairseqEncoder
- sentence_encoder <- TransformerEncoder
- lm_head
###### RobertaClassificationHead
##### roberta_base_architecture ('roberta', 'base')


