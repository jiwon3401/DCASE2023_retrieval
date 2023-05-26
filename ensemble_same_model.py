import re
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from trainer.trainer import Task
from tools.config_loader import get_config
from pathlib import Path
from data_handling.DataLoader import get_dataloader
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.make_csvfile import make_csv
import pickle


from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger


from models.ASE_model import ASE
import lightning.pytorch as pl

from models.ASE_model import ASE


model = ASE(config)

#baseline model
ckp_path1 = '/home/user/audio-text_retrieval/outputs/0511_freeze_True_lr_0.0001_seed_20/models/best_checkpoint.ckpt'
#different seeds
ckp_path2 = '/home/user/audio-text_retrieval/outputs/0511_diff1_freeze_True_lr_0.0001_seed_42/models/best_checkpoint.ckpt'
ckp_path3 = '/home/user/audio-text_retrieval/outputs/0511_diff2_freeze_True_lr_0.0001_seed_1234/models/best_checkpoint.ckpt'
ckp_path4 = '/home/user/audio-text_retrieval/outputs/0511_diff3_freeze_True_lr_0.0001_seed_0/models/best_checkpoint.ckpt'


def ensemble_ckp(ckp_path1, ckp_path2, ckp_path3, ckp_path4):
    model1 = torch.load(ckp_path1)
    model2 = torch.load(ckp_path2)
    model3 = torch.load(ckp_path3)
    model4 = torch.load(ckp_path4)

    sdA = model1['state_dict']
    sdB = model2['state_dict']
    sdC = model3['state_dict']
    sdD = model4['state_dict']
    
    for key in sdA:
        sdD[key] = (sdD[key]+sdC[key]+sdB[key]+sdA[key])/4.
        
    for key in list(sdD.keys()):
        if (key.startswith("audio_enc.audio_enc.") or key.startswith("audio_linear.") 
            or key.startswith("text_enc.") or key.startswith("text_linear.")):
            new_key = "model." + key
            sdD[new_key] = sdD.pop(key)
            
    return sdD

model.load_state_dict(ensemble_ckp(ckp_path1, ckp_path2, ckp_path3, ckp_path4))


# def ensemble_ckp(ckp_path1, ckp_path2, ckp_path3, ckp_path4):
#     model1 = torch.load(ckp_path1)
#     model2 = torch.load(ckp_path2)
#     model3 = torch.load(ckp_path3)
#     model4 = torch.load(ckp_path4)

#     sdA = model1['state_dict']
#     sdB = model2['state_dict']
#     sdC = model3['state_dict']
#     sdD = model4['state_dict']

#     for key in sdA:
#         sdD[key] = (sdD[key]+sdC[key]+sdB[key]+sdA[key])/4.

#     # remove the 'model.' prefix from the keys
#     for key in list(sdD.keys()):
#         if key.startswith('model.'):
#             new_key = key.replace('model.', '')
#             sdD[new_key] = sdD.pop(key)

#     new_state_dict = {}
#     for key, value in sdD.items():
#         # Replace 'auto_encoder' with 'auto_model.encoder'
#         new_key = re.sub(r'auto_encoder', 'auto_model.encoder', key)
#         # Replace 'auto_embeddings' with 'auto_model.embeddings'
#         new_key = new_key.replace('auto_embeddings', 'auto_model.embeddings')
#         # Replace 'auto_pooler' with 'auto_model.pooler'
#         new_key = new_key.replace('auto_pooler', 'auto_model.pooler')
#         # Add the updated key-value pair to the new state dict
#         new_state_dict[new_key] = value

#     print("no missing key?", set(model.state_dict().keys()) == set(new_state_dict.keys()))
    
#     return new_state_dict

#model.load_state_dict(ensemble_ckp(ckp_path1, ckp_path2, ckp_path3, ckp_path4))
    

#score check


'''

Unexpected key(s) in state_dict: "audio_enc.audio_enc.bn0.weight", "audio_enc.audio_enc.bn0.bias", "audio_enc.audio_enc.bn0.running_mean", "audio_enc.audio_enc.bn0.running_var", "audio_enc.audio_enc.bn0.num_batches_tracked", "audio_enc.audio_enc.spectrogram_extractor.stft.conv_real.weight", "audio_enc.audio_enc.spectrogram_extractor.stft.conv_imag.weight", "audio_enc.audio_enc.logmel_extractor.melW", "audio_enc.audio_enc.conv_block1.conv1.weight", "audio_enc.audio_enc.conv_block1.conv2.weight", "audio_enc.audio_enc.conv_block1.bn1.weight", "audio_enc.audio_enc.conv_block1.bn1.bias", "audio_enc.audio_enc.conv_block1.bn1.running_mean", "audio_enc.audio_enc.conv_block1.bn1.running_var", "audio_enc.audio_enc.conv_block1.bn1.num_batches_tracked", "audio_enc.audio_enc.conv_block1.bn2.weight", "audio_enc.audio_enc.conv_block1.bn2.bias", "audio_enc.audio_enc.conv_block1.bn2.running_mean", "audio_enc.audio_enc.conv_block1.bn2.running_var", "audio_enc.audio_enc.conv_block1.bn2.num_batches_tracked", "audio_enc.audio_enc.conv_block2.conv1.weight", "audio_enc.audio_enc.conv_block2.conv2.weight", "audio_enc.audio_enc.conv_block2.bn1.weight", "audio_enc.audio_enc.conv_block2.bn1.bias", "audio_enc.audio_enc.conv_block2.bn1.running_mean", "audio_enc.audio_enc.conv_block2.bn1.running_var", "audio_enc.audio_enc.conv_block2.bn1.num_batches_tracked", "audio_enc.audio_enc.conv_block2.bn2.weight", "audio_enc.audio_enc.conv_block2.bn2.bias", "audio_enc.audio_enc.conv_block2.bn2.running_mean", "audio_enc.audio_enc.conv_block2.bn2.running_var", "audio_enc.audio_enc.conv_block2.bn2.num_batches_tracked", "audio_enc.audio_enc.conv_block3.conv1.weight", "audio_enc.audio_enc.conv_block3.conv2.weight", "audio_enc.audio_enc.conv_block3.bn1.weight", "audio_enc.audio_enc.conv_block3.bn1.bias", "audio_enc.audio_enc.conv_block3.bn1.running_mean", "audio_enc.audio_enc.conv_block3.bn1.running_var", "audio_enc.audio_enc.conv_block3.bn1.num_batches_tracked", "audio_enc.audio_enc.conv_block3.bn2.weight", "audio_enc.audio_enc.conv_block3.bn2.bias", "audio_enc.audio_enc.conv_block3.bn2.running_mean", "audio_enc.audio_enc.conv_block3.bn2.running_var", "audio_enc.audio_enc.conv_block3.bn2.num_batches_tracked", "audio_enc.audio_enc.conv_block4.conv1.weight", "audio_enc.audio_enc.conv_block4.conv2.weight", "audio_enc.audio_enc.conv_block4.bn1.weight", "audio_enc.audio_enc.conv_block4.bn1.bias", "audio_enc.audio_enc.conv_block4.bn1.running_mean", "audio_enc.audio_enc.conv_block4.bn1.running_var", "audio_enc.audio_enc.conv_block4.bn1.num_batches_tracked", "audio_enc.audio_enc.conv_block4.bn2.weight", "audio_enc.audio_enc.conv_block4.bn2.bias", "audio_enc.audio_enc.conv_block4.bn2.running_mean", "audio_enc.audio_enc.conv_block4.bn2.running_var", "audio_enc.audio_enc.conv_block4.bn2.num_batches_tracked", "audio_enc.audio_enc.conv_block5.conv1.weight", "audio_enc.audio_enc.conv_block5.conv2.weight", "audio_enc.audio_enc.conv_block5.bn1.weight", "audio_enc.audio_enc.conv_block5.bn1.bias", "audio_enc.audio_enc.conv_block5.bn1.running_mean", "audio_enc.audio_enc.conv_block5.bn1.running_var", "audio_enc.audio_enc.conv_block5.bn1.num_batches_tracked", "audio_enc.audio_enc.conv_block5.bn2.weight", "audio_enc.audio_enc.conv_block5.bn2.bias", "audio_enc.audio_enc.conv_block5.bn2.running_mean", "audio_enc.audio_enc.conv_block5.bn2.running_var", "audio_enc.audio_enc.conv_block5.bn2.num_batches_tracked", "audio_enc.audio_enc.conv_block6.conv1.weight", "audio_enc.audio_enc.conv_block6.conv2.weight", "audio_enc.audio_enc.conv_block6.bn1.weight", "audio_enc.audio_enc.conv_block6.bn1.bias", "audio_enc.audio_enc.conv_block6.bn1.running_mean", "audio_enc.audio_enc.conv_block6.bn1.running_var"
"audio_enc.audio_enc.fc1.weight", "audio_enc.audio_enc.fc1.bias", "audio_linear.0.weight", "audio_linear.0.bias", "audio_linear.2.weight", "audio_linear.2.bias", "text_enc.bert_encoder.0.auto_model.embeddings.position_ids", "text_enc.bert_encoder.0.auto_model.embeddings.word_embeddings.weight", "text_enc.bert_encoder.0.auto_model.embeddings.position_embeddings.weight", "text_enc.bert_encoder.0.auto_model.embeddings.LayerNorm.weight", "text_enc.bert_encoder.0.auto_model.embeddings.LayerNorm.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.q.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.q.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.k.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.k.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.v.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.v.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.o.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.o.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.LayerNorm.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.LayerNorm.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.intermediate.dense.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.intermediate.dense.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.dense.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.dense.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.LayerNorm.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.LayerNorm.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.q.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.q.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.k.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.k.bias", "text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.v.weight", "text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.v.bias",
 
 





Missing key(s) in state_dict: "model.audio_enc.audio_enc.bn0.weight", "model.audio_enc.audio_enc.bn0.bias", "model.audio_enc.audio_enc.bn0.running_mean", "model.audio_enc.audio_enc.bn0.running_var", "model.audio_enc.audio_enc.spectrogram_extractor.stft.conv_real.weight", "model.audio_enc.audio_enc.spectrogram_extractor.stft.conv_imag.weight", "model.audio_enc.audio_enc.logmel_extractor.melW", "model.audio_enc.audio_enc.conv_block1.conv1.weight", "model.audio_enc.audio_enc.conv_block1.conv2.weight", "model.audio_enc.audio_enc.conv_block1.bn1.weight", "model.audio_enc.audio_enc.conv_block1.bn1.bias", "model.audio_enc.audio_enc.conv_block1.bn1.running_mean", "model.audio_enc.audio_enc.conv_block1.bn1.running_var", "model.audio_enc.audio_enc.conv_block1.bn2.weight", "model.audio_enc.audio_enc.conv_block1.bn2.bias", "model.audio_enc.audio_enc.conv_block1.bn2.running_mean", "model.audio_enc.audio_enc.conv_block1.bn2.running_var", "model.audio_enc.audio_enc.conv_block2.conv1.weight", "model.audio_enc.audio_enc.conv_block2.conv2.weight", "model.audio_enc.audio_enc.conv_block2.bn1.weight", "model.audio_enc.audio_enc.conv_block2.bn1.bias", "model.audio_enc.audio_enc.conv_block2.bn1.running_mean", "model.audio_enc.audio_enc.conv_block2.bn1.running_var", "model.audio_enc.audio_enc.conv_block2.bn2.weight", "model.audio_enc.audio_enc.conv_block2.bn2.bias", "model.audio_enc.audio_enc.conv_block2.bn2.running_mean", "model.audio_enc.audio_enc.conv_block2.bn2.running_var", "model.audio_enc.audio_enc.conv_block3.conv1.weight", "model.audio_enc.audio_enc.conv_block3.conv2.weight", "model.audio_enc.audio_enc.conv_block3.bn1.weight", "model.audio_enc.audio_enc.conv_block3.bn1.bias", "model.audio_enc.audio_enc.conv_block3.bn1.running_mean", "model.audio_enc.audio_enc.conv_block3.bn1.running_var", "model.audio_enc.audio_enc.conv_block3.bn2.weight", "model.audio_enc.audio_enc.conv_block3.bn2.bias", "model.audio_enc.audio_enc.conv_block3.bn2.running_mean", "model.audio_enc.audio_enc.conv_block3.bn2.running_var", "model.audio_enc.audio_enc.conv_block4.conv1.weight", "model.audio_enc.audio_enc.conv_block4.conv2.weight", "model.audio_enc.audio_enc.conv_block4.bn1.weight", "model.audio_enc.audio_enc.conv_block4.bn1.bias", "model.audio_enc.audio_enc.conv_block4.bn1.running_mean", "model.audio_enc.audio_enc.conv_block4.bn1.running_var", "model.audio_enc.audio_enc.conv_block4.bn2.weight", "model.audio_enc.audio_enc.conv_block4.bn2.bias", "model.audio_enc.audio_enc.conv_block4.bn2.running_mean", "model.audio_enc.audio_enc.conv_block4.bn2.running_var", "model.audio_enc.audio_enc.conv_block5.conv1.weight", "model.audio_enc.audio_enc.conv_block5.conv2.weight", "model.audio_enc.audio_enc.conv_block5.bn1.weight", "model.audio_enc.audio_enc.conv_block5.bn1.bias", "model.audio_enc.audio_enc.conv_block5.bn1.running_mean", "model.audio_enc.audio_enc.conv_block5.bn1.running_var", "model.audio_enc.audio_enc.conv_block5.bn2.weight", "model.audio_enc.audio_enc.conv_block5.bn2.bias", "model.audio_enc.audio_enc.conv_block5.bn2.running_mean", "model.audio_enc.audio_enc.conv_block5.bn2.running_var", "model.audio_enc.audio_enc.conv_block6.conv1.weight", "model.audio_enc.audio_enc.conv_block6.conv2.weight", "model.audio_enc.audio_enc.conv_block6.bn1.weight", "model.audio_enc.audio_enc.conv_block6.bn1.bias", "model.audio_enc.audio_enc.conv_block6.bn1.running_mean", "model.audio_enc.audio_enc.conv_block6.bn1.running_var", "model.audio_enc.audio_enc.conv_block6.bn2.weight", "model.audio_enc.audio_enc.conv_block6.bn2.bias", "model.audio_enc.audio_enc.conv_block6.bn2.running_mean", "model.audio_enc.audio_enc.conv_block6.bn2.running_var", "model.audio_enc.audio_enc.fc1.weight", "model.audio_enc.audio_enc.fc1.bias", "model.audio_linear.0.weight", "model.audio_linear.0.bias", "model.audio_linear.2.weight", "model.audio_linear.2.bias", "model.text_enc.bert_encoder.0.auto_model.embeddings.position_ids", "model.text_enc.bert_encoder.0.auto_model.embeddings.word_embeddings.weight", "model.text_enc.bert_encoder.0.auto_model.embeddings.position_embeddings.weight", "model.text_enc.bert_encoder.0.auto_model.embeddings.LayerNorm.weight", "model.text_enc.bert_encoder.0.auto_model.embeddings.LayerNorm.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.q.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.q.bias", 
"model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.k.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.k.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.v.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.v.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.o.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.attn.o.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.LayerNorm.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.attention.LayerNorm.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.intermediate.dense.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.intermediate.dense.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.dense.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.dense.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.LayerNorm.weight", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.0.output.LayerNorm.bias", "model.text_enc.bert_encoder.0.auto_model.encoder.layer.1.attention.attn.q.weight"
'''