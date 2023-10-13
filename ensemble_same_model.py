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
    