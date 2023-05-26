import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
from trainer.trainer import Task
from tools.config_loader import get_config
from pathlib import Path
from data_handling.DataLoader import get_dataloader
# from tools.make_csvfile import make_csv
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
import torch
import random
from sentence_transformers import util
import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary

from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent, VICReg
#from tools.make_csvfile import make_csv
import pickle
from models.ASE_model import ASE
import lightning.pytorch as pl

#pickle path
pickle_path1 = '/home/user/audio-text_retrieval/outputs/0511_freeze_True_lr_0.0001_seed_20/pickle/temporal_embeddings.pkl'
pickle_path2 = '/home/user/audio-text_retrieval/outputs/0511_diff2_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
#pickle_path3 = '/home/user/audio-text_retrieval/outputs/0512_tripletweight_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path3 = '/home/user/audio-text_retrieval/outputs/cnn14triplet_weight/pickle/temporal_embeddings.pkl'
pickle_path4 = '/home/user/audio-text_retrieval/outputs/0512_triplet_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path5 = '/home/user/audio-text_retrieval/outputs/0512_resnetweight_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path6 = '/home/user/audio-text_retrieval/outputs/0512_WLCNN_weight_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path7 = '/home/user/audio-text_retrieval/outputs/0511_resnet_freeze_True_lr_0.0001_seed_20/pickle/temporal_embeddings.pkl'

#pickle open
with open(pickle_path1, 'rb') as f:  
    temporal_dict1=pickle.load(f)
    
with open(pickle_path2, 'rb') as f:  
    temporal_dict2=pickle.load(f)
    
with open(pickle_path3, 'rb') as f:  
    temporal_dict3=pickle.load(f)
    
with open(pickle_path4, 'rb') as f:  
    temporal_dict4=pickle.load(f)
    
with open(pickle_path5, 'rb') as f:  
    temporal_dict5=pickle.load(f)
    
with open(pickle_path6, 'rb') as f:  
    temporal_dict6=pickle.load(f)

with open(pickle_path7, 'rb') as f:  
    temporal_dict7=pickle.load(f)


#Ensemble combinations
#------------------------------------------------------------------------------------------------- 
# baseline+ntxent(0.229), cnn+sbert+triplet_weighted(0.235), cnn+sbert+triplet-max (0.23)  
# resnet+weighted , WLCNN+weighted
audio_mean23456 = np.mean([temporal_dict2['audio_embs'], temporal_dict3['audio_embs'],
                      temporal_dict4['audio_embs'], temporal_dict5['audio_embs'],temporal_dict6['audio_embs']], axis=0)

caption_mean23456 = np.mean([temporal_dict2['cap_embs'], temporal_dict3['cap_embs'],
                      temporal_dict4['cap_embs'], temporal_dict5['cap_embs'],temporal_dict6['cap_embs']], axis=0)
    
t2a(audio_mean23456,caption_mean23456) #24.46


#-------------------------------------------------------------------------------------------------
# baseline+ntxent(0.229), cnn+sbert+triplet_weighted(0.235), cnn+sbert+triplet-max (0.23)  

audio_mean234 = np.mean([temporal_dict2['audio_embs'], temporal_dict3['audio_embs'],
                      temporal_dict4['audio_embs'],], axis=0)

caption_mean234 = np.mean([temporal_dict2['cap_embs'], temporal_dict3['cap_embs'],
                      temporal_dict4['cap_embs'],], axis=0)

t2a(audio_mean234,caption_mean234) #24.01


#-------------------------------------------------------------------------------------------------
#cnn+sbert+triplet-max(0.23), resnet+weighted , WLCNN+weighted, resnet+ntxent

audio_mean4567 = np.mean([temporal_dict4['audio_embs'], temporal_dict5['audio_embs'],
                      temporal_dict6['audio_embs'], temporal_dict7['audio_embs']], axis=0)

caption_mean4567 = np.mean([temporal_dict4['cap_embs'], temporal_dict5['cap_embs'],
                      temporal_dict6['cap_embs'], temporal_dict7['cap_embs']], axis=0)

t2a(audio_mean4567, caption_mean4567) #23.59



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#same model, different loss(ntxent, triplet-max, triplet-sum, triplet-weighted), same seed(=1234)
pickle_path_a = '/home/user/audio-text_retrieval/outputs/0511_diff2_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path_b = '/home/user/audio-text_retrieval/outputs/0512_triplet_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path_c= '/home/user/audio-text_retrieval/outputs/0512_tripletsum_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'
pickle_path_d = '/home/user/audio-text_retrieval/outputs/0512_tripletweight_freeze_True_lr_0.0001_seed_1234/pickle/temporal_embeddings.pkl'

with open(pickle_path_a, 'rb') as f:  
    pickle_a=pickle.load(f)
    
with open(pickle_path_b, 'rb') as f:  
    pickle_b=pickle.load(f)
    
with open(pickle_path_c, 'rb') as f:  
    pickle_c=pickle.load(f)
    
with open(pickle_path_d, 'rb') as f:  
    pickle_d=pickle.load(f)

# print(t2a(pickle_a['audio_embs'],pickle_a['cap_embs']))
# print(t2a(pickle_b['audio_embs'],pickle_b['cap_embs']))
# print(t2a(pickle_c['audio_embs'],pickle_c['cap_embs']))
# print(t2a(pickle_d['audio_embs'],pickle_d['cap_embs']))

audio_4loss = np.mean([pickle_a['audio_embs'], pickle_b['audio_embs'],
                      pickle_c['audio_embs'], pickle_d['audio_embs']], axis=0)

caption_4loss = np.mean([pickle_a['cap_embs'], pickle_b['cap_embs'],
                      pickle_c['cap_embs'], pickle_d['cap_embs']], axis=0)

t2a(audio_4loss,caption_4loss) #23.736