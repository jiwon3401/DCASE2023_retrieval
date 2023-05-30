import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
#from trainer.trainer import Task
from tools.config_loader import get_config
from pathlib import Path
from data_handling.DataLoader import get_dataloader
# from tools.make_csvfile import make_csv
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import re
import sys
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, TripletLoss, WeightTriplet, NTXent, VICReg
from tools.InfoNCE import InfoNCE
from tools.make_csvfile import make_csv
import pickle

from models.ASE_model import ASE
import lightning.pytorch as pl


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str, help='Name of the experiment.')
    parser.add_argument('-d', '--dataset', default='Clotho_new', type=str, help='Dataset used')
    parser.add_argument('-l', '--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('-c', '--config', default='settings', type=str, help='Name of the setting file.')
    parser.add_argument('-o', '--loss', default='ntxent',  type=str, help='Name of the loss function.')
    parser.add_argument('-f', '--freeze', default='False', type=str, help='Freeze or not.')
    parser.add_argument('-e', '--batch', default=24, type=int, help='Batch size.')
    parser.add_argument('-m', '--margin', default=0.2, type=float, help='Margin value for loss')
    parser.add_argument('-s', '--seed', default=20, type=int, help='Training seed')
    parser.add_argument('-p', '--epochs',default=0, type=int, help='Epoch')

    args = parser.parse_args()
    config = get_config(args.config)

    config.exp_name = args.exp_name
    config.dataset = args.dataset
    config.training.lr = args.lr
    config.training.loss = args.loss
    config.training.freeze = eval(args.freeze)
    config.data.batch_size = args.batch
    config.training.margin = args.margin
    config.training.seed = args.seed
    config.training.epochs = args.epochs

    # Set Up Seed
    seed_everything(config.training.seed, workers=True)

    # Set up Path Names
    folder_name = '{}_freeze_{}_lr_{}_' \
                    'seed_{}'.format(config.exp_name, str(config.training.freeze),
                                                config.training.lr,
                                                config.training.seed)
    config.model_output_dir = Path('outputs', folder_name, 'models')
    config.log_output_dir = Path('outputs', folder_name, 'logging')
    config.pickle_output_dir = Path('outputs', folder_name, 'pickle')
    config.folder_name = folder_name
    config.log_output_dir.mkdir(parents=True, exist_ok=True)
    config.model_output_dir.mkdir(parents=True, exist_ok=True)
    config.pickle_output_dir.mkdir(parents=True, exist_ok=True)

    # if config.training.csv:
    #     config.csv_output_dir = Path('outputs', config.folder_name, 'csv')
    #     config.csv_output_dir.mkdir(parents=True, exist_ok=True)

    
    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    config.data.val_datasets_size = len(val_loader.dataset)
    print(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    print(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    
    #test_loader = get_dataloader('test', config)
    #config.data.test_datasets_size = len(test_loader.dataset)
    #print(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')
    
    # Model Defined
    train_model=Task(config)

    # Checkpoint and LR Monitoring
    checkpoint_callback = ModelCheckpoint(monitor='validation_epoch_loss',
        filename="best_checkpoint", dirpath=config.model_output_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    ddp_strategy = DDPStrategy(find_unused_parameters=True)

    # trainer = Trainer(
    #     logger=TensorBoardLogger(save_dir=config.log_output_dir),
    #     max_epochs=config.training.epochs,
    #     strategy=ddp_strategy,
    #     num_sanity_val_steps=-1,
    #     sync_batchnorm=True,
    #     callbacks=[checkpoint_callback, lr_monitor],
    #     default_root_dir=config.log_output_dir,
    #     reload_dataloaders_every_n_epochs=1,
    #     accumulate_grad_batches=1,
    #     log_every_n_steps=1,
    #     )
#     trainer.fit(model=train_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    
    ckp_path = '/home/user/audio-text_retrieval/outputs/0511_freeze_True_lr_0.0001_seed_20/models'
     
    config.data.batch_size=32
    test_loader = get_dataloader('val', config) #val_loader
    
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir=config.log_output_dir),
        accelerator="gpu",
        devices=1
    )
    
    checkpoint = torch.load(os.path.join(ckp_path, "best_checkpoint.ckpt"))
    test_model = Task(config)
    print("model_")
    test_model.load_state_dict(checkpoint['state_dict'])
    trainer.test(model=test_model, dataloaders=test_loader)
    print("Eval Done!")
    
    
    '''         
    #baseline model
    ckp_path1 = '/home/user/audio-text_retrieval/outputs/0511_freeze_True_lr_0.0001_seed_20/models/best_checkpoint.ckpt'
    #different seeds
    ckp_path2 = '/home/user/audio-text_retrieval/outputs/0511_diff1_freeze_True_lr_0.0001_seed_42/models/best_checkpoint.ckpt'
    ckp_path3 = '/home/user/audio-text_retrieval/outputs/0511_diff2_freeze_True_lr_0.0001_seed_1234/models/best_checkpoint.ckpt'
    ckp_path4 = '/home/user/audio-text_retrieval/outputs/0511_diff3_freeze_True_lr_0.0001_seed_0/models/best_checkpoint.ckpt'
    #different loss
    ckp_path5 = '/home/user/audio-text_retrieval/outputs/0512_triplet_freeze_True_lr_0.0001_seed_1234/models/best_checkpoint.ckpt'
    ckp_path6 = '/home/user/audio-text_retrieval/outputs/0512_tripletsum_freeze_True_lr_0.0001_seed_1234/models/best_checkpoint.ckpt'
    ckp_path7 = '/home/user/audio-text_retrieval/outputs/0512_tripletweight_freeze_True_lr_0.0001_seed_1234/models/best_checkpoint.ckpt'


    #ckp_path 1~4 (different seeds) -> 멸망
    #ckp_path 1,5~7 -> 성능 유지
    #ckp_path 3,5~7 -> different loss 미세하게 젤 높음.    
    
    
    def ensemble_ckp(ckp_path1, ckp_path2, ckp_path3, ckp_path4):
        model1 = torch.load(ckp_path1)
        model2 = torch.load(ckp_path2)
        model3 = torch.load(ckp_path3)
        model4 = torch.load(ckp_path4)
        #model5 = torch.load(ckp_path5)

        sdA = model1['state_dict']
        sdB = model2['state_dict']
        sdC = model3['state_dict']
        sdD = model4['state_dict']
        #sdE = model5['state_dict']

        for key in sdA:
            sdD[key] = (sdD[key]+sdC[key]+sdB[key]+sdA[key])/4.
            #sdD[key] = (sdD[key]+sdC[key]+sdB[key]+sdA[key]+sdE[key])/5.

        for key in list(sdD.keys()):
            if (key.startswith("audio_enc.audio_enc.") or key.startswith("audio_linear.") 
                or key.startswith("text_enc.") or key.startswith("text_linear.")):
                new_key = "model." + key
                sdD[new_key] = sdD.pop(key)

        return sdD
    
    test_model = Task(config)
    test_model.load_state_dict(ensemble_ckp(ckp_path3, ckp_path5, ckp_path6, ckp_path7))
    trainer.test(model=test_model, dataloaders=test_loader)


    #test_model.load_state_dict(ensemble_ckp(ckp_path1, ckp_path5, ckp_path6, ckp_path7)) #r1:12.40, r5:34.12, r10:46.33, mAP10:21.44
    #test_model.load_state_dict(ensemble_ckp(ckp_path1, ckp_path3, ckp_path5, ckp_path6, ckp_path7)) #r1:12.76, r5:33.68, r10:46.83, mAP10:21.66
      
    '''

    
################################################################################################

class Task(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = ASE(config)
        # self.return_ranks = config.training.csv
        self.pickle_output_path=Path(config.pickle_output_dir,'temporal_embeddings.pkl')
        self.train_step_outputs = []
        self.validate_step_outputs = []

        #Print SubModules of Task
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            # do nothing, only run on main process
            None
        else:
            summary(self.model.audio_enc)
            summary(self.model.audio_linear)
            summary(self.model.text_enc)
            summary(self.model.text_linear)

        # #Set-up for CSV file
        # if config.training.csv:
        #     self.csv_output_dir = Path('outputs', config.folder_name, 'csv')
        #     self.csv_output_dir.mkdir(parents=True, exist_ok=True)

        

        # set up model
        # if torch.cuda.is_available():
        #     device, device_name = ('cuda',torch.cuda.get_device_name(torch.cuda.current_device()))
        # else: 
        #     device, device_name = ('cpu', None)
        # print(f'Process on {device}:{device_name}')

        # Set up Loss function
        if config.training.loss == 'triplet': #triplet-max
            self.criterion = TripletLoss(margin=config.training.margin)
        
        elif config.training.loss == 'ntxent':
            self.criterion = NTXent()
        
        elif config.training.loss == 'weight':
            self.criterion = WeightTriplet(margin=config.training.margin)
            
        elif config.training.loss == 'infonce':
            self.criterion = InfoNCE()

        elif config.training.loss == 'infonce+vicreg':
            self.criterion = VICReg()
            
        else: #config.training.loss == 'bidirect': #triplet-sum
            self.criterion = BiDirectionalRankingLoss(margin=config.training.margin)

        ep = 1

        # resume from a checkpoint
        if config.training.resume:
            checkpoint = torch.load(config.path.resume_model)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            ep = checkpoint['epoch']

    # def on_train_start(self):
    #     self.recall_sum =[]

    # def on_train_epoch_start(self):
    #     self.epoch_loss = AverageMeter()

    def training_step(self, batch, batch_idx):

        audios, captions, audio_ids, _, _ = batch

        audio_embeds, caption_embeds = self.model(audios, captions)

        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('train_step_loss',loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.train_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.log('train_epoch_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_step_outputs.clear()

    def configure_optimizers(self):
        # set up optimizer
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.training.lr)
        # set up scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,threshold=0.005,threshold_mode='abs',min_lr=0.000001,verbose=True)
        return {"optimizer":optimizer, 
                "lr_scheduler":{"scheduler":scheduler,
                                "monitor": 'validation_epoch_loss',
                                "frequency": 1}}

    # def on_validation_start(self):
    #     self.audio_embs, self.cap_embs , self.audio_names_, self.caption_names= None, None, None, None
        
    def validation_step(self, batch, batch_idx):
        # Tensor(N,E), list, Tensor(N), array, list
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.val_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)
            # if self.return_ranks:
            #     Task.audio_names_ = np.array([None for i in range(data_size)], dtype=object)
            #     Task.caption_names = np.array([None for i in range(data_size)], dtype=object)
        
        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('validation_step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.validate_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validate_step_outputs).mean()
        self.log('validation_epoch_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validate_step_outputs.clear()

    # def on_test_start(self):
    #     self.on_validation_start()
    '''
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
    
    def test_step(self, batch, batch_idx):
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.test_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)

        if Task.audio_embs is None:
            Task.audio_embs = np.zeros((data_size, audio_embeds.shape[1]))
            Task.cap_embs = np.zeros((data_size, caption_embeds.shape[1]))
            if self.return_ranks:
                Task.audio_names_ = np.array([None for i in range(data_size)],dtype=object)
                Task.caption_names = np.array([None for i in range(data_size)],dtype=object)
        
        loss = self.criterion(audio_embeds, caption_embeds, audio_ids)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        Task.audio_embs[indexs] = audio_embeds.cpu().numpy()
        Task.cap_embs[indexs] = caption_embeds.cpu().numpy()

        if self.return_ranks:
            Task.audio_names_[indexs] = np.array(audio_names)
            Task.caption_names[indexs] = np.array(captions)
        return loss

    def on_test_end(self):
        if self.return_ranks:
            r1, r5, r10, mAP10, medr, meanr, ranks, Task.top10 = t2a(Task.audio_embs, Task.cap_embs, return_ranks=True)
            print("Top10 Shape:",Task.top10.shape,"Audio Embeddings:",Task.audio_embs.shape)
        else:
            r1, r5, r10, mAP10, medr, meanr = t2a(Task.audio_embs, Task.cap_embs)
        self.logger.experiment.add_scalars('test_metric',{'r1':r1, 'r5':r5, 'r10':r10, 'mAP10':mAP10, 'medr':medr, 'meanr':meanr})
    
    def on_after_backward(self):
        # call on_test_end() only once after accumulating the results of each process
        if self.trainer.local_rank == 0:
            self.on_test_end()
'''
    def on_test_start(self):
        temporal_dict={'audio_embs':None, 'cap_embs':None, 'audio_names_':None, 'caption_names':None}
        with open(self.pickle_output_path, 'wb') as f:  
            pickle.dump(temporal_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def test_step(self, batch, batch_idx):
        with open(self.pickle_output_path, 'rb') as f:  
            temporal_dict=pickle.load(f)
        # Tensor(N,E), list, Tensor(N), array, list
        audios, captions, audio_ids, indexs, audio_names = batch
        data_size = self.config.data.val_datasets_size
        audio_embeds, caption_embeds = self.model(audios, captions)
        if temporal_dict['audio_embs'] is None:
            temporal_dict['audio_embs'] = np.zeros((data_size, audio_embeds.shape[1]))
            temporal_dict['cap_embs'] = np.zeros((data_size, caption_embeds.shape[1]))
            # if self.return_ranks:
            #     Task.audio_names_ = np.array([None for i in range(data_size)], dtype=object)
            #     Task.caption_names = np.array([None for i in range(data_size)], dtype=object)
        temporal_dict['audio_embs'][indexs] = audio_embeds.cpu().numpy()
        temporal_dict['cap_embs'][indexs] = caption_embeds.cpu().numpy()

        with open(self.pickle_output_path, 'wb') as f:  
            pickle.dump(temporal_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def on_test_end(self):
        with open(self.pickle_output_path, 'rb') as f:  
            temporal_dict=pickle.load(f)
        r1, r5, r10, mAP10, medr, meanr = t2a(temporal_dict['audio_embs'], temporal_dict['cap_embs'])
        print(f'from_temporal_dict   r1:{r1}, r5:{r5}, r10:{r10}, mAP10:{mAP10}')
        self.logger.experiment.add_scalars('metric',{'r1':r1, 'r5':r5, 'r10':r10, 'mAP10':mAP10, 'medr':medr, 'meanr':meanr},self.current_epoch)