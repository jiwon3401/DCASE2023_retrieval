import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tools.utils import l2norm
from models.AudioEncoder import Cnn10, ResNet38, Cnn14, Wavegram_Logmel_Cnn14
from models.TextEncoder import BertEncoder #, W2VEncoder
from models.SBertEncoder import SenBERTEncoder
from models.BERT_Config import MODELS


class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        
        joint_embed = 1024
        self.fc1 = nn.Linear(2048, joint_embed)

        
    def forward(self, x):
        x = self.fc1(x)
        return x
    

class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.fc1 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


    
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

# Create models and load state_dicts    
modelA = MyModelA()
modelB = MyModelB()
# # Load state dicts
# modelA.load_state_dict(torch.load(PATH))
# modelB.load_state_dict(torch.load(PATH))

# model = MyEnsemble(modelA, modelB)
# x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
# output = model(x1, x2)