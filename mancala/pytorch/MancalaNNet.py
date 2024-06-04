import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MancalaNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(MancalaNNet, self).__init__()
        self.fc01 = nn.Linear(self.board_x*(self.board_y+1), 1024) # 14
        self.fc02 = nn.Linear(1024, 1024)
        self.fc03 = nn.Linear(1024, 512)

        self.bn01 = nn.BatchNorm1d(1024)
        self.bn02 = nn.BatchNorm1d(1024)
        self.bn03 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s:  board_x x board_y
        # s = s.view(self.board_x * (self.board_y+1))       
        s = s.view(-1, self.board_x * (self.board_y+1))         
        s = F.relu(self.bn01(self.fc01(s)))                          
        s = F.relu(self.bn02(self.fc02(s)))                          
        s = F.dropout(F.relu(self.bn03(self.fc03(s))), p=self.args.dropout, training=self.training)  

        pi = self.fc3(s)                                                                         # action_size
        v = self.fc4(s)                                                                          # 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
