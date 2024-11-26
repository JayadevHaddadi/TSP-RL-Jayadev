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
        self.fc01 = nn.Linear(self.board_x*(self.board_y+1), 2024) # 14
        self.bn01 = nn.BatchNorm1d(2024)

        self.fc02 = nn.Linear(2024, 2024)
        self.bn02 = nn.BatchNorm1d(2024)

        self.fc03 = nn.Linear(2024, 2024)
        self.bn03 = nn.BatchNorm1d(2024)

        self.fc04 = nn.Linear(2024, 1024)
        self.bn04 = nn.BatchNorm1d(1024)

        self.fc_pi = nn.Linear(1024, self.action_size)
        
        self.fc_v = nn.Linear(1024, 1)

    def forward(self, s):
        #                                                           s:  board_x x board_y
        # s = s.view(self.board_x * (self.board_y+1))       
        s = s.view(-1, self.board_x * (self.board_y+1))         
        s = F.relu(self.bn01(self.fc01(s)))                          
        s = F.relu(self.bn02(self.fc02(s)))                    
        s = F.relu(self.bn03(self.fc03(s)))                    
        s = F.relu(self.bn04(self.fc04(s)))                               
        # s = F.dropout(F.relu(self.bn04(self.fc03(s))), p=self.args.dropout, training=self.training)  

        pi = self.fc_pi(s)              # action_size
        v = self.fc_v(s)                # 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
