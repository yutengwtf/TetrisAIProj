import torch
import torch.nn as nn
import copy
import numpy as np

def phi(t):
    t = torch.as_tensor(np.asarray(t)).float()
    t = t.unsqueeze(dim=0) if t.ndim == 3 else t
    return t.cuda()

class ImageAnal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(16, 4), stride=(8, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(6, 5), stride=(3, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, t):
        return self.net(t)

class FeatAnal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=64*5*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4)
        )

    def forward(self, t):
        return self.net(t)

class FeatValAnal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=64*5*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, t):
        return self.net(t)

class Dueling(nn.Module):
    def __init__(self):
        super().__init__()
        self.val_ly = FeatValAnal()
        self.adv_ly = FeatAnal()
    def forward(self, charateristic):
        advantage = self.adv_ly(charateristic)  
        value = self.val_ly(charateristic)
        return value + advantage - torch.mean(advantage, dim=1).view(-1, 1)  

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ImageAnal()
        self.linear = FeatAnal()

    def forward(self, t):
        return self.linear(self.conv(t))

    def save_CNNnet(self, path):
        torch.save(self.conv.state_dict(), path)
    
    def load_CNNnet(self, path):
        self.conv.load_state_dict(torch.load(path))

    def get_linear_net(self):
        return self.linear.parameters()

class DuelingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ImageAnal()
        self.linear = Dueling()

    def forward(self, t):
        return self.linear(self.conv(t))
    
    def save_CNNnet(self, path):
        torch.save(self.conv.state_dict(), path)
    
    def load_CNNnet(self, path):
        self.conv.load_state_dict(torch.load(path))

    def get_linear_net(self):
        return self.linear.parameters()

class Brain(nn.Module):
    def __init__(self, dueling=False):
        super().__init__()
        if dueling:
            self.policy = DuelingNetwork()
            self.target = DuelingNetwork()
        else:
            self.policy = CNNNetwork()
            self.target = CNNNetwork()
        self.target.load_state_dict(self.policy.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, t):
        return self.policy(t)

    def get_optim_action(self, obs):
        ''' No gradient, return action(int) '''
        with torch.no_grad():
            return torch.argmax(self.policy(obs)).item()    

    def get_Q(self, states_t, actions_t):
        ''' With gradient, return Qs(tensor) '''
        Qs_t = torch.gather(self.policy(states_t), 1, actions_t.view(-1, 1)).view(-1)
        return Qs_t

    def get_Vstar(self, obses):
        ''' Without gradient, return Qs_t'''
        with torch.no_grad():
            actions_t = torch.argmax(self.policy(obses), dim=1)
            y_Q = self.target(obses)
            Vs_t = torch.gather(y_Q, 1, actions_t.view(-1, 1)).view(-1)
            return Vs_t

    def learnable(self):
        return self.policy.parameters()

    def update(self):
        self.target.load_state_dict(self.policy.state_dict())

    def save_learned(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_learned(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.update()
    
    def save_cnn(self, path):
        self.policy.save_CNNnet(path)

    def import_cnn(self, path):
        self.policy.load_CNNnet(path)
        self.update()

    def for_transfer(self):
        return self.policy.get_linear_net()