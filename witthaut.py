import torch
import numpy as np
from tqdm import tqdm
import ipdb

class Witthaut(object):
    def __init__(self, num_units, sigma=0.1):
        self.num_units = num_units
        coupling_init  = torch.rand((num_units,))
        self.coupling  = torch.einsum('i,j->ij', coupling_init, coupling_init)
        self.omega     = torch.clamp(torch.normal(torch.ones((num_units,)), sigma * torch.ones(num_units,)), 0,2)
        
    def leapfrog_update(self, eps=1e-3):
        angle_diffs  = self.angle.unsqueeze(0) - self.angle.unsqueeze(1) 
        action_diffs = self.action.unsqueeze(0) - self.action.unsqueeze(1)
        action_prods = torch.sqrt(torch.einsum('i,j->ij',self.action, self.action))

        # Action half-step
        self.action -= (eps / 2.0) * (2 / self.num_units) * (action_prods * action_diffs * torch.cos(angle_diffs)).sum(1)
        action_diffs = self.action.unsqueeze(0) - self.action.unsqueeze(1)
        action_prods = torch.sqrt(torch.einsum('i,j->ij',self.action, self.action))

        # Phase full-step
        self.angle  += eps *(self.omega + (1. / self.num_units)*(2*action_prods*torch.sin(self.angle)*(1.0 + action_diffs))).sum(1)
        angle_diffs  = self.angle.unsqueeze(0) - self.angle.unsqueeze(1) 

        # Action half-step
        self.action -= (eps / 2.0)* (2.0 / self.num_units) * (action_prods * action_diffs * torch.cos(angle_diffs)).sum(1)

    def run_hamilton(self, num_steps=100, init_angle=None, init_action=None):
        if init_angle is None:
            self.angle = 2 * np.pi*torch.rand((self.num_units,)) 
        if init_action is None:
            self.action = torch.clamp(torch.normal(torch.ones((self.num_units,)), .1 * torch.ones((self.num_units))),0,1)
        for n in tqdm(range(num_steps)):
            self.leapfrog_update()

if __name__=='__main__':
    ham = Witthaut(5)
    ham.run_hamilton(100)

    
