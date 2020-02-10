import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
import numpy as np
from tqdm import tqdm
import os
import ipdb

class Witthaut(object):
    def __init__(self, num_units, sigma=0.1, K = 1.0, omega_std=.1):
        self.num_units = num_units
        self.coupling  = K * (torch.ones(num_units, num_units) - torch.eye(num_units))
        #self.omega     = torch.normal(torch.zeros((num_units,)), omega_std*torch.ones((num_units,)))
        self.omega = torch.tensor([-2.,-1.,3.])

    def hamiltonian(self):
        angle_diffs  = self.angle.unsqueeze(0) - self.angle.unsqueeze(1) 
        action_diffs = self.action.unsqueeze(0) - self.action.unsqueeze(1)
        action_prods = torch.sqrt(torch.einsum('i,j->ij',self.action, self.action))

        return (self.omega * self.action).sum() - (1. / self.num_units) * (self.coupling *action_prods * action_diffs * torch.sin(angle_diffs)).sum()
        
    def update(self, eps=1e-3, leapfrog=True):
        angle_diffs  = self.angle.unsqueeze(1) - self.angle.unsqueeze(0) 
        action_diffs = self.action.unsqueeze(1) - self.action.unsqueeze(0)
        action_prods = torch.sqrt(torch.einsum('i,j->ij',self.action, self.action))

        # Action step
        current_eps = eps / 2.0 if leapfrog else eps
        self.action = self.action -  (current_eps) * (2. / self.num_units) * (self.coupling * action_prods * action_diffs * torch.cos(angle_diffs)).sum(0)
        action_diffs = self.action.unsqueeze(1) - self.action.unsqueeze(0)
        action_prods = torch.sqrt(torch.einsum('i,j->ij',self.action, self.action))
        action_quots = torch.sqrt(torch.einsum('i,j->ij', self.action, 1. / self.action))

        # Phase full-step
        self.angle  = self.angle + eps *(self.omega + (1. / self.num_units)*(self.coupling * torch.sin(angle_diffs) * (2*action_prods - action_quots*action_diffs)).sum(0))
        angle_diffs  = self.angle.unsqueeze(1) - self.angle.unsqueeze(0) 

        if leapfrog:
            # Action half-step
            self.action = self.action - (eps / 2.0)* (2. / self.num_units) * (action_prods * self.coupling*action_diffs * torch.cos(angle_diffs)).sum(0)
            action_diffs = self.action.unsqueeze(1) - self.action.unsqueeze(0)
        return angle_diffs.data.numpy(), action_diffs.data.numpy()

    def run_hamilton(self, num_steps=100, init_angle=None, init_action=None, eps=1e-3, show_every=25):
        if init_angle is None:
            self.angle = 2 * np.pi*torch.rand((self.num_units,)) 
        else:
            self.angle = init_angle
        if init_action is None:
            self.action = torch.clamp(torch.normal(.5*torch.ones((self.num_units,)), 1e-4 * torch.ones((self.num_units))),0,1)
        else:
            self.action = init_action
  
        act_d  = []
        ang_d  = []
        act    = []
        ang    = []
        energy = []    
        ut_inds = np.triu_indices(self.num_units,k=1)
        for n in tqdm(range(num_steps)):
            angle_diffs, action_diffs = self.update(eps, leapfrog=True)
            act_d.append(action_diffs[ut_inds].reshape(-1)) 
            ang_d.append(angle_diffs[ut_inds].reshape(-1)) 
            act.append(self.action.data.numpy())
            ang.append(self.angle.data.numpy())
            energy.append(self.hamiltonian())
           
            if n % show_every == 0:
                for (name, array) in zip(['energy', 'angle_diffs', 'action', 'angle'], [energy, ang_d, act, ang]):
                    plt.plot(array) 
                    plt.savefig(os.path.join('/home/matt/witthaut/', name + '.png'))
                    plt.close()

if __name__=='__main__':
    num_units = 3
    ham = Witthaut(num_units, sigma = 1, K=2.25)
    #init_action = torch.ones((5,))
    ham.run_hamilton(10000,eps=1e-2,init_action=None)
