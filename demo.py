import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import ipdb

class net(object):
    def __init__(self, num_units, batch_size, eps=1e-2):
        self.num_units      = num_units
        self.K              = torch.clamp(torch.normal(1.0*torch.ones((num_units, num_units)), 1e-1*torch.ones((num_units, num_units))), 0, 2)
        self.omega          = 2.0*torch.rand(self.num_units) + 9.0
        self.batch_size     = batch_size
        self.current_batch = Variable(2*np.pi*torch.rand(batch_size, num_units), requires_grad=True)
        self.eps            = eps
        self.ar             = 0

    def potential(self,phase):
        phase_diffs = torch.cos(phase.unsqueeze(0) - phase.unsqueeze(1))
        return -1*(phase*self.omega).sum() + (1. / 2*num_units)*(self.K.unsqueeze(0)*phase_diffs).sum()

class kuramoto(object):
    def __init__(self,couplings, i_freqs, batch_size, eps=1e-2):
        self.batch_size     = batch_size
        self.couplings      = couplings
        self.num_units      = couplings.shape[0]
        self.i_freqs        = i_freqs
        self.current_batch = Variable(2*np.pi*torch.rand(batch_size, num_units), requires_grad=True)
        self.batch_size     = batch_size
        self.eps            = eps
        self.ar             = 0

    def potential(self,phase):
      phase_diffs = torch.cos(phase.unsqueeze(0) - phase.unsqueeze(1))
      return -1*(phase*self.i_freqs).sum() + (1. / 2*num_units)*(self.couplings.unsqueeze(0)*phase_diffs).sum()

def langevin(ham_obj): 
        chain_number  = 0
        batch = []
        batch_size     = ham_obj.batch_size
        potential      = ham_obj.potential
        current_batch  = ham_obj.current_batch
        current_batch.requires_grad = True
        eps            = ham_obj.eps

        #pbar = tqdm(total=batch_size)
        for b in range(batch_size):
            current_sample = current_batch[b,:]
            keep_sample = False
            while not keep_sample:
                chain_number+=1
                U1 = potential(current_sample)
                U1.backward()
                with torch.no_grad():
                    momentum = torch.normal(torch.zeros(num_units), torch.ones(num_units))
                    dU1    = current_batch.grad[b,:]
                    candidate_sample = torch.clamp(current_sample - (eps**2 / 2.0)*dU1 + eps*momentum, 0, 2*np.pi)
                candidate_sample.requires_grad=True
                U2 = potential(candidate_sample)
                U2.backward()
                with torch.no_grad():
                    dU2 = candidate_sample.grad
                    candidate_momentum = momentum - (eps / 2.0) * dU1 - (eps / 2.0) * dU2

                    # MH Step
                    BF_exp = 0
                    BF_exp += (potential(candidate_sample) - potential(current_sample)) 
                    BF_exp += .5*((candidate_momentum**2).sum() - (momentum**2).sum())
                    accept_prob = min(1,torch.exp(-1*BF_exp))
                    keep_sample = accept_prob > np.random.rand()

                    if keep_sample:
                        batch.append(candidate_sample.data) 
                    current_batch.grad.zero_()
        batch                  = torch.stack(batch)
        ham_obj.current_batch  = batch
        ham_obj.ar             = batch_size / float(chain_number)
        return batch

def fit(model_obj, data_obj, true_params, lr=1e-3, num_steps=1024, show_every=32):
    true_coupling = true_params[0]
    true_i_freq   = true_params[1]
    num_units     = true_coupling.shape[0]

    coupling_mse = []
    i_freq_mse   = []
    model_ar     = []
    data_ar      = []
    K_grad       = []
    omega_grad   = []

    for n in range(num_steps):
        batch = langevin(data_obj)
        data_ar.append(data_obj.ar)
        
        # Positive Statistics
        pos_phase       = batch.clone()
        pos_phase_diffs = torch.cos(batch.unsqueeze(1) - batch.unsqueeze(2))
        
        # Negative Statistics
        neg_phase       = langevin(model_obj)
        model_ar.append(model_obj.ar)
        neg_phase_diffs = torch.cos(neg_phase.unsqueeze(1) - neg_phase.unsqueeze(2))

        # Update
        delta_omega = -1*(pos_phase - neg_phase).mean(0)
        omega_grad.append(torch.abs(delta_omega).mean().data.numpy())
        delta_K     = -1*(1./ 2*num_units)*(pos_phase_diffs - neg_phase_diffs).mean(0)
        K_grad.append(torch.abs(delta_K).mean().data.numpy())

        my_net.omega   += lr*delta_omega 
        my_net.K       += lr*delta_K

        # Display
        if n % show_every == 0:
            coupling_mse.append(((my_net.K - true_coupling)**2).mean())
            i_freq_mse.append(((my_net.omega - true_i_freq)**2).mean())
            print('Coupling MSE: {}\nInt. Freq. MSE: {}'.format(coupling_mse[-1], i_freq_mse[-1]))

            # Plot
            plt.plot(coupling_mse)
            plt.savefig('/home/matt/figs/spike_learn/coupling_mse.png')
            plt.close()

            plt.plot(i_freq_mse)
            plt.savefig('/home/matt/figs/spike_learn/i_freq_mse.png')
            plt.close()

            plt.plot(model_ar)
            plt.plot(data_ar)
            plt.legend(('Model AR', 'Data AR'))
            plt.savefig('/home/matt/figs/spike_learn/acceptance_rates.png')
            plt.close()
  
            plt.plot(K_grad) 
            plt.title('Coupling abs gradient')
            plt.savefig('/home/matt/figs/spike_learn/coupling_abs_grad.png')
            plt.close()

            plt.plot(omega_grad)
            plt.title('Ifreq abs gradient')
            plt.savefig('/home/matt/figs/spike_learn/ifreq_abs_grad.png')
            plt.close()

            bins = np.linspace(my_net.omega.data.numpy().min(), my_net.omega.data.numpy().max(), 30)
            plt.hist(my_net.omega.data.numpy(), bins, alpha=0.5, label='Estimate')
            #plt.xlim([true_i_freq.min(),true_i_freq.max()])
            plt.xlim([my_net.omega.data.numpy().min(), my_net.omega.data.numpy().max()])
            plt.hist(true_i_freq.data.numpy(), bins, alpha=0.5, label='True')
            plt.xlim([my_net.omega.data.numpy().min(), my_net.omega.data.numpy().max()])
            #plt.xlim([true_i_freq.min(), true_i_freq.max()])
            plt.legend(loc='upper right')
            plt.title('Intrinsic Frequencies')
            plt.savefig('/home/matt/figs/spike_learn/i_freqs.png')
            plt.close()

            bins = np.linspace(0, 2, 30)
            plt.hist(my_net.K.data.view(-1).numpy(), bins=bins, alpha=0.5, label='Estimate')
            plt.xlim([0.0,2.0])
            plt.hist(true_coupling.view(-1).data.numpy(), bins=bins, alpha=0.5, label='True')
            plt.xlim([0.0,2.0])
            plt.legend(loc='upper right')
            plt.title('Couplings')
            plt.savefig('/home/matt/figs/spike_learn/couplings.png')
            plt.close()

if __name__=='__main__':
   
    # Parameters 
    num_units         = 64
    num_burn_in       = 64
    eps               = 1e-3
    batch_size        = 128
    i_freqs           = torch.clamp(torch.normal(10.0*torch.ones(num_units),0.01*torch.ones(num_units)), 0,20)
    coupling_strength = 1.0
    couplings         = coupling_strength*torch.ones((num_units,num_units))

    #Initialize model and data
    km     = kuramoto(couplings, i_freqs, batch_size,eps=eps)
    # Warm up data model
    print('Running burn in for KM.')
    for n in tqdm(range(num_burn_in)):
        burn_in = langevin(km)
    my_net = net(num_units, batch_size,eps=eps)

    # Fit
    fit(my_net, km, [couplings, i_freqs], lr=1e-3)
