import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu)**2) / (2 * sigma**2)
                    ) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class SG(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = .15
        temp = gaussian(input, mu=0., sigma=0.5) * (1. + hight) \
            - gaussian(input, mu=0.5, sigma=scale * 0.5) * hight \
            - gaussian(input, mu=-0.5, sigma=scale * 0.5) * hight
        return grad_input * temp.float() * 0.5


def mem_update_pra(inputs, mem, spike, v_th, tau_m, dt=1, device=None):
    """
    neural model with soft reset
    """
    alpha = torch.sigmoid(tau_m)
    mem = mem*alpha + (1-alpha) * inputs - v_th*spike
    inputs_ = mem - v_th

    spike = SG.apply(inputs_)
    return mem, spike


def output_Neuron_pra(inputs, mem, tau_m, dt=1, device=None):
    """
    The read out neuron is leaky integrator without spike
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        tau_m(float): time factors of soma
    """
    alpha = torch.sigmoid(tau_m).to(device)
    mem = mem*alpha + (1-alpha) * inputs
    return mem


class spike_dense_test_origin(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        tau_minitializer='uniform',
        low_m=0,
        high_m=4,
        vth=0.5,
        dt=1,
        device='cuda',
        bias=True
    ):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim, output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

    def set_neuron_state(self, batch_size):

        self.mem = Variable(torch.rand(batch_size,
                                       self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,
                                         self.output_dim)).to(self.device)

        self.v_th = Variable(
            torch.ones(batch_size, self.output_dim) * self.vth
        ).to(self.device)

    def forward(self, input_spike):
        k_input = input_spike.float()

        d_input = self.dense(k_input)
        self.mem, self.spike = mem_update_pra(
            d_input,
            self.mem,
            self.spike,
            self.v_th,
            self.tau_m,
            self.dt,
            device=self.device
        )
        return self.mem, self.spike


class readout_integrator_test(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        tau_minitializer='uniform',
        low_m=0,
        high_m=4,
        device='cuda',
        bias=True,
        dt=1
    ):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
        """
        super(readout_integrator_test, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dt = dt
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

    def set_neuron_state(self, batch_size):
        self.mem = (torch.rand(batch_size, self.output_dim)).to(self.device)

    def forward(self, input_spike):
        #synaptic inputs
        d_input = self.dense(input_spike.float())
        # neuron model without spiking
        self.mem = output_Neuron_pra(
            d_input, self.mem, self.tau_m, self.dt, device=self.device
        )
        return self.mem


class LIFFCNet(nn.Module):

    def __init__(self):
        super().__init__()
        n = 64

        self.dense_1 = spike_dense_test_origin(700, n, vth=1, dt=1)

        #readout layer
        self.dense_2 = readout_integrator_test(n, 20, dt=1)
        nn.init.xavier_normal_(self.dense_2.dense.weight)
        nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, input):
        b, seq_length, input_dim = input.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        output = 0
        for i in range(seq_length):
            input_x = input[:, i, :].reshape(b, input_dim)
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2 = self.dense_2.forward(spike_layer1)
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)
        return output
