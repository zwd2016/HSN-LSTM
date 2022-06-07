# -*- coding: utf-8 -*-
"""

@author: Wendong
"""

import torch
from torch import nn
import math

b_j0 = 0.01  # neural threshold baseline
R_m = 1  # membrane resistance
dt = 1  #

gamma = .5  # gradient scale
lens = 0.5 # hyper-parameters of approximate function

alpha = 0.4
beta = 0.5

class ActFun_our(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        temp = torch.tanh(alpha * torch.exp(-1 * input**2) ** beta)
        return grad_input * temp.float() * gamma

act_fun_our = ActFun_our.apply

def mem_update_ahs(inputs, mem, spike, tau_adp=32, b=0.1, tau_m=4, dt=0.01, isAdapt=1):
    alpha = torch.exp(-1. * (dt / tau_m) * 0.5 * mem).cuda()
    ro = torch.exp(-1. * (dt / tau_adp) * 0.5 * spike).cuda()
    b_j0 = 0.04
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_our(inputs_)    
    return mem, spike, B, b

def output_SpikeNeuron(inputs, mem, tau_m, dt=0.01):
    """
    The read out neuron is leaky integrator without spike
    """
    b_decayed = 0.08
    alpha = torch.exp(-1. * dt / tau_m)
    mem = mem * alpha + (1. - alpha) * R_m * inputs + b_decayed
    return mem


class HSN_LSTM(nn.Module):  #2021-12-22
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
        
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        
        outputs = [] 
        hidden_spike_ = []
        hidden_mem_ = []
        for t in range(x.shape[1]):
            
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            
            
            
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            
            
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            
            
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            
            mem_info_j_tilda_t, j_tilda_t_1, B, b = mem_update_ahs(h_tilda_t,j_tilda_t,i_tilda_t)
            tau_m_o = torch.ones_like(j_tilda_t_1)
            mem_output = output_SpikeNeuron(j_tilda_t_1, mem_info_j_tilda_t, tau_m_o)
            
            hidden_spike_.append(j_tilda_t_1.data.cpu().numpy())
            hidden_mem_.append(mem_info_j_tilda_t.data.cpu().numpy())
            
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t_1 * mem_output
            
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        betas = 1e-2 * -torch.log2(betas)
        mean = torch.sum(betas*mu, dim=1)
        
        return mean