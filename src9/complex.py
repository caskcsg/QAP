import copy
import json
import logging
import math
import os
import shutil
import torch.nn.functional as F
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
#from utils12 import *

def model_num(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
def ComplexMultiply(amplitude, phase):

    # print(amplitude.shape)
    # print(phase.shape)
    # exit(0)
    if amplitude.dim() == phase.dim()+1: # Assigning each dimension with same phase
        #print(250)
        cos = torch.unsqueeze(torch.cos(phase), dim=-1)
        sin = torch.unsqueeze(torch.sin(phase), dim=-1)
    elif amplitude.dim() == phase.dim(): #Each dimension has different phases
        #print(260)
        cos = torch.cos(phase)
        sin = torch.sin(phase)
    else:
        raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

    real_part = cos*amplitude
    imag_part = sin*amplitude

    return [real_part, imag_part]

def Qouter(x):
    # real = x[0]
    # imag = x[1]
    # real1 = real.unsqueeze(-1)
    # real2 = real.unsqueeze(-2)
    # imag1 = imag.unsqueeze(-1)
    # imag2 = imag.unsqueeze(-2)
    # output_rr = torch.matmul(real1,real2)+torch.matmul(imag1,imag2)
    # output_ii = -torch.matmul(real1,imag2)+torch.matmul(imag1,real2)
    # output=[output_rr,output_ii]

    c=torch.complex(x[0],x[1])
    c1=c.unsqueeze(-1)
    c2=torch.conj(c1).permute(0,1,3,2)
    output=torch.matmul(c1,c2)

    return output
def step_unitary1(G_r, G_i, W_r, W_i, lr, shape0):

    # A = G^H W - W^H G
    # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
    # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)

    if len(G_r.shape)==2:
        A_skew_r = torch.mm(G_r.t(),W_r) + torch.mm(G_i.t(),W_i) - torch.mm(W_r.t(),G_r) -  torch.mm(W_i.t(),G_i)
        A_skew_i = torch.mm(G_r.t(),W_i) - torch.mm(G_i.t(),W_r) - torch.mm(W_r.t(),G_i) +  torch.mm(W_i.t(),G_r)
        
        #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
        idm = torch.eye(G_r.shape[0]).to(G_r.device)

        
        # cayley_numer = I-lr/2 * A
        cayley_numer_r = idm - (lr/2)* A_skew_r
        cayley_numer_i = - (lr/2)* A_skew_i
        
        # cayley_demon = (I + lr/2 * A)^(-1)
        X = idm + (lr/2)* A_skew_r
        Y = + (lr/2)* A_skew_i
        
        #(X + i*Y)^-1 = (X + Y*X^-1*Y)^-1 - i*(Y + X*Y^-1*X)^-1
        
        #cayley_denom_r = (X + torch.mm(Y,torch.mm(X.inverse(),Y))).inverse()
        
        if X.det() == 0:
            X.add_(idm,alpha=1e-5)
        
        if Y.det() == 0:
            Y.add_(idm,alpha=1e-5)
        
        inv_cayley_denom_r = X + torch.mm(Y,torch.mm(X.inverse(),Y))
        if inv_cayley_denom_r.det() == 0:
            inv_cayley_denom_r.add_(idm,alpha=1e-5)
        
        cayley_denom_r = inv_cayley_denom_r.inverse()
        
        #cayley_denom_i = - (Y + torch.mm(X,torch.mm(Y.inverse(),X))).inverse()
        inv_cayley_denom_i = Y + torch.mm(X,torch.mm(Y.inverse(),X))
        if inv_cayley_denom_i.det() == 0:
            inv_cayley_denom_i.add_(idm,alpha=1e-5)
        
        cayley_denom_i = - inv_cayley_denom_i.inverse()
        
        #W_new = cayley_denom*cayley_numer*W
        W_new_r = torch.mm(cayley_denom_r, cayley_numer_r) - torch.mm(cayley_denom_i, cayley_numer_i)
        W_new_i = torch.mm(cayley_denom_r, cayley_numer_i) + torch.mm(cayley_denom_i, cayley_numer_r)            
        
        W_new_r_2 = torch.mm(W_new_r, W_r) - torch.mm(W_new_i, W_i)
        W_new_i_2 = torch.mm(W_new_r, W_i) + torch.mm(W_new_i, W_r)
    elif len(G_r.shape)==4:
        batch_size = G_r.shape[0]
        seq_len = G_r.shape[1]

        cha = batch_size - shape0
        G_r1 = G_r[:shape0,:,:,:]
        G_i1 = G_i[:shape0,:,:,:]
        W_r1 = W_r[:shape0,:,:,:]
        W_i1 = W_i[:shape0,:,:,:]
        A_skew_r = torch.matmul(G_r1.permute(0,1,3,2),W_r1) + torch.matmul(G_i1.permute(0,1,3,2),W_i1) - torch.matmul(W_r1.permute(0,1,3,2),G_r1) -  torch.matmul(W_i1.permute(0,1,3,2),G_i1)
        A_skew_i = torch.matmul(G_r1.permute(0,1,3,2),W_i1) - torch.matmul(G_i1.permute(0,1,3,2),W_r1) - torch.matmul(W_r1.permute(0,1,3,2),G_i1) +  torch.matmul(W_i1.permute(0,1,3,2),G_r1)
        
        #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
        idm1 = torch.eye(G_r.shape[0]).to(G_r.device)
        
        idm = torch.eye(G_r.shape[-1]).unsqueeze(0).repeat(seq_len,1,1).unsqueeze(0).repeat(shape0,1,1,1).to(G_r.device)
        
        # cayley_numer = I-lr/2 * A
        cayley_numer_r = idm - (lr/2)* A_skew_r
        cayley_numer_i = - (lr/2)* A_skew_i
        
        # cayley_demon = (I + lr/2 * A)^(-1)
        X = idm + (lr/2)* A_skew_r
        Y = + (lr/2)* A_skew_i
        
        #(X + i*Y)^-1 = (X + Y*X^-1*Y)^-1 - i*(Y + X*Y^-1*X)^-1
        
        #cayley_denom_r = (X + torch.mm(Y,torch.mm(X.inverse(),Y))).inverse()
        
        # if X.det() == 0:
        #     X.add_(idm1,alpha=1e-5)
        
        # if Y.det() == 0:
        #     Y.add_(idm1,alpha=1e-5)
        
        inv_cayley_denom_r = X + torch.matmul(Y,torch.matmul(X.inverse(),Y))
        # if inv_cayley_denom_r.det() == 0:
        #     inv_cayley_denom_r.add_(idm1,alpha=1e-5)
        
        cayley_denom_r = inv_cayley_denom_r.inverse()
        
        #cayley_denom_i = - (Y + torch.mm(X,torch.mm(Y.inverse(),X))).inverse()
        try:
            inv_cayley_denom_i = Y + torch.matmul(X,torch.matmul(Y.inverse(),X))
        except:
            print(Y)
            # print(G_r)
            # print(G_i)
            # print(W_r)
            # print(W_i)
            exit(0)
        # if inv_cayley_denom_i.det() == 0:
        #     inv_cayley_denom_i.add_(idm1,alpha=1e-5) 
        
        cayley_denom_i = - inv_cayley_denom_i.inverse()
        
        #W_new = cayley_denom*cayley_numer*W
        W_new_r = torch.matmul(cayley_denom_r, cayley_numer_r) - torch.matmul(cayley_denom_i, cayley_numer_i)
        W_new_i = torch.matmul(cayley_denom_r, cayley_numer_i) + torch.matmul(cayley_denom_i, cayley_numer_r)            
        
        W_new_r_2 = torch.matmul(W_new_r, W_r1) - torch.matmul(W_new_i, W_i1)
        W_new_i_2 = torch.matmul(W_new_r, W_i1) + torch.matmul(W_new_i, W_r1)

        if cha>0:
            W_new_r_2 = torch.cat((W_new_r_2,W_r[shape0:,:,:,:]),dim=0)
            W_new_i_2 = torch.cat((W_new_i_2,W_i[shape0:,:,:,:]),dim=0)
            #print(W_new_r_2.shape)
            #print(W_new_i_2.shape)
    else:
        print("the shape of grad is error!")
        exit(0)
    return torch.stack([W_new_r_2, W_new_i_2], dim= -1)

def step_unitary(G_r, G_i, W_r, W_i, lr, shape0):
    #print(lr)
    X=torch.complex(W_r,W_i)
    # A = G^H W - W^H G
    # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
    # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)
    
    if len(G_r.shape)==2:
        A_skew_r = torch.mm(G_r.t(),W_r) + torch.mm(G_i.t(),W_i) - torch.mm(W_r.t(),G_r) -  torch.mm(W_i.t(),G_i)
        A_skew_i = torch.mm(G_r.t(),W_i) - torch.mm(G_i.t(),W_r) - torch.mm(W_r.t(),G_i) +  torch.mm(W_i.t(),G_r)
        
        A = torch.complex(A_skew_r,A_skew_i)
        idm = torch.eye(G_r.shape[0]).to(G_r.device)

        zuo = idm + (lr/2)* A
        you = idm - (lr/2)* A

        X_bar=torch.matmul(torch.matmul(zuo.inverse(),you),X)

        X_bar=clear_dia(X_bar)

        W_new_r_2 = X_bar.real
        W_new_i_2 = X_bar.imag
    elif len(G_r.shape)==4:
        batch_size = G_r.shape[0]
        seq_len = G_r.shape[1]

        cha = batch_size - shape0
        G_r1 = G_r[:shape0,:,:,:]
        G_i1 = G_i[:shape0,:,:,:]
        W_r1 = W_r[:shape0,:,:,:]
        W_i1 = W_i[:shape0,:,:,:]
        X = X[:shape0,:,:,:]

        A_skew_r = torch.matmul(G_r1.permute(0,1,3,2),W_r1) + torch.matmul(G_i1.permute(0,1,3,2),W_i1) - torch.matmul(W_r1.permute(0,1,3,2),G_r1) -  torch.matmul(W_i1.permute(0,1,3,2),G_i1)
        A_skew_i = torch.matmul(G_r1.permute(0,1,3,2),W_i1) - torch.matmul(G_i1.permute(0,1,3,2),W_r1) - torch.matmul(W_r1.permute(0,1,3,2),G_i1) +  torch.matmul(W_i1.permute(0,1,3,2),G_r1)
        A = torch.complex(A_skew_r,A_skew_i)
        
        idm = torch.eye(G_r.shape[-1]).unsqueeze(0).repeat(seq_len,1,1).unsqueeze(0).repeat(shape0,1,1,1).to(G_r.device)
        zuo = idm + (lr/2)* A
        you = idm - (lr/2)* A 

        X_bar=torch.matmul(torch.matmul(zuo.inverse(),you),X)
        X_bar=clear_dia(X_bar)        
        
        W_new_r_2 = X_bar.real
        W_new_i_2 = X_bar.imag

        if cha>0:
            W_new_r_2 = torch.cat((W_new_r_2,W_r[shape0:,:,:,:]),dim=0)
            W_new_i_2 = torch.cat((W_new_i_2,W_i[shape0:,:,:,:]),dim=0)
            #print(W_new_r_2.shape)
            #print(W_new_i_2.shape)
    else:
        print("the shape of grad is error!")
        exit(0)
    return torch.stack([W_new_r_2, W_new_i_2], dim= -1)

def clear_dia(x):
    real=x.real
    imag=x.imag
    eye=torch.eye(real.shape[-1],real.shape[-1]).to(x.device)
    eye=eye*-1+1
    imag=imag*eye
    r=torch.complex(real,imag)
    return r

class ImageEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim = 2001):
        super(ImageEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        #Vaswani et al.
        frequency_inits = 1/torch.pow(10000, torch.arange(embed_dim).to(torch.float32)/torch.Tensor([embed_dim]*embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix, freeze=False)
        
        phase_matrix = torch.rand(self.input_dim, self.embed_dim)       
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix, freeze=False)

        
        #self.frequencies = nn.Parameter(frequency_inits.unsqueeze(dim = 0).to(self.device))
    
        
    def forward(self, x):
        #print(x[0][0])
        # print(x.shape)
        # print(x)
        
        x1=(x*1000+1000).long()
        # print(x1.shape)
        # print(x1)
        # exit(0)
        phases = self.phase_embedding(x1)
        phases = 2*3.14*nn.Sigmoid()(phases)
        time_stamps = x1.shape[1]
        # print(type(time_stamps))
        # print(time_stamps)
        # exit(0)
        positions = torch.arange(time_stamps).unsqueeze(-1).to(x.device)
        #positions = torch.arange(50).unsqueeze(-1).to(device)
        #xit(0)
        pos_embed = positions.repeat(1, self.embed_dim)* self.frequency_embedding(x1) + phases

        return pos_embed

class linear_c1(nn.Module):#对复矩阵进行类线性变换
    def __init__(self, embed_dim,batch_size,seq_len):
        super(linear_c, self).__init__()
        self.unitary = torch.nn.Parameter(torch.stack([torch.eye(embed_dim).unsqueeze(0).repeat(seq_len,1,1).unsqueeze(0).repeat(batch_size,1,1,1),torch.zeros(embed_dim, embed_dim).unsqueeze(0).repeat(seq_len,1,1).unsqueeze(0).repeat(batch_size,1,1,1)],dim = -1))
        # print(self.unitary.shape)
        #self.unitary = torch.nn.Parameter(torch.stack([torch.randn(embed_dim,embed_dim),torch.randn(embed_dim, embed_dim)],dim = -1))
        self.batch_size = batch_size
        self.seq_len = seq_len
    def forward(self, x):#batch_size*seq_len*embedding*embedding
        # x_real = x[0]
        # x_imag = x[1]
        # len1=len(x_real)
        # cha=self.batch_size-len(x_real)
        # if cha>0:
        #     padding = torch.zeros(cha,x_real.shape[1],x_real.shape[2],x_real.shape[3]).to(device)
        #     x_real=torch.cat((x_real,padding),dim=0)
        #     x_imag=torch.cat((x_imag,padding),dim=0)
        # U_real = self.unitary[:,:,:,:,0]
        # U_imag = self.unitary[:,:,:,:,1]

        # r1 = torch.matmul(U_real, x_real) - torch.matmul(U_imag, x_imag)
        # i1 = torch.matmul(U_imag, x_real) + torch.matmul(U_real, x_imag)
        # output_real1 = torch.matmul(r1, U_real.permute(0,1,3,2)) + torch.matmul(i1, U_imag.permute(0,1,3,2))
        # output_imag1 = torch.matmul(i1, U_real.permute(0,1,3,2)) - torch.matmul(r1, U_imag.permute(0,1,3,2))

        len1=len(x)
        cha=self.batch_size-len1
        if cha>0:
            padding = torch.zeros(cha,x.shape[1],x.shape[2],x.shape[3]).to(x.device)
            x_bu=torch.cat((x,padding),dim=0)
        else:
            x_bu=x
        U_real = self.unitary[:,:,:,:,0]
        U_imag = self.unitary[:,:,:,:,1]
        U = torch.complex(U_real,U_imag)
        U_H = torch.conj(U).permute(0,1,3,2)
        output=torch.matmul(torch.matmul(U,x_bu),U_H)
        output=clear_dia(output)
        return output[:len1,:,:,:]

class linear_c(nn.Module):#对复矩阵进行类线性变换
    def __init__(self, embed_dim,batch_size,seq_len):
        super(linear_c, self).__init__()
        self.unitary = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1))
        # print(self.unitary.shape)
        #self.unitary = torch.nn.Parameter(torch.stack([torch.randn(embed_dim,embed_dim),torch.randn(embed_dim, embed_dim)],dim = -1))
        self.batch_size = batch_size
        self.seq_len = seq_len
    def forward(self, x):#batch_size*seq_len*embedding*embedding
        # x_real = x[0]
        # x_imag = x[1]
        # len1=len(x_real)
        # cha=self.batch_size-len(x_real)
        # if cha>0:
        #     padding = torch.zeros(cha,x_real.shape[1],x_real.shape[2],x_real.shape[3]).to(device)
        #     x_real=torch.cat((x_real,padding),dim=0)
        #     x_imag=torch.cat((x_imag,padding),dim=0)
        # U_real = self.unitary[:,:,:,:,0]
        # U_imag = self.unitary[:,:,:,:,1]

        # r1 = torch.matmul(U_real, x_real) - torch.matmul(U_imag, x_imag)
        # i1 = torch.matmul(U_imag, x_real) + torch.matmul(U_real, x_imag)
        # output_real1 = torch.matmul(r1, U_real.permute(0,1,3,2)) + torch.matmul(i1, U_imag.permute(0,1,3,2))
        # output_imag1 = torch.matmul(i1, U_real.permute(0,1,3,2)) - torch.matmul(r1, U_imag.permute(0,1,3,2))

        
        U_real = self.unitary[:,:,0]
        U_imag = self.unitary[:,:,1]
        U = torch.complex(U_real,U_imag)
        U_H = torch.conj(U).permute(1,0)
        output = torch.matmul(torch.matmul(U,x),U_H)
        # list1 = []
        # for xx in x:
        #     list2=[]
        #     for xxx in xx:
        #         output=torch.mm(torch.mm(U,xxx),U_H)
        #         list2.append(output)
        #     list2=torch.stack(list2, dim = 0)
        #     list1.append(list2)
        # list1=torch.stack(list1, dim = 0)
        # list1=clear_dia(list1)
        # print(list1.shape)
        # exit(0)
        return output

class QDropout1(torch.nn.Module):
    def __init__(self, p=0.2):
        super(QDropout1, self).__init__()
        self.p = p
    def forward(self, x):
        x_real = x[0]
        x_imag = x[1]
        #print(x_real.shape)
        # x_imag = torch.tensor([[[0,3],[-3,0]]],dtype=torch.float32)
        batch_size = len(x_real)
        seq_len = x_real.shape[1]
        dimension = x_real.shape[-1]
        # print(batch_size)
        # print(dimension)
        binary_ids = torch.bernoulli(torch.ones(batch_size,seq_len,dimension)*(1-self.p)).to(x.device)
        #print(binary_ids)
        mask_tensor = torch.ones_like(x_real)
        #print(mask_tensor)
        mask_tensor[binary_ids == 0,:] = 0
        #print(mask_tensor)
        temp = mask_tensor.permute(0,1,3,2) 
        temp[binary_ids == 0,:] = 0
        #print(temp)
        mask_tensor = temp
        output_real = torch.diag_embed(torch.diagonal(x_real,0,2,3),0,-2,-1)
        #print(output_real)
        output_imag = torch.diag_embed(torch.diagonal(x_imag,0,2,3),0,-2,-1)
        output_real[mask_tensor ==1] = x_real[mask_tensor ==1]
        #print(output_real)
        output_imag[mask_tensor ==1] = x_imag[mask_tensor ==1]
        return [output_real,output_imag]
class QDropout(torch.nn.Module):
    def __init__(self, p=0.2):
        super(QDropout, self).__init__()
        self.p = p
    def forward(self, x):
        batch_size = len(x)
        seq_len = x.shape[1]
        dimension = x.shape[-1]
        eye=torch.eye(dimension).to(x.device)

        x_eye=x*eye
        # print(y_eye)
        b=torch.triu(torch.bernoulli(torch.ones(batch_size,seq_len,dimension,dimension)*(1-self.p))).to(x.device)
        b=b+b.permute(0,1,3,2)
        eye_fan=(eye==0)
        b=b*eye_fan

        y=x*b
        y=y+x_eye        
        return y
class QMeasurement1(torch.nn.Module):
    def __init__(self, embed_dim):
        super(QMeasurement1, self).__init__()
        self.embed_dim = embed_dim
        self.kernel_unitary = torch.nn.Parameter(torch.stack([torch.eye(embed_dim).to(device),torch.zeros(embed_dim, embed_dim).to(device)],dim = -1))
        #print(self.kernel_unitary.shape)

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
    
    
        input_real = inputs[0]
        input_imag = inputs[1]
        # print(input_real.shape)
        # print(input_imag.shape)
        real_kernel = self.kernel_unitary[:,:,0]
        imag_kernel = self.kernel_unitary[:,:,1]
        # print(real_kernel.shape)
        # print(imag_kernel.shape)
        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)
        # print(real_kernel.shape)
        # print(imag_kernel.shape)
        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
            + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))  
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
            - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # print(projector_real.shape)
        # print(projector_imag.shape)
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [num_measurements, embed_dim, embed_dim]
        # print(input_real.shape)
        # print(torch.flatten(input_real, start_dim = -2, end_dim = -1).shape)
        # print(torch.flatten(projector_real, start_dim = -2, end_dim = -1).t().shape)
        # print(torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1), torch.flatten(projector_real, start_dim = -2, end_dim = -1).t()).shape)
        # exit(0)
        output_real = torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1), torch.flatten(projector_real, start_dim = -2, end_dim = -1).t())\
            - torch.matmul(torch.flatten(input_imag, start_dim = -2, end_dim = -1), torch.flatten(projector_imag, start_dim = -2, end_dim = -1).t())
        # print(output_real.shape)
        # exit(0)
        return output_real

class QMeasurement(torch.nn.Module):
    def __init__(self, embed_dim):
        super(QMeasurement, self).__init__()
        self.embed_dim = embed_dim
        self.kernel_unitary = torch.nn.Parameter(torch.stack([torch.eye(embed_dim),torch.zeros(embed_dim, embed_dim)],dim = -1))
        #print(self.kernel_unitary.shape)

    def forward(self, inputs):
        # input_real = inputs[0]
        # input_imag = inputs[1]
        # print(input_real.shape)
        # print(input_imag.shape)
        real_kernel = self.kernel_unitary[:,:,0]
        imag_kernel = self.kernel_unitary[:,:,1]

        kernel = torch.complex(real_kernel,imag_kernel)

        #print(kernel)
        # print(kernel.shape)
        # print(real_kernel.shape)
        # print(imag_kernel.shape)
        # real_kernel = real_kernel.unsqueeze(-1)
        # imag_kernel = imag_kernel.unsqueeze(-1)
        kernel = kernel.unsqueeze(-1)
        kernel_H = torch.conj(kernel).permute(0,2,1)
        # print(real_kernel.shape)
        # print(imag_kernel.shape)
        projector = torch.matmul(kernel,kernel_H)
        # print(projector.shape)
        # exit(0)
        # projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
        #     + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))  
        # projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
        #     - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # print(projector_real.shape)
        # print(projector_imag.shape)
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [num_measurements, embed_dim, embed_dim]
        # print(input_real.shape)
        # print(torch.flatten(input_real, start_dim = -2, end_dim = -1).shape)
        # print(torch.flatten(projector_real, start_dim = -2, end_dim = -1).t().shape)
        # print(torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1), torch.flatten(projector_real, start_dim = -2, end_dim = -1).t()).shape)
        # exit(0)

        result = torch.matmul(torch.flatten(inputs, start_dim = -2, end_dim = -1),torch.flatten(projector, start_dim = -2, end_dim = -1).t())
        result=result.real
        input_imag=inputs.imag
        projector_imag=projector.imag
        result_bu=torch.matmul(torch.flatten(input_imag, start_dim = -2, end_dim = -1),torch.flatten(projector_imag, start_dim = -2, end_dim = -1).t())
        # print(result.shape)
        # print(result_bu.shape)
        # exit(0)
        output_real=result+2*result_bu
        # output_real = torch.matmul(torch.flatten(input_real, start_dim = -2, end_dim = -1), torch.flatten(projector_real, start_dim = -2, end_dim = -1).t())\
        #     - torch.matmul(torch.flatten(input_imag, start_dim = -2, end_dim = -1), torch.flatten(projector_imag, start_dim = -2, end_dim = -1).t())
        # print(output_real.shape)
        # exit(0)
        #print(kernel)
        return output_real

class QActivation(torch.nn.Module):
    def __init__(self):
        super(QActivation, self).__init__()
    def forward(self,x):
        x_real = x[0]
        x_imag = x[1]
        
        diagonal_values = torch.diagonal(x_real,0,2,3)

        new_diagonal_values = F.softmax(diagonal_values, dim = -1)

        max_ratio = torch.max(diagonal_values/new_diagonal_values, dim = -1).values

        x_real = x_real/max_ratio.unsqueeze(-1).unsqueeze(-1)

        x_imag = x_imag/max_ratio.unsqueeze(-1).unsqueeze(-1)

        eyes=torch.eye(x_real.shape[3],x_real.shape[3]).unsqueeze(0).repeat(x_real.shape[1],1,1).unsqueeze(0).repeat(x_real.shape[0],1,1,1).to(x.device)
        eyes=eyes*-1+1
        x_real=x_real*eyes
        x_imag = x_imag*eyes

        x_real = x_real + torch.diag_embed(new_diagonal_values)
        
        
        return [x_real, x_imag]    

class QNorm(torch.nn.Module):
    def __init__(self, embed_dim):
        super(QNorm, self).__init__()
        # self.norm1=nn.LayerNorm(embed_dim)
        # self.norm2=nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
    def forward(self,x):
                               
        x_complex = x

        v1,v2=torch.linalg.eigh(x_complex)


        real=v2.real

        imag=v2.imag

        real1=real.permute(0,1,3,2)
        imag1=-1*imag.permute(0,1,3,2)
        v2_H = torch.complex(real1,imag1)
        dia = torch.diag_embed(v1,0,-2,-1).type(torch.complex64)

        result = torch.matmul(torch.matmul(v2,dia),v2_H)
        result = clear_dia(result)
        # output_real=result.real
        # output_imag=result.imag

        # dia_imag=torch.diag_embed(torch.diagonal(output_imag,0,2,3),0,-2,-1)
        # output_imag=output_imag-dia_imag
        return result




class RMSprop_Unitary(Optimizer):
    """Implements RMSprop gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        
    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrent neural networks. In NIPS 2016. 

        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None, shape0=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # print(type(self.param_groups))
        # exit(0)
        # pp=self.param_groups[0]
        # print(type(pp))
        # print(pp.keys())
        # params=pp['params'][0]
        # print(type(params))
        # print(len(params))
        # for p in pp['params']:
        #     print(p.grad.data.shape)
        #     print(p.data.shape)
        # exit(0)
        #print(250)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data 
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                        
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    grad = buf
                else:
                    grad = torch.zeros_like(p.data).addcdiv(grad, avg)
                
                lr = group['lr']
                

                if len(grad.shape)==3:
                    G_r = grad[:,:,0]
                    G_i = grad[:,:,1]
                    W_r = p.data[:,:,0]
                    W_i = p.data[:,:,1]
                elif len(grad.shape)==5:
                    G_r = grad[:,:,:,:,0]
                    G_i = grad[:,:,:,:,1]
                    W_r = p.data[:,:,:,:,0]
                    W_i = p.data[:,:,:,:,1]
                else:
                    print("the shape of grad is error!")
                    exit(0)
                #cc1=torch.complex(W_r,W_i)
                #print(cc1.shape)
                # if len(grad.shape)==3:
                #     cc1=torch.complex(W_r,W_i)
                #     print(cc1.shape)
                #     print(cc1)
                p.data = step_unitary(G_r, G_i, W_r, W_i, lr, shape0)
                # if len(grad.shape)==3:
                #     cc2=torch.complex(p.data[:,:,0],p.data[:,:,1])
                #     print(cc2.shape)
                #     print(cc2)
                #     exit(0)
        return loss