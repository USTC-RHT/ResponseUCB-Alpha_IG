import torch
import numpy as np
import torch.optim.adam
from alpha_rank_copy import Trans_Matrix,alpha_rank

# N = 200
# num_iterations = 10000
payoffs = np.load('D:\InfoGainalpharank\soccer200.npy')


def Try(payoffs,N):
    num_iterations = 50000
    P = Trans_Matrix(payoffs,alpha=1000)    # alpha充分大时 接近准确求出 α-rank
    # print(P)
    P = torch.tensor(P,dtype=float).to(device='cuda')

    # 初始化参数
    theta = torch.zeros(N, requires_grad=True)

    # 定义优化器
    optimizer = torch.optim.Adam([theta], lr=0.001)

    for _ in range(num_iterations):
        # 计算当前分布 π
        pi = torch.softmax(theta, dim=0).to(device='cuda',dtype=float)
        
        # 随机采样状态 i
        i = torch.randint(0, N, (1,))
        
        # 计算 (πP)_i (假设 P 是已知矩阵)
        pi_P_i = torch.dot(pi, P[:, i].squeeze())
        
        # 计算损失 L_i
        loss = (pi_P_i - pi[i])**2
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()

    return pi

    # true_pi = torch.tensor(alpha_rank([payoffs],alpha=1000)).to(device='cuda')
    # print(max(abs(pi - true_pi)))
    # print(pi @ P - pi)