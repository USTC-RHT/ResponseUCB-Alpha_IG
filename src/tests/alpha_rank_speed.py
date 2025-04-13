import numpy as np
import sys
from Try import Try
from alpha_rank_copy import alpha_rank
from functools import partial
import time
import torch

# Used for profiling the alpha rank implementation when used repeatedly
if __name__ == "__main__":
    N = 30
    P = 2

    alpha = partial(alpha_rank, alpha=1000)

    K = 1
    print("Starting {:,} iterations.".format(K))

    # Crude timing, but its sufficient
    # start_time = time.time()
    for i in range(K):
        payoff = np.random.random(size=(P, N, N))
        start_time = time.time()
        phi = alpha(payoff)
        end_time = time.time()
        print("原 {:,} seconds.".format(end_time - start_time))

        start_time = time.time()
        phi_new = Try(payoff,N ** P)
        end_time = time.time()
        print("新 {:,} seconds.".format(end_time - start_time))

        print(max(abs(torch.tensor(phi).to(device='cuda') - phi_new)))
        print(phi_new)
        # n_phi = phi + 1 # Double checking phi is being used
    # end_time = time.time()
    print('DONE')