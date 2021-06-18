import numpy as np
from numpy import random
from numpy import linalg
import scipy.constants as sc


def init_state(n):
    # initializing state
    psi0= random.rand(n)
    norm_psi0 = np.linalg.norm(psi0)
    psi0= psi0/norm_psi0
    # print(psi0)

    return psi0

# function that flips the i-th and j-th bits
def flip_ij(n,i,j,k):
    # initialize column number in binary - same as row number for now
    new_k_binary = np.base_repr(k, base=2)
    new_k_binary = new_k_binary.zfill(n)
    # print("new_k_binary=", new_k_binary)

    # checking whether we will flip to 0 or 1
    add_i = (int(new_k_binary[i]) + 1) % 2
    add_j = (int(new_k_binary[j]) + 1) % 2
    # print("add_i=", add_i)
    # print("add_j=", add_j)

    # column number after the X_iX_j gate in base 2
    new_k_binary = str(int(new_k_binary)  - ((-1)** add_i) * (10 ** (n - i - 1))- ((-1)** add_j)  * (10 ** (n - j - 1)))
    # print("new_k_binary=", new_k_binary)

    # column number after the X_iX_j gate in base 2, filled to n characters
    new_k_binary = new_k_binary.zfill(n)
    # print(new_k_binary)

    # column number after the X_iX_j gate in base 10
    new_k = int(str(new_k_binary), base=2)
    # print(new_k)

    return new_k

# function that generates the Hamiltonian
def gen_H(n):
    # initializing empty Hamiltonian
    H = np.zeros((2 ** n, 2 ** n), dtype=int)

    for i in range(n):
        for j in range(i+1,n):
            for k in range(2**n): # k-th row of X_iX_j
                # print("i,j=",i,j)
                # print("k=",k)

                new_k= flip_ij(n,i,j,k)

                # adding the contribution of gate X_iX_j to the Hamiltonian matrix for row k
                H[k,new_k] += 1

    return H # if we were to count all pairs (i,j), then the Hamiltonian would have a factor of 2


def time_ev_state(n,t):
    # psi0 = init_state(n)
    psi0 = np.zeros((n,))
    print(psi0)
    (evalues,v)= np.linalg.eig(gen_H(n))
    exp_evalues = np.zeros((2**n,2**n))
    for k in range(2**n):
        exp_evalues[k][k] = np.exp(1j*t*evalues[k]/sc.hbar)

    psi_t= np.matmul(v, np.matmul(exp_evalues, np.matmul(v.cong().T,psi0)))

    return psi_t

print(time_ev_state(3,1))
