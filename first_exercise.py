import numpy as np
from numpy import random
from numpy import linalg
import scipy.constants as sc
import csv



# function that flips the i-th and j-th bits
def gen_X_ij(n,i,j,k):
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

                new_k= gen_X_ij(n,i,j,k)

                # adding the contribution of gate X_iX_j to the Hamiltonian matrix for row k
                H[k,new_k] += 1

    return H # if we were to count all pairs (i,j), then the Hamiltonian would have a factor of 2

# r=0 if we want uniform superposition or r=1 if we want random superposition
def init_state(n,r):
    if r==1:
        # initializing state
        psi0= random.rand(2**n)
        norm_psi0 = np.linalg.norm(psi0)
        psi0= psi0/norm_psi0
        # print(psi0)
    else:
        psi0 = (1 / np.sqrt(2 ** n)) * np.ones((2 ** n,))

    return psi0

def time_ev_state(n,t,psi0):
    # just to remember that we consider hbar =1
    hbar =1

    # e-values and vectors of the Hamiltonian
    (evalues,v)= np.linalg.eig(gen_H(n))
    # print(evalues,v)

    # constructing diagonal matrix with e-values as entries
    exp_evalues = np.zeros((2**n,2**n), dtype=np.complex_)
    for k in range(2**n):
        exp_evalues[k][k] = complex(np.exp(-1j*t*evalues[k]/hbar))
    # print(exp_evalues)

    psi_t= np.matmul(v, np.matmul(exp_evalues, np.matmul(v.conj().T,psi0)))

    return psi_t

def operator_average(n,t,psi0,op):
    psi_t = time_ev_state(n,t,psi0)
    return np.matmul(psi_t.conj().T,np.matmul(op,psi_t))

# operator |psi0\rangle \langle psi0| to measure overlap with the initial state
def gen_overlap_op(n,r):
    psi0= init_state(n,r)
    init_overlap_op = np.kron(psi0, psi0.conj().T)
    return init_overlap_op

def gen_Z_i(n,i):
    if i == 0:
        Z_i = np.array([[1, 0], [0, -1]])
    else:
        Z_i = np.identity(2)
    for k in range(1, n):
        if k == i:
            new_Z_i = np.array([[1, 0], [0, -1]])
        else:
            new_Z_i = np.identity(2)
        Z_i = np.kron(Z_i, new_Z_i)

    return Z_i

def gen_Z_tot(n):
    Z_tot = np.zeros((2**n,2**n))
    for i in range(n):
        Z_tot = np.add(Z_tot,gen_Z_i(n,i))
    return Z_tot

############################


# Opens a CSV file for writing, with the file name
# `filename` and the header `header`.
#
# Returns a tuple in the form (writer object, file object).
#
# Don't forget to close the file object after you're done writing!
def csv_init_write(filename, header):
    csvfile = open(filename, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(header)
    return writer, csvfile


for n in range(2,7):
    # Open the CSV files for writing
    header = ["n", "t", "op_avg"]
    writer_gen_overlap, gen_overlap_file = csv_init_write("gen_overlap.csv", header)
    writer_gen_Z, gen_Z_file = csv_init_write("gen_Z.csv", header)

    t = 0
    while t <= 10:
        # Compute the result and write to the file

        ### NOTE: The `operator_average` function is not working properly
        ###       at the moment, but I tested this with other values.

        res_gen_overlap = operator_average(n,t,init_state(n,0),gen_overlap_op(n,0))
        writer_gen_overlap.writerow([n, t, res_gen_overlap])

        res_gen_Z = operator_average(n, t, init_state(n, 0), gen_Z_tot(n))
        writer_gen_Z.writerow([n, t, res_gen_Z])
        t += 0.5

    gen_overlap_file.close()
    gen_Z_file.close()


########################################################

def alt_gen_X_ij(n,i,j):
    if i==0:
        X_ij= np.array([[0, 1], [1, 0]])
    else:
        X_ij = np.identity(2)
    for k in range(1,n):
        if k== i or k==j:
            new_X_ij = np.array([[0, 1], [1, 0]])
        else:
            new_X_ij = np.identity(2)
        X_ij = np.kron(X_ij, new_X_ij)

    return X_ij

# alternative function that generates the Hamiltonian
def alt_gen_H(n):
    # initializing empty Hamiltonian
    H = np.zeros((2 ** n, 2 ** n), dtype=int)

    for i in range(n):
        for j in range(i+1,n):
            H = np.add(H,alt_gen_X_ij(n,i,j))

    return H

##########################################################






