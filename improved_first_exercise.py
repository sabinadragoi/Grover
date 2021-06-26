import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
import csv
import time

start = time.time()

def gen_sparse_X_i(n,i): # X_i = I \tensor I ...\tensor X_i \tensor .. I
    if i==0:
        X_i= sparse.csc_matrix(np.array([[0, 1], [1, 0]]))
    else:
        X_i = sparse.identity(2)
    for k in range(1,n):
        if k== i:
            new_X_i = sparse.csc_matrix(np.array([[0, 1], [1, 0]]))
        else:
            new_X_i = sparse.identity(2)
        X_i = scipy.sparse.kron(X_i, new_X_i)

    return X_i


def gen_sparse_X_ij(n,i,j): # X_iX_j
    return np.multiply(gen_sparse_X_i(n,i),gen_sparse_X_i(n,j))


def gen_sparse_H(n): # H = \sum_{i,j} X_iX_j
    # initializing empty Hamiltonian
    H = sparse.csc_matrix((2 ** n, 2 ** n), dtype=int)

    for i in range(n):
        for j in range(i+1,n):
            H = np.add(H,gen_sparse_X_ij(n,i,j))

    return H


def init_sparse_state(n,r):
    if r==1: # |0^n\rangle state
        psi0= sparse.csc_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2**n, 1))
    else: # uniform superposition
        psi0 = (1 / np.sqrt(2 ** n)) * np.ones((2 ** n, 1))

    return psi0

def init_boson_sparse_state(m,k):
    # starts in pure H.O. state k
    return sparse.csc_matrix((np.array([1]), (np.array([k]), np.array([0]))), shape=(m+1, 1))

def coupled_init_state(m,k,n,r):
    return sparse.kron(init_boson_sparse_state(m,k),init_sparse_state(n,r))


def time_ev_sparse_state(t,psi0,H):
    # H = gen_sparse_H(n) or H = gen_sparse_H_coupled(m,n)
    # just to remember that we consider hbar =1
    hbar =1

    psi_t= sparse.linalg.expm_multiply(-1j*t*H/hbar, psi0)

    return psi_t


def sparse_operator_average(t,psi0,op,H):
    psi_t = time_ev_sparse_state(t,psi0,H)
    return ((psi_t.conj().T @ (op @ psi_t)).real.toarray())[0][0]



# operator |psi0\rangle \langle psi0| to measure overlap with the initial state
def gen_sparse_overlap_op(n,r):
    psi0= init_sparse_state(n,r)
    init_overlap_op = sparse.kron(psi0, psi0.conj().T)
    return init_overlap_op


def gen_sparse_Z_i(n,i): # Z_i = I \tensor I ...\tensor Z_i \tensor .. I
    if i == 0:
        Z_i = sparse.csc_matrix(np.array([[1, 0], [0, -1]]))
    else:
        Z_i = sparse.identity(2)
    for k in range(1, n):
        if k == i:
            new_Z_i = sparse.csc_matrix(np.array([[1, 0], [0, -1]]))
        else:
            new_Z_i = sparse.identity(2)
        Z_i = sparse.kron(Z_i, new_Z_i)

    return Z_i


def gen_sparse_Z_tot(n):
    Z_tot = sparse.csc_matrix((2 ** n, 2 ** n), dtype=int)

    for i in range(n):
        Z_tot = np.add(Z_tot,gen_sparse_Z_i(n,i))
    return Z_tot

def gen_sparse_Z_tot_coupled(m,n):
    return sparse.kron(sparse.identity(m+1),gen_sparse_Z_tot(n)/n)

def low_raise_op(m): # m = max number of bosons, a = lowering operator
    a = sparse.csc_matrix((m+1, m+1))
    a_dagger = sparse.csc_matrix((m+1, m+1))
    for i in range(1, m+1):
        # for lowering operator
        data = np.array([np.sqrt(1 / i)])
        row = np.array([i -1])
        col = np.array([i])
        add = sparse.csc_matrix((data, (row, col)), shape=(m+1, m+1))
        a = np.add(a, add)

        # for raising operator
        row_dagger = np.array([i])
        col_dagger = np.array([i-1])
        add_dagger = sparse.csc_matrix((data, (row_dagger, col_dagger)), shape=(m+1, m+1))
        a_dagger = np.add(a_dagger, add_dagger)


    return (a,a_dagger)


def gen_bosonic_mode(m): # m = max number of bosons
    # define a+ a
    return np.add(low_raise_op(m)[0],low_raise_op(m)[1])


def gen_sparse_sigma_i_pm(n,i): # sigma_i_plus/minus = I \tensor I ...\tensor sigma_plus/minus \tensor .. I
    # shape should be (2**n, 2**n)
    sigma_plus = np.array([[0, 1], [0, 0]])
    sigma_minus = np.array([[0, 0], [1, 0]])

    if i == 0:
        sigma_i_plus = sparse.csc_matrix(sigma_plus)
        sigma_i_minus = sparse.csc_matrix(sigma_minus)
    else:
        sigma_i_plus = sparse.identity(2)
        sigma_i_minus = sparse.identity(2)
    for k in range(1, n):
        if k == i:
            new_sigma_i_plus = sparse.csc_matrix(sigma_plus)
            new_sigma_i_minus = sparse.csc_matrix(sigma_minus)
        else:
            new_sigma_i_plus = sparse.identity(2)
            new_sigma_i_minus = sparse.identity(2)
        sigma_i_minus = sparse.kron(sigma_i_minus, new_sigma_i_minus)
        sigma_i_plus = sparse.kron(sigma_i_plus, new_sigma_i_plus)

    return (sigma_i_minus, sigma_i_plus)


def gen_sparse_H_coupled(m,n):
    # H = sum_i (a^{dagger}\sigma_i^- + a\sigma_i^+)
    # a - raising/lowering operators acting on the bosonic mode
    # \sigma - acts on the spins
    (a,a_dagger)= low_raise_op(m)
    H = sparse.csc_matrix((((2 ** n) * (m+1)), ((2 ** n) * (m+1))), dtype=int)

    for i in range(n):
        (sigma_i_minus, sigma_i_plus) = gen_sparse_sigma_i_pm(n, i)
        add = np.add(sparse.kron(a_dagger,sigma_i_minus),sparse.kron(a,sigma_i_plus))
        H = np.add(H, add)

    return H
n=3
t=2.5
H= gen_sparse_H_coupled(1,n)
psi0 = coupled_init_state(1,1,n,1)
psit = sparse.linalg.expm_multiply(-1j*t*H, psi0)
print(psit)

# print(time_ev_sparse_state(t,coupled_init_state(1,1,n,1),gen_sparse_H_coupled(1,n)).real)
# print(sparse_operator_average(t,coupled_init_state(1,1,n,1),gen_sparse_Z_tot_coupled(1,n),gen_sparse_H_coupled(1,n)))


# print(sparse_operator_average(t, init_sparse_state(n, 1), gen_sparse_Z_tot(n),gen_sparse_H(n)))
# print(time_ev_sparse_state(1,init_sparse_state(n, 1),gen_sparse_H(n)))


###################################################################
# Averages of Z_tot and overlap operators for system of spins (without bosons)

# def csv_init_write(filename, header):
#     csvfile = open(filename, 'w')
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     return writer, csvfile
#
# # Open the CSV files for writing
# header = ["n", "t", "op_avg"]
# writer_gen_overlap, gen_overlap_file = csv_init_write("gen_sparse_overlap.csv", header)
# writer_gen_Z, gen_Z_file = csv_init_write("gen_sparse_Z.csv", header)
#
# for n in range(3,11):
#     t = 0
#     while t <= 17:
#         # Compute the result and write to the file
#
#         res_gen_overlap = sparse_operator_average(t,init_sparse_state(n,1),gen_sparse_overlap_op(n,1),gen_sparse_H(n))
#         writer_gen_overlap.writerow([n, t, res_gen_overlap])
#
#         res_gen_Z = sparse_operator_average(t, init_sparse_state(n, 1), gen_sparse_Z_tot(n),gen_sparse_H(n))
#         writer_gen_Z.writerow([n, t, res_gen_Z])
#         t += 0.25
#
# gen_overlap_file.close()
# gen_Z_file.close()

#################################################
# Average of average spin magnetization for coupled cavity

# def csv_init_write(filename, header):
#     csvfile = open(filename, 'w')
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     return writer, csvfile
#
# # Open the CSV files for writing
# header = ["n", "t", "op_avg"]
# writer_gen_Z, gen_Z_file = csv_init_write("gen_sparse_Z_couples.csv", header)
#
# for n in range(3,11):
#     t = 0
#     while t <= 17:
#         # Compute the result and write to the file
#
#         res_gen_Z = sparse_operator_average(t, coupled_init_state(1,1,n,1), gen_sparse_Z_tot_coupled(1,n), gen_sparse_H_coupled(1,n))
#         writer_gen_Z.writerow([n, t, res_gen_Z])
#         t += 0.25
#
# gen_Z_file.close()

#####################################################


end = time.time()
print("time=", end - start)

