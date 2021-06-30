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
import random

start = time.time()

pauli_X = sparse.csc_matrix((np.array([1,1]), (np.array([0,1]), np.array([1,0]))), shape=(2, 2))
pauli_Z = sparse.csc_matrix((np.array([1,-1]), (np.array([0,1]), np.array([0,1]))), shape=(2, 2))

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
    if r==1: # |1^n\rangle state
        psi0= sparse.csc_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2**n, 1))
    elif r==0: # |0^n\rangle state
        psi0 = sparse.csc_matrix((np.array([1]), (np.array([2**n-1]), np.array([0]))), shape=(2 ** n, 1))
    elif r==2: # uniform superposition
        psi0 = (1 / np.sqrt(2 ** n)) * np.ones((2 ** n, 1))
    else: # random basis state
        state = [random.randrange(0, 2, 1) for i in range(2 ** n)]
        psi0 = np.reshape(np.array(state),(2**n,1))

    return psi0


def init_boson_sparse_state(m,k,n):
    # starts in pure H.O. state k
    return sparse.csc_matrix((np.array([1]), (np.array([k]), np.array([0]))), shape=(m+n+1, 1))

def coupled_init_state(m,k,n,r):
    return sparse.kron(init_boson_sparse_state(m,k,n),init_sparse_state(n,r))


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
    return sparse.kron(sparse.identity(m+n+1),gen_sparse_Z_tot(n)/n)

def low_raise_op(m,n): # m = max number of bosons, a = lowering operator
    a = sparse.csc_matrix((m+n+1, m+n+1))
    a_dagger = sparse.csc_matrix((m+n+1, m+n+1))
    for i in range(1, m+n+1):
        # for lowering operator
        data = np.array([np.sqrt(i)])
        row = np.array([i -1])
        col = np.array([i])
        add = sparse.csc_matrix((data, (row, col)), shape=(m+n+1, m+n+1))
        a = np.add(a, add)

        # for raising operator
        row_dagger = np.array([i])
        col_dagger = np.array([i-1])
        add_dagger = sparse.csc_matrix((data, (row_dagger, col_dagger)), shape=(m+n+1, m+n+1))
        a_dagger = np.add(a_dagger, add_dagger)


    return (a,a_dagger)


def gen_bosonic_mode(m,n): # m = max number of bosons
    # define a+ a
    return np.add(low_raise_op(m,n)[0],low_raise_op(m,n)[1])


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
    (a,a_dagger)= low_raise_op(m,n)
    H = sparse.csc_matrix((((2 ** n) * (m+1+n)), ((2 ** n) * (m+1+n))), dtype=int)

    sigma_minus = sparse.csc_matrix(((2 ** n, 2 ** n)), dtype=int)
    sigma_plus = sparse.csc_matrix(((2 ** n, 2 ** n)), dtype=int)
    for i in range(n):
        (sigma_i_minus, sigma_i_plus) = gen_sparse_sigma_i_pm(n, i)
        sigma_minus = np.add(sigma_minus, sigma_i_minus)
        sigma_plus = np.add(sigma_plus, sigma_i_plus)

    add = np.add(sparse.kron(a_dagger,sigma_minus),sparse.kron(a,sigma_plus))
    H = np.add(H, add)

    return H

#########################################################
# MONIKA'S PAPER HAMILTONIAN EQ 4

def gen_sparse_sigma_i_z(n,i): # sigma_i_plus/minus = I \tensor I ...\tensor sigma_plus/minus \tensor .. I
    # shape should be (2**n, 2**n)
    sigma_z = np.array([[0, 1], [0, 0]])

    if i == 0:
        sigma_i_z = sparse.csc_matrix(sigma_z)
    else:
        sigma_i_z = sparse.identity(2)
    for k in range(1, n):
        if k == i:
            new_sigma_i_z = sparse.csc_matrix(sigma_z)
        else:
            new_sigma_i_z = sparse.identity(2)
        sigma_i_z = sparse.kron(sigma_i_z, new_sigma_i_z)


    return sigma_i_z

def gen_weights(n,random):
    # random = 1 if numbers are randomly generated
    # random = 0 if they are all 1
    if random ==1:
        return random.rand(n)
    else:
        return np.ones((n, ))

def S_z(n,random):
    S_z = sparse.csc_matrix((2 ** n, 2 ** n ), dtype=int)
    for i in range(n):
        S_z = np.add(S_z,gen_weights(n,random)[i]*gen_sparse_sigma_i_z(n,i))
    return S_z

def gen_H_central_boson_model(m,n,random,w_rabi):
    (a, a_dagger) = low_raise_op(m, n)
    driving_H = sparse.kron( [[w_rabi*x for x in y] for y in np.add(a,a_dagger)] ,sparse.identity(2**n))
    pure_H = sparse.kron(np.multiply(a,a_dagger),S_z(n,random))
    return np.add(pure_H,driving_H)

def gen_H_central_spin_model(n,random,w_rabi,delta):
    driving_H = sparse.kron( [[w_rabi * x for x in y] for y in np.array([[0,1],[1,0]])] ,sparse.identity(2**n))
    detuning_H= sparse.kron( [[delta * x for x in y] for y in np.array([[1,0],[0,-1]])] ,sparse.identity(2**n))
    pure_H = sparse.kron(np.array([[1,0],[0,-1]]), S_z(n, random))
    return np.add(detuning_H,np.add(pure_H,driving_H))


def init_state_central_spin(ancilla_state,n,r):
    if ancilla_state == 0:
        ancilla = np.array([[1],[0]])
        return sparse.kron(ancilla,init_sparse_state(n,r))
    elif ancilla_state ==1:
        ancilla = np.array([[0], [1]])
        return sparse.kron(ancilla,init_sparse_state(n,r))
    else:
        return 'wrong initial state'

def I_z(n):
    return sparse.kron(pauli_Z,sparse.identity(2**n))

# random_number =0
# ancilla_state =0
# t= 1
# n=5
# r=0
# w_rabi = 3
# delta = 2
#
#
# init_state = init_state_central_spin(ancilla_state,n,r)
# av =sparse_operator_average(t, init_state, I_z(n),gen_H_central_spin_model(n,random_number,w_rabi,delta))
# print(av)


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
#         # r=1
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
# writer_gen_Z, gen_Z_file = csv_init_write("gen_sparse_Z_coupled_1.csv", header)
#
# for n in range(3,11):
#     t = 0
#     while t <= 17:
#         # Compute the result and write to the file
#         # r=0, m=1,k=1
#         res_gen_Z = sparse_operator_average(t, coupled_init_state(1,1,n,0), gen_sparse_Z_tot_coupled(1,n), gen_sparse_H_coupled(1,n))
#         writer_gen_Z.writerow([n, t, res_gen_Z])
#         t += 0.25
#
# gen_Z_file.close()

#####################################################





#################################################
# Average of I_z for central spin model

ancilla_state = 0
r=0
random_number =0

# def csv_init_write(filename, header):
#     csvfile = open(filename, 'w')
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     return writer, csvfile
#
# # Open the CSV files for writing
# header = ["n", "t", "w_rabi","delta","op_avg"]
# writer_gen_Z, gen_Z_file = csv_init_write("gen_sparse_I_z.csv", header)
#

# for n in range(3,12):
#     for w_rabi in range(0,11):
#         delta =0
#         while delta <=4:
#             t = 0
#             while t <= 17:
#                 # Compute the result and write to the file
#                 # r=0, m=1,k=1
#
#                 init_state = init_state_central_spin(ancilla_state, n, r)
#                 res_gen_Z = sparse_operator_average(t, init_state, I_z(n),gen_H_central_spin_model(n,random_number,w_rabi,delta))
#                 writer_gen_Z.writerow([n, t,w_rabi,delta, res_gen_Z])
#                 t += 0.25
#             delta += 0.25
#
# gen_Z_file.close()

#####################################################


end = time.time()
print("time=", end - start)

#########################################################

# data = "gen_overlap.csv" or data  = "gen_Z.csv"
# def spliting_data(n,data):
#     data_to_split = np.genfromtxt(data, delimiter=",", names=["x", "y", "z"])
#     times=[]
#     averages=[]
#     for i in range(1,len(data_to_split)):
#         if data_to_split[i][0]==n:
#             averages.append(data_to_split[i][2])
#             times.append(data_to_split[i][1])
#
#     return [np.array(times),np.array(averages)]

# for n in range(3,11):
#     [times, averages] = spliting_data(n, "gen_sparse_Z_coupled_1.csv")
#
#     plt.plot(times,averages,label="no fit", linewidth=1)
#     plt.scatter(times,averages)
#     plt.title('n='+ str(n))
#     plt.xlabel('time')
#     plt.ylabel('average of Z_tot, coupled system')
#
#     plt.savefig('n='+ str(n))
#     plt.show()

def spliting_data(n,w_rabi,delta,data):
    data_to_split = np.genfromtxt(data, delimiter=",", names=["x", "y", "z","a","b"])
    times=[]
    averages=[]
    for i in range(1,len(data_to_split)):
        # print(data_to_split[i])
        if data_to_split[i][0]==n:
            if data_to_split[i][2] == w_rabi:
                if data_to_split[i][3] == delta:
                    averages.append(data_to_split[i][4])
                    times.append(data_to_split[i][1])

    return [np.array(times),np.array(averages)]

for n in range(3,4):
    for w_rabi in range(1,8):
        for delta in range(0,5):
            [times, averages] = spliting_data(n, w_rabi,delta, "gen_sparse_I_z.csv")

            plt.plot(times,averages,label="no fit", linewidth=1)
            plt.scatter(times,averages)
            plt.title('n='+ str(n)+'w_rabi='+str(w_rabi)+'delta='+str(delta))
            plt.xlabel('time')
            plt.ylabel('average of I_z, central spin system')

            plt.savefig('n='+ str(n)+'w_rabi='+str(w_rabi)+'delta='+str(delta))
            plt.show()