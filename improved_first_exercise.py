import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from scipy import sparse
from scipy import integrate
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
import csv
import time
import random

start = time.time()

pauli_X = sparse.csc_matrix((np.array([1,1]), (np.array([0,1]), np.array([1,0]))), shape=(2, 2))
pauli_Z = sparse.csc_matrix((np.array([1,-1]), (np.array([0,1]), np.array([0,1]))), shape=(2, 2))

def gen_sparse_X_i(n,i):
    # X_i = I \tensor I ...\tensor X_i \tensor .. I
    # operator that applies gate X to atom i and Identity to all others
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
    # operator that applies gate X to atom i and atom j, and Identity to all others
    return np.multiply(gen_sparse_X_i(n,i),gen_sparse_X_i(n,j))


def gen_sparse_H(n):
    # H = \sum_{i,j} X_iX_j
    # generating Hamiltonian for all-to-all interaction

    # initializing empty Hamiltonian
    H = sparse.csc_matrix((2 ** n, 2 ** n), dtype=int)

    for i in range(n):
        for j in range(i+1,n):
            H = np.add(H,gen_sparse_X_ij(n,i,j))

    return H

def init_sparse_state(n,psi0_string):
    # generating initial state as sparse from string

    # sanity check
    if n != len(psi0_string):
        print("wrong input for initial state")

    # any string in binary of n characters corresponds to a base 10 integer up to 2**n, which is the initial state
    number_of_state = int(psi0_string, base =2)

    # represent initial state as a column vector with just one non-zero number
    return sparse.csc_matrix((np.array([1]), (np.array([number_of_state]), np.array([0]))), shape=(2**n, 1))

def init_sparse_state_old(n,r):
    # generates initial state for the 2 extreme cases and a random superposition
    # NO LONGER IN USE

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
    # generate initial state of system with m initial bosons, and m+n potential bosons at the end of the evolution
    # starts in pure Harmonic Oscillator state k
    return sparse.csc_matrix((np.array([1]), (np.array([k]), np.array([0]))), shape=(m+n+1, 1))

def coupled_init_state(m,k,n,r):
    # generate initial state of system with m initial bosons and n atoms
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
def gen_sparse_overlap_op(n,psi0_string):
    psi0= init_sparse_state(n,psi0_string)
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
    sigma_z = np.array([[1, 0], [0, -1]])

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
    S_z_variable = sparse.csc_matrix((2 ** n, 2 ** n ), dtype=int)
    for i in range(n):
        S_z_variable = np.add(S_z_variable,gen_weights(n,random)[i]* gen_sparse_sigma_i_z(n,i))
    return S_z_variable

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


def init_state_central_spin(ancilla_state,n,psi0_string):
    if ancilla_state == 0:
        ancilla = np.array([[1],[0]])
        return sparse.kron(ancilla,init_sparse_state(n,psi0_string))
    elif ancilla_state ==1:
        ancilla = np.array([[0], [1]])
        return sparse.kron(ancilla,init_sparse_state(n,psi0_string))
    else:
        return 'wrong initial state'

def I_z(n):
    return sparse.kron(pauli_Z,sparse.identity(2**n))


def expected_value_I_z(n, psi0_string, random):  # central spin model

    weights = gen_weights(n, random)
    expected_value = 0
    for i in range(n):

        if int(psi0_string[i]) == 0:
            expected_value += weights[i]
        elif int(psi0_string[i]) == 1:
            expected_value = expected_value - weights[i]
        else:
            print("wrong input")
            continue

    return expected_value

# Checking that everything works so far

# ancilla_state = 0
# psi0_string = "000"
# random_number =0
#
# w_rabi = 3
# delta = 1
#
# n=3
# t=2
#
# init_state = init_state_central_spin(ancilla_state, n, psi0_string)
#
# H = gen_H_central_spin_model(n,random_number,w_rabi,delta)
#
# psi_t = time_ev_sparse_state(t,init_state,H)
#
# # new_psi_t = sparse.csc_matrix([[x/np.linalg.norm(psi_t.toarray()) for x in y] for y in psi_t.toarray()])
# norm_psi_t = np.linalg.norm(psi_t.toarray())
#
# av_I_z = sparse_operator_average(t, init_state, I_z(n),H)
# print(av_I_z/ (norm_psi_t**2))



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
random_number =0
w_rabi = 0.5
t_max = 20

delta_range = 2
delta_step = 0.1

# psi0_string = '0000000000000000'
# n=16
# if len(psi0_string) != n:
#     print('error')
# delta = - expected_value_I_z(n, psi0_string, random_number)
# H = gen_H_central_spin_model(n, random_number, w_rabi, delta)


# def csv_init_write(filename, header):
#     csvfile = open(filename, 'w')
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     return writer, csvfile
#
#
# for n in range(2,3):
#     # Open the CSV files for writing
#     header = ["n", "t", "w_rabi", "delta", "op_avg"]
#     writer_gen_Z, gen_Z_file = csv_init_write("gen_sparse_I_z_n"+str(n)+".csv", header)
#
#
#     # list_of_psi0_string = []
#     # for i in range(1,2): # range used to be 2**n
#     #     list_of_psi0_string.append("{0:b}".format(i).zfill(n))
#
#     list_of_psi0_string = ['00']
#
#     for psi0_string in list_of_psi0_string:
#         delta = - expected_value_I_z(n, psi0_string, random_number)
#         pairs = []
#
#         for i in range(int(delta_range*10+1)):
#             pairs.append((w_rabi, delta - delta_range/2 + i * delta_step))
#
#         for (w_rabi,delta) in pairs:
#             H = gen_H_central_spin_model(n, random_number, w_rabi, delta)
#             # print(H.toarray())
#             t = 0
#             while t <= t_max:
#
#                 init_state = init_state_central_spin(ancilla_state, n, psi0_string)
#
#                 psi_t = time_ev_sparse_state(t,init_state,H)
#                 av_I_z = sparse_operator_average(t, init_state, I_z(n),H)
#                 norm_psi_t = np.linalg.norm(psi_t.toarray())
#
#                 res_gen_Z = av_I_z/ (norm_psi_t**2)
#                 writer_gen_Z.writerow([n, t,w_rabi,delta, res_gen_Z])
#                 t += 1
#
#
#     gen_Z_file.close()

#####################################################


end = time.time()
print("time=", end - start)

#########################################################
# Analyzing data for Coupled Boson/Spins system

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

#################################################
# Analyzing data of I_z

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

#######################################################
# Analyzing I_z(t) for different psi0


# for n in range(2,7):
#     list_of_psi0_string = []
#     for i in range(2**n):
#         list_of_psi0_string.append("{0:b}".format(i).zfill(n))
#
#     for psi0_string in list_of_psi0_string:
#         delta = - expected_value_I_z(n, psi0_string, random_number)
#         pairs = []
#
#         for i in range(int(delta_range*10+1)):
#             pairs.append((w_rabi, delta - delta_range/2 + i * delta_step))
#
#         for (w_rabi,delta) in pairs:
#
#             [times, averages] = spliting_data(n, w_rabi,delta, "gen_sparse_I_z_n"+str(n)+".csv")
#
#             # H = gen_H_central_spin_model(n, random_number, w_rabi, delta)
#             # print('w,delta=', w_rabi, delta)
#             # print()
#             # print('H=', H.toarray())
#
#
#             plt.plot(times,averages,label="no fit", linewidth=1)
#             plt.scatter(times,averages)
#             plt.title("psi0= "+psi0_string+' n='+ str(n)+' w_rabi='+str(w_rabi)+' delta='+str(delta))
#             plt.xlabel('time')
#             plt.ylabel('average of I_z, central spin system')
#
#
#             # plt.savefig('psi0='+psi0_string)
#             plt.show()


#################################################################
# Analyzing the average of I_z vs delta


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


# N = 10000
# for n in range(2,3):
#     # list_of_psi0_string = []
#     # for i in range(2**n):
#     #     list_of_psi0_string.append("{0:b}".format(i).zfill(n))
#
#     list_of_psi0_string = ['00']
#
#     for psi0_string in list_of_psi0_string:
#         averages_I_z_vs_delta = []
#         Deltas= []
#         delta = - expected_value_I_z(n, psi0_string, random_number)
#         pairs = []
#
#         for i in range(int(delta_range*10+1)):
#             pairs.append((w_rabi, delta - delta_range/2 + i * delta_step))
#
#         for (w_rabi,delta) in pairs:
#
#             [times, averages] = spliting_data(n, w_rabi,delta, "gen_sparse_I_z_n"+str(n)+".csv")
#
#             res = fit_sin(times, averages)
#             period = '%(period)s' %res
#             times2= np.linspace(0,4*float(period),N)
#             sum= 0
#             for i in range(N):
#                 sum+= res["fitfunc"](times2)[i]
#             average_value_I_z = sum/len(times2)
#
#             averages_I_z_vs_delta.append(average_value_I_z)
#             Deltas.append(delta)
#
#         plt.plot(Deltas,averages_I_z_vs_delta,label="no fit", linewidth=1)
#
#         plt.title("psi0= "+psi0_string)
#         plt.xlabel('delta')
#         plt.ylabel('averages of I_z, central spin system')
#
#
#         plt.savefig('averages of I_z vs delta, central spin system'+'psi0='+psi0_string)
#         plt.show()

##################################################
# Solving master's equation

def commutator(A,B): # A,B are operators
    return np.matmul(A,B)-np.matmul(B,A)

def anti_commutator(A,B): # A,B are operators
    return np.matmul(A,B)+np.matmul(B,A)

def masters_eq(rho, n, random_number, w_rabi, delta):
     # rho is the density matrix of a system of n qubits, (2^n,2^n)
     full_rho = np.kron(np.identity(2),rho)
     H = gen_H_central_spin_model(n, random_number, w_rabi, delta)
     (a,a_dagger)= low_raise_op(1,n)
     sum = 0
     for i in range(n):
         sum = sum + np.matmul(np.matmul(a,full_rho), a_dagger) - anti_commutator(np.matmul(a_dagger,a),full_rho)/2
     return -1j*commutator(H,full_rho)+sum

# def solve_masters_eq(rho,t, n, random_number, w_rabi, delta,rho_0):
#     (times,values) = integrate.solve_ivp(masters_eq(rho, n, random_number, w_rabi, delta), (0,t), rho_0, method='RK45')
#     return (times,values)



def solve_Schrodinger_RK45(psi0_string,n,random,w_rabi,delta,t):
    # initial state psi0 has size (2**(n+1),1) and we convert it to a ndarray
    psi0= init_state_central_spin(int(psi0_string[0]),n,psi0_string).toarray()

    # H has size (2**(n+1),2**(n+1)) and we convert it to a ndarray
    H = gen_H_central_spin_model(n,random,w_rabi,delta).toarray()

    # Schrodinger's equation, which we want to be of size (2**(n+1),)
    Schrodinger_func = lambda t, psi: -1j*np.matmul(H,psi).reshape((2**(n+1),))

    # assigning complex type for initial state
    new_psi0 =  np.array(psi0.reshape((2**(n+1),)),dtype = complex)

    # solving the equation
    sol = integrate.solve_ivp(Schrodinger_func, [0, t], new_psi0, method='RK45',vectorized=True)

    # return just the solutions for y
    return sol.y[0]

psi0_string = '00'
n=2
delta = 2
t=10


psi0= init_state_central_spin(int(psi0_string[0]),n,psi0_string)
H = gen_H_central_spin_model(n,random_number,w_rabi,delta)
psi_t = time_ev_sparse_state(t,psi0,H).toarray().reshape((2**(n+1),))
print('psi_t=',psi_t)
print('psi_t_ODE_solution=',solve_Schrodinger_RK45(psi0_string,n,random_number,w_rabi,delta,t))

