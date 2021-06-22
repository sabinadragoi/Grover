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

def gen_sparse_X_ij(n,i,j):
    if i==0:
        X_ij= sparse.csc_matrix(np.array([[0, 1], [1, 0]]))
    else:
        X_ij = sparse.identity(2)
    for k in range(1,n):
        if k== i or k==j:
            new_X_ij = sparse.csc_matrix(np.array([[0, 1], [1, 0]]))
        else:
            new_X_ij = sparse.identity(2)
        X_ij = scipy.sparse.kron(X_ij, new_X_ij)

    return X_ij


def gen_sparse_H(n):
    # initializing empty Hamiltonian
    H = sparse.csc_matrix((2 ** n, 2 ** n), dtype=int)

    for i in range(n):
        for j in range(i+1,n):
            H = np.add(H,gen_sparse_X_ij(n,i,j))

    return H


def init_sparse_state(n,r):
    if r==1:
        psi0= sparse.csc_matrix((np.array([1]), (np.array([0]), np.array([0]))), shape=(2**n, 1))
    else:
        psi0 = (1 / np.sqrt(2 ** n)) * np.ones((2 ** n, 1))

    return psi0


def time_ev_sparse_state(n,t,psi0):
    # just to remember that we consider hbar =1
    hbar =1

    psi_t= sparse.linalg.expm_multiply(-1j*t*gen_sparse_H(n)/hbar, psi0)

    return psi_t


def sparse_operator_average(n,t,psi0,op):
    psi_t = time_ev_sparse_state(n,t,psi0)
    return ((psi_t.conj().T @ (op @ psi_t)).real.toarray())[0][0]



# operator |psi0\rangle \langle psi0| to measure overlap with the initial state
def gen_sparse_overlap_op(n,r):
    psi0= init_sparse_state(n,r)
    init_overlap_op = sparse.kron(psi0, psi0.conj().T)
    return init_overlap_op


def gen_sparse_Z_i(n,i):
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

###################################################################

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
#         res_gen_overlap = sparse_operator_average(n,t,init_sparse_state(n,1),gen_sparse_overlap_op(n,1))
#         writer_gen_overlap.writerow([n, t, res_gen_overlap])
#
#         res_gen_Z = sparse_operator_average(n, t, init_sparse_state(n, 1), gen_sparse_Z_tot(n))
#         writer_gen_Z.writerow([n, t, res_gen_Z])
#         t += 0.25
#
# gen_overlap_file.close()
# gen_Z_file.close()

##########################################


end = time.time()
print("time=", end - start)

# data = "gen_overlap.csv" or data  = "gen_Z.csv"
def spliting_data(n,data):
    data_to_split = np.genfromtxt(data, delimiter=",", names=["x", "y", "z"])
    times=[]
    averages=[]
    for i in range(1,len(data_to_split)):
        if data_to_split[i][0]==n:
            averages.append(data_to_split[i][2])
            times.append(data_to_split[i][1])

    return [np.array(times),np.array(averages)]

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

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t+p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c

    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

N=500
times2= np.linspace(0, 17, 10*N)


for n in range(3,11):
    for data in ["gen_sparse_overlap.csv","gen_sparse_Z.csv"]:

        [times, averages] = spliting_data(n, data)
        res = fit_sin(times, averages)
        # print("Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res)


        plt.plot(times2, res["fitfunc"](times2), "r-", label="fit curve", linewidth=1.5)

        # plt.plot(times,averages,label="no fit", linewidth=1)
        plt.scatter(times,averages)
        plt.title('n='+ str(n))
        plt.xlabel('time')

        if data == "gen_sparse_overlap.csv":
            operator = "overlap"
            plt.ylabel('average of overlap_op')
        else:
            operator = "Z_tot"
            plt.ylabel('average of Z_tot op')

        plt.savefig('n='+ str(n)+', operator='+ operator)
        # plt.show()