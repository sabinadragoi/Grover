import numpy as np
from numpy import random
n=2

# initializing state
psi0= random.rand(n)
norm_psi0 = np.linalg.norm(psi0)
psi0= psi0/norm_psi0
print(psi0)

# initializing empty Hamiltonian
H = np.zeros((2**n,2**n),dtype=int)

for i in range(n):
    for j in range(i+1,n):

        for k in range(2**n-1): # k-th row of X_iX_j
            print("i,j=",i,j)
            print("k=",k)

            # representing k (row number) in binary (string)
            k_binary = np.base_repr(k, base=2)
            print("k_binary=", k_binary)


            # binary k but filled to n bits
            k_binary_full = np.base_repr(k,base=2, padding = n-len(str(k_binary))+1)
            print(k_binary_full)

            # initialize column number in binary - same as row number for now
            new_k_binary = np.base_repr(k,base=2, padding = n-len(str(k_binary))+1)
            print("new_k_binary=",new_k_binary)


            # checking whether we will flip to 0 or 1
            add_i = (int(new_k_binary[i])+1) % 2
            add_j = (int(new_k_binary[j])+1) % 2
            print("add_i=",add_i)
            print("add_j=",add_j)

            # column number after the X_iX_j gate in base 2
            new_k_binary= str(int(new_k_binary) + add_i* (10**(n-i-1))+ add_j* (10**(n-j-1)))
            print("new_k_binary=",new_k_binary)

            # column number after the X_iX_j gate in base 2, filled to n characters
            new_k_binary = new_k_binary.zfill(n)
            print(new_k_binary)

            # column number after the X_iX_j gate in base 10
            new_k = int(str(new_k_binary), base = 2)
            print(new_k)

            # adding the contribution of gate X_iX_j to the Hamiltonian matrix for row k
            H[k,new_k] += 1

print(H)
