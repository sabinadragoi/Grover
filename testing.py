import numpy as np
import matplotlib.pyplot as plt
import math

########################
# Schrodiger model

def c(n,n_minus):
    return (math.factorial(n)/(math.factorial(n_minus)*math.factorial(n-n_minus)))


def E(n,n_minus):
    return ((2*n_minus-n)**2)

def exact_F(n,t):
    sum =  0
    for i in range(n+1):
        a_i = (c(n,i)/(2**n))*np.exp(-1j*E(n,i)*t)
        sum += a_i

    return  (np.abs(sum))**2

def approx_F(n,t):
    sum1 = 0
    sum2 =0
    for i in range(n+1):
        a_i =  (c(n,i)/(2**n))*E(n,i)
        sum1 += a_i

        b_i = (c(n,i)/(2**n))*(E(n,i)**2)
        sum2 += b_i

    F = 1+ (t**2)*(sum1**2)-(t**2)*sum2
    return F

# print(exact_F(3,0))
# print(approx_F(3,0))





N=500
times = np.linspace(0, 4, 10 * N)

for n in range(3,14):
    periods= []
    approx_F_values = []
    exact_F_values = []

    for t in times:
        # approx_F_values.append(approx_F(n,t))
        exact_F_values.append(exact_F(n,t))
        if np.abs(exact_F(n,t)-1)<0.00001:
            periods.append(t)
    # print('n=',n,periods)

        # plt.plot(times, approx_F_values, "r-", label="approx curve", linewidth=1.5)
    plt.plot(times, exact_F_values, "b-", label="exact curve", linewidth=1.5)
    plt.title('n=' + str(n))
    plt.xlabel('time')
    plt.ylabel('F(t)')

    # plt.savefig('n='+ str(n)+', F(t), Schrodinger')
    # plt.show()


########################
# Heisenberg model

J=1

def alpha_k(n,k):
    return (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*((-1j)**k)

def exact_F_Heisenberg(n,t):
    sum =  0
    for k in range(n+1):
        a_k = alpha_k(n,k)*((np.cos(J*t/n))**(n-k))*((np.sin(J*t/n))**k)
        sum += a_k

    return  (np.abs(sum))**2

def approx_F_Heisenberg(n,t):
    sum1 = 0
    sum2 =0
    for i in range(n+1):
        a_i =  (c(n,i)/(2**n))*E(n,i)
        sum1 += a_i

        b_i = (c(n,i)/(2**n))*(E(n,i)**2)
        sum2 += b_i

    F = 1+ (t**2)*(sum1**2)-(t**2)*sum2
    return F

approx_F_values = []
exact_F_values = []

n=10
N=500
times= np.linspace(0, 0.01, 10*N)
for t in times:
    approx_F_values.append(approx_F_Heisenberg(n,t))
    exact_F_values.append(exact_F_Heisenberg(n,t))

# plt.plot(times, approx_F_values, "r-", label="approx curve", linewidth=1.5)
# plt.plot(times, exact_F_values, "b-", label="exact curve", linewidth=1.5)
plt.show()