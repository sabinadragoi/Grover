import numpy as np
import time
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

start_time = time.time()

# def CZ(qbit1,qbit2):
#     # qbit1 = [c_0,c_1]
#     # qbit2 = [d_0,d_1]
#     return [qbit1[0]*qbit2[0],qbit1[0]*qbit2[1],-qbit1[1]*qbit2[0],-qbit1[1]*qbit2[1]]
#
# def CNOT(qbit1,qbit2):
#     # qbit1 = [c_0,c_1]
#     # qbit2 = [d_0,d_1]
#     return [qbit1[0]*qbit2[0],qbit1[0]*qbit2[1],qbit1[1]*qbit2[1],qbit1[1]*qbit2[0]]

def normalize(state):
    # takes np.array representing state of system of qubits
    # outputs state normalized using Euclidean norm

    sum_of_squares = 0
    for i in range(len(state)):
        sum_of_squares += np.abs(state[i]) ** 2
    return [j/np.sqrt(sum_of_squares) for j in state]


def total_CNOT(init_state, i1,i2, normalized):
    # init state is a list of 2**n entries, each corresponding to the coefficient of that basis state
    # n = total number of qubits considered
    n = int(np.log2(len(init_state)))
    # i1 = index of control qubit, i2 = index of target qubit
    new_state = init_state.copy()
    for i in range(len(init_state)):
        basis_state = "{0:b}".format(i).zfill(n)
        basis_state_entry = init_state[i]
        if basis_state[i1] == '1':
            basis_state_list = list(basis_state)
            basis_state_list[i2] = str(1 - int(basis_state[i2]))
            new_basis_state= ''.join([str(item) for item in basis_state_list])
            new_basis_state_index = int(new_basis_state,2)
            new_state[new_basis_state_index] += basis_state_entry
            new_state[i] = new_state[i] - basis_state_entry

    if normalized == False:
        return new_state
    if normalized == True:
        return normalize(new_state)



def total_CZ(init_state, i1,i2):
    # init state is a list of 2**n entries, each corresponding to the coefficient of that basis state
    # NOTE: after applying a CZ gate, the state remains normalized

    # n = total number of qubits considered
    n = int(np.log2(len(init_state)))
    # i1 = index of control qubit, i2 = index of target qubit
    new_state = init_state.copy()
    for i in range(len(init_state)):
        basis_state = "{0:b}".format(i).zfill(n)
        basis_state_entry = init_state[i]

        if basis_state[i1] == '0' and basis_state[i2] == '0':
            new_state[i] = - basis_state_entry

    return new_state

def add_two_qbits(state1, state2):
    # takes 2 states, each an arbitrary number of qubits n & m, written in their respective bases
    # 1st state has 2**n basis vectors, 2nd state has 2**m basis vectors
    # e.g. for 1 qubit: qbit = [c_0,c_1]
    # outputs the overall state, written in the total basis, which has 2**(n+m) basis vectors

    n = int(np.log2(len(state1))) # number of qubits of 1st state
    m = int(np.log2(len(state2))) # number of qubits of 2nd state
    total_state = np.zeros(2**(n+m)) # 2**(n+m) = len(state1)+len(state1)
    for i in range(len(state1)):
        basis_state_1 = "{0:b}".format(i).zfill(n)
        basis_state_1_entry = state1[i]
        for j in range((len(state2))):
            basis_state_2 = "{0:b}".format(j).zfill(m)
            basis_state_2_entry = state2[j]

            basis_state_total = basis_state_1 + basis_state_2
            basis_state_total_index = int(basis_state_total,2)
            total_state[basis_state_total_index] = basis_state_1_entry*basis_state_2_entry
    return normalize(total_state)


# add_qbits(*argv)
#    A wrapper around `add_two_qbits`.
#    Takes an arbitrary number of arguments, and performs
#    addition on all of them.
#
#    Should not be called with fewer than 2 arguments.
#
#    Should not be called with an extremely large number
#    of arguments, as the recursion depth limit might be
#    exceeded.

def add_qbits(*argv):
    assert len(argv) >= 2, ("add_qbits cannot be called "
                            "with fewer than 2 arguments")

    if len(argv) == 2:
        return add_two_qbits(*argv)

    return add_qbits(add_two_qbits(argv[0], argv[1]), *argv[2:])



def un_entangle_1_qubit(state,chosen_qubit):
    n = int(np.log2(len(state)))  # number of qubits of state
    new_state_0 = np.zeros(2**(n-1))
    new_state_1 = np.zeros(2**(n-1))

    for i in range(len(state)):
        # print('i=', i)

        basis_state = "{0:b}".format(i).zfill(n)
        # print('basis_state=', basis_state)

        basis_state_entry = state[i]
        # print('basis_state_entry=',basis_state_entry)

        basis_state_list = list(basis_state)

        del basis_state_list[chosen_qubit]
        new_basis_state = ''.join([str(item) for item in basis_state_list])
        new_basis_state_index = int(new_basis_state, 2)
        # print('basis_state[chosen_qubit]=',basis_state[chosen_qubit])


        if basis_state[chosen_qubit] == '0':
            new_state_0[new_basis_state_index] = basis_state_entry
        else:
            new_state_1[new_basis_state_index] = basis_state_entry
        # print('new_state_0=',new_state_0)
        # print('new_state_1=', new_state_1)


    all_0_for_0 = np.all((new_state_0 == 0))
    all_0_for_1 = np.all((new_state_1 == 0))
    # print(all_0_for_0)
    # print(all_0_for_1)

    if all_0_for_0 == False and all_0_for_1 == False:
        ratio = 0
        for j in range(2**(n-1)):
            if new_state_0[j] == 0 and new_state_1[j] != 0:
                return 'Cannot un-entangle this qubit from the overall state.'
            elif new_state_1[j] == 0 and new_state_0[j] != 0:
                return 'Cannot un-entangle this qubit from the overall state.'
            elif new_state_1[j] != 0 and new_state_0[j] != 0:
                ratio += new_state_0[j]/new_state_1[j]
                break
        # print('ratio=',ratio)

        for j in range(2**(n-1)):
            if new_state_0[j] == 0 and new_state_1[j] != 0:
                return 'Cannot un-entangle this qubit from the overall state.'
            elif new_state_1[j] == 0 and new_state_0[j] != 0:
                return 'Cannot un-entangle this qubit from the overall state.'
            elif new_state_1[j] != 0 and new_state_0[j] != 0:
                if ratio != new_state_0[j]/new_state_1[j]:
                    return 'Cannot un-entangle this qubit from the overall state.'

        return (normalize(new_state_0), normalize([1,1/ratio]))


    elif all_0_for_0 == True:
        return (normalize(new_state_1), [0, 1])

    elif all_0_for_1 == True:
        return (normalize(new_state_0), [1, 0])




c_0 = np.sqrt(1/2)
c_1 = np.sqrt(1/2)
d_0 = np.sqrt(1/3)
# d_1 = np.sqrt(2/3)
d_1 = 0
q_1 = [c_0,c_1]
q_2 = [d_0,d_1]
a1a2 = [np.sqrt(1/2),0,0,np.sqrt(1/2)] # Bell state

q1a1a2q2 = add_qbits(q_1,a1a2,q_2)
# print(un_entangle(q1a1a2q2,2))
print(q1a1a2q2)

inter_state = total_CNOT(q1a1a2q2,0,1,True)
inter_state = total_CZ(inter_state,1,2)
inter_state = total_CNOT(inter_state,0,1,True)
inter_state = total_CNOT(inter_state,3,2,True)

print(inter_state)
print(un_entangle_1_qubit(inter_state,0))

# Function meant to untangle ANY NUMBER of qubits from the total state
# NOT YET FINISHED

# def un_entangle(state,chosen_qubits):
#     # chosen_qubits = list of qubits to unentangle
#     n = int(np.log2(len(state)))  # number of total qubits of state
#     m = len(chosen_qubits) # number of qubits to un_entangle
#     output_state = np.zeros(2**(n-m))
#     output_state_unentangled = np.zeros(2 ** m)
#     list_of_lists = []
#
#     for i in range(2**(m)):
#         state_of_one_qubit = np.zeros(2**(n-m))
#         list_of_lists.append(state_of_one_qubit)
#
#     for i in range(len(state)):
#         # print('i=', i)
#
#         basis_state = "{0:b}".format(i).zfill(n)
#         # print('basis_state=', basis_state)
#
#         basis_state_entry = state[i]
#         # print('basis_state_entry=',basis_state_entry)
#
#         basis_state_list = list(basis_state)
#
#         list_of_untangled_basis_states = [] # list of strings
#         for k in range(len(chosen_qubits)):
#             list_of_untangled_basis_states.append(basis_state_list[chosen_qubits[k]])
#             del basis_state_list[chosen_qubits[k]]
#
#         new_basis_state = ''.join([str(item) for item in basis_state_list])
#         new_basis_state_index = int(new_basis_state, 2)
#
#         real_basis_vector_state = ''.join([str(item) for item in list_of_untangled_basis_states])
#         number_of_basis_vector = int(real_basis_vector_state, 2)
#
#         list_of_lists[number_of_basis_vector][new_basis_state_index] = basis_state_entry
#
#     list_of_unexistent_states = []
#     for j in range(2 ** m):
#         if np.all((list_of_lists[j] == 0)) == True:
#             output_state_unentangled[j] = 0
#             binary_unexistent_basis_state = "{0:b}".format(j).zfill(m)
#
#     if np.all((list_of_all_0_boolvalues == False)) == True:
#         ratio = 0
#         for j in range(2**(n-m)):
#             if new_state_0[j] == 0 and new_state_1[j] != 0:
#                 return 'Cannot un-entangle this qubit from the overall state.'
#             elif new_state_1[j] == 0 and new_state_0[j] != 0:
#                 return 'Cannot un-entangle this qubit from the overall state.'
#             elif new_state_1[j] != 0 and new_state_0[j] != 0:
#                 ratio += new_state_0[j]/new_state_1[j]
#                 break
#         # print('ratio=',ratio)
#
#         for j in range(2**(n-1)):
#             if new_state_0[j] == 0 and new_state_1[j] != 0:
#                 return 'Cannot un-entangle this qubit from the overall state.'
#             elif new_state_1[j] == 0 and new_state_0[j] != 0:
#                 return 'Cannot un-entangle this qubit from the overall state.'
#             elif new_state_1[j] != 0 and new_state_0[j] != 0:
#                 if ratio != new_state_0[j]/new_state_1[j]:
#                     return 'Cannot un-entangle this qubit from the overall state.'
#
#         return (normalize(new_state_0), normalize([1,1/ratio]))
#
#
#     else:




def draw_circuit():
    anc = QuantumRegister(2, 'a')
    qr = QuantumRegister(2, 'q')
    # cr = ClassicalRegister(1, 'c')
    cr = QuantumRegister(1, 'c')
    circuit = QuantumCircuit(anc,qr, cr)

    circuit.cx(qr[0],anc[0])
    circuit.cx(qr[1],anc[1])
    circuit.cz(qr[0],anc[0],label = 'CZ')

    circuit.cx(qr[0],cr[0])
    circuit.cx(qr[1],cr[0])
    circuit.fredkin(cr[0],anc[0],anc[1])
    circuit.cz(qr[0],anc[0],label = 'CZ')

    circuit.barrier()

    circuit.cx(qr[0],anc[0])
    circuit.cx(qr[1],anc[1])

    circuit.draw(output='mpl')
    plt.show()

end_time = time.time()
print('time=', end_time-start_time)