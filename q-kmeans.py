import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi
import numpy as np
import random

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.circuit.library import RYGate, RXGate, RZGate
import qiskit.quantum_info as qi
from qiskit.extensions import *
from qiskit.aqua.utils import tensorproduct


'''
According to the pseudo codes in Ref[1], we here present vanilla codes of the quantum-enhanced K-means algorithm. So, you can enjoy the modifications due to your application.

References:
[1] H. Ohno, A quantum algorithm of K-means toward practical use, Quantum Information Processing, Published online: 05 April 2022.
'''

# Set a quantum simulator to the backend
backend = Aer.get_backend('qasm_simulator')

# Define qsub_md for K = 3.
'''
Quantum subroutine: qsub_md
Input
v0: sample in training data, v1: cluster centroid, and shots: number of shots
Output
result: Euclidean distance between v0 and v1
'''
def qsub_md(v0, v1, shots):
    # Eq. (11) in the text
    x_1 = [v0[0]] + [x[0] for x in v1]
    x_2 = [v0[1]] + [x[1] for x in v1]

    M = 3

    # Eq. (9) in the text
    sin_t = np.sqrt((M+1)/(2*M))
    cos_t = np.sqrt((M-1)/(2*M))

    I = np.array([[1, 0], [0, 1]])

    # Set Hadamard matrix
    H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])
    H2 = tensorproduct(H, H)
    I2 = tensorproduct(I, I)

    # Set V matrix (Eq. (5) in the text)
    V = UnitaryGate(cos_t*I2 - 1j*sin_t*H2, label='V')
    V_d = UnitaryGate(cos_t*I2 + 1j*sin_t*H2, label='V^dagger')
    
    def dist(v, u1, u2, u3):
        nq = 2 + 1 + 1
        qr = QuantumRegister(nq)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
    
        v = 2.*np.arcsin(-v)
        u1 = 2.*np.arcsin(u1)
        u2 = 2.*np.arcsin(u2)
        u3 = 2.*np.arcsin(u3)

        # Apply V
        qc.append(V, [0,1])
        qc.x(0)
        qc.x(1)
        ccry=RYGate(v).control(2)
        qc.append(ccry, [0,1,2])
        qc.x(0)

        # Apply U
        ccry=RYGate(u1).control(2)
        qc.append(ccry, [0,1,2])
        qc.x(1)
        qc.x(0)
        ccry=RYGate(u2).control(2)
        qc.append(ccry, [0,1,2])
        qc.x(0)
        ccry=RYGate(u3).control(2)
        qc.append(ccry, [0,1,2])

        # Apply V^dagger
        qc.append(V_d, [0,1])
        qc.x(0)
        qc.x(1)
        qc.mct([0,1,2], 3)
        qc.x(1)
        qc.x(0)
        qc.measure([3], cr)

        job = execute(qc, backend=backend, shots=shots, seed_simulator=12345)
        counts = job.result().get_counts(qc)

        if ('0' in counts) and counts['0'] == shots:
            p1 = 0.
        else:
            p1 = counts['1']/shots
        return p1

    p1 = dist(x_1[0], x_1[1], x_1[2], x_1[3])
    p2 = dist(x_2[0], x_2[1], x_2[2], x_2[3])
    result = 2*np.sqrt(p1 + p2)
    return result

# Define qsub_min for K = 3.
'''
Quantum subroutine: qsub_min
Input
t_m: number of iterations, v: data, and shots: number of shots
Output
y: index with the minimum value in data
'''
def qsub_min(t_m, v, shots):
    # Set dummy value for 2^n
    v_ext = np.append(v, 10.0)

    # Set oracle for Grover's algorithm
    def oracle(i, dt):
        f1 = -1. if dt[0] < dt[i] else 1.
        f2 = -1. if dt[1] < dt[i] else 1.
        f3 = -1. if dt[2] < dt[i] else 1.
        f4 = -1. if dt[3] < dt[i] else 1.
        op = qi.Operator([[f1, 0, 0, 0],
                          [0, f2, 0, 0],
                          [0, 0, f3, 0],
                          [0, 0, 0, f4]])
        return op

    y = np.random.choice(len(v))
    for _ in range(t_m):
        qc = QuantumCircuit(2)
        oracle_op = oracle(y, v_ext)
        qc.h([0,1])

        # Set number of iterations for Grover's iteration at random
        t_r = np.random.choice(len(v_ext))+1

        # Grover's iteration
        for i in range(t_r):
            qc.unitary(oracle_op, [0,1], label='oracle')
            qc.h([0,1])
            qc.x([0,1])
            qc.h([1])
            qc.cx(0,1)
            qc.h([1])
            qc.x([0,1])
            qc.h([0,1])
        qc.measure_all()
    
        job = execute(qc, backend=backend, shots=shots, seed_simulator=12345)
        counts = job.result().get_counts(qc)
        max_c = 0
        for c in counts:
            if max_c < counts[c]:
                max_c = counts[c]
                new_y = int(c, 2)
        if v_ext[new_y] < v_ext[y]:
            y = new_y
    return(y)

# Load data (synthetic data).
dt = np.loadtxt('DATA/data-x.txt')
Y = np.loadtxt('DATA/data-label.txt')
M, num_features = dt.shape

# Normalize data on x-y positive region.
min = dt.min(axis=0, keepdims=True)
max = dt.max(axis=0, keepdims=True)
dt = (dt - min)/(max - min)

np.random.seed(1)

# Set parameter values.
K = 3
Tm = 2
T = 5
shots = 8192

class_list = np.zeros(M, dtype=int)
for i in range(M):
    class_list[i] = np.random.randint(0, K)

# Run.
for s in range(T):
    for i in range(M):
        distances = []
        for j in range(K):
            dt_j = dt[class_list==j]
            lst_r = np.random.choice(len(dt_j), len(dt_j) if len(dt_j) < K else K, replace=False)
            d = qsub_md(dt[i], dt_j[lst_r], shots)
            distances.append(d)
        class_list[i] = qsub_min(Tm, distances, shots)

# Set data for graphs.
df = pd.DataFrame(dt)
df0 = df[class_list==0]
df1 = df[class_list==1]
df2 = df[class_list==2]

# Plot graph.
plt.scatter(x=df0[0], y=df0[1], s=120, marker='o')
plt.scatter(x=df1[0], y=df1[1], s=120, marker='s')
plt.scatter(x=df2[0], y=df2[1], s=120, marker='x')
plt.axes().set_aspect('equal')
plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.tick_params(labelsize=14)
plt.show()
plt.savefig("sample.png")
plt.clf()
