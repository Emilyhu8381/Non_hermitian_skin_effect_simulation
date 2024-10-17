# 1.Jordan-Wigner transformation
# 2.Hatano-Nelson Hamiltonian and the 2**7 by 2**7 global operator
# 3.SSH Hamiltonian
# 4.physical and ancilla qubits register (global and local)
# 5.encoding non-unitary U_{\pm} in bigger unitary matrices R_{R_{\pm}}

# 6. quimb tensornetwork optimization
# 7. measurement (aer simualtor, shots)
# 8. dynamical density evolution ni(T) encoding
# 9. dynamical density evolution plot (heatmap and colorbar)

import qiskit
import qiskit_aer
import qiskit_algorithms 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, AncillaRegister 
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, UnitaryGate, RXGate
from qiskit import transpile 
from qiskit_aer import AerSimulator

import quimb as qu
import quimb.tensor as qtn

import numpy as np
import math 
import matplotlib.pyplot as plt 


# 1.Jordan-Winger transformation (spin operators onto fermionic annihilation/creation operators)
# specific in a sense that it is a 1D spin chain model
# general in a sense that it applies to all N number of sites, with each being i or j 

# 1.Pauli matrices 
pauli_x = Pauli('X')
pauli_y = Pauli('Y')
pauli_z = Pauli('Z')

# 2.Pauli lowering/rising operators
sigma_x = pauli_x.to_matrix()
sigma_y = pauli_y.to_matrix()
sigma_z = pauli_z.to_matrix()
identity = [[1,0], [0,1]]
identity = np.array(identity)
sigma_plus = (sigma_x + 1j *sigma_y)/2
sigma_minus = (sigma_x - 1j *sigma_y)/2



# 2.Hatano-Nelson Hamiltonian
# H_HN= -\sum^{L-2}_{j=0} ( (J+\gamma)X^+_j*X^-_{j+1} + (J-\gamma)X^-_j*X^+_j+1)
# X^+_j = \ket{\uparrow}\bra{\uparrow},  X^-_j = \ket{\downarrow}\bra{\downarrow}

def HN_ham (length_l:int, J, gamma):
    H1=np.zeros((2**(length_l), 2**(length_l)), dtype=complex)
    H2=np.zeros((2**(length_l), 2**(length_l)), dtype=complex)

    # (J+\gamma) H1 part 
    for i in range(length_l-1): # 0,1,2,3,4
        sigma_prod = np.kron(sigma_minus, sigma_plus)

        if i < (length_l-2):
            pre_sigma = identity
            for _ in range(length_l-i-3): 
                pre_sigma= np.kron(pre_sigma, identity)
            result = np.kron(pre_sigma, sigma_prod)
        else:
            result = sigma_prod 
        

        if i > 0:
            post_sigma = identity 
            for _ in range(i-1): 
                post_sigma = np.kron(post_sigma, identity)
            result = np.kron(result, post_sigma)
    
        H1 += result
    H1 = -1 * (J+gamma) * H1 
    #print(f"final H1 is {H1}")


    # (J-\gamma) H2 part 
    for i in range(length_l-1): # 0,1,2,3,4
        sigma_prod = np.kron(sigma_plus, sigma_minus)

        if i < (length_l-2):
            pre_sigma = identity
            for _ in range(length_l-i-3): 
                pre_sigma= np.kron(pre_sigma, identity)
            result = np.kron(pre_sigma, sigma_prod)
        else:
            result = sigma_prod 
        

        if i > 0:
            post_sigma = identity 
            for _ in range(i-1): 
                post_sigma = np.kron(post_sigma, identity)
            result = np.kron(result, post_sigma)
    
        H2 += result
    H2 = -1 * (J+gamma) * H1 
    #print(f"final H1 is {H2}")

    H = H1 + H2 
    return H 


# 3.SSH Hamiltonian 
# H_SSH= -\sum^{L/2}_j=0 ( (J+\gamma)X^+_{2j}*X^-_{2j+1} + (J-\gamma)X^-_{2j}*X^+_{2j+1}) +
# J/2 -\sum^{L/2-1}_{j=0} (X_{2j+1}*X_{2j+2} + Y_{2j+1}*Y_{2j+2} ))
def ssh_ham (length_l, J, gamma):
    H1=np.zeros((2**(length_l+2), 2**(length_l+2)), dtype=complex)
    H2=np.zeros((2**(length_l+2), 2**(length_l+2)), dtype=complex)
    H3=np.zeros((2**(length_l+2), 2**(length_l+2)), dtype=complex)
    H4=np.zeros((2**(length_l+2), 2**(length_l+2)), dtype=complex)
    #print(f"shape of H {H.shape}")

    # H1: (J+\gamma) part 
    for i in range(length_l/2):
        sigma_prod = np.kron(sigma_plus, sigma_minus)
        identities=np.kron(identity, identity)
        pre_sigma = np.kron(identity, identity)
        post_sigma = np.kron(identity, identity)
            
        if i > 0:
            for _ in range(i-2):
                pre_sigma = np.kron(pre_sigma, identities)
            result = np.kron(pre_sigma, sigma_prod)
        
        else: 
            result = sigma_prod


        if i < (length_l/2): # i=0,1,2
            for _ in range(length_l/2-i-1): # i=0, 1,2
                post_sigma=np.kron(post_sigma, identities)
                
            result = np.kron(result, post_sigma)
        else:
            result = result 
            
        H1 += result 
    H1 = -1 * (J+gamma) * H1 


    # H2: (J-\gamma) part 
    for i in range(length_l/2):
        sigma_prod = np.kron(sigma_minus, sigma_plus)
        identities=np.kron(identity, identity)
        pre_sigma = np.kron(identity, identity)
        post_sigma = np.kron(identity, identity)
            
        if i > 0:
            for _ in range(i-2):
                pre_sigma = np.kron(pre_sigma, identities)
            result = np.kron(pre_sigma, sigma_prod)
        
        else: 
            result = sigma_prod


        if i < (length_l/2): # i=0,1,2
            for _ in range(length_l/2-i-1): # i=0, 1,2
                post_sigma=np.kron(post_sigma, identities)
                
            result = np.kron(result, post_sigma)
        else:
            result = result 
            
        H2 += result 
    H2 = -1 * (J-gamma) * H2


    # H3: sigma-x part 
    for i in range(length_l/2-1):  # i = 0,1,2
        sigmax_prod = np.kron(sigma_x, sigma_x)
        identities = np.kron(identity, identity)
        pre_sigma = identity 
        post_sigma = identity
        
        for _ in range(i): # i=0,1,2
            pre_sigma = np.kron(pre_sigma, identities)
            result = np.kron(pre_sigma, sigmax_prod)
        
        for _ in range(length_l/2-i-1): 
            post_sigma = np.kron(post_sigma, identities)
            result = np.kron(result, post_sigma)
        
        H3 += result 
    H3 = -1 * (J/2) H3


    # H4: sigma-y part 
    for i in range(length_l/2-1):  # i = 0,1,2
        sigmay_prod = np.kron(sigma_y, sigma_y)
        identities = np.kron(identity, identity)
        pre_sigma = identity 
        post_sigma = identity
        
        for _ in range(i): # i=0,1,2
            pre_sigma = np.kron(pre_sigma, identities)
            result = np.kron(pre_sigma, sigmax_prod)
        
        for _ in range(length_l/2-i-1): 
            post_sigma = np.kron(post_sigma, identities)
            result = np.kron(result, post_sigma)
        
        H4 += result 
    H4 = -1 * (J/2) H4
    
    H = H1 + H2 + H3 + H4
    return H 


# 4.1 HN model global ancilla qubit register 
n=9
psi0= QuantumRegister(n) 
circ = QuantumCircuit(psi0)
circ.draw()

# prepare initial state (but to flip the qubits to spin-up state)
# Apply X gate to all qubits to prepare the |111...1> state
for i in range(n):
    circ.x(i)

# one ancilla qubit is "globally" attached to 6 physical qubit 
#n= 9

#pq = QuantumRegister(7, name = "physical")
#aq = QuantumRegister(2, name = "ancilla")

#qc_global = QuantumCircuit(aq,pq) # fig. 7(b)


# 4.2 SSH model local ancilla qubit register 

#pq = QuantumRegister(2, name = "physical")
#aq = QuantumRegister(1, name = "ancilla")

# fig. 7(a)
#qc_local = QuantumCircuit(aq,pq) 


# 5.encoding non-unitary U_{\pm} in bigger unitary matrices R_{R_{\pm}}
# 7 qubits to represent a 2**7 by 2**7 matrix 

# U3 gates, eqn. (27)
theta = Parameter('theta')
phi = Parameter('phi')
lam= Parameter('lambda')

u3_matrix = np.array([
    [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
    [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
])


# 7. eqn. (S9) 
phi = Parameter('phi')
H = HN_ham(6, 1, 0.5)

H_daggar = (H. transpose()).conjugate()
identity_64 = identity 
for i in range(5):
    identity_64 = np.kron(identity_64, identity)
 
C_H = -1* np.sqrt(identity_64 - H * H_daggar )
B_H = -1 * C_H
D_H= H
top_H = np.hstack((H, B_H))
bottom_H = np.hstack((C_H, D_H))
unitary_H = np.vstack((top_H, bottom_H))


# 6. tensornetwork for NH model 

# 6.1 U3, CX gates
def single_qubit_layer(circ, gate_round=None):
    """Apply a parametrizable layer of single qubit ``U3`` gates.
    """
    for i in range(circ.N):
        # initialize with random parameters
        params = qu.randn(3, dist='uniform')
        circ.apply_gate(
            'U3', *params, i, 
            gate_round=gate_round, parametrize=True)


def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
    """Apply a layer of constant entangling gates.
    """
    regs = range(0, circ.N - 1)
    if reverse:
        regs = reversed(regs)
    
    for i in regs:
        circ.apply_gate(
            gate2, i, i + 1, gate_round=gate_round)


def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
    """Apply a layer of constant entangling gates.
    """
    regs = range(0, circ.N - 1)
    if reverse:
        regs = reversed(regs)
    
    for i in regs:
        circ.apply_gate(
            gate2, i, i + 1, gate_round=gate_round)


def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
    """Apply a layer of constant entangling gates.
    """
    regs = range(0, circ.N - 1)
    if reverse:
        regs = reversed(regs)
    
    for i in regs:
        circ.apply_gate(
            gate2, i, i + 1, gate_round=gate_round)



def ansatz_circuit(n, depth, gate2='CX', **kwargs):
    """Construct a circuit of single qubit and entangling layers.
    """
    circ = qtn.Circuit(n, **kwargs)
    
    for r in range(depth):
        # single qubit gate layer
        single_qubit_layer(circ, gate_round=r)
        
        # alternate between forward and backward CX layers
        two_qubit_layer(
            circ, gate2=gate2, gate_round=r, reverse=r % 2 == 0)

    # add a final single qubit layer
    single_qubit_layer(circ, gate_round=r + 1)
    
    return circ


# 6.2 initialize a circuit
n = 7
depth = 8 # "8 ansatz layers can ensure fidelity exceeds 90%"
gate2 = 'CX'

circ = ansatz_circuit(n, depth, gate2=gate2)
circ 


# 6.3 the current unitary circuit operator as a tensor network 
V = circ.uni
V.draw(color=['U3', gate2], show_inds=True) 
V.draw(color=[f'ROUND_{i}' for i in range(depth + 1)], show_inds=True)
V.draw(color=[f'I{i}' for i in range(n)], show_inds=True)


# 6.4 the HN model hamiltonian (target operator) 
n=7 # 7 qubits for a 6 lattice site implementation  
H = Hamiltonian_spin_HN(6, 1, 0.5) # site=6, J=1, gamma=0.5

# 6.5 Trotterized gates operator with time evolution
tot_t = 1 # total duration of state evolution 
delta_t = 0.1 
n_steps= tot_t / delta_t 

U_dense = qu.expm(-1j*H*delta_t)**n_steps

# 'tensorized' version of the unitary propagator 
U = qtn.Tensor(
    data=U_dense.reshape([2] * (2 * n)),   
    inds=[f'k{i}' for i in range(n)] + [f'b{i}' for i in range(n)],
    tags={'U_TARGET'}
)
U.draw(color=['U3', gate2, 'U_TARGET']) 
(V.H & U).draw(color=['U3', gate2, 'U_TARGET'])


# 6.6 Tensor Network optimization, jax based 
def loss(V, U):
    return 1 - abs((V.H & U).contract(all, optimize='auto-hq')) / 2**n  

loss(V, U)

tnopt = qtn.TNOptimizer(
    V,
    loss,
    loss_constants = {'U':U}, # supply U to the loss fuction as a constant TN
    tags=['U3'],              # only optimize U3 tensors 
    autodiff_backend='jax',   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',      # the optimization algorithm  
)

# allow 10 hops with 500 steps in each 'basin'
V_opt = tnopt.optimize_basinhopping(n=500, nhop=10)

# first we turn the tensor network version of V into a dense matrix 
V_opt_dense = V_opt.to_dense([f'k{i}' for i in range(n)], [f'b{i}' for i in range(n)])

# create a random initial state, and evolve it with the
psi0 = qu.rand_ket(2**n)

# this is the exact state we want
psif_exact =  U_dense @ psi0 

# this is the state our circuit will produce if fed 'psi0'
psif_approx = V_opt_dense @ psi0

# (in)fidelity
f"Fidelity:{100 * qu.fidelity(psif_approx, psif_exact): .2f} %"

circ.update_params_from(V_opt)
circ.gates


# 7. measurement (aer simualtor, shots)
simulator = AerSimulator()
circ = transpile(circ, simulator)

# Run and get counts and memory
# memory=true to get store 10,000 runs in a list  
result = simulator.run(circ, shots=10000, memory=True).result() 
counts = result.get_counts(circ) # an averaged result from the 10,000 runs
memory = result.get_memory(circ)
print(memory)
print(len(memory))


# 8. dynamical density evolution ni(T) encoding


# 9. dynamical density evolution plot (heatmap and colorbar)
dataset = np.random.rand(6,10)
plt.imshow(dataset, aspect='auto', cmap='inferno') 
plt.colorbar()
plt.show()

