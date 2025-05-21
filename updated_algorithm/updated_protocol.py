from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.nodes import Node, DirectConnection
from netsquid.qubits import qubitapi as qapi
import netsquid as ns
from netsquid.components.qmemory import QuantumMemory
import random
from _generate_cliffords import generate_cliffords
import numpy as np
from netsquid.components.models.qerrormodels import T1T2NoiseModel
from netsquid.qubits.operators import Operator, X as PAULI_X
from netsquid.components.instructions import IGate
import netsquid.qubits.operators as ns
import pickle as pk
from netsquid.qubits.qubitapi import measure
from netsquid.protocols import Signals


class PPProtocol(NodeProtocol):
    def __init__(self, alice, bob, m_bounces, n_values):
        super().__init__(alice)
        self.alice = alice
        self.bob = bob
        self.m_bounces = m_bounces
        self.n_values = n_values


                # output results
        self.bits_raw        = {}   # m - list[float]
        self.bits_mean       = {}   # m - float 
        self.bits_mean_noisy = {}   # m - float
        self.t_gate = 0.1

    def run(self):
        cliffords = generate_cliffords()
        port_a = self.alice.ports["port_to_channel"]
        port_b = self.bob.ports["port_to_channel"]

        for m in range(1, self.m_bounces + 1):
            bm_values = []
            for n in range(self.n_values):
                qubit, = qapi.create_qubits(1)
                self.alice.qmemory.put(qubit, positions=[0])
                gates_applied = []
                for i in range(m):
                    # Alice: apply gate + send
                    GA = random.choice(cliffords)
                    self.alice.qmemory.execute_instruction(GA,[0], physical=True )
                    q = self.alice.qmemory.pop([0])[0]
                    port_a.tx_output(q)

                    # Bob: receive, apply gate, send back
                    yield self.await_port_input(port_b)
                    q = port_b.rx_input().items[0]
                    self.bob.qmemory.put(q, [0])
                    GB = random.choice(cliffords)
                    self.bob.qmemory.execute_instruction(GB, [0], physical=True)
                    q = self.bob.qmemory.pop([0])[0]
                    port_b.tx_output(q)

                    # Alice: receive again
                    yield self.await_port_input(port_a)
                    q = port_a.rx_input().items[0]
                    self.alice.qmemory.put(q, [0])
                    gates_applied.append((GA, GB))

                #choose randomly from set {1, P} (P must be orthogonal to the initial state) - X as initial qubit was \0>
                P_label, P_op = random.choice([                                 # Pauli twirl: I or X
                ("I", np.eye(2, dtype=complex)),
                ("X", np.array([[0, 1], [1, 0]], dtype=complex))
                ])

                #inverse matrix
                print(gates_applied)
                U = np.eye(2, dtype=complex)
                for GA, GB in reversed(gates_applied):
                    U = GB._operator._matrix @ GA._operator._matrix @ U 
                #apply the dagger
                U_dag = U.conj().T
                inv_op  = Operator("inv", P_op @ U_dag)
                inv_gate = IGate("inv_gate", inv_op)
                self.alice.qmemory.execute_instruction(inv_gate, [0], physical=True)
                yield self.await_timer(self.t_gate) 

                #measure the state rho_a using POVM
                qubit_to_measure = self.alice.qmemory.pop([0])[0]
                bm, p = measure(qubit_to_measure, observable=ns.Z) #m is either 0. or 1, and p is the probability

                if P_label == "X":
                    print(f"Pauli twirl: {P_label} - I")
                    print("bm before the flipping"  , bm)
                    bm = -bm
                print("bm after the flipping"  , bm)    
                
                # save the results
                print(f'bm and prob {bm}, {p}')
                print(f"bm = {bm} (type: {type(bm)})")
                bm_values.append(bm)
            print(f"cyle, {m}, -----------------------------------")
            print("the list of bm_values are"   , bm_values)
            b_mean = float(np.mean(bm_values))
            self.bits_mean[m] = b_mean
            self.bits_raw[m] = bm_values
        
            # noise calculus
            sigma       = np.sqrt((1 + b_mean)*(1 - b_mean)) / np.sqrt(4000)
            b_noisy     = b_mean + np.random.normal(0, sigma)
            self.bits_mean_noisy[m] = float(b_noisy)


            print(f"\n>> m={m}: ⟨b⟩={b_mean:+.4f}, noisy={self.bits_mean_noisy[m]:+.4f}")
            print("b_mean is", b_mean)
        # --------------------------------------------
        with open("simulation_data.pickle", "wb") as f:
            pk.dump({
                "b_mean"       : self.bits_mean,
                "b_mean_noisy" : self.bits_mean_noisy,
                "b_samples"    : self.bits_raw,
                "params"       : {
                    "max_bounces" : self.m_bounces,
                    "n_samples"   : self.n_values
            
                }
            }, f)
        print("\n[✓] Protocol finished → bounce_data.pickle written")
        self.send_signal(signal_label=Signals.SUCCESS)
