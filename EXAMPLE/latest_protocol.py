
import random
import numpy as np
import netsquid as ns
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.nodes.node import Node
from netsquid.qubits.operators import Operator, X as PAULI_X
from netsquid.qubits.ketstates import b00
from netsquid.qubits.qubitapi import create_qubits
from netsquid.util.datacollector import DataCollector
from pydynaa import EventExpression
from netsquid.protocols import NodeProtocol
from pydynaa import EventType
from netsquid.qubits.qformalism import QFormalism
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.models.qerrormodels import T1T2NoiseModel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.connections import DirectConnection
import netsquid.qubits.ketstates as ks
from updated_algorithm.updated_teleportation import TeleportationNoiseModel 
from netsquid.components.component  import Port
from updated_algorithm._generate_cliffords import generate_cliffords
from netsquid.components.instructions import IGate
from netsquid.protocols import Signals
from netsquid.qubits.operators import Operator
from netsquid.components import DepolarNoiseModel
import pickle as pk
from netsquid.qubits import exp_value


class TwoNodeHalfBounce(NodeProtocol):
    """
    Random-bounce RB with one travelling qubit. (entangled qubit)
    Slot-0 moves; slot-1 stays in Alice as the entangled partner to keep PB pair existing (condition fo entanglement)
    """

    def __init__(self, alice, bob, cliffords,
                max_bounces, min_bounces, n_samples,
                gate_time, gate_noise, n_shots=4000, debug=True):
        super().__init__(alice)          # home node = Alice
        self.a, self.b  = alice, bob
        # self.qsrc       = qsrc
        self.C          = cliffords
        self.m_max      = max_bounces
        self.m_min      = min_bounces
        self.n_samp     = n_samples
        self.t_gate     = gate_time
        self.gate_noise = gate_noise
        self.n_shots    = n_shots        # for shot-noise
        self.debug      = debug

        # output results
        self.bits_raw        = {}   # m - list[float]
        self.bits_mean       = {}   # m - float 
        self.bits_mean_noisy = {}   # m - float

    # Function for debugging
    def _dprint(self, *args):
        if self.debug:
            print(*args)
            
        def get_fidelity(self):
            """
            Returns:
            [ bits_mean_dict, bits_raw_dict ]
            where
            bits_mean_dict[m] is the mean ⟨b_m⟩ over samples at bounce-length m,
            bits_raw_dict[m]  is the list of individual ±1 results at m.
            """

            return [self.bits_mean, self.bits_raw]
    # -----------------------------------------------------------------
    def run(self):

        for m in range(self.m_min, self.m_max + 1):
            self._dprint(f"\n=== Bounce length m = {m} ===")
            raw_bits = []

            for s in range(self.n_samp):
                self._dprint(f"\n-- sample {s+1}/{self.n_samp}")
                # self.a.qmemory.reset()
                self.b.qmemory.reset()

                # trigger a single |0> in Alice slot 0
                # self.qsrc.trigger()
                q = create_qubits(1)[0]
                self.a.qmemory.put(q, positions=[0])
                yield self.await_timer(self.t_gate)
                #debug printing what it comes from memory
                qubit = self.a.qmemory.peek([0])[0]
                rho = qubit.qstate.dm
                self._dprint(f"Sample {s+1}: slot 0 density matrix:\n{rho}")

            

                self._dprint(f"{ns.sim_time():.0f} ns: Bell pair created")

                # start the bouncing loop
                gate_seq = []
                q = None

                for i in range(m):
                    # applying GA on Alice at slot [0] of memory
                    GA = random.choice(self.C)
                    self.a.qmemory.execute_instruction(GA, [0], physical=True)  #apply GA on qubit in slot 0
                    yield self.await_timer(self.t_gate)
                    self._dprint(f"{ns.sim_time():.0f}: GA={GA.name}")

                    #prinitng before piping out
                    qubit = self.a.qmemory.peek([0])[0]
                    rho = qubit.qstate.dm
                    self._dprint(f"Sample {s+1}: slot 0 density matrix:\n{rho} with matrix {GA}")


                    # Sending message from alice to bob
                    q = self.a.qmemory.pop(0)[0]                                #pop from alice memory
                    self.a.ports["chan_A2B"].tx_output(q)                       #place in the chan_A2B 
                    self._dprint(f"{ns.sim_time():.0f}: qubit sent to Bob")     #the connections is managed using driver's wiring
                    yield self.await_port_input(self.b.ports["qin0"])           #wait till qubit is getting into Bob
                    q = self.b.ports["qin0"].rx_input().items[0]                #receive the message 
                    self.b.qmemory.put(q, positions=[0])                        #place into the memory of Bob
                    self._dprint(f"{ns.sim_time():.0f}: Bob received qubit")
                    #prinitng before piping out
                    qubit = self.b.qmemory.peek([0])[0]
                    rho = qubit.qstate.dm
                    self._dprint(f"Sample {i+1}: slot 0 density matrix:\n{rho} arrived and is saved at bob")

                    # Applying GB on Bob at slot [0] of memory
                    GB = random.choice(self.C)                                  #random choice from Gate set
                    self.b.qmemory.execute_instruction(GB, [0], physical=True)  #apply GB on qubit in slot 0
                    yield self.await_timer(self.t_gate)                         #wait till the gate is applied
                    self._dprint(f"{ns.sim_time():.0f}: GB={GB.name}")
                    #priniting after appllyting G_B in Bobs memory
                    qubit = self.b.qmemory.peek([0])[0]
                    rho = qubit.qstate.dm
                    self._dprint(f"Sample {i+1}: slot 0 density matrix:\n{rho} with matrix {GB}")

                    # Sending the message back from bob to alice
                    q = self.b.qmemory.pop(0)[0]                                #pop from bob memory
                    self.b.ports["chan_B2A"].tx_output(q)                       #place in the chan_B2A
                    self._dprint(f"{ns.sim_time():.0f}: qubit sent back")
                    # yield self.await_port_input(self.a.ports["qin0"])     #wait till it appears on the other and it is autosaved because of the forward_input in the driver
                    yield self.await_port_input(self.a.ports["qin_return"]) 
                    # yield self.await_port_input(self.a.qmemory.ports["qin0"])

                    self._dprint(f"{ns.sim_time():.0f}: Alice received qubit")
                    #prinitng before piping out
                    qubit = self.a.qmemory.peek([0])[0]
                    rho = qubit.qstate.dm
                    self._dprint(f"Sample {i+1}: slot 0 density matrix:\n{rho} arrived and is saved at alice")
                    gate_seq.append((GA, GB))      #save the gates

                
                # gate_seq.append((GA, GB))      #save the gates
                #Final qubit after all bounces                             
                qubit = self.a.qmemory.peek([0])[0]
                rho = qubit.qstate.dm
                self._dprint(f"Sample {s+1}: slot 0 density matrix:\n{rho} getting out of the loop")
                # Inverse matrix of the gate sequence
                U = np.eye(2, dtype=complex)
                for GA, GB in reversed(gate_seq):
                    U = GB._operator._matrix @ GA._operator._matrix @ U         #order is important
                    print(f'matrix GB {GB._operator._matrix}')
                    print(f'matrix GA {GA._operator._matrix}')
                    print(f'matrix U {U}')
                        
                U_dag = U.conj().T
                print(f"U_dag {U_dag}")

                P_label, P_op = random.choice([                                 # Pauli twirl: I or X
                    ("I", np.eye(2, dtype=complex)),
                    ("X", np.array([[0, 1], [1, 0]], dtype=complex))
                ])
                print(f"Pauli twirl: {P_op}")
                # 2) Apply the inverse gate sequence

                inv_op  = Operator("inv", P_op @ U_dag)                         #building the operator P U†
                print(f"inv_op {inv_op}")
                inv_gate = IGate("inv_gate", inv_op)                            #trasnforming it into a gate
                self.a.qmemory.execute_instruction(inv_gate, [0], physical=True) #apply into alice's memory at slot 0, (ideally) in the pure state
                yield self.await_timer(self.t_gate)                             #wait till it is done
                
                #prinitng before piping out
                qubit = self.a.qmemory.peek([0])[0]
                rho = qubit.qstate.dm
                self._dprint(f" density matrix: \n{rho} after applying inverse gate")

                # yield self.a.qmemory.execute_instruction("MEASURE", qubit_mapping=[0], output_key="meas")
                # do this:
                results, probs = self.a.qmemory.measure(
                    positions=[0],
                    meas_operators=None,  # None means standard computational POVM { |0><0|, |1><1| }
                    discard=True
                )
                # measure() returns a tuple (results_list, probs_list)
                outcome = results[0]
                # outcome = self.a.qmemory.get_measurement_outcome("meas")


                # building the POVM
                # print("bbuilding the POVM")
                # psi0       = np.array([1,0], dtype=complex)                  # |0>
                # psi_target = P_op @ (U_dag @ psi0)                          # P U^† |0>
                # E_mat      = np.outer(psi_target, psi_target.conj()) 
                # E_op       = Operator("E",        E_mat)
                # I_minus_E  = Operator("I_minus_E", np.eye(2, dtype=complex) - E_mat)
                

                # # 3) perform the POVM on slot 0
                # results, probs = self.a.qmemory.measure(
                #     positions      = [0],            # only the bounced qubit
                #     meas_operators = [E_op, I_minus_E],
                #     discard        = True,
                #     skip_noise     = True
                # )

                

                # print("POVM is done")
                # print(f"results {results}")
                # # map {0,1}→{+1,−1} and apply Pauli-twirl sign
                # b_nm = +1 if results[0] == 0 else -1

                b_nm = +1 if outcome == 0 else -1
                if P_label == "X":
                    b_nm = -b_nm
                raw_bits.append(b_nm)
                # print("multiplying times -1 if PA == P")

    

                # reset Alice slot-0 for next sample (copy partner)
                # self.a.qmemory.reset()     # empties Alice slots 0 and 1
                # self.b.qmemory.reset()     # empties Bob  slot 0

                qubit_a = self.a.qmemory.peek([0])[0]
                if qubit_a is None:
                    print("Alice slot 0 is empty")
                else:
                    print("Alice slot 0 has a density matrix")

                # Bob
                qubit_b = self.b.qmemory.peek([0])[0]
                if qubit_b is None:
                    print("Bob slot 0 is empty")
                else:
                    print("Bob slot 0 has density matrix")
                
                print("sample done, here we go again")

            # mean and assigning the results in the tables
            print(f"actually we are done with the bounces {m} in {self.n_samp} samples")
            b_mean = float(np.mean(raw_bits))
            self.bits_mean[m] = b_mean
            self.bits_raw[m] = raw_bits

            # noise calculus
            sigma       = np.sqrt((1 + b_mean)*(1 - b_mean)) / np.sqrt(self.n_shots)
            b_noisy     = b_mean + np.random.normal(0, sigma)
            self.bits_mean_noisy[m] = float(b_noisy)


            self._dprint(f"\n>> m={m}: ⟨b⟩={b_mean:+.4f}, noisy={self.bits_mean_noisy[m]:+.4f}")

        # --------------------------------------------
        with open("data_4b.pickle", "wb") as f:
            pk.dump({
                "b_mean"       : self.bits_mean,
                "b_mean_noisy" : self.bits_mean_noisy,
                "b_samples"    : self.bits_raw,
                "params"       : {
                    "min_bounces" : self.m_min,
                    "max_bounces" : self.m_max,
                    "n_samples"   : self.n_samp,
                    "n_shots"     : self.n_shots
                }
            }, f)
        print("\n[✓] Protocol finished → bounce_data.pickle written")
        self.send_signal(signal_label=Signals.SUCCESS)
