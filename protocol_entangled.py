
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
from teleportationnoisemodel_v1 import TeleportationNoiseModel 
from netsquid.components.component  import Port
from _generate_cliffords import generate_cliffords
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

    def __init__(self, alice, bob, qsrc, cliffords,
                 max_bounces, min_bounces, n_samples,
                 gate_time, gate_noise, n_shots=4000, debug=True):
        super().__init__(alice)          # home node = Alice
        self.a, self.b  = alice, bob
        self.qsrc       = qsrc
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

    # -----------------------------------------------------------------
    def run(self):

        for m in range(self.m_min, self.m_max + 1):
            self._dprint(f"\n=== Bounce length m = {m} ===")
            raw_bits = []

            for s in range(self.n_samp):
                self._dprint(f"\n-- sample {s+1}/{self.n_samp}")

                # trigger the source qubits generated in Alice slots 0 & 1
                self.qsrc.trigger()
                yield self.await_timer(self.t_gate)
                self._dprint(f"{ns.sim_time():.0f} ns: Bell pair created")

                # start the bouncing loop
                gate_seq = []

                for i in range(m):
                    # applying GA on Alice at slot [0] of memory
                    GA = random.choice(self.C)
                    self.a.qmemory.execute_instruction(GA, [0], physical=True)  #apply GA on qubit in slot 0
                    yield self.await_timer(self.t_gate)
                    self._dprint(f"{ns.sim_time():.0f}: GA={GA.name}")

                    # Sending message from alice to bob
                    q = self.a.qmemory.pop(0)[0]                                #pop from alice memory
                    self.a.ports["chan_A2B"].tx_output(q)                       #place in the chan_A2B 
                    self._dprint(f"{ns.sim_time():.0f}: qubit sent to Bob")     #the connections is managed using driver's wiring
                    yield self.await_port_input(self.b.ports["qin0"])           #wait till qubit is getting into Bob
                    q = self.b.ports["qin0"].rx_input().items[0]                #receive the message 
                    self.b.qmemory.put(q, positions=[0])                        #place into the memory of Bob
                    self._dprint(f"{ns.sim_time():.0f}: Bob received qubit")

                    # Applying GB on Bob at slot [0] of memory
                    GB = random.choice(self.C)                                  #random choice from Gate set
                    self.b.qmemory.execute_instruction(GB, [0], physical=True)  #apply GB on qubit in slot 0
                    yield self.await_timer(self.t_gate)                         #wait till the gate is applied
                    self._dprint(f"{ns.sim_time():.0f}: GB={GB.name}")

                    # Sending the message back from bob to alice
                    q = self.b.qmemory.pop(0)[0]                                #pop from bob memory
                    self.b.ports["chan_B2A"].tx_output(q)                       #place in the chan_B2A
                    self._dprint(f"{ns.sim_time():.0f}: qubit sent back")
                    yield self.await_port_input(self.a.ports["qin_return"])     #wait till it appears on the other and it is autosaved because of the forward_input in the driver
                    self._dprint(f"{ns.sim_time():.0f}: Alice received qubit")

                    gate_seq.append((GA, GB))                                   #save the gates

                # Inverse matrix of the gate sequence
                U = np.eye(2, dtype=complex)
                for GA, GB in reversed(gate_seq):
                    U = GB._operator._matrix @ GA._operator._matrix @ U         #order is important
                U_dag = U.conj().T

                P_label, P_op = random.choice([                                 # Pauli twirl: I or X
                    ("I", np.eye(2, dtype=complex)),
                    ("X", np.array([[0, 1], [1, 0]], dtype=complex))
                ])

                inv_op  = Operator("inv", P_op @ U_dag)                         #building the operator P U†
                inv_gate = IGate("inv_gate", inv_op)                            #trasnforming it into a gate
                self.a.qmemory.execute_instruction(inv_gate, [0], physical=True) #apply into alice's memory at slot 0
                yield self.await_timer(self.t_gate)                             #wait till it is done

                # building the POVM
                proj_phi = np.outer(b00, b00.conj())                            # 4×4 rank-1 projector
                M0 = Operator("BellProj", proj_phi)  
                M1 = Operator("NotBell", np.eye(4, dtype=complex) - proj_phi)

                # measure the qubit in slot-0 and the reference in slot-1
                results, probs = self.a.qmemory.measure(
                    positions      = [0, 1],                                    # *both* qubits: bounced qubit + untouched reference
                    meas_operators = [M0, M1],                                  # {Φ+-projector, I−projector}
                    discard        = True,
                    skip_noise     = True
                )

                print("POVM is done")

                b_nm = results[0]   # 0 if you landed back in |Φ+⟩, 1 otherwise
                if P_label == "X":
                    b_nm = -b_nm
                raw_bits.append(b_nm)
                print("multiplying times -1 if PA == P")

     

                # reset Alice slot-0 for next sample (copy partner)
                self.a.qmemory.reset()     # empties Alice slots 0 and 1
                self.b.qmemory.reset()     # empties Bob  slot 0

            # mean and assigning the results in the tables
            b_mean = float(np.mean(raw_bits))
            self.bits_mean[m] = b_mean
            self.bits_raw[m] = raw_bits

            # noise calculus
            sigma       = np.sqrt((1 + b_mean)*(1 - b_mean)) / np.sqrt(self.n_shots)
            b_noisy     = b_mean + np.random.normal(0, sigma)
            self.bits_mean_noisy[m] = float(b_noisy)


            self._dprint(f"\n>> m={m}: ⟨b⟩={b_mean:+.4f}, noisy={self.bits_mean_noisy[m]:+.4f}")

        # --------------------------------------------
        with open("data_final.pickle", "wb") as f:
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
