import netsquid as ns
from pydynaa import EventType
from netsquid.qubits.qformalism import QFormalism
from netsquid.nodes.node import Node
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.models.qerrormodels import T1T2NoiseModel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.connections import DirectConnection
import netsquid.qubits.ketstates as ks
from teleportationnoisemodel_v1 import TeleportationNoiseModel 
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.component  import Port
from _generate_cliffords import generate_cliffords
from protocol_entangled import TwoNodeHalfBounce
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.qubits.qubitapi import create_qubits
import random
from netsquid.components.component import Message
from netsquid.qubits import qubitapi as qapi
import numpy as np
from netsquid.qubits.ketstates import b00
from netsquid.components import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction

#clear out any previous simulation

ns.sim_reset()

#test density matrices
ns.set_qstate_formalism(QFormalism.DM)
print("Simulator reset; using density-matrix formalism.")


# -------------------------------
# Define Nodes and Memories
# -------------------------------
T1, T2 =  10**9 , 12 * 10**6  # Long decoherence times
# gate_noise = T1T2NoiseModel(T1=1e9, T2=12e6)


gate_time = 39*10**(3) # ns
p_gate = 0.01  #ho much depolarization per gate
depolar_rate_per_ns = p_gate / gate_time
gate_noise = DepolarNoiseModel(depolar_rate=depolar_rate_per_ns)

cliffords = generate_cliffords()
clifford_instrs = [
    PhysicalInstruction(gate, duration=5, q_noise_model=gate_noise)
    for gate in cliffords
]

alice_qmem = QuantumProcessor("alice_proc", num_positions=2, phys_instructions=clifford_instrs,
    mem_noise_models=[T1T2NoiseModel(T1=T1, T2=T2)] * 2, fallback_to_nonphysical=True)

bob_qmem = QuantumProcessor("bob_proc", num_positions=1, phys_instructions=clifford_instrs,
    mem_noise_models=[T1T2NoiseModel(T1=T1, T2=T2)], fallback_to_nonphysical=True)

alice = Node("Alice", port_names=["chan_A2B", "chan_B2A"])
bob = Node("Bob", port_names=["qin0", "chan_B2A"])
alice.qmemory = alice_qmem
bob.qmemory = bob_qmem


# -------------------------------
# Setup Bi-directional Quantum Channel
# -------------------------------
alpha = 0.95  # For teleportation noise model
models = {
    "delay_model": FibreDelayModel(speed=2e5), #adjust speed
    "quantum_noise_model": TeleportationNoiseModel(alpha=alpha)
}


chan_A2B = QuantumChannel("QChan_A2B", length=40, models=models)
chan_B2A = QuantumChannel("QChan_B2A", length=40, models=models)



alice.ports["chan_A2B"].connect(chan_A2B.ports["send"])
chan_A2B.ports["recv"].connect(bob.ports["qin0"])


bob.ports["chan_B2A"].connect(chan_B2A.ports["send"])
# --- add a return port on the Alice node ----------------------------
alice.add_ports(['qin_return'])                       # node-level port
alice.ports['qin_return'].forward_input(
    alice.qmemory.ports['qin0'])                     # auto-store in slot-0

# --- connect B→A channel to that port -------------------------------
chan_B2A.ports["recv"].connect(alice.ports["qin_return"])
# -------------------------------
#  BP Pair lives inside Alice
# -------------------------------
bell_sampler = StateSampler([ks.b00], [1.0])
qsrc = QSource("BellSource",
               state_sampler=bell_sampler,
               num_ports=2,
               status=SourceStatus.EXTERNAL)
alice.add_subcomponent(qsrc, name="BellSource")

#   half-A → Alice.memory[0]
qsrc.ports["qout0"].connect(alice.qmemory.ports["qin0"])
#   half-B → Alice.memory[1]
qsrc.ports["qout1"].connect(alice.qmemory.ports["qin1"])


# -------------------------------
# Set up & run protocol
# -------------------------------

protocol = TwoNodeHalfBounce(               # or TwoNodeRBProtocol
    alice        = alice,
    bob          = bob,
    qsrc         = qsrc,
    cliffords    = cliffords,
    max_bounces  = 20,
    min_bounces  = 1,
    n_samples    = 40,
    gate_time    = gate_time,
    gate_noise=gate_noise
)

protocol.start()
ns.sim_run()

# Optionally load results from saved pickle
# import pickle
# with open("bounce_data.pickle1", "rb") as f:
#     data = pickle.load(f)
#     print(data["b_mean"])