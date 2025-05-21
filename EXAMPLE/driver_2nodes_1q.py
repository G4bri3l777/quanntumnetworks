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
from updated_algorithm.updated_teleportation import TeleportationNoiseModel 
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.component  import Port
from updated_algorithm._generate_cliffords import generate_cliffords
from protocol_final_1q import TwoNodeHalfBounce
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.qubits.qubitapi import create_qubits
import random
from netsquid.components.component import Message
from netsquid.qubits import qubitapi as qapi
import numpy as np
from netsquid.qubits.ketstates import s0, b00
from netsquid.components import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction

#clear out any previous simulation

ns.sim_reset()

#test density matrices
ns.set_qstate_formalism(QFormalism.DM)
print("Simulator reset; using density-matrix formalism.")


# -------------------------------
# Define Nodes and Memories

T1, T2 =  10**9 , 12 * 10**6  # Long decoherence times

gate_time = 39*10**(3) # ns

depolar_rate_per_ns = 1 - np.exp(-gate_time/T2) #gate independent noise assumption    
gate_noise = DepolarNoiseModel(depolar_rate=depolar_rate_per_ns) #added to see how noise affect?

cliffords = generate_cliffords()
clifford_instrs = [
    PhysicalInstruction(gate, duration=5, q_noise_model=gate_noise)
    for gate in cliffords
]

alice_qmem = QuantumProcessor("alice_proc", num_positions=2, phys_instructions=clifford_instrs,
    mem_noise_models=[T1T2NoiseModel(T1=T1, T2=T2)] * 2, fallback_to_nonphysical=True)

bob_qmem = QuantumProcessor("bob_proc", num_positions=1, phys_instructions=clifford_instrs,
    mem_noise_models=[T1T2NoiseModel(T1=T1, T2=T2)], fallback_to_nonphysical=True)

alice = Node("Alice", port_names=["qin0", "chan_A2B"])
bob = Node("Bob", port_names=["qin0", "chan_B2A"])
alice.qmemory = alice_qmem
bob.qmemory = bob_qmem
# ---------------------------------------------------
# Single-qubit source at Alice (|0⟩ each trigger)
qsrc = QSource(
    "QSource",
    state_sampler=StateSampler([s0], [1.0]),    
    num_ports=1,
    status=SourceStatus.EXTERNAL
)
alice.add_subcomponent(qsrc, name="QSource")
# feed that qubit into slot-0 of Alice
qsrc.ports["qout0"].connect(alice.qmemory.ports["qin0"])


# -------------------------------
# Setup Bi-directional Quantum Channel
# -------------------------------
alpha = 0.95  # For teleportation noise model
models = {
    "delay_model": FibreDelayModel(speed=2e5), #adjust speed
    "quantum_noise_model": TeleportationNoiseModel(alpha=alpha)
}

#Alice to Bob

channel_A2B = QuantumChannel("QChan_A2B", length=40, models=models)
alice.ports["chan_A2B"].connect(channel_A2B.ports["send"])
channel_A2B.ports["recv"].connect(bob.ports["qin0"])


# B→A (return)
channel_B2A = QuantumChannel("QChan_B2A", length=40, models=models)
bob.ports["chan_B2A"].connect(channel_B2A.ports["send"])
# --- add a return port on the Alice node ----------------------------
alice.add_ports(['qin_return'])                       # node-level port
alice.ports['qin_return'].forward_input(
    alice.qmemory.ports['qin0'])                     # auto-store in slot-0

# --- connect B→A channel to that port -------------------------------
channel_B2A.ports["recv"].connect(alice.ports["qin_return"])

# -----------------------------------------------------------------------------
# Instantiate & run the RB protocol
# -----------------------------------------------------------------------------
protocol = TwoNodeHalfBounce(
    alice,              # first positional arg
    bob,                # second positional arg
    qsrc,               # third positional arg
    cliffords,          # fourth positional arg
    max_bounces=20,      # keyword
    min_bounces=1,      # keyword
    n_samples=40,        # keyword
    gate_time=gate_time,# keyword
    gate_noise=gate_noise,  # keyword (match init signature)
    n_shots=4000        # keyword
)

protocol.start()
ns.sim_run()


# # now pull out the fidelity data
# mean_dict, raw_dict = protocol.get_fidelity()

# # and save it for plotting
# import pickle as pk
# with open("two_node_data.pickle", "wb") as f:
#     pk.dump({
#       "decay data": [mean_dict, raw_dict],
#       "endpoints" : [protocol.m_min, protocol.m_max]
#     }, f)