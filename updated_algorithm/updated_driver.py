from netsquid.qubits.qubitapi import create_qubits
from updated_teleportation import TeleportationNoiseModel
from netsquid.components import QuantumProcessor
from netsquid.components.models.qerrormodels import T1T2NoiseModel
from netsquid.nodes.node import Node
from netsquid.components.qchannel import QuantumChannel
from _generate_cliffords import generate_cliffords
import random
from updated_protocol import PPProtocol
from netsquid.nodes.connections import DirectConnection
import netsquid as ns
from netsquid.components.models.delaymodels import FibreDelayModel

# #importing cliffords
# cliffords = generate_cliffords()
# print(cliffords)
#generating channels
ns.sim_reset()
T1, T2 =  10**9 , 12 * 10**6  # Long decoherence times
alice_processor = QuantumProcessor("alice_processor", num_positions=1, mem_noise_models=[T1T2NoiseModel(T1=10**9, T2=12*10**6)], fallback_to_nonphysical=True)
bob_processor = QuantumProcessor("bob_processor", num_positions=1, mem_noise_models=[T1T2NoiseModel(T1=10**9, T2=12*10**6)], fallback_to_nonphysical=True)

#adjuncting each processor to a node
alice_node = Node("Alice", port_names=["port_to_channel"])
bob_node = Node("Bob", port_names=["port_to_channel"])

#implanting memories into nodes
alice_node.qmemory = alice_processor
bob_node.qmemory = bob_processor

#creating the commmunication channel
alpha = 0.95  # For teleportation noise model
models = {
    "delay_model": FibreDelayModel(speed=2e5), #adjust speed
    "quantum_noise_model": TeleportationNoiseModel(alpha=alpha)

}
connection = DirectConnection("Connection",
                              QuantumChannel("Channel_LR", delay=10),
                              QuantumChannel("Channel_RL", delay=10))
alice_node.ports["port_to_channel"].connect(connection.ports["A"])
bob_node.ports["port_to_channel"].connect(connection.ports["B"])
#creating the protocol
alice = alice_node
bob = bob_node
m_bounces = 20
n_values = 400

protocol = PPProtocol(alice, bob, m_bounces, n_values)
protocol.start()
#running the simulation

stats = ns.sim_run()