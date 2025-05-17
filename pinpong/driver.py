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
# from teleportationnoisemodel import TeleportationNoiseModel 
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.component  import Port
# from debug._generate_cliffords import generate_cliffords
# from new_protocol import TwoNodeRBProtocol
from netsquid.qubits import set_qstate_formalism, QFormalism
from netsquid.qubits.qubitapi import create_qubits
import random
from netsquid.components.component import Message
from netsquid.qubits import qubitapi as qapi

from protocol import PingProtocol, PongProtocol

ns.sim_reset()
ns.set_random_state(seed=42)
node_ping = Node("Ping", port_names=["port_to_channel"])
node_pong = Node("Pong", port_names=["port_to_channel"])
connection = DirectConnection("Connection",
                              QuantumChannel("Channel_LR", delay=10),
                              QuantumChannel("Channel_RL", delay=10))
node_ping.ports["port_to_channel"].connect(connection.ports["A"])
node_pong.ports["port_to_channel"].connect(connection.ports["B"])
ping_protocol = PingProtocol(node_ping)
pong_protocol = PongProtocol(node_pong)

ping_protocol.start()
pong_protocol.start()
stats = ns.sim_run(91)