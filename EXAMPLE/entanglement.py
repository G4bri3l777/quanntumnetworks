from netsquid.components.qchannel import QuantumChannel
from netsquid.qubits import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models import FixedDelayModel, DepolarNoiseModel
import netsquid.qubits.ketstates as ks
from netsquid.nodes.connections import Connection
from netsquid.components import ClassicalChannel
from netsquid.components.models import FibreDelayModel
from netsquid.nodes.node import Node
from netsquid.components.qmemory import QuantumMemory



class ClassicalConnection(Connection):
    def __init__(self, length):
        super().__init__(name="ClassicalConnection")
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length,
            models={"delay_model": FibreDelayModel()}))
        self.ports['A'].forward_input( 
            self.subcomponents["Channel_A2B"].ports['send']) #When the outside world (e.g. Alice's node) sends a message into port "A", it gets forwarded into the "Channel_A2B"’s send port.
        self.subcomponents["Channel_A2B"].ports['recv'].forward_output(
            self.ports['B']) #It forwards whatever is received at the internal channel’s "recv" port to the external output port "B".
        
class EntanglingConnection(Connection):
    def __init__(self, length, source_frequency):
        super().__init__(name="EntanglingConnection")
        timing_model = FixedDelayModel(delay=(1e9 / source_frequency)) #timining the modek
        # Create a quantum source that generates entangled pairs of qubits
        qsource = QSource("qsource", StateSampler([ks.b00], [1.0]), num_ports=2,
                          timing_model=timing_model,
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource) #adding the source to the connection
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")]) #automatically forward the channel’s recv port output to the connection’s external port A
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")]) #automatically forward the channel’s recv port output to the connection’s external port Bob
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"]) #from source to send port in quantum channel
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])  



def example_network_setup(node_distance=4e-3, depolar_rate=1e7):
    # Setup nodes Alice and Bob with quantum memories:
    noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    alice = Node(
        "Alice", port_names=['qin_charlie', 'cout_bob'],
        qmemory=QuantumMemory("AliceMemory", num_positions=2,
                              memory_noise_models=[noise_model] * 2))
    alice.ports['qin_charlie'].forward_input(alice.qmemory.ports['qin1'])
    bob = Node(
        "Bob", port_names=['qin_charlie', 'cin_alice'],
        qmemory=QuantumMemory("BobMemory", num_positions=1,
                              memory_noise_models=[noise_model]))
    bob.ports['qin_charlie'].forward_input(bob.qmemory.ports['qin0'])
    # Setup classical connection between nodes:
    c_conn = ClassicalConnection(length=node_distance)
    alice.ports['cout_bob'].connect(c_conn.ports['A'])
    bob.ports['cin_alice'].connect(c_conn.ports['B'])
    # Setup entangling connection between nodes:
    q_conn = EntanglingConnection(length=node_distance, source_frequency=2e7)
    alice.ports['qin_charlie'].connect(q_conn.ports['A'])
    bob.ports['qin_charlie'].connect(q_conn.ports['B'])
    return alice, bob, q_conn, c_conn