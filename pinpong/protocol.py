from netsquid.qubits.qubitapi import *
from netsquid.qubits import operators as ops
from pydynaa import EventType
from netsquid.protocols import Protocol
from netsquid.components.component import Message
from netsquid.components.qprogram import QuantumProgram
from netsquid.nodes.node import Node
from netsquid.components.qsource import QSource
from netsquid.qubits.state_sampler import StateSampler
import netsquid.qubits.ketstates as ks
from netsquid.qubits import operators as ops
from netsquid.components.instructions import *
import random as rd
from netsquid.components import DepolarNoiseModel
import numpy as np


class MultiNodeRB(Protocol):
    """ 
    Description: Class that simulates network RB on an n-node network. Subclasses the netsquid Protocol class.

    Parameters:
        n_nodes: number of nodes in the network, must match the number of nodes physically present in the network
        min_bounces: The smallest sequence length in the netRB protocol
        max_bounces:  The largest sequence length in the netRB protocol
            The protocol will loop over every length between min_bounces and max_bounces
        n_samples: The number of random sequences sampled at each sequence lenght 

    Notes:
        Adapted from an earlier version of the Netsquid-QRepeater snippet (see netsquid.org/snippets/)
    """
    def __init__(self,n_nodes, min_bounces, max_bounces, n_samples):
        super().__init__()
        self._max_bounces_current_round = min_bounces
        self._max_bounces = max_bounces
        self._n_nodes = n_nodes
        self._counter = 1
        self.nodes= []
        self._n_samples = n_samples
        self._current_sample = 1
        self._end_fidelity = []
        self._mean_fidelity_bounce = {}
        self._array_fidelity_bounce = {}
        self._gates = []
        self._initial = 0


        #Set up possible events
        self.evtype_trigger = EventType("TRIGGER", "Start the protocol at node A_0")
        self.evtype_list_qubit_stored = [ EventType(f"STORED_{i}", f"Qubit stored_at_A_{i}") for i in range(self._n_nodes)]
        self.evtype_list_qubit_operated = [ EventType(f"OPER_{i}", f"Qubit operated_at_A_{i}") for i in range(self._n_nodes)]
        self.evtype_qubit_inverted = EventType("INVERTED", "Qubit inverted_at_A_0")

        # Set up the Clifford gates
        self.cliffords = self._generate_cliffords()
    def _send_qubit(self, node, event=None):
        """
        Sends a qubit to the next qubit in the chain (identified by the node's remote_ID
        """

        protoID = self.uid
        # Take the qubit from the processor and send it through the channel
        node[protoID]["busy_operating"] = False
        qubit = node.qmemory.pop(0)

        port = node.get_conn_port(
            node[protoID]["remote_ID"], label=node[protoID]["channel_label"]
        )
        port.tx_output(Message(qubit))
        # Tell the node to start listening for new qubits. 
        port_listen = node.get_conn_port(
            node[protoID]["remote_ID"], label=node[protoID]["channel_label"]
        )
        self.wait_for_event(
            f"qubit sent_to_A_{node.name}",
            entity=port_listen,
            event_type=port_listen.evtype_input,
        )

