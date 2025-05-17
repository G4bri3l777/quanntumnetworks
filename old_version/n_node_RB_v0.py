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

    def _operate_qubit(self, node, event=None):
        """
        Applies a random Clifford  to a state in the local processor, unless the maximal number of iterations is reached, 
        in which case the qubit is read out and a new sequence is started.

        """
        protoID = self.uid
        node[protoID]["busy_operating"] = True
        #Check if the maximal number of bounces has been reached
        if self._counter <= 2* (self._n_nodes-1) * self._max_bounces_current_round:
            #Choose a random Clifford gate
            instr = rd.choice(self.cliffords)
            #Apply the Clifford gate in the processor
            node.qmemory.execute_instruction(
                instr, [0], error_on_fail=True, physical=True
            )
            #Wait for processor to report back
            self.wait_for_event(
                f"qubit operated_at_A_{node.name}",
                event_type=node.qmemory.evtype_program_done,
            )

            node[protoID]["busy_operating"] = False
            #Store records of all applied gates (globally)
            self._gates.append(instr)
        else:
            #retrieve qubit from memory
            [qubit] = node.qmemory.pop(0)
            #Create two reference states to compute fidelity (two because of the extra pauli average step)
            [ref_qubit1, ref_qubit2] = create_qubits(2)
            operate(ref_qubit2, ops.X)
            for gate_instr in self._gates:
                gate = gate_instr._operator
                operate(ref_qubit1, gate)
                operate(ref_qubit2, gate)

            # Do the Pauli average 
            #Note the minus sign!
            self._end_fidelity.append(
                exp_value(
                    qubit,
                    ops.Operator(
                        "ref", (ref_qubit1.qstate.dm - ref_qubit2.qstate.dm) / 2
                )                    ),

            )

            if self._current_sample < self._n_samples:
                #reset simulator for next random sequence
                print(f"sample {self._current_sample} of {self._n_samples} at {self._max_bounces_current_round}")
                self._current_sample = self._current_sample + 1
                self._counter = 1
                node.qmemory.reset()
                self._gates = []
                self._schedule_now(self.evtype_trigger)
            else:
                #Reset simulator for the next sequence length
                print(
                    "all samples gathered, mean fidelity at {} bounces: ".format(
                        self._max_bounces_current_round
                    ),
                    np.mean(self._end_fidelity),
                )

                self._mean_fidelity_bounce[self._max_bounces_current_round] = np.mean(
                    self._end_fidelity
                )
                

                 # Add random shot noise simulating 4000 measurement average
                self._array_fidelity_bounce[
                    self._max_bounces_current_round
                ] = self._end_fidelity + np.random.normal(
                    0,
                    np.sqrt(
                        (
                            (1+ np.mean(self._end_fidelity))
                            * (1 - np.mean(self._end_fidelity))
                        )
                    )
                    / np.sqrt(4000),
                    1,
                )

                if self._max_bounces_current_round < self._max_bounces:
                    #Reset simulator for the next sequence length
                    self._end_fidelity = []
                    self._max_bounces_current_round = (
                        self._max_bounces_current_round + 1
                    )
                    self._counter = 1
                    self._current_sample = 1
                    node.qmemory.reset()                
                    self._gates = []
                    self._schedule_now(self.evtype_trigger)

                else:
                    #End simulation
                    self.stop()



    def _receive_qubit(self, node, event=None):
        """
        Receive a qubit and place it in the local processor
        """

        protoID = self.uid
        # Store qubit and signal to processor that it is stored
        port = node.get_conn_port(
            node[protoID]["remote_ID"], label=node[protoID]["channel_label"]
        )
        msg = port.rx_input()

        qubit = msg.items[0]
        try:
            position = node.qmemory.unused_positions[0]
        except IndexError:
            raise RuntimeError("No empty memory positions on {}.".format(node.qmemory))

        node.qmemory.put(
            qubit, positions=[position], ready_evtype=node[protoID]["evtype_stored"]
        )

        self._counter = self._counter + 1
        # Logic for the port switching at nodes: 
        # After receiving a qubit it changes the remote_ID for sending and vice versa
        if (node.name != "0" and node.name != f"{self._n_nodes-1}"):
                label = node[protoID]["channel_label"]
                #Swap listening channel
                node[protoID]["channel_label"] = label[:-2]+label[-2:][::-1]
                #Swap listening node
                if node[protoID]["direction"] == 0:
                    node[protoID]["direction"] = 1
                    node[protoID]["remote_ID"] = self.nodes[int(node.name)+1].ID
                    node[protoID]["channel_label"] = label[:-2] + f"{int(label[-2])+1}"+ f"{int(label[-1])+1}"
                else:
                    node[protoID]["direction"] = 0
                    node[protoID]["remote_ID"] = self.nodes[int(node.name)-1].ID    
                    node[protoID]["channel_label"] = label[:-2] + f"{int(label[-2])-1}"+ f"{int(label[-1])-1}"
               

    def _output_handler(self, message, node, protoID):
        #Stores the output of the qsource in the local processor.
        pos = node.qmemory.unused_positions[0]

        node.qmemory.put(
            message.items, positions=[pos], ready_evtype=node[protoID]["evtype_stored"]
        )

    def _handle_trigger(self, event, node):
        #Generates a starting state from the qsource when the protocol starts ("trigger" event)
        protoID = self.uid
        qsource = node.subcomponents[node[protoID]["qsource_name"]]
        qsource.trigger()
        qsource.ports["qout0"].bind_output_handler(
            lambda message, node=node, protoID=protoID: self._output_handler(
                message, node, protoID
            )
        )

    def start(self):
        """
        Begin protocol for nodes.

        Resets any internal state.

        """
        super().start()

        # Setup node local contexts
        protoID = self.uid
        #Set up first node
        self.nodes[0][protoID] = {
            "remote_ID": self.nodes[1].ID,
            "busy_operating": False,
            "qsource_name": self._get_qsource_name(self.nodes[0]),
            "evtype_trigger": self.evtype_trigger,
            "evtype_stored": self.evtype_list_qubit_stored[0],
            "evtype_operated": self.nodes[0].qmemory.evtype_program_done,
            "evtype_inverted": self.evtype_qubit_inverted,
            "channel_label": "quantum01",

        }
        #set up intermediate nodes
        for i in range(1,self._n_nodes-1):
            self.nodes[i][protoID] = {
                "remote_ID": self.nodes[i-1].ID,
                "channel_label": f"quantum{i-1}{i}",
                "evtype_stored": self.evtype_list_qubit_stored[i],
                "evtype_operated": self.nodes[i].qmemory.evtype_program_done,
                "direction":0 #Indicated wether node i is expecting a qubit from node i-1 (0) or node i+1 (1)

            }
        #set up last node
        self.nodes[self._n_nodes-1][protoID] = {
            "remote_ID": self.nodes[self._n_nodes-2].ID,
                "channel_label": f"quantum{self._n_nodes-2}{self._n_nodes-1}",
                "evtype_stored": self.evtype_list_qubit_stored[self._n_nodes-1],
                "evtype_operated": self.nodes[self._n_nodes-1].qmemory.evtype_program_done,

        }

        #Set up callback functions
        self.add_callback("trigger", self._handle_trigger, node=self.nodes[0])

        for i in range(0,self._n_nodes):
            self.add_callback(f"qubit operated_at_A_{i}", self._send_qubit, node=self.nodes[i])
            self.add_callback(f"qubit sent_to_A_{i}", self._receive_qubit, node=self.nodes[i])
            self.add_callback(f"qubit stored_at_A_{i}", self._operate_qubit, node=self.nodes[i])


        # Tell node A_1 to listen for trigger
        self.wait_for_event("trigger",once=False, event_type=self.evtype_trigger)

        # Tell all nodes to listen for storage events
        for i in range(0,self._n_nodes):
            self.wait_for_event(
                f"qubit stored_at_A_{i}", once=False, event_type=self.evtype_list_qubit_stored[i]
            )


        # Tell node A_i to listen for a received qubit from node A_{i-1 }
        for i in range(1,self._n_nodes):
            port= self.nodes[i].get_conn_port(self.nodes[i-1].ID, label=f"quantum{i-1}{i}")
            self.wait_for_event(
                f"qubit sent_to_A_{i}",
                entity=port,
                event_type=port.evtype_input,
            )  # Sending messages automatically generates events of the type evtype_input


    def stop(self):
        """Stop protocol.

        Does not clear internal state.

        """
        super().stop()
        self.dismiss_callbacks()

    @staticmethod
    def _get_qsource_name(node):
        """
        Get the name of a QSource if present.
        """
        for name in node.subcomponents:
            if isinstance(node.subcomponents[name], QSource):
                return name
        else:
            return None

    @property
    def is_connected(self):
        """
        bool: True if protocol fully and correctly connected, otherwise False.
        """
        if not self.check_assigned([self.nodes[i] for i in range(self._n_nodes)], Node):
            return False
        if (any([not self.nodes[i].get_conn_port(self.nodes[i+1].ID, label=f"quantum{i}{i+1}")
            for i in range(self._n_nodes-1)]) or
            any([self.nodes[i].qmemory.num_positions == 0 for i in range(self._n_nodes-1)])
            or self._get_qsource_name(self.nodes[0]) is None
        ):
            return False
        return True

    def get_fidelity(self):
        """Return the fidelity of a single m bounce experiment."""
        return [self._mean_fidelity_bounce, self._array_fidelity_bounce]

    def _generate_cliffords(self):
    # Sets up an array of Clifford instructions for the Clifford gates
        cliff_ops = [
            ops.I,
            ops.X,
            ops.Y,
            ops.Z,
            ops.H,
            ops.S,
            ops.X * ops.H,
            ops.Y * ops.H,
            ops.Z * ops.H,
            ops.X * ops.S,
            ops.Y * ops.S,
            ops.Z * ops.S,
            ops.X * ops.H * ops.S,
            ops.Y * ops.H * ops.S,
            ops.Z * ops.H * ops.S,
            ops.H * ops.S * ops.H,
            ops.X * ops.H * ops.S * ops.H,
            ops.Y * ops.H * ops.S * ops.H,
            ops.Z * ops.H * ops.S * ops.H,
            ops.S * ops.H * ops.S,
            ops.X * ops.S * ops.H * ops.S,
            ops.Y * ops.S * ops.H * ops.S,
            ops.Z * ops.S * ops.H * ops.S,
        ]

        cliff_instrs = [
            IGate("Clifford_{}".format(str(i)), op) for i, op in enumerate(cliff_ops)
        ]
        return cliff_instrs
