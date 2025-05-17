from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumChannel
from netsquid.nodes import Node, DirectConnection
from netsquid.qubits import qubitapi as qapi
import netsquid as ns

class PingProtocol(NodeProtocol):
    def run(self):
        print(f"Starting ping at t={ns.sim_time()}")
        port = self.node.ports["port_to_channel"]
        qubit, = qapi.create_qubits(1)
        port.tx_output(qubit)  # Send qubit to Pong
        while True:
            # Wait for qubit to be received back
            yield self.await_port_input(port)
            qubit = port.rx_input().items[0]
            m, prob = qapi.measure(qubit, ns.Z)
            labels_z =  ("|0>", "|1>")
            print(f"{ns.sim_time()}: Pong event! {self.node.name} measured "
                  f"{labels_z[m]} with probability {prob:.2f}")
            port.tx_output(qubit)  # Send qubit to B


class PongProtocol(NodeProtocol):
    def run(self):
        print("Starting pong at t={}".format(ns.sim_time()))
        port = self.node.ports["port_to_channel"]
        while True:
            yield self.await_port_input(port)
            qubit = port.rx_input().items[0]
            m, prob = qapi.measure(qubit, ns.X)
            labels_x = ("|+>", "|->")
            print(f"{ns.sim_time()}: Ping event! {self.node.name} measured "
                  f"{labels_x[m]} with probability {prob:.2f}")
            port.tx_output(qubit)  # send qubit to Ping