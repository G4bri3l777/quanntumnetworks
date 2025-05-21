import netsquid as ns
import pandas as pd
from functions import example_network_setup, example_sim_setup
from functions import run_experiment

ns.set_qstate_formalism(ns.QFormalism.DM)

# Corrected unpacking
network = example_network_setup()
alice = network.get_node("Alice")
bob = network.get_node("Bob")

# Run simulation (15 ns, or whatever your timing model expects)
stats = ns.sim_run(15)

# Peek qubits from memory
qA, = alice.qmemory.peek(positions=[1])
qB, = bob.qmemory.peek(positions=[0])

# Compute entanglement fidelity
fidelity = ns.qubits.fidelity([qA, qB], ns.b00)
print(f"Entangled fidelity (after 15 ns wait) = {fidelity:.3f}")
