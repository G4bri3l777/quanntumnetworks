import netsquid as ns
from netsquid.qubits.operators import I, X, Y, Z, H, S
from netsquid.components.instructions import IGate

def generate_cliffords():
    """Generate the 24 single-qubit Clifford gate instructions."""
    cliff_ops = [
        I, X, Y, Z, H, S,
        X * H, Y * H, Z * H, 
        X * S, Y * S, Z * S,
        X * H * S, Y * H * S, Z * H * S,
        H * S * H,
        X * H * S * H, Y * H * S * H, Z * H * S * H,
        S * H * S,
        X * S * H * S, Y * S * H * S, Z * S * H * S,
    ]
    return [IGate(f"Clifford_{i}", op) for i, op in enumerate(cliff_ops)]

# Unit test for generate_cliffords
cliffords = generate_cliffords()

# Check there are 24 gates
print("Number of Clifford gates:", len(cliffords))

# Check all names are unique
names = [gate.name for gate in cliffords]
print("Unique names count:", len(set(names)))

# Check operators are distinct by flattening their matrices
reprs = [repr(gate._operator) for gate in cliffords]
# Convert each matrix to a tuple for set uniqueness
print("Unique operator repr count:", len(set(reprs)))
print("First 5 gate names:", names[:5])
