import numpy as np
from teleportationnoisemodel_v1 import TeleportationNoiseModel
import pandas as pd

# Dummy qubit to carry a density matrix
class DummyQubit:
    def __init__(self, dm):
        self.qstate = type("QState", (), {"dm": dm.copy()})

# Helper to test a single case
def test_case(alpha, initial_dm):
    qubit = DummyQubit(initial_dm)
    model = TeleportationNoiseModel(alpha)
    model.apply_noise(qubit)
    return qubit.qstate.dm


# Define initial density matrices
dm_zero = np.array([[1, 0], [0, 0]])  # |0><0|
dm_one  = np.array([[0, 0], [0, 1]])  # |1><1|


# Run test cases
cases = [
    {"alpha": 1.0, "initial": dm_one},
    {"alpha": 0.0, "initial": dm_one},
    {"alpha": 0.3, "initial": dm_one},
    {"alpha": 0.5, "initial": dm_zero},
]

results = []
for case in cases:
    alpha = case["alpha"]
    init = case["initial"]
    out_dm = test_case(alpha, init)
    results.append({
        "alpha": alpha,
        "initial_dm": init,
        "result_dm": np.round(out_dm, 6)
    })


df = pd.DataFrame(results)
print(df)