# teleportationnoisemodel.py
from netsquid.components.models.qerrormodels import QuantumErrorModel
from netsquid.qubits.qubitapi import reduced_dm, assign_qstate
import numpy as np


class TeleportationNoiseModel(QuantumErrorModel):
    """Implements the map  ρ ↦ α ρ + (1−α)|0⟩⟨0|."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.add_property("alpha", alpha, mutable=False)

    @property
    def alpha(self) -> float:
        return self.properties["alpha"]

    def error_operation(self, qubits, delta_time=0, **kwargs):
        if self.alpha == 1.0:# ideal channel
            return

        a  = self.alpha
        # creation of |0⟩⟨0|
        k0 = np.array([[1, 0],      
                       [0, 0]], dtype=complex)

        for q in qubits:
            if q is None:                     
                continue
            rho   = reduced_dm(q)             # 2 × 2 
            noisy = a * rho + (1 - a) * k0    # convex mix
            assign_qstate([q], noisy)         # update the cahannel
