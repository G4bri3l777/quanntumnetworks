
import math
import numpy as np
import netsquid as ns
import netsquid.qubits.qubitapi as qapi
import netsquid.util.simtools as simtools
from netsquid.qubits.qubit import Qubit
from netsquid.components.models.qnoisemodels import QuantumNoiseModel
from netsquid.util.simlog import warn_deprecated




class TeleportationNoiseModel(QuantumNoiseModel):
	"""Noise model that emulates the noise produces when performing teleportation with a noisy entangled state of the form

		alpha|PHI><PHI| + (1-alpha)|00><00|.




	Parameters
	----------
	alpha : float
		Bright state population
	


	Raises
	------
	ValueError
	    If alpha is <0 or >1


	"""

	def __init__(self, alpha =1, **kwargs):
	    super().__init__(**kwargs)
	    self._properties.update({'alpha': alpha})
	    if alpha < 0:
	        raise ValueError("alpha {} is negative".format(self.alpha))
	    if alpha>1:
	        raise ValueError("alpha {} is larger than one".format(self.alpha))
	    # If both types of noise are applied, then this should be satisfied


	    self._properties.update({'alpha': alpha})

	@property
	def alpha(self):
	    """ float: alpha, dictating probability of entangled state."""
	    return self._properties['alpha']

	@alpha.setter
	def alpha(self, value):
	    self._properties['alpha'] = value




	def noise_operation(self, qubits,delta_time=0, **kwargs):
	    """Error operation to apply to qubits.

	    Parameters
	    ----------
	    qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
	        Qubits to apply noise to.
	    delta_time : float, optional
	        Time qubits have spent on component [ns].

	    """
	    for qubit in qubits:
	    	self.apply_noise(qubit)



	def apply_noise(self, qubit):
		"""Applies noise to the qubit, depending on alpha

		"""
		# Check whether the memory is empty, if so we do nothing
		if qubit is None:
			return

			# If no alpha is given, no noise is applied
		if self.alpha == 1:
			return
		
		# Apply noise

		qubit.qstate.dm = self.alpha*qubit.qstate.dm + (1-self.alpha)*np.array([[1, 0],[0,0]])
