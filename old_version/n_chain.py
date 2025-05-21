import numpy as np

import matplotlib.pyplot as plt


import netsquid as ns
from netsquid import pydynaa
from netsquid.qubits.ketstates import b00
from netsquid.qubits import QFormalism
from netsquid.nodes.connections import DirectConnection
from netsquid.components.qmemory import QuantumMemory
from netsquid.components.qsource import QSource
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol  # , localmethod, nonlocalmethod
from netsquid.nodes.node import Node

from netsquid.components import T1T2NoiseModel,QuantumChannel,ClassicalChannel, FibreLossModel, FibreDelayModel, DepolarNoiseModel,DephaseNoiseModel

from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *

from netsquid.util.simstats import *

from netsquid.qubits import ketstates as ks
from prot_test import MultiNodeRB


from tel import TeleportationNoiseModel

import pickle as pk
ns.sim_reset()


#Set T1 and T2 times for the nodes

T1 = 10**9 
#one second

T2 = 12 * 10**6
#12 milliseconds

gate_time = 39*10**(3)
#39 microseconds

#Set bright state population fraction (decides the fidelity of the "teleportation channel")
alpha = 0.95


ns.set_qstate_formalism(QFormalism.DM)

n_nodes_list = [2,3,4,5,6]

#Set up an experiment with num_nodes nodes in a linear chain

for num_nodes in n_nodes_list:

    n_nodes =num_nodes

    #Define local nodes
    nodes = [Node(f"{i}") for i in range(n_nodes)]


    #define teleportation quantum channels
    qchannelsUP ={i:QuantumChannel(f"TelChannel{i}{i+1}", length=40, models = {"delay_model":FibreDelayModel(),
                               "quantum_noise_model":TeleportationNoiseModel(alpha=alpha)}) for i in range(n_nodes)}
    qchannelsDOWN ={i:QuantumChannel(f"TelChannel{i+1}{i}", length=40, models = {"delay_model":FibreDelayModel(),
                               "quantum_noise_model":TeleportationNoiseModel(alpha=alpha)}) for i in range(n_nodes)}

    #Add memories to the nodes
    qproc0 = QuantumProcessor("QPD-0", num_positions=2,
         mem_noise_models=[T1T2NoiseModel(T1=T1,T2=T2)]*2)

    qprocs = {i:QuantumProcessor(f"QPD-{i}", num_positions=1,
         mem_noise_models=[T1T2NoiseModel(T1=T1,T2=T2)]) for i in range(1,n_nodes)}
    nodes[0].add_subcomponent(qproc0)
    for i in range(1, n_nodes):
      nodes[i].add_subcomponent(qprocs[i])




    #Add a source of states to node A_0

    nodes[0].add_subcomponent(QSource("QSource",status=2))

    #Add a quantum channel connecting A_i and A_i+1
    for i in range(n_nodes-1):
      nodes[i].connect_to(nodes[i+1],DirectConnection(f"quantum_connect{i}{i+1}",qchannelsUP[i],qchannelsDOWN[i]), label=f"quantum{i}{i+1}")







    #Set up a protocol instance
    min_bounces = 1
    max_bounces= 10
    multi_RB = MultiNodeRB(n_nodes=n_nodes,min_bounces = min_bounces,max_bounces = max_bounces,n_samples =40)



    #Assign nodes to the RB protocol
    for i in range(n_nodes):
      multi_RB.nodes.append(nodes[i])



    #Assign clifford gate instructions to the node processors with T1T2 noise
    for instr in multi_RB.cliffords:
        for i in range(n_nodes):
          multi_RB.nodes[i].qmemory.add_instruction(instr, duration =gate_time,topology=[0], q_noise_model=T1T2NoiseModel(T1=T1,T2=T2) )


    #set up the starting conditions for the protocol
    multi_RB.start()
    god = pydynaa.Entity()  # helper entity
    god._schedule_now(multi_RB.evtype_trigger)

    #Start the simulation
    ns.sim_run()

    #Record fidelity results and pickle.
    with open(f"{n_nodes}_RB_decay.pickle", 'wb') as f:
      pk.dump({"decay data":multi_RB.get_fidelity(),"endpoints":[min_bounces,max_bounces]},f)


