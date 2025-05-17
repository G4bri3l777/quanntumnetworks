import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t


#Plotting script for figure b


plt.close()
def exp(m,A,f):
	return A* f**m




fid  = {}
popt = {}
pcov = {}
g = plt.figure(1)
ax = plt.subplot()

n_nodes = [2,3,4,5,6]

symbols = {2:"|",3:"^", 4:"s", 5:"p", 6:"H"}

for i in n_nodes:
	#Read out data from pickles 
	with open(f"{i}_RB_decay.pickle", 'rb') as f:
		fid[i] = pk.load(f)

	endp = fid[i]["endpoints"]
	datp= [ fid[i]["decay data"][0][x] for x in range(*endp)]
	#Fit exponential
	popt[i],pcov[i] = curve_fit(exp,np.array(range(*endp)), datp )

	ax.axis([0.5,endp[1]-0.5, 0,0.6])
	#Plot sequence average 
	ax.scatter(np.array(range(*endp)), datp, marker = symbols[i], color = "b")
	#Plot exponential fit
	ax.plot(np.linspace(*endp,100), 
		[exp(m, *popt[i]) for m in np.linspace(*endp,100)],alpha = 0.5,color = "b")
	for k in range(*endp):
		dats =  fid[i]["decay data"][1][k]
		loc = [k for x in range(len(dats))]

#Define inset figure
sub = plt.axes([0.62,0.59,0.27,0.27])
sub.axis([1,7,0.49,1])

#Fit fidelity against number of nodes
fidopt,fidcov = curve_fit(exp, n_nodes, [popt[i][1] for  i in n_nodes])

#plot network link fidelity against number of nodes
sub.plot(np.linspace(n_nodes[0]-1,n_nodes[-1]+1,100), 
		[exp(m, *fidopt) for m in np.linspace(n_nodes[0]-1,n_nodes[-1]+1,100)],alpha = 0.5)
sub.set_xlabel("Number of nodes",fontsize=13)
sub.set_ylabel("Network fidelity",fontsize=13)

#Compute studentized error bars on network link fidelities
h = t.ppf((1 +0.95) / 2., 18-2)
sub.errorbar(n_nodes, [popt[i][1] for i in n_nodes],
	yerr=[h* np.sqrt(pcov[i][1,1]) for i in n_nodes],
	capsize=3,marker="o",ms=3,ls="",color="b")
print([popt[i][1] for i in n_nodes])
print([h* np.sqrt(pcov[i][1,1]) for i in n_nodes])

ax.set_xlabel("Number of bounces",fontsize=18)
ax.set_ylabel("Sequence mean $b_m$",fontsize=18)

g.savefig("n_node_netrb",transparent=True)

g.show()

plt.show()

