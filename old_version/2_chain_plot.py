import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

#Plotting script for figure a

def exp(m,A,f):
	return A* f**m


plt.close()

with open('two_node_data.pickle', 'rb') as f:
	fid_AB = pk.load(f)
	endpoints_AB = fid_AB["endpoints"]
	fid_AB_means = fid_AB["decay data"][0]   # dict: {m: mean_b_m}
	fid_AB_data  = fid_AB["decay data"][1]   # dict: {m: array_of_b_m_values}
	import os
	print("File exists?   ", os.path.exists('AB_decay.pickle'))
	print("Loaded keys:  ", fid_AB_means.keys())
	print("Full pickle:  ", fid_AB)
	# endpoints_AB = fid_AB["endpoints"]
	# fid_AB_means = fid_AB["decay data"][0] 
	# fid_AB_data = fid_AB["decay data"][1]

# decay data: [mean_dict, array_dict]


m_vals = sorted(fid_AB_means.keys())

#Compute exponential fit
# popt_AB,pcov_AB = curve_fit(exp,np.array(range(endpoints_AB[0],endpoints_AB[1]+1)), [fid_AB_means[i] for i in range(endpoints_AB[0],endpoints_AB[1]+1)])
x = np.array(m_vals)
y = np.array([fid_AB_means[m] for m in m_vals])

# Fit A * f^m
popt_AB, pcov_AB = curve_fit(exp, x, y)

# 95% Student‑t confidence interval on the fit parameters
dof = len(x) - len(popt_AB)             # degrees of freedom
h   = t.ppf((1 + 0.95) / 2., dof)       # two‑sided 95%

# #Set up figure
# g = plt.figure(2)
# ax2 = plt.subplot()


# #Plot sequence length averages
# ax2.scatter(range(endpoints_AB[0],endpoints_AB[1]+1), [fid_AB_means[i] for i in range(endpoints_AB[0],endpoints_AB[1]+1)],color = "b",label = r"$\alpha =0.97$")

# #Plot individual sequence means
# for k in range(len(fid_AB_data[endpoints_AB[0]])):
# 		ax2.scatter(range(endpoints_AB[0],endpoints_AB[1]+1), [fid_AB_data[i][k] for i in range(endpoints_AB[0],endpoints_AB[1]+1)],color = "b",alpha = 0.2,s=5)

# #Add network link fidelity
# ab_fid  = f"Network link fidelity = {popt_AB[1]:.3f}" f" $\\pm${h*np.sqrt(pcov_AB[1,1]):.3f}"
# ax2.text(5.5,0.45,ab_fid,fontsize = 15)
# #Plot exponential decay
# ax2.plot(range(endpoints_AB[0],endpoints_AB[1]+1), [exp(m, popt_AB[0],popt_AB[1]) for m in range(endpoints_AB[0],endpoints_AB[1]+1)] ,color = "b",alpha = 0.7)


# #set axes labels
# ax2.set_xlabel("Number of A $\\to$ B $\\to$ A bounces",fontsize =18)
# ax2.set_ylabel("Sequence mean $b_m$",fontsize =18)
# ax2.set_xticks(np.arange(2,21,2))


# #Save figure
# g.savefig("two_node_netrb",transparent=True)

# g.show()

# plt.show()

plt.close('all')
fig, ax = plt.subplots(figsize=(8,6))
# (a) Sequence‑length means
ax.scatter(x, y,
           color='b',
           label=r"Mean $b_m$ ($\alpha=0.97$)",
           s=50)

# (b) Individual sequence points
# assume each fid_AB_data[m] is a list/array of length N_samples
N_samples = len(fid_AB_data[m_vals[0]])
for k in range(N_samples):
    ax.scatter(x,
               [fid_AB_data[m][k] for m in m_vals],
               color='b', alpha=0.2, s=5)

# (c) Exponential fit curve
A_fit, f_fit = popt_AB
ax.plot(x,
        [exp(m, A_fit, f_fit) for m in m_vals],
        color='b', alpha=0.7,
        label=f"Fit: $A={A_fit:.2f},\\ f={f_fit:.3f}\\pm{h*np.sqrt(pcov_AB[1,1]):.3f}$")

# Labels & ticks
ax.set_xlabel(r"Number of $A\to B\to A$ bounces, $m$", fontsize=14)
ax.set_ylabel(r"Sequence mean $b_m$",            fontsize=14)
ax.set_xticks(m_vals)
ax.legend()

plt.tight_layout()
plt.show()

# Save transparent
fig.savefig("two_node_netrb.png", dpi=300, transparent=True)