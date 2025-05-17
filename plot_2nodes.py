import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

# Load saved protocol output
with open('data_2NODE_entg.pickle', 'rb') as f:
    data = pickle.load(f)

b_mean    = data['b_mean']      # dict: m -> mean b_m
b_samples = data['b_samples']   # dict: m -> list of individual b_m values
params    = data['params']
m_vals    = sorted(b_mean.keys())

# Prepare arrays for fitting
x = np.array(m_vals)
y = np.array([b_mean[m] for m in m_vals])

# Exponential model
def exp_model(m, A, f):
    return A * f**m

# Fit A * f^m to (x, y)
popt, pcov = curve_fit(exp_model, x, y)
dof   = len(x) - len(popt)
h_val = t.ppf((1 + 0.95) / 2, dof)

# Plotting
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 6))

# (a) Mean sequence points
ax.scatter(x, y, label=r"Mean $b_m$", s=50)

# (b) Individual samples
N_samples = len(b_samples[m_vals[0]])
for k in range(N_samples):
    ax.scatter(
        x,
        [b_samples[m][k] for m in m_vals],
        alpha=0.2, s=5
    )

# (c) Exponential fit curve
A_fit, f_fit = popt
ci = h_val * np.sqrt(pcov[1, 1])
ax.plot(
    x,
    exp_model(x, A_fit, f_fit),
    alpha=0.7,
    label=fr"Fit: $A={A_fit:.2f},\ f={f_fit:.3f}\pm{ci:.3f}$"
)

# Labels & ticks
ax.set_xlabel(r"Number of $A\to B\to A$ bounces, $m$")
ax.set_ylabel(r"Sequence mean $b_m$")
ax.set_xticks(m_vals)
ax.legend()

plt.tight_layout()
plt.show()

# Save transparent PNG
fig.savefig("two_node_data.png", dpi=300, transparent=True)
