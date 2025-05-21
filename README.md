# A benchmarking procedure for quantum networks

## Prerequisites

- Python 3.9x - (make sure the ctypes exists: python -c "import _ctypes; print('ctypes OK')"  )
Steps: (let us use brew)
1. brew install python@3.9
2. python3.9 -m venv .venv
3. source .venv/bin/activate
4. use any of these versions to avoid incompatibilities numpy<2.0,>=1.24: pip install "numpy<2.0,>=1.24"


5. **Clone the repo**  
pip install --extra-index-url https://YourUser:YourPassword@pypi.netsquid.org netsquid

6. **Structure**
Drivers:

Name: driver_2nodes_1q.py --> this is apparently the one that is expected in teh code
Description: we start with one qubit wired to alice's memory.
Protocol associated: protocol_final_1q.py

Name: driver_2nodes_entangled.py
Description: we start with one PB pair. One of them is saved in Alice's memory slot 0, the other half is saved in alice's memory slot 1.
Protocol associated: protocol_entangled.py


