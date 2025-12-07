# Parameter_Reconstruction_Bloch_Torrey_Master_Thesis
Code implementations of algorithms developed in the master thesis "Reconstruction methodology for MRI based on the Bloch-Torrey equation".

The File Bloch_Torrey_Solver.py provides a numerical solver for the Bloch-Torrey equation in the rotating frame, where the solution is saved in a File solution.xdmf.
An example is provided in the file too.
The functions $T_1$, $T_2$, $M^{eq}$ and $B$ in the Bloch-Torrey equation are required to be symbolic expressions.
It is also possible to run the code 

The code is fully parallelized using MPI and can be executed on multiple processors.  
To run the simulation in parallel, use:

mpirun -np 2 python3 Bloch_Torrey_Solver.py

# Third-Party Libraries:
The following open source libraries are used in this project:

- NumPy (numpy)  
  License:  BSD license  
  Link: https://numpy.org

- MPI for Python (mpi4y)
  License: BSD-3-Clause license  
  Link: https://mpi4py.readthedocs.io/

- DOLFINx (dolfinx)    
  License: LGPL-3.0 license  
  Link: https://docs.fenicsproject.org

- Unified Form Language (ufl)  
  License: LGPL-3.0 license  
  Link: https://www.fenicsproject.org

- Basix (basix)  
  License: MIT License  
  Link: https://docs.fenicsproject.org/basix/main/
