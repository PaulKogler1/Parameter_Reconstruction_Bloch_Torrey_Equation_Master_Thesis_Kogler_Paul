#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the following functions:
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, default_real_type, fem, io
import ufl
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import form, assemble_scalar
import time

def Bloch_Torrey_solver(domain, time_duration, time_partitions,
                 gamma,
                        
                 M_function_class, M_function_class_regularity,
                 M_0_expression,
                        
                 B_function_class, B_function_class_regularity,
                 B_expression,
                        
                 T_Meq_function_class, T_Meq_function_class_regularity,
                 T1_expression, T2_expression, Meq_expression,
                        
                 D_function_class, D_function_class_regularity,
                 alpha,
                 ):
    
    
    """
    This function solves the Bloch-Torrey equation in the rotating frame without a velocity term numerically on the given "domain".
    The domain should be a unit cube with an arbitrary fine grid. 
    The unit cube can be replaced with any domain if one adjusts the boundary conditions in the function.
    The function saves the result as an xdmf.file in solution.xdmf.
    
    time_duration...enpoint of the interevall [0, time_duration] where the equation is solved
    time_partition... number of time steps in the Backward Euler method
    gamma...constant in the Bloch-Torrey equation
    expression...symbolic expression of the function
    function_class... function class (e.g. Lagrange) in which the expression is interpolated on the domain
    function_class_regularity... regularity of the function class (e.g. 1 for Lagrange corresponds to spline interpolation)
    T_Meq_function_class... the function class of T_1, T_2 and M^{eq}, they all need the same function class
    """
    
    t_start = 0
    t_end = time_duration
    #The solution will be computed on the intervall [t_start, t_end]
    
    delta_t = (t_end - t_start)/time_partitions
    one_over_delta_t = 1/delta_t
    
    #Define all function spaces before we go into the loop for efficiency
    
    #M_0
    #Define one function on a cell element of the domain with the required function class and regularity
    # Vector space for M : ℝ³ → ℝ³
    M_element = element(M_function_class, domain.basix_cell(), M_function_class_regularity, shape=(3,), dtype=default_real_type)
    M_function_space = fem.functionspace(domain, M_element)
    M_0 = fem.Function(M_function_space) #Define the function M_0
    M_0.interpolate(M_0_expression)   #interpolate the values for M_0 from the given expression
    
    #B:
    B_element = element(B_function_class, domain.basix_cell(), B_function_class_regularity, shape=(3,), dtype=default_real_type)
    B_function_space = fem.functionspace(domain, B_element)
    B_t = fem.Function(B_function_space)
    
    #f: 
    #We need the same function class as for Meq an T1 but vector valued
    f_element = element(T_Meq_function_class, domain.basix_cell(), T_Meq_function_class_regularity, shape=(3,), dtype=default_real_type)
    # Vector space for T : ℝ³ → ℝ
    f_function_space = fem.functionspace(domain, f_element)
    
    #We split f into an time dependent f_t and time independent part. The reson for this is that the time independent part can
    #be defined outside the time loop and this increases the efficiency.
    
    f = fem.Function(f_function_space)   #time independent term of f
    f_t = fem.Function(M_function_space) #time dependent term of f needs the same function sapce as M because we add later M to it
    
    #Relaxation times T_1 & T_2:
    
    #Note that the Relaxing times and the equilibrium magnetization is for every step the same since it is not time dependent!!!!!
    
    T_Meq_element = element(T_Meq_function_class, domain.basix_cell(), T_Meq_function_class_regularity, shape=(), dtype=default_real_type)
    # Vector space for T_1,T_2, M^{eq} : ℝ³ → ℝ
    T_Meq_function_space = fem.functionspace(domain, T_Meq_element)
    T1 = fem.Function(T_Meq_function_space)
    T2 = fem.Function(T_Meq_function_space)
    Meq = fem.Function(T_Meq_function_space)
    
    #interpolate the values for the functions from the given expressions
    T1.interpolate(T1_expression)
    T1.x.scatter_forward()
    T2.interpolate(T2_expression)
    T2.x.scatter_forward()
    Meq.interpolate(Meq_expression)
    Meq.x.scatter_forward()
    
    
    # Generate the time independet term of the right hand side
    Meq_values = Meq.x.array
    T1_values = T1.x.array

    # Generate an empty array for f which is of the size of a vector with 3 elements
    f_array = np.zeros((3 * len(Meq_values),), dtype=Meq_values.dtype)

    # z-component: Meq / T1
    f_array[2::3] = Meq_values / T1_values  #set the z component

    # Assign the values to the function f
    f.x.array[:] = f_array
    f.x.scatter_forward()
    
    #Diffusion D :  
    
    #Note that the diffusion is for every step the same since it is not time dependent!!!!!
    
    #Define one function on a cell element of the domain with the required function class and regularity
    D_element = element(D_function_class, domain.basix_cell(), D_function_class_regularity, shape=(3, 3, 3, 3), dtype=default_real_type)
    D_function_space = fem.functionspace(domain, D_element)
    D = fem.Function(D_function_space)
    const_tensor_4d = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    const_tensor_4d[i,j,k,l] = alpha if (i==k and j==l) else 0.0
                
    values = np.tile(const_tensor_4d.flatten(), len(D.x.array) // const_tensor_4d.size)
       
    D.x.array[:] = values
    D.x.scatter_forward()

    
    #Create the part of the bilinear form and right hand side, that is not time depending:
    #The bilinear form consists of the terms: diffusion, relaxation, gamma * MxB, one_over_delta_t * M
    M = ufl.TrialFunction(M_function_space)    #search for function
    phi = ufl.TestFunction(M_function_space)  #Testfunctions
    
    #Creating the acting of the tensor D on the gradient of M
    D_grad_M_components = []
    for i in range(3):
        row = []
        for j in range(3):
            s = 0
            for k in range(3):
                for l in range(3):
                    s += D[i, j, k, l] * ufl.grad(M)[k, l]
            row.append(s)
        D_grad_M_components.append(row)
        
    D_grad_M = ufl.as_tensor(D_grad_M_components)
    diffusion_ufl = ufl.inner(D_grad_M, ufl.grad(phi)) * ufl.dx
    
    # relaxing terms:
    relaxation = ufl.as_vector([M[0] / T2, M[1] / T2, M[2] / T1])
    
    #Not time depending right hand side 
    f_ufl = ufl.inner(f, phi) * ufl.dx
    #Define the boundary conditions:
    #Dirichlet with 0 on the boundary:

    def boundary(x):
        return np.isclose(x[0], 0) | np.isclose(x[0], 1) | \
               np.isclose(x[1], 0) | np.isclose(x[1], 1) | \
               np.isclose(x[2], 0) | np.isclose(x[2], 1)

    # mark the points on the boundary
    boundary_facets = fem.locate_dofs_geometrical(M_function_space, boundary)

    # create the boundary conditions M = 0
    zero_vec = np.zeros(3, dtype=default_real_type)
    bc = fem.dirichletbc(zero_vec, boundary_facets, M_function_space)

    
    #Define the initial value for the loop:
    M_t_old = fem.Function(M_function_space)
    M_t_old.x.array[:] = M_0.x.array
    M_t_old.x.scatter_forward()
    
    #Save the results
    xdmf = io.XDMFFile(domain.comm, "solution.xdmf", "w")
    xdmf.write_mesh(domain)  # save mesh
    
    #To save a function it must be interpolated to "Lagrange" with regularity 1
    M_regularity_1 = element(M_function_class, domain.basix_cell(), 1, shape=(3,), dtype=default_real_type)
    M_regularity_1_function_space = fem.functionspace(domain, M_regularity_1)
    M_regularity_1_function = fem.Function(M_regularity_1_function_space)
    
    M_regularity_1_function.interpolate(M_t_old)
    
    
    xdmf.write_function(M_regularity_1_function, 0) #write M_0
    
    #For loop for the implizit Euler:
    for time_step in range(1,time_partitions + 1):
        t = t_start + delta_t * time_step
        
        def B_t_expression(x):
            return B_expression(x, t)
        B_t.interpolate(B_t_expression)
        
        #Creating the bilinear form

        #cross product of the magnetization and the linear terms

        # cross product: M × B_t
        MxB = ufl.cross(M, B_t)

        bilinear_form_without_diffusion = -gamma * MxB + relaxation + one_over_delta_t * M
        bilinear_form_without_diffusion_ufl = ufl.inner(bilinear_form_without_diffusion, phi) * ufl.dx

        #Put the terms of the bilinear form together:
        bilinear_form = diffusion_ufl  + bilinear_form_without_diffusion_ufl

        #Define the time dependent right hand side f_t:
        M_t_old.x.array[:] *= one_over_delta_t
        M_t_old.x.scatter_forward()

        #Right hand side f
        f_t_ufl = ufl.inner(M_t_old, phi) * ufl.dx
        
        #Combine the time and not time depending term:
        f_combined = f_t_ufl + f_ufl
        
        #Solve the linear problem:
        problem = LinearProblem(bilinear_form, f_combined, bcs=[bc], petsc_options={
        "ksp_type": "preonly",  # keine Iterationen, sondern direkter Solve
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist" }) # direkter LU-Preconditioner = direkter Löser
             
        M_t = problem.solve()
        #save the function
        M_regularity_1_function.interpolate(M_t)
        xdmf.write_function(M_regularity_1_function, t)

        #Increment M_t_old
        M_t_old.x.array[:] = M_t.x.array
        M_t_old.x.scatter_forward()
         
    #End for loop
    xdmf.close()
    print("Process completed!")
    
#######################################################################


#Example for the Bloch-torrey solver:
grid_partitions = 8   #Try to stay under 20 to keep the runtime moderate
time_duration = 1   #in seconds, time goes from [0,time_duration]
time_partitions = 30

#Create the domain of the equation:
domain = mesh.create_unit_cube(MPI.COMM_WORLD, grid_partitions, grid_partitions, grid_partitions)

gamma = 2.675e8  #gyromagnetic ration

#unknown function of the magnetization M
#This is the subspace where we search for the magnetization
M_function_class = "Lagrange"
M_function_class_regularity = 1  #Try to keep this low to keep the runtime moderate <= 4

#Magnetization at time 0:
def M_0_expression(x):      #The eqilibrium Magnetization
    values = np.zeros((3, x.shape[1])) #set all values to 0
    values[0] = 0     #it is already 0 but we write it for readability
    values[1] = 0.00324 *300* x[0]*(1 - x[0])  * x[1]*(1 - x[1])* x[2]*(1 - x[2])   #this satisfies that we are zero on the boundary
    values[2] = 0   #Assume at first that it  is constant one like the equilibrium magnetization
    return values


#Information about the known functions in the Bloch-Torrey equation:

#Magnetic field B:
#Accuracy of interpolation
B_function_class = "Lagrange"
B_function_class_regularity = 1
def B_expression(x,t):
    z_direction_field = 1 #This is the constant strong field in the z-direction
    values = np.zeros((3, x.shape[1]))
    values[0] = 0   #  + 0.000000001 * np.cos(5 * t)*(1/(10*t + 1))
    values[1] = 0
    values[2] = 1  - z_direction_field   #+ 0.04*t*(0.2 - x[0])*(0.2 - x[1])*(0.2 - x[2])
    return values


#Relaxation times T_1 & T_2 and M_eq:
#The same function space is used for all three
#Accuracy of interpolation
T_Meq_function_class = "DG"         #discontinuous
T_Meq_function_class_regularity = 0

def T1_expression(x):
    return np.full(x.shape[1], 1)  # constant function with value 1
def T2_expression(x):
    return np.full(x.shape[1], 0.1)
def Meq_expression(x):
    return np.full(x.shape[1], 0.00324) #For water with 25 degreees


#Diffusion D:
#Accuracy of interpolation of the coefficients of the diffusion
D_function_class = "DG"         #discontinuous
D_function_class_regularity = 0
#We assume that D ist the identity multplied with some constant \alpha
alpha = 2e-9

#######################################################################

Bloch_Torrey_solver(domain = domain, time_duration = time_duration, time_partitions = time_partitions,
                 gamma = gamma,
                        
                 M_function_class = M_function_class, M_function_class_regularity = M_function_class_regularity,
                 M_0_expression = M_0_expression,
                        
                 B_function_class = B_function_class, B_function_class_regularity = B_function_class_regularity,
                 B_expression = B_expression,
                        
                 T_Meq_function_class = T_Meq_function_class, T_Meq_function_class_regularity= T_Meq_function_class_regularity,
                 T1_expression = T1_expression, T2_expression = T2_expression, Meq_expression = Meq_expression,
                        
                 D_function_class = D_function_class, D_function_class_regularity = D_function_class_regularity,
                 alpha = alpha,
                 )

