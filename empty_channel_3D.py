from fenics import *
from mshr import *
import numpy as np
import math 
#import matplotlib.pyplot as plt
from vtkplotter.dolfin import datadir, plot
import my_functions as mf
import mshr




T = 1.0           # final time
num_steps = 100    # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
rho = 1            # density

#MY mesh

mesh = Mesh()
# Create list of polygonal domain vertices
channel_file=mf.import_file('/home/matteo/Desktop/Python_Code/FEniCS_Simulations/Rheoflu_simulations/Channel_shape','.txt');
c=np.loadtxt(channel_file[0]);
a=c[-2980:-2790:5]
a[0,:]=np.round(a[0,:])
a[-1,:]=np.round(a[-1,:])
a[:,2]=np.flip(a[:,2])
a[:,3]=np.flip(a[:,3])



domain_vertices=[]


for j in range(len(a[:,0])):
    domain_vertices.append(Point(a[j,0],a[j,1],1))
for j in range(len(a[:,0])):
    domain_vertices.append(Point(a[j,2],a[j,3],1))
    



domain = mshr.Polygon(domain_vertices)
g=mshr.Extrude2D(domain,80)
mesh = generate_mesh(g, 40)


#plot(mesh)

# Define function spaces for pressure and velocity
V = VectorFunctionSpace(mesh, "Lagrange", degree=3, dim=3)
Q = FunctionSpace(mesh, "Lagrange", 1)


#boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

inflow  = 'near(x[1], 2365)'
outflow = 'near(x[1], 4207)'
walls   = 'on_boundary'

# Define boundary conditions
bcu_noslip  = DirichletBC(V, Constant((0, 0,0)), walls)
bcp_inflow  = DirichletBC(Q, Constant(10), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# Define expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)


# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))


# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))


# Define variational problem for step 1
F1 = rho * dot((u - u_n) / k, v) * dx \
     + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx \
     + inner(sigma(U, p_n), epsilon(v)) * dx \
     + dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds \
     - dot(f, v) * dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx

# Define variational problem for step 3
a3 = dot(u, v) * dx
L3 = dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "petsc_amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

t = 0
for n in range(num_steps):
    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, "bicgstab", "default")

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, "bicgstab", prec)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    [bc.apply(b3) for bc in bcu]
    solve(A3, u_.vector(), b3, "bicgstab", "default")
	
    #plot(u_)
    u_n.assign(u_)
    p_n.assign(p_)


# Save to file
#ufile << u_
#pfile << p_
plot(u_)
plot(p_)
