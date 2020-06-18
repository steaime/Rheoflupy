from fenics import *
from dolfin import *
from mshr import *
from vtkplotter.dolfin import plot, datadir, Latex
from scipy.integrate import simps
import matplotlib.pyplot as plt
import numpy as np
import my_functions as mf
import mshr
import math 




mesh = Mesh()
# Create list of polygonal domain vertices
channel_file=mf.import_file('/home/matteo/Desktop/Python_Code/FEniCS_Simulations/Rheoflu_simulations/Channel_shape','.txt');
c=np.loadtxt(channel_file[0]);
a=c[-3080:-2700:10]
a[0,:]=np.round(a[0,:])
a[-1,:]=np.round(a[-1,:])
a[:,2]=np.flip(a[:,2])
a[:,3]=np.flip(a[:,3])



domain_vertices=[]


for j in range(len(a[:,0])):
    domain_vertices.append(Point(a[j,0],a[j,1],1))
for j in range(len(a[:,0])):
    domain_vertices.append(Point(a[j,2],a[j,3],1))
    



h=60
domain = mshr.Polygon(domain_vertices)
g=mshr.Extrude2D(domain,h)
mesh = generate_mesh(g, 200)


P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# No-slip boundary condition for velocity

def inflow(x, on_boundary):
    return near(x[1], 1162)
def outflow(x, on_boundary):
    return near(x[1], 5170)
def walls(x, on_boundary):
    return on_boundary
#inflow  = 'near(x[1], 1162)'
#outflow = 'near(x[1], 5170)'
#walls   = 'on_boundary'

# Define boundary conditions
noslip  = DirichletBC(W.sub(0), Constant((0, 0,0)), walls)
inflow  = DirichletBC(W.sub(1), Constant(10), inflow)
outflow = DirichletBC(W.sub(1), Constant(0), outflow)




bcs = [noslip, inflow,outflow]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0,0))
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = inner(f, v)*dx
w = Function(W)
#F = inner(grad(u)*u, v)*dx + nu*inner(grad(u), grad(v))*dx \
#     - inner(p, div(v))*dx + inner(q, div(u))*dx + inner(f, v)*dx

solve(a == L, w, bcs)

# Split the mixed solution using a shallow copy
(u, p) = w.split(True)




(u_x,u_y,u_z) = u.split(True)


tol = 10  # avoid hitting points outside the domain
y = np.linspace(1162 + tol, 5170 -5, 101)
y_p=3399
x_p=132
x_sec = np.linspace(-x_p,x_p,300)
z_sec=np.linspace(0,h,25)

points_y_12 = [(0, y_,z_sec[12]) for y_ in y] 
points_y_11 = [(0, y_,z_sec[11]) for y_ in y] 
points_y_10 = [(0, y_,z_sec[10]) for y_ in y] 
points_y_09 = [(0, y_,z_sec[9]) for y_ in y]
points_y_08 = [(0, y_,z_sec[8]) for y_ in y] 
points_y_07 = [(0, y_,z_sec[7]) for y_ in y] 
points_y_06 = [(0, y_,z_sec[6]) for y_ in y] 
points_y_05 = [(0, y_,z_sec[5]) for y_ in y] 
points_y_04 = [(0, y_,z_sec[4]) for y_ in y] 
points_y_03 = [(0, y_,z_sec[3]) for y_ in y] 
points_y_02 = [(0, y_,z_sec[2]) for y_ in y]
points_y_01 = [(0, y_,z_sec[1]) for y_ in y] 
points_y_00 = [(0, y_,z_sec[0]) for y_ in y] 
u_y_line_12 = np.array([u_y(point) for point in points_y_12])
u_y_line_11 = np.array([u_y(point) for point in points_y_11])
u_y_line_10 = np.array([u_y(point) for point in points_y_10])
u_y_line_09 = np.array([u_y(point) for point in points_y_09])
u_y_line_08 = np.array([u_y(point) for point in points_y_08])
u_y_line_07 = np.array([u_y(point) for point in points_y_07])
u_y_line_06 = np.array([u_y(point) for point in points_y_06])
u_y_line_05 = np.array([u_y(point) for point in points_y_05])
u_y_line_04 = np.array([u_y(point) for point in points_y_04])
u_y_line_03 = np.array([u_y(point) for point in points_y_03])
u_y_line_02 = np.array([u_y(point) for point in points_y_02])
u_y_line_01 = np.array([u_y(point) for point in points_y_01])
u_y_line_00 = np.array([u_y(point) for point in points_y_00])


TFS_CG1 = FunctionSpace(mesh, P2)
grad_u = project(grad(p), TFS_CG1)
# Plot du/dx
pdy=grad_u.sub(1)
p_line=np.array([pdy(point) for point in points_y_06])
# plot of the pressure 
#plot(pdy)
 

# Manual flux
#set the trasversal velocity
points_x_12 = [(x_, y_p,z_sec[12]) for x_ in x_sec]
points_x_11 = [(x_, y_p,z_sec[11]) for x_ in x_sec]
points_x_10 = [(x_, y_p,z_sec[10]) for x_ in x_sec]
points_x_09 = [(x_, y_p,z_sec[9]) for x_ in x_sec]
points_x_08 = [(x_, y_p,z_sec[8]) for x_ in x_sec]
points_x_07 = [(x_, y_p,z_sec[7]) for x_ in x_sec]
points_x_06 = [(x_, y_p,z_sec[6]) for x_ in x_sec]
points_x_05 = [(x_, y_p,z_sec[5]) for x_ in x_sec]
points_x_04 = [(x_, y_p,z_sec[4]) for x_ in x_sec]
points_x_03 = [(x_, y_p,z_sec[3]) for x_ in x_sec]
points_x_02 = [(x_, y_p,z_sec[2]) for x_ in x_sec]
points_x_01 = [(x_, y_p,z_sec[1]) for x_ in x_sec]
points_x_00 = [(x_, y_p,z_sec[0]) for x_ in x_sec]
u_x_line_12 = np.array([u_y(point) for point in points_x_12])
u_x_line_11 = np.array([u_y(point) for point in points_x_11])
u_x_line_10 = np.array([u_y(point) for point in points_x_10])
u_x_line_09 = np.array([u_y(point) for point in points_x_09])
u_x_line_08 = np.array([u_y(point) for point in points_x_08])
u_x_line_07 = np.array([u_y(point) for point in points_x_07])
u_x_line_06 = np.array([u_y(point) for point in points_x_06])
u_x_line_05 = np.array([u_y(point) for point in points_x_05])
u_x_line_04 = np.array([u_y(point) for point in points_x_04])
u_x_line_03 = np.array([u_y(point) for point in points_x_03])
u_x_line_02 = np.array([u_y(point) for point in points_x_02])
u_x_line_01 = np.array([u_y(point) for point in points_x_01])
u_x_line_00 = np.array([u_y(point) for point in points_x_00])
#u_x_prova=np.array([u_y(point) for point in points_x_prova])



Q_12=simps(u_x_line_12, x_sec)
Q_11=simps(u_x_line_11, x_sec)
Q_10=simps(u_x_line_10, x_sec)
Q_09=simps(u_x_line_09, x_sec)
Q_08=simps(u_x_line_08, x_sec)
Q_07=simps(u_x_line_07, x_sec)
Q_06=simps(u_x_line_06, x_sec)
Q_05=simps(u_x_line_05, x_sec)
Q_04=simps(u_x_line_04, x_sec)
Q_03=simps(u_x_line_03, x_sec)
Q_02=simps(u_x_line_02, x_sec)
Q_01=simps(u_x_line_01, x_sec)
Q_00=simps(u_x_line_00, x_sec)

Q_x=np.array([Q_00,Q_01,Q_02,Q_03,Q_04,Q_05,Q_06,Q_07,Q_08,Q_09,Q_10,Q_11,Q_12,Q_11,Q_10,Q_09,Q_08,Q_07,Q_06,Q_05,Q_04,Q_03,Q_02,Q_01,Q_00])
print(z_sec)
print(len(Q_x))
Q=simps(Q_x,z_sec)
Q_appoximation=Q_12*h

print(z_sec)
print(3*Q/(4*h)) 

b=c[-3080:-2700:1]
x_channel=b[:,3]
l_x_channel=b[:,0]
fig, ax = plt.subplots()
ax.plot(b[:,1],3*Q/(4*b[:,0]*h),'r--',label='3*Q/4L(x)h')
#ax.plot(b[:,1],3*Q_appoximation/(4*b[:,0]*h),'g--',label='3*Q/4L(x)h')
ax.plot(y, u_y_line_06, 'b.', label='velocity')
ax.plot(b[:,1],0.01*b[:,0],'k--',label='Channel')


#ax.axis('equal')
leg = ax.legend();

plt.xlabel('x (um)')
plt.ylabel('amplitude (m/s)')
#plt.title('A sine wave with a gap of NaNs between 0.4 and 0.6')
plt.grid(True)


fig_1, ax_1 = plt.subplots()
ax_1.plot(y, u_y_line_06, 'c.', linewidth=2,label='velocity_30')
ax_1.plot(y, u_y_line_05, 'k.', linewidth=2,label='velocity_25')
ax_1.plot(y, u_y_line_04, 'b.', linewidth=2,label='velocity_20')
ax_1.plot(y, u_y_line_03, 'm.', linewidth=2,label='velocity_15')
ax_1.plot(y, u_y_line_02, 'g.', linewidth=2,label='velocity_10')
ax_1.plot(y, u_y_line_01, 'r.', linewidth=2,label='velocity_05')
ax_1.plot(y, u_y_line_00, 'y.', linewidth=2,label='velocity_00')
leg = ax_1.legend();

plt.xlabel('x (um)')
plt.ylabel('amplitude (m/s)')
plt.title('A sine wave with a gap of NaNs between 0.4 and 0.6')
plt.grid(True)



plt.show()

# 



'''
##################################################################### vtkplotter
f = r'-\nabla \cdot(\nabla u+p I)=f ~\mathrm{in}~\Omega'
formula = Latex(f, pos=(0.55,0.45,-.05), s=0.1)

plot(u, formula, at=0, N=2,
     mode='mesh ', scale=10,
     wireframe=True, scalarbar=False, style=1)
plot(p, at=1, text="pressure", cmap='rainbow')


##################################################################### streamlines
# A list of seed points (can be automatic: just comment out 'probes')
ally = np.linspace(0,50, num=100)
probes = np.c_[np.ones_like(ally), ally, np.zeros_like(ally)]

plot(u, 
     mode='mesh with streamlines',
     streamlines={'tol':0.1,            # control density of streams
                  'lw':1,                # line width 
                  'direction':'forward', # direction of integration
                  'maxPropagation':18,  # max length of propagation
                  'probes':probes,       # custom list of point in space as seeds
                 },
     c='white',                          # mesh color
     alpha=0.3,                          # mesh alpha
     lw=0,                               # mesh line width
     wireframe=True,                     # show as wireframe
     bg='blackboard',                    # background color
     newPlotter=True,                    # new window
     pos=(200,200),                      # window position on screen
     )
     '''
