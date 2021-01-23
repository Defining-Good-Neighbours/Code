from fenics import *
import random
import numpy as np
# Time in hours
T = 24 *2             # Final time
num_steps = 240 *2   # Number of time steps
dt = T / num_steps # Time step size

# Coefficients affecting the domain
Lv = Constant(100)
xr = Constant(1 / sqrt(np.pi * Lv)) # Radius of zone of root influence (along r-axis)
a = Constant(0.001)                 # Root radius 
Len0 = Constant(0.5)                # Initial length (renamed L to Len)
GpD = Constant(0.2)                 # Growth per day
LenMax = Constant(Len0 + 2*GpD)     # Max length in 1 day 

# Create mesh and define function space
nx = ny = 100
mesh = RectangleMesh(Point(a,-LenMax), Point(xr, 0), nx, ny) # x-axis = r, y-axis = z

# Define function space for concentrations
P1 = FiniteElement('CG', triangle, 1)
element = MixedElement([P1, P1])
V = FunctionSpace(mesh, element)

# Define the coefficients (converted from seconds to hours, M to nanoM)
theta = Constant(0.7)
f = Constant(0.5)
DLx = Constant(2.52 * pow(10,-4))
DLy = Constant(2.52 * pow(10,-4))
Dx = Constant(DLx * theta * f)
Dy = Constant(DLy * theta * f)
bx = Constant(200)
by = Constant(1)
kx = Constant(5 * pow(10,3))
ky = Constant(0)
Vmax = Constant(3600*(2.5 - 10 * pow(10,-9)))
KM = Constant(pow(10,5)) #Constant(100 * pow(10,-6))
rho = Constant(1)
alpha = Constant(5.4)
Fy = Constant(1.44 * pow(10,-7))
nu = Constant(3.6 * pow(10,-3))


# Root growth rate in dm/h 
G = Constant(GpD / 24)
# Len = Constant(Len0) # initialise length
dLenY = Constant(0.2)
tol = 1E-14

# Initial values (converted M to nanoM)
X0 = Constant(10) #Constant(0.01*pow(10,-6))
Y0 = Constant(0)        

# Define initial condition for the system
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = X0 # Initial value for X (=zinc)
        values[1] = Y0 # Initial value for Y (=PS)
    def value_shape(self):
        return (2,)

# Class for interfacing with the solver
class PlantSys(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

# Define test functions
v_1, v_2 = TestFunctions(V)

# Define functions for concentrations
du = TrialFunction(V)   # For getting the Jacobian
u = Function(V)         # The unknown functions
u_n = Function(V)       # Values of the function at previous time step

# Define initial value
u_init = InitialConditions(degree=0)
u.interpolate(u_init)
u_n.interpolate(u_init)

# Split functions to access components
# dW, dO = split(du)
X, Y = split(u)
X_n, Y_n = split(u_n)

tol = 1E-14
#Sides of soil (2D version)
class Soil_top(SubDomain): # z=0
    def inside(self, x, on_boundary):
        return(on_boundary and near(x[1], 0.0, tol))

class Soil_bottom(SubDomain): # z=-Lmax
    def inside(self, x, on_boundary):
        return(on_boundary and near(x[1], -LenMax, tol))    

class Soil_left(SubDomain): # r=a
    def inside(self, x, on_boundary):
        return(on_boundary and near(x[0], a, tol))

class Soil_right(SubDomain): # r=x
    def inside(self, x, on_boundary):
        return(on_boundary and near(x[0], xr, tol))

#Initialise subdomains
soil_top = Soil_top()
soil_bottom = Soil_bottom()       
soil_left = Soil_left()
soil_right = Soil_right()

#Initialise mesh function for boundaries
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)      

soil_left.mark(boundaries, 0)
soil_top.mark(boundaries, 1)
soil_right.mark(boundaries, 2)
soil_bottom.mark(boundaries, 3)

# Define measures corresponding to boundary surfaces
ds = Measure('ds', domain = mesh, subdomain_data = boundaries)

onesvec = Constant([1,1])
def vec1(X):
    return Dx*grad(X) - nu*X * onesvec
def vec2(Y):
    return Dy*grad(Y) - nu*Y * onesvec

# Define Expressions for the boundary integrals (indicator functions for the BCs)
indic1 = Expression('(near(x[0], a, tol) and x[1] <= 0 and x[1] >= -(Len0 + G*t) ) ? 1: 0', degree=0, a=a, tol=tol, Len0=Len0, G=G, t=0)
# indic1 = Expression('near(x[0], a, tol) ? 1: 0', degree=0, a=a, tol=tol)
indic2 = Expression('(near(x[0], a, tol) and x[1] <= -(Len0 + G*t-dLenY) and x[1] >= -(Len0 + G*t) ) ? 1: 0', degree=0, a=a, tol=tol, Len0=Len0, G=G, dLenY=dLenY, t=0)

# Define variational problem
L1 = (theta + bx/(1 + kx*bx*Y_n)) * (X - X_n) *v_1*dx \
- kx*pow(bx,2) * X_n * pow((1 + kx*bx*Y_n),-2) * (Y - Y_n) * v_1*dx \
+ dt * dot(vec1(X),grad(v_1)) * dx \
+ dt * alpha * X * indic1 *v_1*ds(0) 
# + dt * gx(X,Y) * v_1 * dx # this term is zero anyway

# a1 = 0
L2 = (theta + by/(1 + ky*by*X_n)) * (Y - Y_n) *v_2*dx \
- ky*pow(by,2)*Y_n * pow((1 + ky*by*X_n),-2) * (X - X_n) * v_2*dx \
+ dt * dot(vec2(Y),grad(v_2)) * dx \
- dt * Fy * indic2 *v_2*ds(0) \
+ dt * rho*Vmax*Y/(KM+Y_n) * v_2 * dx

# a2 = dt * Fy * indic2 *v_2*ds(0)

L = L1 + L2
# atot = a1 + a2

a = derivative(L, u, du)
problem = PlantSys(a, L)

# Create VTK files for visualization output
vtkfile_u1 = File('X_lin/X.pvd')
vtkfile_u2 = File('Y_lin/Y.pvd')

# solver = PETScSNESSolver()
# solver.parameters['line_search'] = 'basic'
# solver.parameters['linear_solver']= 'lu'

solver = NewtonSolver()
# solver.parameters["linear_solver"] = "lu"
# solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-3

# info(NonlinearVariationalSolver.default_parameters(), 1)
# solver.parameters["newton_solver"]["relative_tolerance"] = 1.0e-3


# Time stepping
t = 0
toler = 1.e-6
for n in range(num_steps):
    # Update current time
    t += dt
    # print(0.5 + 0.2/24*t - 0.2)
    # Root growth update:
    indic1.t += dt
    indic2.t += dt

    # Solve variational problem for time step
    # solve(L == a, u)    # Save solution to file 
    solver.solve(problem, u.vector())

    _X, _Y = u.split()
    vtkfile_u1 << (_X, t)
    vtkfile_u2 << (_Y, t)
    
    # Update previous solution
    u_n.assign(u)
    print('t=',t)
