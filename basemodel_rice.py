from fenics import *
import random
import numpy as np
from numpy import pi as π

# -------------------------------------------------------- #
# *************** Parameters and constants *************** #
# -------------------------------------------------------- #

# Time in hours
T = 24                          # Final time
dt = 0.01                       # Step size

# Coefficients affecting the domain
Lv = 100.0                      # Maximum root length density
xr = 1.0 / ((π * Lv)**(0.5))    # Radius of zone of root influence (along r-axis)
# w  = 0.08                       # *Space between the two roots (both plants same type)*
a  = 1e-3                       # Root radius
# wr = w + a                      # Location of the other root

L_0 = 0.5                       # Initial length
GpD =  0.2                      # Growth per day
LenMax = L_0 + 3*GpD            # Max length in 3 day

# Root growth rate in dm/h 
G   = GpD/(24.0 * 3600)         # Growth in s
δLy = 0.2                       # Length or PS secretion zone
δLx = 0                         # Length or Zn uptake zone
δlt = 0                         # excudation 2 cm behind the tip

# Coefficients for the PDEs
θ = 0.7                         # Solution volume fraction
f = 0.5                         # Diffusion impedance factor
DLx = 7e-8                      # Diffusion coefficient of Zn species in free solution
DLy = 7e-8                      # Diffusion of coefficient of DMA in free solution
Dx = DLx * θ * f                # Diffusion coefficient for X
Dy = DLy * θ * f                # Diffusion coefficient for Y
bx = 200.0                      # Zn buffer power
by = 1.0                        # DMA buffer power
kx = 5e3                        # Zn-DMA interaction coefficient
ky = 0.0                        # DMA-Zn interaction coefficient
α  = 1.5 * 1e-3                 # Zn absorbing power of root       
Fy = 4.0 * 1e-11                # Rate of DMA exudation over 24 h
ν  = 1e-6                       # Water flux (agreed to not include, considered 0)
ρ  = 1.0                        # Soil bulk density

# Constants for DMA decomposition
Vmax = 2.5 * 1e-9
KM = 100 * 1e-6

# Initial values
X0 = 1e-8                      
Y0 = 0.0


# ***** Non-dim constants *****
ε_t = 3.6e3                     # Scaling constant for time T (in s)
ε_z = LenMax                    # Scaling constant for z axis (in dm)
ε_r = xr-a                      # Scaling constant for r axis (in dm)
ε_y = 1e-6                      # Scaling constant for Y (in mol/dm^3 = M)
ε_x = 1e-9                      # Scaling constant for X (in mol/dm^3)

# Define the non-dim coefficients
δLy = δLy/ε_z                                       # non-dim δLy
δLx = δLx/ε_z                                       # non-dim δLx
δlt = δlt/ε_z                                       # non-dim δlt
L_0 = L_0/ε_z                                       # non-dim L_0

KM = Constant(KM/ε_y)                               # non-dim KM
Fy = Constant(Fy * ε_t / (24 * ε_r * ε_y) )         # non-dim Fy / 24 
α  = Constant(α * ε_t/ε_r)                          # non-dim α
G  = G * ε_t / ε_z                                  # non-dim G
ρ  = Constant(ρ * Vmax * ε_t/ε_y)                   # non-dim Vmax or ρ, replaces hat_Vmax


Dx = Dx * ε_t                                       # Pre-scaling
Dy = Dy * ε_t                                       # Pre-scaling
Dx = Constant([ [Dx/(ε_r**2),0], [0,Dx/(ε_z**2)] ]) # Scale in r and z
Dy = Constant([ [Dy/(ε_r**2),0], [0,Dy/(ε_z**2)] ]) # Scale in r and z

ν = ν * ε_t                                         # Pre-scaling
ν_r = Constant(ν / ε_r)                             # Scale in r
ν_z = Constant(ν / ε_z)                             # Scale in z


kbx = Constant(ε_y * kx * bx)                       # kx * bx    
kby = Constant(ky * by)                             # ky * by (is zero: not included)
kx = Constant(ε_y * kx)                             # non-dim kx
ky = Constant(ε_x * ky)                             # non-dim ky
bx = Constant(bx)                                   # do not require normalisation
by = Constant(by)                                   # do not require normalisation

a = (a - a)/ε_r;                                    # Translation in a and non-dim
xr = (xr-a)/ε_r;                                    # Translation in a 

# Initial values (converted M to nanoM)
X0 = Constant(X0/ε_x)
Y0 = Constant(Y0/ε_y)


# -------------------------------------------------------- #
# ************************* Mesh ************************* #
# -------------------------------------------------------- #

# Create mesh and define function space
nx = 60
ny = 80
mesh = RectangleMesh(Point(a,-LenMax/ε_z), Point(xr, 0), nx, ny) # x-axis = r, y-axis = z

# Define function space for concentrations
V = FunctionSpace(mesh, 'CG', 1)

tol = 1E-15

#Sides of soil (2D version)
class Soil_top(SubDomain): # z=0
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class Soil_bottom(SubDomain): # z=-Lmax
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -LenMax/ε_z, tol)

class Soil_left(SubDomain): # r=a
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], a, tol)

class Soil_right(SubDomain): # r=x
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], xr, tol)

#Initialise subdomains
soil_top = Soil_top()
soil_bottom = Soil_bottom()       
soil_left = Soil_left()
soil_right = Soil_right()

#Initialise mesh function for boundaries
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)    

soil_left.mark(boundaries, 1)
soil_top.mark(boundaries, 2)
soil_right.mark(boundaries, 3)
soil_bottom.mark(boundaries, 4)

 
# Define measures corresponding to boundary surfaces
ds = Measure('ds', domain = mesh, subdomain_data = boundaries)


# Define initial value
XL_n=interpolate(X0, V)
YL_n=interpolate(Y0, V)
YLs = interpolate(Y0, V)

ε_s = 10**(-4) # to use for steepness of indicator functions

# Define Expressions for the boundary integrals (indicator functions for the BCs)
#Approx indicator function
indic1 = Expression('0.5*(1+ (2/ppi)*atan((x[1] +(Len0 + G*t))/epss ))', degree=6, Len0=L_0, ppi = π, epss = ε_s, G=G, t=0)

#Approx indicator function x Fy
indic2 = Expression('Fy*(1/ppi)*(atan((x[1] +(Len0 + G*t - dlt))/(5*epss)) - atan((x[1] +(Len0 + G*t-dlt -dLenY))/(5*epss) ))', degree=6, Len0=L_0, ppi = π, epss = ε_s, G=G, Fy=Fy, dLenY=δLy, dlt=dlt, t=0)

# Define test functions
v_1 = TestFunction(V)
v_2 = TestFunction(V)

# Define trail functions
X = TrialFunction(V) 
Y = TrialFunction(V) 

# Define variational problem for step 1 (assuming ky=0)

a1 = (θ + by) * Y * v_2*dx + dt * dot(Dy * grad(Y),grad(v_2)) * dx + dt * (ρ/(KM+YL_n))*Y * v_2 * dx
b1 = (θ + by) * YL_n *v_2*dx + dt * indic2 *v_2*ds(1) 

# Variational problem for step 2

a2 = (θ + bx/(1 + kbx*YL_n)) * X*v_1*dx - (kbx*bx /((1 + kbx*YL_n)**2))* (YLs - YL_n) * X* v_1*dx + dt * dot(Dx * grad(X),grad(v_1)) * dx + dt * indic1*α *X *v_1*ds(1) 
b2 = (θ + bx/(1 + kbx*YL_n)) * XL_n *v_1*dx


# -------------------------------------------------------- #
# ********************** Solve PDEs ********************** #
# -------------------------------------------------------- #
X  = Function(V)
Y  = Function(V)

# Time stepping
t = 0

# Create VTK files for visualization output
#vtkfile_u1 = File('Zn_lin_dec/Zn.pvd')
#vtkfile_u2 = File('DMA_lin_dec/DMA.pvd')
# Create HDMF5 files to visualise output
# if asked in ParaView, use Xdmf3ReaderS
xdmffile_X = XDMFFile('Zn/Zn.xdmf')
xdmffile_Y = XDMFFile('DMA/DMA.xdmf')
xdmffile_X.parameters["flush_output"] = True
xdmffile_Y.parameters["flush_output"] = True

#vtkfile_u1 << (X, 0)
#vtkfile_u2 << (Y, 0)
xdmffile_X.write(X, 0.0)
xdmffile_Y.write(Y, 0.0)


while t<=T:
    # Update current time
    t += dt 
    # Root growth update:
    indic1.t = t
    indic2.t = t

    solve(a1 == b1, Y) 
    
    YLs.assign(Y)   

    solve(a2 == b2, X)   

    XL_n.assign(X)
    YL_n.assign(Y)
    

    xdmffile_X.write(X, t)
    xdmffile_Y.write(Y, t)

    print('t=',t)
    
    if round(t,2) == 24.00:
        file = File("Zn_24h.pvd")
        file << X

        file = File("DMA_24h.pvd")
        file << Y
    if round(t,2) == 48.00:
        file = File("Zn_48h.pvd")
        file << X

        file = File("DMA_48h.pvd")
        file << Y