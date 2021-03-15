from fenics import *
import random
import numpy as np
from numpy import pi as π
# ---------------------------------------------------------------
# ------- 2 plants, 2 exudates, 1 metal -------------------------
# ---------------------------------------------------------------
# left-hand side  = plant 1 = barley
# right-hand side = plant 2 = tobacco
# X = phosphate
# Y1 = phytase
# Y2 = citrate
# -------------------------------------------------------- #
# *************** Parameters and constants *************** #
# -------------------------------------------------------- #

# Time in hours
T = 24                           # Final time
dt = 0.01                       # Step size

# Coefficients affecting the domain
w  = 0.04                       # *Space between the two roots*
a  = 0.005                      # Root radius
wr = w + a                      # Location of the other root
δLy = 0.2                       # Length of secretion zone 
δLx = 0                         # Length of  uptake zone
δlt = 0.2                       # Exudation 2 cm behind the tip

### Plant 1 (barley)
L_01 = 0.5                      # Initial length
GpD1 =  0.1728                  # Growth per day
LenMax1 = L_01 + 3*GpD1         # Max length in 3 days
G1   = GpD1/(24.0 * 3600)       # Growth in s

### Plant 2 (tobacco)
L_02 = 0.5                      # Initial length
GpD2 =  0.2                     # Growth per day
LenMax2 = L_02 + 3*GpD2         # Max length in 3 days
G2   = GpD2/(24.0 * 3600)       # Growth in s
LenMax = max(LenMax1, LenMax2)

# Coefficients for the PDEs
θ = 0.25                        # Solution volume fraction  
f = 0.5                         # Diffusion impedance factor
DLx = 9 * 1e-8                  # Diffusion coefficient of the nutrient X (phosphate) in free solution
DLy1 = 4.5 * 1e-8 *1.2              # Diffusion of coefficient of the nutrient Y1 (phytase) in free solution
DLy2 = 2.1 * 1e-8               # Diffusion of coefficient of the nutrient Y2 (citrate) in free solution
Dx = DLx * θ * f                # Diffusion coefficient for X
Dy1 = DLy1 * θ * f              # Diffusion coefficient for Y1
Dy2 = DLy2 * θ * f              # Diffusion coefficient for Y1
beta1 = 3.7 * 1e-6              # adsorption of phosphate to soil particles
beta2 = 4.68 * 1e-9             # desorption of phosphate from soil particles
beta4 = 3.41 * 1e-4             # phosphate-enhanced desorption from soil solid due to absorbed citrate
bx = beta1 / beta2 #/2             # X buffer power (ranges 300 to 2200)
by1 = 1.0                       # Y1 buffer power 
by2 = 1.0                       # Y2 buffer power
kx2 = beta4 / beta1             # X-Y2 interaction coefficient (5e3 for Zn-DMA)
kx1 = kx2 / 5                   # X-Y1 interaction coefficient (5e3 for Zn-DMA)
ky1 = 0.0                       # Y1-X interaction coefficient
ky2 = 0.0                       # Y2-X interaction coefficient
ν  = 1e-6                       # Water flux (agreed to not include, considered 0)
ρ  = 1.2                        # Soil bulk density 

### Plant 1 (barley)
α1  = 5.6 * 1e-3 #/10               # X absorbing power of root (was 1.5*1e-3 for Zn)     
Fy11 = 2.503 * 1e-10  *10          # Rate of Y1 exudation 
Fy21 = 1.006 * 1e-10            # Rate of Y2 exudation 

### Plant 2 (tobacco)
α2  = 5.6 * 1e-3                # X absorbing power of root (was 1.5*1e-3 for Zn)     
Fy12 = 1.316 * 1e-10            # Rate of Y1 exudation
Fy22 = 2.722 * 1e-10            # Rate of Y2 exudation

# Constants for Y decomposition (assume this is the same for Y1, Y2 for now)
Vmax1 = 0                       # Y1 (phytase) consumption by microbes
Vmax2 = 0 #2.5 * 1e-9           # Y2 (citrate) consumption by microbes
KM = 100 * 1e-6

# Initial values
X0 = 0.5 *1e-6  # initial phosphate in soil = 0.5 mikroM (advice by Tim, 17/02 email)                    
Y10 = 0.0
Y20 = 0.0


# ***** Non-dim constants *****
ε_t = 3.6e3                     # Scaling constant for time T (in s)
ε_z = LenMax                    # Scaling constant for z axis (in dm)
ε_r = wr-a                      # Scaling constant for r axis (in dm)
ε_y = 1e-6                      # Scaling constant for Y (in mol/dm^3 = M)
ε_x = 1e-6                      # Scaling constant for X (in mol/dm^3)

# Define the non-dim coefficients
δLy = δLy/ε_z                                       # non-dim δLy
δLx = δLx/ε_z                                       # non-dim δLx
δlt = δlt/ε_z                                       # non-dim δlt
L_01 = L_01/ε_z                                     # non-dim L_0
L_02 = L_02/ε_z                                     # non-dim L_0


KM = Constant(KM/ε_y)                               # non-dim KM
Fy11 = Constant(Fy11 * ε_t / (24 * ε_r * ε_y) )     # non-dim Fy / 24 
Fy12 = Constant(Fy12 * ε_t / (24 * ε_r * ε_y) )     # non-dim Fy / 24 
Fy21 = Constant(Fy21 * ε_t / (24 * ε_r * ε_y) )     # non-dim Fy / 24 
Fy22 = Constant(Fy22 * ε_t / (24 * ε_r * ε_y) )     # non-dim Fy / 24 
α1  = Constant(α1 * ε_t/ε_r)                        # non-dim α
α2  = Constant(α2 * ε_t/ε_r)                        # non-dim α

G1  = G1 * ε_t / ε_z                                # non-dim G
G2  = G2 * ε_t / ε_z                                # non-dim G
ρ1  = Constant(ρ * Vmax1 * ε_t/ε_y)                   # non-dim Vmax or ρ, replaces hat_Vmax
ρ2  = Constant(ρ * Vmax2 * ε_t/ε_y)                   # non-dim Vmax or ρ, replaces hat_Vmax

Dx = Dx * ε_t                                       # Pre-scaling
Dy1 = Dy1 * ε_t                                     # Pre-scaling
Dy2 = Dy2 * ε_t                                     # Pre-scaling
Dx = Constant([ [Dx/(ε_r**2),0], [0,Dx/(ε_z**2)] ]) # Scale in r and z
Dy1 = Constant([ [Dy1/(ε_r**2),0], [0,Dy2/(ε_z**2)] ]) # Scale in r and z
Dy2 = Constant([ [Dy2/(ε_r**2),0], [0,Dy2/(ε_z**2)] ]) # Scale in r and z

ν = ν * ε_t                                         # Pre-scaling
ν_r = Constant(ν / ε_r)                             # Scale in r
ν_z = Constant(ν / ε_z)                             # Scale in z


kbx1 = Constant(ε_y * kx1 * bx)                     # kx1 * bx    
kbx2 = Constant(ε_y * kx2 * bx)                     # kx1 * bx    
kby1 = Constant(ky1 * by1)                          # ky * by (is zero: not included)
kby2 = Constant(ky2 * by2)                          # ky * by (is zero: not included)
kx1 = Constant(ε_y * kx1)                           # non-dim kx
kx2 = Constant(ε_y * kx2)                           # non-dim kx
ky1 = Constant(ε_x * ky1)                           # non-dim ky
ky2 = Constant(ε_x * ky2)                           # non-dim ky
bx = Constant(bx)                                   # do not require normalisation
by1 = Constant(by1)                                 # do not require normalisation
by2 = Constant(by2)                                 # do not require normalisation

a = (a - a)/ε_r;                                    # Translation in a and non-dim
w = (w - a)/ε_r;                                    # Translation in a (not used here)
wr = (wr-a)/ε_r;                                    # Translation in a and non-dim

# Initial values (converted M to nanoM)
X0 = Constant(X0/ε_x)
Y10 = Constant(Y10/ε_y)
Y20 = Constant(Y20/ε_y)


# -------------------------------------------------------- #
# ************************* Mesh ************************* #
# -------------------------------------------------------- #

# Create mesh and define function space
nx = 90
ny = 90
mesh = RectangleMesh(Point(a,-LenMax/ε_z), Point(wr, 0), nx, ny) # x-axis = r, y-axis = z
# In the above we have LenMax=max(LenMax1, LenMax2) to get length of the domain

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

class Soil_right(SubDomain): # r=w
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], wr, tol)

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
XL_n = interpolate(X0, V)
YL1_n = interpolate(Y10, V)
YL1s = interpolate(Y10, V)
YL2_n = interpolate(Y20, V)
YL2s = interpolate(Y20, V)

ε_s = 10**(-4) # to use for steepness of indicator functions

# Define Expressions for the boundary integrals (indicator functions for the BCs)
# Indicator functions for uptake
indicLX = Expression('0.5*(1+ (2/ppi)*atan((x[1] +(Len0 + G*t))/epss ))', degree=6, Len0=L_01, ppi = π, epss = ε_s, G=G1, t=0) # Plant 1
indicRX = Expression('0.5*(1+ (2/ppi)*atan((x[1] +(Len0 + G*t))/epss ))', degree=6, Len0=L_02, ppi = π, epss = ε_s, G=G2, t=0) # Plant 2

# Indicator functions for exudation of Y1
indicLY1 = Expression('Fy*(1/ppi)*(atan((x[1] +(Len0 + G*t - dlt))/(5*epss)) - atan((x[1] +(Len0 + G*t-dlt -dLenY))/(5*epss) ))', degree=6, Len0=L_01, ppi = π, epss = ε_s, G=G1, Fy=Fy11, dLenY=δLy, dlt=δlt, t=0) # Plant 1
indicRY1 = Expression('Fy*(1/ppi)*(atan((x[1] +(Len0 + G*t - dlt))/(5*epss)) - atan((x[1] +(Len0 + G*t-dlt -dLenY))/(5*epss) ))', degree=6, Len0=L_02, ppi = π, epss = ε_s, G=G2, Fy=Fy12, dLenY=δLy, dlt=δlt, t=0) # Plant 2

# Indicator functions for exudation of Y2
indicLY2 = Expression('Fy*(1/ppi)*(atan((x[1] +(Len0 + G*t - dlt))/(5*epss)) - atan((x[1] +(Len0 + G*t-dlt -dLenY))/(5*epss) ))', degree=6, Len0=L_01, ppi = π, epss = ε_s, G=G1, Fy=Fy21, dLenY=δLy, dlt=δlt, t=0) # Plant 1
indicRY2 = Expression('Fy*(1/ppi)*(atan((x[1] +(Len0 + G*t - dlt))/(5*epss)) - atan((x[1] +(Len0 + G*t-dlt -dLenY))/(5*epss) ))', degree=6, Len0=L_02, ppi = π, epss = ε_s, G=G2, Fy=Fy22, dLenY=δLy, dlt=δlt, t=0) # Plant 2

# Define test functions
v_1 = TestFunction(V)
v_2 = TestFunction(V)
v_3 = TestFunction(V)

# Define trail functions
X = TrialFunction(V) 
Y1 = TrialFunction(V) 
Y2 = TrialFunction(V) 

# Define variational problem for step 1 (solve for Y1, assuming ky1=0)

a1 = (θ + by1) * Y1 * v_2*dx + dt * dot(Dy1 * grad(Y1),grad(v_2)) * dx + dt * (ρ1/(KM+YL1_n))*Y1 * v_2 * dx

b1 = (θ + by1) * YL1_n *v_2*dx + dt * indicLY1 *v_2*ds(1) + dt * indicRY1 *v_2*ds(3)

# Define variational problem for step 2 (solve for Y2, assuming ky2=0)

a2 = (θ + by2) * Y2 * v_3*dx + dt * dot(Dy2 * grad(Y2),grad(v_3)) * dx + dt * (ρ2/(KM+YL2_n))*Y2 * v_3 * dx

b2 = (θ + by2) * YL2_n *v_3*dx + dt * indicLY2 *v_3*ds(1) + dt * indicRY2 *v_3*ds(3)

# Variational problem for step 3 (solve for X)

a3 = (θ + bx/(1 + kbx1*YL1_n + kbx2*YL2_n)) * X*v_1*dx - (kbx1*bx /((1 + kbx1*YL1_n + kbx2*YL2_n)**2))* (YL1s - YL1_n) * X* v_1*dx - (kbx2*bx /((1 + kbx1*YL1_n + kbx2*YL2_n)**2))* (YL2s - YL2_n) * X* v_1*dx + dt * dot(Dx * grad(X),grad(v_1)) * dx + dt * indicLX*α1 *X *v_1*ds(1) + dt * indicRX*α2 *X *v_1*ds(3)

b3 = (θ + bx/(1 + kbx1*YL1_n + kbx2*YL2_n)) * XL_n *v_1*dx


# -------------------------------------------------------- #
# ********************** Solve PDEs ********************** #
# -------------------------------------------------------- #
X  = Function(V)
Y1  = Function(V)
Y2  = Function(V)

# Time stepping
t = 0

# Create HDMF5 files to visualise output
# if asked in ParaView, use Xdmf3ReaderS
xdmffile_X = XDMFFile('Fy11times10DY1up20pc/Phosphate_base/X.xdmf')
xdmffile_Y1 = XDMFFile('Fy11times10DY1up20pc/Phytase_base/Y1.xdmf')
xdmffile_Y2 = XDMFFile('Fy11times10DY1up20pc/Citrate_base/Y2.xdmf')
xdmffile_X.parameters["flush_output"] = True
xdmffile_Y1.parameters["flush_output"] = True
xdmffile_Y2.parameters["flush_output"] = True


n = FacetNormal(mesh)
file_flux_left = open('Fy11times10DY1up20pc/flux_left.txt', 'w')
file_flux_right = open('Fy11times10DY1up20pc/flux_right.txt', 'w')

while t<=T:

    # Solve the system  
    solve(a1 == b1, Y1) 
    YL1s.assign(Y1)

    solve(a2 == b2, Y2) 
    YL2s.assign(Y2)   

    solve(a3 == b3, X)   
    XL_n.assign(X)

    YL1_n.assign(Y1)
    YL2_n.assign(Y2)

    flux_left = α1*X*indicLX*ds(1)
    total_flux_left = assemble(flux_left)
    # print('Total flux: ', total_flux_left)
    file_flux_left.write(str(round(t,2)) + " " + str(total_flux_left))
    file_flux_left.write('\n')

    flux_right = α2*X*indicRX*ds(3)
    total_flux_right = assemble(flux_right)
    # print('Total flux: ', total_flux_right)
    file_flux_right.write(str(round(t,2)) + " " + str(total_flux_right))
    file_flux_right.write('\n')

    # Update current time
    t += dt 
    # Root growth update:
    indicLX.t = t
    indicRX.t = t
    indicLY1.t = t
    indicLY2.t = t
    indicRY1.t = t
    indicRY2.t = t
    # Write into file
    xdmffile_X.write(X, t)
    xdmffile_Y1.write(Y1, t)
    xdmffile_Y2.write(Y2, t)

    print('t=',round(t,2))
    
    # if round(t,2) == 24.00:
    #     file = File("X_24h.pvd")
    #     file << X
    #     file = File("Y1_24h.pvd")
    #     file << Y1
    #     file = File("Y2_24h.pvd")
    #     file << Y2
    # if round(t,2) == 48.00:
    #     file = File("X_48h.pvd")
    #     file << X
    #     file = File("Y1_48h.pvd")
    #     file << Y1
    #     file = File("Y2_48h.pvd")
    #     file << Y2

file_flux_left.close()
file_flux_right.close()

# file = File("X_24h.pvd")
# file << X
# file = File("Y1_24h.pvd")
# file << Y1
# file = File("Y2_24h.pvd")
# file << Y2