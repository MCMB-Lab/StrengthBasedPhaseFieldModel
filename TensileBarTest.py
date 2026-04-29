from dolfin import *
from fenics import *
from mshr import *
import numpy as np

# material constants
E=25000.0# Young's modulus, [MPa]
nu=0.3# Poisson's ratio, [1/1]
lmbda=E*nu/(1.0+nu)/(1.0-2.0*nu)# Lame's constant, [MPa]
mu=E/2.0/(1.0+nu)# Lame's constant, [MPa]
k=lmbda+2.0*mu/3.0# bulk modulus, [MPa]
lc=1.0# length scale, [mm]
Gc=0.05# critical energy release rate, [N/mm]
ANISOSWT=0# 0, no split; 1, with split
ATSWT=4# 1, AT1 model; 2, AT2 model; 3, AT3 model; 4, strength-based PFM (low sigc/sigt); 5, strength-based PFM (high sigc/sigt)
# extra parameters for ATSWT=3, https://doi.org/10.1016/j.tafmec.2023.103902
xixi=0.5# 0.0<XI<1.0, XI in the generalized quadratic geometric function, [1/1]
caca=2.440# normalization constant, [1/1]
# extra parameters for ATSWT=4&5, ANISOSWT has to be 0, https://dx.doi.org/10.2139/ssrn.4473238
sigt=5.0# tensile strength, [MPa]
sigc=28.0# compressive strength, [MPa]
delta=0.0# parameter to adjust fracture strength surface, [1/1]
tol=1.0E-5# tolerance, [1/1]
lIrwin=3.0*Gc*E/8.0/(sigt**2)# Irwin characteristic length, 18.75 mm
if abs(ATSWT-4.0)<tol or abs(ATSWT-5.0)<tol:
    lc=lIrwin/2.0# 1.875, length scale, [mm], lc<4.0*lIrwin should work according to Dr. Kumar's code

# create mesh
WW=1.0# width of the domain, [mm]
HH=10.0# height of the domain, [mm]
#NX=1# number of elements in x direction, [1/1]
#NY=1# number of elements in y direction, [1/1]
#mesh=RectangleMesh(Point(0.0,0.0),Point(WW,HH),NX,NY)
NOTCHW=0.1# notch width for cross section reduction, [mm]
NOTCHH=0.1# notch height for cross section reduction, [mm]
ELERES=50# element resolution, [1/1]
domain=Polygon([dolfin.Point(0.0,0.0),dolfin.Point(WW,0.0),dolfin.Point(WW,HH/2.0-2.0*NOTCHH),dolfin.Point(WW-NOTCHW,HH/2.0-NOTCHH),dolfin.Point(WW-NOTCHW,HH/2.0+NOTCHH),dolfin.Point(WW,HH/2.0+2.0*NOTCHH),dolfin.Point(WW,HH),dolfin.Point(0.0,HH),dolfin.Point(0.0,HH/2.0+2.0*NOTCHH),dolfin.Point(NOTCHW,HH/2.0+NOTCHH),dolfin.Point(NOTCHW,HH/2.0-NOTCHH),dolfin.Point(0.0,HH/2.0-2.0*NOTCHH)])
mesh=generate_mesh(domain,ELERES)

# define spaces
Vu=VectorFunctionSpace(mesh,'CG',1)# displacement field space (u)
Vp=FunctionSpace(mesh,'CG',1)# phase field (p), maximum history strain energy (mhh), and maximum history phase field space (mhp)
# define variables
u=TrialFunction(Vu)
v=TestFunction(Vu)
uold=Function(Vu)
unew=Function(Vu)
p=TrialFunction(Vp)
q=TestFunction(Vp)
pold=Function(Vp)
pnew=Function(Vp)
pmin=Function(Vp)
pmax=Function(Vp)
mhh=Function(Vp)
mhp=Function(Vp)
# initialize variables
u_0=Expression(('0.0','0.0'),degree=1)
uold=project(u_0,Vu)
unew=project(u_0,Vu)
p_0=Expression(('0.0'),degree=1)
p_1=Expression(('0.99'),degree=1)
pold=project(p_0,Vp)
pnew=project(p_0,Vp)
pmin=project(p_0,Vp)
pmax=project(p_1,Vp)
mhh=project(p_0,Vp)
mhp=project(p_0,Vp)

# define boundaries
bot=CompiledSubDomain("abs(x[1])<tol && on_boundary",tol=tol)
sid=CompiledSubDomain("(abs(x[0])<tol || abs(x[0]-WW)<tol) && on_boundary",tol=tol,WW=WW)
top=CompiledSubDomain("abs(x[1]-HH)<tol && on_boundary",HH=HH,tol=tol)
# define boundary conditions
bcu_bot=DirichletBC(Vu,Constant((0.0,0.0)),bot)
bcu_sid=DirichletBC(Vu.sub(0),Constant(0.0),sid)
umax=0.04
load=Expression(("0.0*t","umax*t"),t=0.0,umax=umax,degree=1)
bcu_top=DirichletBC(Vu,load,top)
bcu=[bcu_bot,bcu_sid,bcu_top]
bcp_bot=DirichletBC(Vp,Constant(0.0),bot)
bcp_top=DirichletBC(Vp,Constant(0.0),top)
bcp=[]
# define domain to extract traction later
boundaries=MeshFunction("size_t",mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
top.mark(boundaries,1)
ds=Measure("ds")(subdomain_data=boundaries)
nnorm=FacetNormal(mesh)

# define functions
# strain matrix
def epsilon(u):
	return sym(grad(u))
# stress matrix
def sigma(u):
	return lmbda*tr(sym(grad(u)))*Identity(len(u))+2.0*mu*sym(grad(u))
# von Mises stress
def sigmavm(u,sig):
	return sqrt(3.0*inner(sig-tr(sig)*Identity(len(u))/3.0,sig-tr(sig)*Identity(len(u))/3.0)/2.0)
# strain energy
def energy(u):
	return inner(lmbda*tr(sym(grad(u)))*Identity(len(u))+2.0*mu*sym(grad(u)),sym(grad(u)))/2.0
# positive strain energy
def energyp(u):
    # (1) Miehe strain spectral decomposition, canot run in this code
    #return lmbda*((tr(sym(grad(u)))+abs(tr(sym(grad(u)))))/2.0)**2/2.0+mu*inner((sym(grad(u))+abs(sym(grad(u))))/2.0,(sym(grad(u))+abs(sym(grad(u))))/2.0)
    # (2) Amor strain deviatoric decomposition
    #epsd=sym(grad(u))-tr(sym(grad(u)))*Identity(len(u))/3.0
    #return k*((tr(sym(grad(u)))+abs(tr(sym(grad(u)))))/2.0)**2/2.0+mu*inner(epsd,epsd)
    # (3) expansion and compression comparison
    coep=(abs(div(u)+tol**3)+div(u)+tol**3)/2.0/abs(div(u)+tol**3)
    return coep*inner(lmbda*tr(sym(grad(u)))*Identity(len(u))+2.0*mu*sym(grad(u)),sym(grad(u)))/2.0
# negative strain energy
def energyn(u):
    # (1) Miehe strain spectral decomposition, canot run in this code
    #return lmbda*((tr(sym(grad(u)))-abs(tr(sym(grad(u)))))/2.0)**2/2.0+mu*inner((sym(grad(u))-abs(sym(grad(u))))/2.0,(sym(grad(u))-abs(sym(grad(u))))/2.0)
    # (2) Amor strain deviatoric decomposition
    #return k*((tr(sym(grad(u)))-abs(tr(sym(grad(u)))))/2.0)**2/2.0
    # (3) expansion and compression comparison
    coen=(abs(div(u)+tol**3)-div(u)-tol**3)/2.0/abs(div(u)+tol**3)
    return coen*inner(lmbda*tr(sym(grad(u)))*Identity(len(u))+2.0*mu*sym(grad(u)),sym(grad(u)))/2.0
# monotonically incresing phase
def comparelt(p,mhp):
	return conditional(le(p,mhp),mhp,p)
# monotonically incresing phase
def comparegt(p,mhp):
	return conditional(ge(p,mhp),mhp,p)

# settings for constructing variational form
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=4
parameters["linear_algebra_backend"]="PETSc"
ffc_options={"eliminate_zeros": True,\
			 "optimize": True,\
             "precompute_basis_const": True,\
             "precompute_ip_const": True}

# define displacement field
if ANISOSWT<0.5:
    Pu=(1.0-pold)**2*energy(unew)*dx
elif ANISOSWT>0.5:
    Pu=(1.0-pold)**2*energyp(unew)*dx+energyn(unew)*dx
Ru=derivative(Pu,unew,v)
Jacu=derivative(Ru,unew,u)

# define phase field AT1
if abs(ATSWT-1.0)<tol and ANISOSWT<0.5:
    mhh=comparelt(energy(uold),mhh)
    Pp=(1-pnew)**2*mhh*dx+3.0*Gc*(pnew/lc+lc*dot(grad(pnew),grad(pnew)))/8.0*dx
elif abs(ATSWT-1.0)<tol and ANISOSWT>0.5:
    mhh=comparelt(energyp(uold),mhh)
    Pp=(1-pnew)**2*mhh*dx+3.0*Gc*(pnew/lc+lc*dot(grad(pnew),grad(pnew)))/8.0*dx

# define phase field AT2
if abs(ATSWT-2.0)<tol and ANISOSWT<0.5:
    mhh=comparelt(energy(uold),mhh)
    Pp=(1-pnew)**2*mhh*dx+Gc*(pnew**2/lc+lc*dot(grad(pnew),grad(pnew)))/2.0*dx
elif abs(ATSWT-2.0)<tol and ANISOSWT>0.5:
    mhh=comparelt(energyp(uold),mhh)
    Pp=(1-pnew)**2*mhh*dx+Gc*(pnew**2/lc+lc*dot(grad(pnew),grad(pnew)))/2.0*dx

# define phase field AT3
if abs(ATSWT-3.0)<tol and ANISOSWT<0.5:
    mhh=comparelt(energy(uold),mhh)
    Pp=(1-pnew)**2*mhh*dx+Gc*((xixi*pnew+(1.0-xixi)*pnew**2)/lc+lc*dot(grad(pnew),grad(pnew)))/caca*dx
elif abs(ATSWT-3.0)<tol and ANISOSWT>0.5:
    mhh=comparelt(energyp(uold),mhh)
    Pp=(1-pnew)**2*mhh*dx+Gc*((xixi*pnew+(1.0-xixi)*pnew**2)/lc+lc*dot(grad(pnew),grad(pnew)))/caca*dx

# define phase field AT4 and AT5
if abs(ATSWT-4.0)<tol:
    beta2=-sqrt(3.0)*(1.0+delta)*(sigc+sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sqrt(3.0)*lmbda*(sigc+sigt)/2.0/mu/(3.0*lmbda+2.0*mu)+sqrt(3.0)*(sigc+sigt)/2.0/(3.0*lmbda+2.0*mu)
    beta1=-(1.0+delta)*(sigc-sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc-lmbda*(sigc-sigt)/2.0/mu/(3.0*lmbda+2.0*mu)-(sigc-sigt)/2.0/(3.0*lmbda+2.0*mu)
    beta0=delta*3.0*Gc/8.0/lc
    ce=(1.0-pold)**2*beta2*sigmavm(uold,sigma(uold))/sqrt(3.0)+(1.0-pold)**2*beta1*tr(sigma(uold))+beta0
    Pp=(1.0-pnew)**2*energy(uold)*dx+3.0*Gc*(pnew/lc+lc*dot(grad(pnew),grad(pnew)))/8.0*dx-(1.0-pnew)*ce*dx
elif abs(ATSWT-5.0)<tol:
    beta2=-sqrt(3.0)*(1.0+delta)*(sigc+sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sigt/2.0/sqrt(3.0)/(3.0*lmbda+2.0*mu)+sigt/2.0/sqrt(3.0)/mu
    beta1=-(1.0+delta)*(sigc-sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sigt/6.0/(3.0*lmbda+2.0*mu)+sigt/6.0/mu
    beta0=delta*3.0*Gc/8.0/lc
    ce=(1.0-pold)**2*beta2*sigmavm(uold,sigma(uold))/sqrt(3.0)+(1.0-pold)**2*beta1*tr(sigma(uold))+beta0+(1.0-pold)*(1.0-abs(tr(sigma(uold))+tol**3)/(tr(sigma(uold))+tol**3))*((sigmavm(uold,sigma(uold)))**2/3.0/2.0/mu+(tr(sigma(uold)))**2/6.0/(3.0*lmbda+2.0*mu))
    Pp=(1.0-pnew)**2*energy(uold)*dx+3.0*Gc*(pnew/lc+lc*dot(grad(pnew),grad(pnew)))/8.0*dx-(1.0-pnew)*ce*dx

# bound phase in [0,1]
# method 1: penalty method to bound phase in [0,1]
#pencon=5000.0
#pen=pencon*((abs(pnew-0.0)-(pnew-0.0))**2+(abs(1.0-pnew)-(1.0-pnew))**2)*dx
#Rp=derivative(Pp+pen,pnew,q)
# method 2: problem_p.set_bounds(pmin,pmax) to bound phase in [0,1], see below later
Rp=derivative(Pp,pnew,q)
Jacp=derivative(Rp,pnew,p)

# define the solver parameters
set_log_level(40)# warning level is 30, error level is 40
snes_solver_parameters={"nonlinear_solver": "snes",\
                        "snes_solver": {"error_on_nonconvergence": False,\
                                        "line_search": "basic",\
                                        "linear_solver": "cg",\
                                        "maximum_iterations": 100,\
                                        "preconditioner": "amg",\
                                        "report": False}}

# define solution time and the number of steps for the problem
T=1.0# total time
numstep=500# total number of steps
dt=T/numstep# delat t at each step
t=0.0# initial time
nstep=0# step counter
b=PETScVector()# memory allocation for b vector

# create file to save variable
vtk_displacement=File("./Results/displacement.pvd")
vtk_phase=File("./Results/phase.pvd")
vtk_mhh=File("./Results/mhh.pvd")
vtk_stress=File("./Results/stress.pvd")
fname=open('./Results/AForceDisplacement.txt','w')

# calculation
while t<T and nstep<500:
    load.t=t
    niter=1
    err=1
    while niter<10 and err>1.0E-5:
# displacement field
        problem_u=NonlinearVariationalProblem(Ru,unew,bcu,J=Jacu)
        solver_u=NonlinearVariationalSolver(problem_u)
        solver_u.parameters.update(snes_solver_parameters)
        (iter,converged)=solver_u.solve()
# phase field
        problem_p=NonlinearVariationalProblem(Rp,pnew,bcp,J=Jacp)
        problem_p.set_bounds(pmin,pmax)
        solver_p=NonlinearVariationalSolver(problem_p)
        solver_p.parameters.update(snes_solver_parameters)
        (iter,converged)=solver_p.solve()
# check the number of iterations and error
        niter+=1
        b=assemble(-Ru,tensor=b)
        for bc in bcu:
            bc.apply(b)
        err=b.norm('l2')
# update solution
    uold.assign(unew)
    mhp=comparelt(pnew,mhp)
    mhp=comparelt(mhp,pmin)
    mhp=comparegt(mhp,pmax)
    mhp=project(mhp,Vp)
    pold.assign(mhp)
    mhhoutput=project(mhh,Vp)
    stress=(1-pold)**2*sigmavm(uold,sigma(uold))
    stress=project(stress,Vp)
    force=dot((1-pold)**2*sigma(uold),nnorm)
    forcey=force[1]*ds(1)
# save variables
    vtk_displacement << (uold,t)
    vtk_phase << (pold,t)
    vtk_mhh << (mhhoutput,t)
    vtk_stress << (stress,t)# only coreect for isotropic case or case 4&5
    fname.write(str(pold(0.5,0.5))+"\t")
    fname.write(str(umax*t)+"\t")
    fname.write(str(assemble(forcey))+"\n")
# update time and step counter
    t+=dt
    nstep+=1
fname.close()

