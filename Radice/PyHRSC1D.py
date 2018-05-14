"""
PyHRSC1D: a simple python library for the solution of balance laws

Copyright (C) 2018 David Radice <dradice@astro.princeton.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import abc
import sys

from math import copysign, fabs
# Numba seems to be broken in Anaconda-5.1, so I have disabled it
#from numba import jit, jitclass, float64, int32, prange
prange = range
import numpy as np
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class CLaw(abc.ABC):
# -----------------------------------------------------------------------------
  """
  Abstract class representing a conservation law

    \partial_t F^0(u) + \partial_x F^x(u) = S(u)

  * u   are the primitive variables
  * F^0 are the conserved variables
  * F^x are the fluxes
  * S   are the source terms

  * nvars is a static variable with the number of variables to be evolved
  * varnames is a list of names for the primitive variables
  """
  nvars    = 0
  varnames = []

  @abc.abstractmethod
  def prim2con(self, prim, cons, argv=None):
    """
    Compute F^0 given u

    * prim  : primitive variables
    * cons  : conservative variables
    * argv  : extra parameters (optional)
    """
    pass

  @abc.abstractmethod
  def con2prim(self, cons, prim, argv=None):
    """
    Compute u given F^0

    * prim  : primitive variables
    * cons  : conservative variables
    * argv  : extra parameters (optional)
    """
    pass

  @abc.abstractmethod
  def speeds(self, prim, cons, char, argv=None):
    """
    Get characteristic speeds

    * prim  : primitive variables
    * cons  : conservative variables
    * char  : characteristic speeds
    * argv  : extra parameters (optional)
    """
    pass

  @abc.abstractmethod
  def fluxes(self, prim, cons, flux, argv=None):
    """
    Compute F^x given u

    * prim  : primitive variables
    * cons  : conservative variables
    * flux  : fluxes
    * argv  : extra parameters (optional)
    """
    pass

  @abc.abstractmethod
  def sources(self, prim, cons, source, argv=None):
    """
    Compute :math:`S` given `u`

    * prim   : primitive variables
    * cons  : conservative variables
    * source : sources
    * argv   : extra parameters (optional)
    """
    pass
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#@jitclass([
#    ('xf', float64[:]),
#    ('xc', float64[:]),
#    ('ng', int32),
#  ])
class Grid(object):
# -----------------------------------------------------------------------------
  """
  A simple 1D grid

  Attributes:

  xf : location of the cell faces
  xc : location of the cell centers
  ng : number of ghost zones on each side


  The grid is made of cells that look like this

      xf[0]  xc[0] xf[1] xc[1] xf[2] ...
        |-----x-----|-----x-----|-----x-----|...

  Grid functions are stored with this convention

        uR[i-1] uL[i]  u[i] uR[i] uL[i+1]

      ...------|--------X--------|--------X---...

  The overall structure of the grid is

   ghost  xmin                               xmax  ghost
      \    |                                   |    /
       \   |        I N T E R I O R            |   /
        |  |                                   |  |
     |--X--|--X--|--X--|--X--|--X--| ... |--X--|--X--|
  """
  def __init__(self, xf, ng):
    """
    Initializes the grid

    * xf : location of the cell faces
    * ng : number of ghost zones on either side
    """
    self.xf = xf
    self.xc = 0.5*(self.xf[:-1] + self.xf[1:])
    self.ng = ng
def make_uniform_grid(xmin, xmax, ncells, nghost=2):
  """
  Generates a uniform grid

  * xmin   : lower boundary of the grid
  * xmax   : upper boundary of the grid
  * ncells : number of cells in the grid interior
  * nghost : number of ghost zones on each side
  """
  dx = (xmax - xmin)/(ncells - 1)
  xf = xmin + dx*np.arange(-nghost, ncells+nghost)
  return Grid(xf, nghost)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class BoundaryGeneric(abc.ABC):
# -----------------------------------------------------------------------------
  """
  Abstract class representing a boundary condition
  """
  @abc.abstractmethod
  def apply(self, grid, cons, argv=None):
    """
    Boundary conditions are applied to the conserved variables
    """
    pass
class BoundaryStatic(BoundaryGeneric):
  """
  Class implementing trivial boundary conditions
  """
  def apply(self, grid, cons, argv=None):
    pass
class BoundaryPeriodic(BoundaryGeneric):
  """
  Class implementing periodic boundary conditions
  """
  def apply(self, grid, cons, argv=None):
    np = grid.xc.shape[0]
    for i in range(grid.ng):
      cons[:,i] = cons[:,np-2*grid.ng+i]
      cons[:,np-grid.ng+i] = cons[:,grid.ng+i]
class BoundaryFlat(BoundaryGeneric):
  """
  Class implementing flat boundary conditions
  """
  def apply(self, grid, cons, argv=None):
    for i in range(grid.ng):
      cons[:,i] = cons[:,grid.ng]
      cons[:,-i-1] = cons[:,-grid.ng-1]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class InitDataGeneric(abc.ABC):
# -----------------------------------------------------------------------------
  """
  Abstract class representing an initial data setter
  """
  @abc.abstractmethod
  def apply(self, grid, prim, argv=None):
    """
    The initial condition class should set the primitive variables only
    """
    pass
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------
#@jit(nopython=True)
def minmod(xm, xc, xp, theta=2.0):
  """
  The classical minmod limiter
  """
  if(xm*xc > 0.0 and xc*xp > 0.0):
    return copysign(min(theta*fabs(xm), fabs(xc), theta*fabs(xp)), xm)
  else:
    return 0.0

#@jit(nopython=True, parallel=True)
def minmod_reconstruct(grid, u, uL, uR, theta=2.0):
  """
  Reconstruct a function using the minmod method

  * grid : computational grid
  * u    : variable to reconstruct
  * uL   : reconstructed values at the left interface 
  * uR   : reconstructed values at the right interface 
  """
  assert(u.shape[1] == uL.shape[1] == uR.shape[1] == grid.xc.shape[0])
  uL[:, 0] = uR[:, 0] = u[:, 0]
  uL[:,-1] = uR[:,-1] = u[:,-1]
  for ix in prange(1, grid.xc.shape[0]-1):
    for iu in range(u.shape[0]):
      sm = (u[iu,ix+0] - u[iu,ix-1])/(grid.xc[ix+0] - grid.xc[ix-1])
      sc = (u[iu,ix+1] - u[iu,ix-1])/(grid.xc[ix+1] - grid.xc[ix-1])
      sp = (u[iu,ix+1] - u[iu,ix-0])/(grid.xc[ix+1] - grid.xc[ix-0])
      slope = minmod(sm, sc, sp, theta)
      uL[iu,ix] = u[iu,ix] + slope*(grid.xf[ix+0] - grid.xc[ix])
      uR[iu,ix] = u[iu,ix] + slope*(grid.xf[ix+1] - grid.xc[ix])

#@jit(nopython=True, parallel=True)
def HLLE_flux(numflux,
    primL, consL, fluxL, charL,
    primR, consR, fluxR, charR):
  """
  Computes the numerical flux using the HLLE formula

  * primL : primitive variables at the left interface
  * consL : conservative variables at the left interface
  * fluxL : fluxes at the left interface
  * charL : characteristic velocities at the left interface
  * primR : primitive variables at the right interface
  * consR : conservative variables at the right interface
  * fluxR : fluxes at the right interface
  * charR : characteristic velocities at the right interface

  Note the different convention for "L" and "R" compared to the reconstruction!

  Einfeldt, B., On Godunov type methods for the Euler equations with a general
  equation of state, Shock tubes and waves; Proceedings of the Sixteenth
  International Symposium, Aachen, Germany, July 26--31, 1987. VCH Verlag,
  Weinheim, Germany
  """
  assert(numflux.shape == primL.shape == primR.shape)
  assert(primL.shape == consL.shape == fluxL.shape == charL.shape)
  assert(primR.shape == consR.shape == fluxR.shape == charR.shape)
  for ix in prange(numflux.shape[1]):
    alm = min(np.min(charL[:,ix]), np.min(charR[:,ix]))
    arp = max(np.max(charL[:,ix]), np.max(charR[:,ix]))
    numflux[:,ix] = (arp*fluxL[:,ix] - alm*fluxR[:,ix] +
        alm*arp*(consR[:,ix] - consL[:,ix]))/(arp - alm)

#@jit(nopython=True, parallel=True)
def Rusanov_flux(numflux,
    primL, consL, fluxL, charL,
    primR, consR, fluxR, charR):
  """
  Computes the numerical flux using the Rusanov flux formula

  * grid  : computational grid
  * primL : primitive variables at the left interface
  * consL : conservative variables at the left interface
  * fluxL : fluxes at the left interface
  * charL : characteristic velocities at the left interface
  * primR : primitive variables at the right interface
  * consR : conservative variables at the right interface
  * fluxR : fluxes at the right interface
  * charR : characteristic velocities at the right interface

  Note the different convention for "L" and "R" compared to the reconstruction!

  Kurganov and Tadmor, New High-Resolution Central Schemes for Nonlinear
  Conservation Laws and Convection–Diffusion Equations,
  J. Comput. Phys 160:241-282 (2000)
  """
  assert(numflux.shape == primL.shape == primR.shape)
  assert(primL.shape == consL.shape == fluxL.shape == charL.shape)
  assert(primR.shape == consR.shape == fluxR.shape == charR.shape)

  for ix in prange(numflux.shape[1]):
    s = max(fabs(charL[0,ix]), fabs(charR[0,ix]))
    for ic in range(1, charL.shape[0]):
      s = max(s, fabs(charL[ic,ix]), fabs(charR[ic,ix]))
    numflux[:,ix] = 0.5*((fluxL[:,ix] + fluxR[:,ix]) -
        s*(consR[:,ix] - consL[:,ix]))

class KurganovTadmorSolver(object):
  """
  Solve a conservation law with the Kurganov-Tadmor scheme.
  It uses 2nd order TVD reconstruction of the primitive quantities and
  Runge-Kutta 2 time integration.

  Kurganov and Tadmor, New High-Resolution Central Schemes for Nonlinear
  Conservation Laws and Convection–Diffusion Equations,
  J. Comput. Phys 160:241-282 (2000)
  """
  def __init__(self, grid, claw, ic, bc, dtype=np.float64,
      minmod_theta=2.0, flux_formula="HLLE"):
    """
    Initializes the HRSC solver

    * grid : computational grid
    * claw : conservation law to solve
    * ic   : initial conditions
    * bc   : boundary conditions

    * minmod_theta :
        0 --> piecewise constant reconstruction (most dissipative)
        1 --> classical minmod scheme
        2 --> monotonized central limiter (Van Leer)
    * flux_formula :
        HLLE --> Harten, Lax, van Leer and Einfeldt approximate Riemann solver
        Rusanov --> Rusanov flux formula (also known as local Lax-Friedrichs)

    """
    assert(isinstance(bc, BoundaryGeneric)) # bc must be a class derived from boundary.generic
    assert(isinstance(claw, CLaw)) # claw must be a class derived from CLaw
    assert(isinstance(grid, Grid)) # grid must be an object of class Grid
    assert(isinstance(ic, InitDataGeneric)) # ic must be a class derived from initdata.Generic

    self.grid   = grid
    self.claw   = claw
    self.ic     = ic
    self.bc     = bc

    assert(0 <= minmod_theta <= 2) # theta values outside of [0,2] are unstable
    self.minmod_theta = minmod_theta
    assert(flux_formula in ["HLLE", "Rusanov"]) # supported flux formulas
    self.flux_formula = flux_formula

    # Variables stored at cell centers
    self.prim   = np.zeros((self.claw.nvars, self.grid.xc.shape[0]), dtype=dtype)
    self.cons   = [np.zeros_like(self.prim) for i in range(3)]
    self.char   = np.zeros_like(self.prim)
    self.source = np.zeros_like(self.prim)
    self.rhs    = np.zeros_like(self.prim)

    # Variables stored at the cell interfaces
    self.primL  = np.zeros_like(self.prim)
    self.primR  = np.zeros_like(self.prim)
    self.consL  = np.zeros_like(self.prim)
    self.consR  = np.zeros_like(self.prim)
    self.fluxL  = np.zeros_like(self.prim)
    self.fluxR  = np.zeros_like(self.prim)
    self.charL  = np.zeros_like(self.prim)
    self.charR  = np.zeros_like(self.prim)

    # Numerical fluxes
    self.nflux  = np.zeros((self.claw.nvars, self.grid.xf.shape[0]), dtype=dtype)

  def initialize(self, argbc=None, argclaw=None, argic=None):
    """
    Sets up the initial conditions

    * argbc   : this is passed to self.bc
    * argclaw : this is passed to self.claw
    * argic   : this is passed to the ic generator
    """
    self.ic.apply(self.grid, self.prim, argic)
    self.claw.prim2con(self.prim, self.cons[0], argclaw)
    self.bc.apply(self.grid, self.cons[0], argbc)
    self.claw.con2prim(self.cons[0], self.prim, argclaw)

  def compute_rhs(self, argclaw=None, argclawL=None, argclawR=None):
    """
    Computes the RHS of the conservation law

    * argclaw   : this is passed to self.claw
    * argclawL  : this is passed to self.claw when computing things on the left faces
    * argclawR  : this is passed to self.claw when computing things on the right faces

    NOTE: this method is not meant to be called by the end user
    """
    minmod_reconstruct(self.grid, self.prim, self.primL, self.primR,
        self.minmod_theta)

    self.claw.prim2con(self.primL, self.consL, argclawL)
    self.claw.prim2con(self.primR, self.consR, argclawR)
    self.claw.fluxes(self.primL, self.consL, self.fluxL, argclawL)
    self.claw.fluxes(self.primR, self.consR, self.fluxR, argclawR)
    self.claw.speeds(self.primL, self.consL, self.charL, argclawL)
    self.claw.speeds(self.primR, self.consR, self.charR, argclawR)

    self.nflux[:,0] = 0.
    self.nflux[:,-1] = 0.
    if self.flux_formula == "Rusanov":
      Rusanov_flux(self.nflux[:,1:-1],
          self.primR[:,:-1], self.consR[:,:-1], self.fluxR[:,:-1], self.charR[:,:-1],
          self.primL[:,1:], self.consL[:,1:], self.fluxL[:,1:], self.charL[:,1:])
    elif self.flux_formula == "HLLE":
      HLLE_flux(self.nflux[:,1:-1],
          self.primR[:,:-1], self.consR[:,:-1], self.fluxR[:,:-1], self.charR[:,:-1],
          self.primL[:,1:], self.consL[:,1:], self.fluxL[:,1:], self.charL[:,1:])
    else:
      raise ValueError("Unkown flux formula: {}".format(self.scheme_flux))
    self.rhs[:] = 0.
    self.rhs[:] -= np.diff(self.nflux, axis=1)
    self.rhs[:] /= np.diff(self.grid.xf)[np.newaxis,:]

    self.claw.sources(self.prim, self.cons, self.source, argclaw)
    self.rhs[:] += self.source[:]

    self.rhs[:,:self.grid.ng] = 0.
    self.rhs[:,-self.grid.ng:] = 0.

  def estimate_timestep(self, cfl=1.0, argclaw=None):
    """
    * cfl     : CFL condition to be used for the next timestep
    * argclaw : this is passed to self.claw
    
    Returns the timestep
    """
    self.claw.speeds(self.prim, self.cons, self.char, argclaw)
    return cfl*np.min(np.diff(self.grid.xf)/np.max(np.abs(self.char), axis=0))

  def step(self, dt, argbc=None, argclaw=None, argclawL=None, argclawR=None):
    """
    Advances the evolution using the strongly stability preserving (SSP) Runge-Kutta 2 scheme

        u^*     = u^k + dt RHS[u^k]
        u^{k+1} = 1/2 (u^k + u^* + dt RHS[u^*]) 

    * dt        : timestep to be taken
    * argbc     : this is passed to self.bc
    * argclaw   : this is passed to self.claw
    * argclawL  : this is passed to self.claw when computing things on the left faces
    * argclawR  : this is passed to self.claw when computing things on the right faces
    """
    # First step
    self.compute_rhs(argclaw=argclaw, argclawL=argclawL, argclawR=argclawR)
    self.cons[1] = self.cons[0] + dt*self.rhs
    self.bc.apply(self.grid, self.cons[1], argbc)
    self.claw.con2prim(self.cons[1], self.prim, argclaw)

    # Second step
    self.compute_rhs(argclaw=argclaw, argclawL=argclawL, argclawR=argclawR)
    self.cons[2] = 0.5*(self.cons[0] + self.cons[1] + dt*self.rhs)
    self.bc.apply(self.grid, self.cons[2], argbc)
    self.claw.con2prim(self.cons[2], self.prim, argclaw)

    # Rotate timelevels
    u = self.cons[0]
    self.cons[0] = self.cons[2]
    self.cons[2] = self.cons[1]
    self.cons[1] = u
# -----------------------------------------------------------------------------

# vim: set ft=python ts=2 sw=2 expandtab :
