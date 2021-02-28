"""
Created: January 2021

@author: sarahbawabe
@author: joshualamstein

This script was inspired by and largely based off of joshualamstein's code for
an N-Body simulation. This code is optimized for better efficiency and extensibility
by utilizing numpy matrices and matrix operations to avoid excess runtime and memory.

Instead of making this file one runnable code script, I chose to create a class
structure, where each body is a bodyObject (see bodyObject.py), and this simulation
class takes in a list of bodyObjects, and a coordinate dimensionality (meaning that
this simulation is no longer limited to 3-dimensional space).

"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from code.starData import *
import code.bodyObject as bObj

class NBody:

    def __init__(self, bodies, ndim, iters):
        self.bodies = bodies
        self.N = len(bodies)
        self.ndim = ndim
        self._iters = iters # number of iterations
        self.coords = []
        self.masses = []
        self.vels = []

        for body in bodies:
            self.coords.append(body.coords)
            self.masses.append(body.mass)
            self.vels.append(body.vels)

        self._mTot = np.sum(self.masses)

        # conversion factors
        self._solar = 1.988544*10**30 # in kg
        self._year = 3.154*10**7 # year in seconds
        self._AU = 1.496*10**11 # in m
        self._dayToYear = 365.25

        # convert gravitational constant to astronomical units (AU) and solar masses
        G_old = 6.67259e-11 # in m^3/(kg-s^2)
        self._G = G_old*(1/self._AU)**3/(1/self._solar*(1/(self._year))**2) # gravitational constant

        self._dt = .01 # time step (years)

        # initialize and update position & velocity matrices
        self.pos_matrix = np.zeros((self.N, self.ndim, self._iters+1))
        self.vel_matrix = np.zeros((self.N, self.ndim, self._iters+1))
        for i in range(self.N):
            self.pos_matrix[i,:,0] = self.coords[i]
            self.vel_matrix[i,:,0] = self.vels[i] * self._dayToYear

        self._timeSpace = np.arange(0,self._dt*self._iters,self._dt)
        self._timePeriod = np.arange(0,self._dt*10,self._dt)
        self._deltaEnergy = np.zeros(self._iters) # record change in energy to check validity of code
        self._h = np.zeros((self._iters,self.N,self.ndim)) # variable for momentum
        self._deltaH = np.zeros(self._iters) # change in momentum
        self._totalH = np.zeros((self._iters,self.ndim)) # total momentum
        self._hChange = np.zeros(self._iters) # placeholder for momentum to check code
        self._hRatio = 0
        self._hTotalMag = 0
        self._hTotComb = 0
        self._energy = np.zeros(self._iters)

        self._mag = np.zeros((self.N,self.N)) # distances between bodies

        self.acc_matrix = np.zeros((self.N, self.ndim, self._iters))
        self.acc_pull_matrix = np.zeros((self.N, self.ndim, self._iters+1))

        self._vBary = np.zeros((self._iters,self.ndim)) # velocity barycenter
        self._rBary = np.zeros((self._iters,self.ndim)) # barycenter position


    #==============================================================================
    #  helper function definitions
    #==============================================================================

    def magnitude(self,coords):
        # coords : coordinate array of body
        mag = 0
        for coord in coords:
            mag += coord**2
        return sqrt(mag)

    def momentum(self,m,coords,vels):
        # m : mass of second body
        # coords : coordinate array of body's position
        # vels : coordinate array of body's velocity
        return m * np.cross(coords,vels,axis=0);

    ''' Update position of planet with Leap frog method, where
        2*dt is the length of the time step for leap frog'''
    def position(self,coords,vels):
        # coords : coordinate array of body's position
        # vels : coordinate array of body's velocity
        return coords + (vels * 2 * self._dt)

    def position_euler(self,coords,vels): # Get position using the Euler method
        # coords : coordinate array of body's position
        # vels : coordinate array of body's velocity
        return coords + (vels * self._dt)

    def acceleration(self,coords1,coords2,mag,m): # Get acceleration of celestial body
        # coords1 : coordinate array of first body's position
        # coords2 : coordinate array of second body's position
        # mag : magnitude of distance between two bodies
        # m : mass of second body
        return (self._G * m * np.subtract(coords2,coords1)/mag**self.ndim)

    def velocity(self,vels,accs): # get velocity of planet with leap frog method
        # vels : coordinate array of body's velocity
        # accs : coordinate array of body's acceleration
        return vels + (2.0 * self._dt * accs)

    def velocity_euler(self,vels,accs): # euler method
        # vels : coordinate array of body's velocity
        # accs : coordinate array of body's acceleration
        return vels + (accs * self._dt)

    def kinetic(self,m,coords):
        # m : mass of second body
        # coords : coordinate array of body's position
        return 0.5 * m * np.sum(coords**2)

    def potential(self,m1,m2,mag): # Gravitational Potential
        # m1 : mass of planet
        # m2 : mass of other object (probably the Sun)
        # mag : magnitude displacement
        U = (-1 * self._G * m1 * m2) / mag
        return U

    def barycenter(self,m,coords): # Find Barycenter
        # m : mass of planet
        # coords : coordinate array of body's position
        return (np.multiply(m,coords)) / self._mTot

    def perform_simulation(self): # performs nBody simulation on given data

        for i in range(self._iters):

            #==============================================================================
            # acceleration
            #==============================================================================

            # get distance between all included celestial bodies
            for j in range(self.N):
                for k in range(j,self.N,1):
                    temp_coords = np.subtract(self.pos_matrix[j,:,i], self.pos_matrix[k,:,i])
                    self._mag[j][k] = self.magnitude(temp_coords)
                if j > 0:
                    for p in range(j):
                        self._mag[j][p] = self._mag[p][j] #symmetry, distance between one planet is the same as the distance from the other

            # calculate acceleration due to gravity
            for j in range(self.N):
                for k in range(self.N):
                    if j != k:
                        self.acc_matrix[j,:,i] += self.acceleration(self.pos_matrix[j,:,i],self.pos_matrix[k,:,i],self._mag[j][k],self.masses[k])

            # print("ACC", self.acc_matrix[:,:,i])
            # print("=========")

            #==============================================================================
            # velocity
            #==============================================================================

            # at the beginning of the array, you can't use leap frog, so it suffices to use the 1st order Euler method
            if i == 0:
                for j in range(self.N):
                    self.vel_matrix[j,:,i+1] = self.velocity_euler(self.vel_matrix[j,:,i], self.acc_matrix[j,:,i])

            else: # Use leap frog method to update velocity
                for j in range(self.N):
                    self.vel_matrix[j,:,i+1] = self.velocity(self.vel_matrix[j,:,i-1], self.acc_matrix[j,:,i])

            # print("VEL", self.vel_matrix[:,:,i])
            # print("=========")

            #==============================================================================
            # Energy
            #==============================================================================
            # Because energy is conserved,
            # the change in energy should be due to round off error in the simulation.

            KEhold = 0; # kinetic energy placeholder
            PEhold = 0; # potential energy placeholder

            for j in range(self.N):
                KEhold += self.kinetic(self.masses[j], self.vel_matrix[j,:,i])

                for k in range(j,self.N):
                    if j != k:
                        PEhold += self.potential(self.masses[j],self.masses[k],self._mag[j][k])

            self._energy[i] = KEhold + PEhold
            self._deltaEnergy[i] = abs(self._energy[i])-abs(self._energy[0])

            rAdd = 0
            vAdd = 0

            ##==============================================================================
            ## check barycenter which should be 0
            ##==============================================================================
            for j in range(self.N):
                rAdd = self.barycenter(self.masses[j], self.pos_matrix[j,:,i])
                self._rBary[i] += rAdd
                vAdd = self.barycenter(self.masses[j], self.vel_matrix[j,:,i])
                self._vBary[i] += vAdd

            rAdd = 0
            vAdd = 0

            #==============================================================================
            #  new position
            #==============================================================================
            # get position with Euler method for i==0
            if i == 0:
                for j in range(self.N):
                    self.pos_matrix[j,:,i+1] = self.position_euler(self.pos_matrix[j,:,i], self.vel_matrix[j,:,i])

            # update position with leap frog method
            else:
                for j in range(self.N):
                    self.pos_matrix[j,:,i+1] = self.position(self.pos_matrix[j,:,i-1], self.vel_matrix[j,:,i])

            # print("POS", self.pos_matrix[:,:,i])
            # print("=========")

            ##==============================================================================
            ## momentum (no mass)
            ##==============================================================================
            for j in range(self.N):
                self._h[i][j] = self.momentum(self.masses[j],self.pos_matrix[j,:,i],self.vel_matrix[j,:,i])
                self._totalH[i] = np.sum(self._h[i],axis=0)

        self._hTotalMag = np.linalg.norm(self._totalH)
        self._hTotComb = np.linalg.norm(self._totalH,axis=1)
        self._deltaH = self._hTotComb - self._hTotComb[0]
        self._hRatio = self._deltaH / self._hTotalMag


    def plot(self):

        #==============================================================================
        # Plot the orbits of the planets. The sun gets a little covered up by the orbit
        # of Mercury and Venus.
        #==============================================================================
        datestr = "June 20, 1988 - June 20, 2028"

        plt.figure(0)
        for i in range(self.N):
            plt.plot(self.pos_matrix[i,0,:],self.pos_matrix[i,1,:],color = [(i%2)/2,(i%3)/3,(((self.N-1)+i)%4)/4])
        # plt.legend(['Sun','Mercury','Venus','Earth','Mars','Jupiter','Saturn','Neptune','Uranus','Pluto'])
        plt.title(str(self.N) + " Body Simulation, "+datestr,{'size':'14'});
        plt.grid('on')
        plt.axis('equal')
        plt.xlabel("x (AU)", {'size':'14'});
        plt.ylabel("y (AU)", {'size':'14'});
        plt.show()

        #==============================================================================
        # Plot the change in total energy of the solar system. The energy oscillates due to machine error.
        # If the time step is smaller, the change in total energy reduces. For 10 years, with a time step of .001 years for
        # 10,000 iterations, the change in energy was order 10^7.
        #==============================================================================

        plt.figure(1)
        plt.plot(self._timeSpace,self._deltaEnergy,'k')
        plt.title("Change in Energy, "+datestr,{'size':'14'});
        plt.xlabel("dt (Year)", {'size':'14'});
        plt.ylabel("$M_J * AU^2 / (2\pi*year)^2$", {'size':'14'});
        plt.show();

        #==============================================================================
        #  Plot the change in momentum divided by total momentum of the solar system. Again, the error is due to machine error.
        #  The machine error doesn't go away over time, but it oscillates, which I assume means the errors cancel out, most likely
        #  due to the symmetry and oscillation of the orbits of the planets and that the leap frog method conserves energy.
        #  The change in momentum is not quite centered at zero, which suggests the barycenter is not exactly zero.
        #==============================================================================

        plt.figure(2)
        plt.plot(self._timeSpace, self._hRatio,'k')
        plt.title("Angular Momentum, \n"+datestr,{'size':'14'});
        plt.xlabel("dt (Year)", {'size':'14'});
        plt.ylabel("Change in Momentum / Total Momentum", {'size':'14'});
        plt.show();

def main():
    #==============================================================================
    #  RUN SCRIPT
    #==============================================================================
    # obj_list = bObj.convert_to_obj_list(m, coords_matrix, vels_matrix)
    obj_list = bObj.generate_rand_obj_list(N=10,ndim=3)
    nBody = NBody(obj_list, ndim=3,iters=4000)
    nBody.perform_simulation()
    nBody.plot()


if __name__ == '__main__':
    main()
