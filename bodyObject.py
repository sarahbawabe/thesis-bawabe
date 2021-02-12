"""
created: January 31 2021

@author: sarahbawabe

This script models the bodyObject class, and also creates helper methods for
creating random objects (in position, mass, and velocity) in space.

"""
import random
import numpy as np

''' Models a bodyObject, which has knowledge of its mass, position coordinates,
    and velocity vector. Can be any arbitrary dimensionality, but the dimensionality
    of coords and vels must be the same.
    :: mass     : mass of object (in kg)
    :: coords   : N-dimensional coordinate position of object in space
    :: vels     : N-dimensional velocity vector of object '''
class bodyObject:
    def __init__(self, mass, coords, vels):
        self.mass = mass
        self.coords = coords
        self.vels = vels

def generate_random_obj(ndim):
    rand1 = random.random()
    rand2 = int(random.triangular(20,30,22)) # random number between 20 and 30, with most numbers falling around 23
    mass = rand1 * (10**rand2)
    coords = np.zeros(ndim)
    vels = np.zeros(ndim)

    # since we don't want massive objects to also move really fast, we divide
    # this into an if-else to limit the randomness to more reasonable outputs
    if mass > 10**26:
        for i in range(ndim):
            # get random position
            rand1 = random.random() * 10
            rand2 = int(random.triangular(-3,3))
            coords[i] = rand1 * 10**rand2
            # get random velocity
            rand1 = random.random() * 10
            rand2 = int(random.triangular(-10,-6,-8))
            vels[i] = rand1 * 10**rand2
    else:
        for i in range(ndim):
            # get random position
            rand1 = random.random() * 10
            rand2 = int(random.triangular(-10,10))
            coords[i] = rand1 * 10**rand2
            # get random velocity
            rand1 = random.random() * 10
            rand2 = int(random.triangular(-7,-2,-4))
            vels[i] = rand1 * 10**rand2

    # print("MASS:", mass)
    # print("COORDS:", coords)
    # print("VEL:", vels)
    return bodyObject(mass,coords,vels)

''' Generates a list of random bodyObjects of size N in ndim dimensional space. '''
def generate_rand_obj_list(N,ndim):
    # N : desired size out of output list
    # ndim : dimensionality of space
    obj_list = []
    for i in range(N):
        obj_list.append(generate_random_obj(ndim))
    return obj_list

''' Converts a list of mass data, coordinate data, and velocity data into
    a list of bodyObjects. '''
def convert_to_obj_list(masses, coords, vels):
    obj_list = []
    for i in range(coords.shape[0]):
        mass = masses[i]
        pos = coords[i]
        vel = vels[i]
        body = bodyObject(mass,pos,vel)
        obj_list.append(body)
    return obj_list
