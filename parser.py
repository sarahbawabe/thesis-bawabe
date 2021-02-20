import readgadget
import readsnap as rs
import graph as g
import kdtree

def read_snapshot(snapshot, printOut=False):
    # input files
    ptype = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
    header = rs.snapshot_header(snapshot) # reads snapshot header

    coords = rs.read_block(snapshot,"POS ") # reads mass for particles of type 5, using block names should work for both format 1 and 2 snapshots
    ids = rs.read_block(snapshot,"ID  ") # reads mass for particles of type 5, using block names should work for both format 1 and 2 snapshots

    if printOut == True:
        print("coordinates for", coords.size, "particles read")
        print(coords[0:10])
        print("ids for", ids.size, "particles read")
        print(ids[0:10])

    return [ids, coords]

    # # read header
    # header   = readgadget.header(snapshot)
    # BoxSize  = header.boxsize/1e3  #Mpc/h
    # Nall     = header.nall         #Total number of particles
    # Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    # # Omega_m  = header.omega_m      #value of Omega_m
    # # Omega_l  = header.omega_l      #value of Omega_l
    # # h        = header.hubble       #value of h
    # # redshift = header.redshift     #redshift of the snapshot
    # # Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)
    #
    # # read positions, velocities and IDs of the particles
    # pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    # print(pos[:10])
    # # vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    # ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
    # print(ids[:10])


def main():
    snapshot = 'snap_000.0'
    ids, coords = read_snapshot(snapshot)
    graph = kdtree.build_nneigh_graph(coords[:100], 3000)
    # graph = gr.build_graph(coords[:100])
    g.plot_3d_graph(graph)

if __name__ == '__main__':
    main()
