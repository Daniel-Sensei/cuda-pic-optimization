import os
import numpy as np
import sys
import h5py

def reshape(field,verbose=False):
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]
    nz = np.shape(field)[2]
    NX = nx+1
    NY = ny+1
    NZ = nz+1
    if verbose:
     print ('Reshaping', nx,ny,nz,NX,NY,NZ)
    new_field = np.empty((NX,NY,NZ))
    new_field[0:nx,0:ny,0:nz] = field
    new_field[nx,0:ny,0:nz]   = field[0,0:ny,0:nz]
    new_field[:,ny,0:nz]      = new_field[:,0,0:nz]
    new_field[:,:,nz]         = new_field[:,:,0]
    return new_field

inicyc = 0
fincyc = 11
istep  = 2
nx=128
ny=128
nz=2
ns=2

ipath = './../out/' 
opath = './../out_h5'

if not os.path.exists(opath):
    os.makedirs(opath)

for i in range(inicyc,fincyc,istep):
    ifile=os.path.join(ipath,'Exn_'+str(i).zfill(6)+'.dat')
    Exn=np.reshape(np.fromfile(ifile,dtype=float),(nx,ny,nz),order='C')
    ifile=os.path.join(ipath,'Eyn_'+str(i).zfill(6)+'.dat')
    Eyn=np.reshape(np.fromfile(ifile,dtype=float),(nx,ny,nz),order='C')
    ifile=os.path.join(ipath,'Ezn_'+str(i).zfill(6)+'.dat')
    Ezn=np.reshape(np.fromfile(ifile,dtype=float),(nx,ny,nz),order='C')
    ifile=os.path.join(ipath,'rhos_'+str(i).zfill(6)+'.dat')
    rhos=np.reshape(np.fromfile(ifile,dtype=float),(ns,nx,ny,nz),order='C')
    ofilen=os.path.join(opath, 'fields_'+str(i).zfill(6)+'.h5')
    print ('Writing .h5 file ', ofilen)
    hf = h5py.File(ofilen,'w')
    g1 = hf.create_group('Step#0')
    g2 = g1.create_group('Block')
    g3 = g2.create_group('Exn')
    g3.create_dataset('0',data=reshape(Exn).T)
    g3 = g2.create_group('Eyn')
    g3.create_dataset('0',data=reshape(Eyn).T)
    g3 = g2.create_group('Ezn')
    g3.create_dataset('0',data=reshape(Ezn).T)
    for iss in range(ns):
     g3 = g2.create_group('rho'+str(iss))
     g3.create_dataset('0',data=reshape(rhos[iss]).T)
    hf.close()

