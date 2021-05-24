import numpy as np
import scipy.linalg as sLA
import numpy.linalg as LA

class Hamiltonian_Heisenberg:
    def __init__(self,L,J_vec,B_vec,bc):
        self.size = np.int(L)
        self.J_vec = J_vec
        self.B_vec = B_vec
        self.boundary_conditions = bc
        assert bc=='open' or bc=='closed', "The last argument can only be `open` or `closed`."

    def spin_x(self,k):
        L=self.size
        n=np.int(k)
        assert n<=L-1 and n>=0, 'Argument must be between 0 and size-1'
        #Local Pauli matrices
        sx = np.array([[0,1],[1,0]])
        #Set up the local operators that allow you to build the Hamiltonian
        SX = []
        for i in range(L):
            left = np.identity(2**(i))
            right = np.identity(2**(L-i-1))
            tens1 = np.kron(left, sx)
            tens2 = np.kron(tens1,right)
            SX.append(tens2)
        return SX[n]

    def mag_x(self,k,psi):
        L=self.size
        n=np.int(k)
        assert n<=L-1 and n>=0, 'Argument must be between 0 and size-1'
        return np.vdot(psi,np.matmul(self.spin_x(n),psi))


    def spin_y(self,k):
        n=np.int(k)
        L = self.size
        assert n<=L-1 and n>=0, 'Argument must be between 0 and size-1'
        sy = np.array([[0, -1j],[1j,0]])

        SY = []
        for i in range(L):
            left = np.identity(2**(i))
            right = np.identity(2**(L-i-1))
            tens1 = np.kron(left, sy)
            tens2 = np.kron(tens1,right)
            SY.append(tens2)
        return SY[n]

    def mag_y(self,k,psi):
        L=self.size
        n=np.int(k)
        assert n<=L-1 and n>=0, 'Argument must be between 0 and size-1'
        return np.vdot(psi,np.matmul(self.spin_y(n),psi))

    def spin_z(self,k):
        L = self.size
        n=np.int(k)
        assert n<=L-1 and n>=0, 'Argument must be between 1 and L-1'

        sz = np.array([[1,0],[0,-1]])

        SZ = []
        for i in range(L):
            left = np.identity(2**(i))
            right = np.identity(2**(L-i-1))
            tens1 = np.kron(left, sz)
            tens2 = np.kron(tens1,right)
            SZ.append(tens2)
        return SZ[n]


    def mag_z(self,k,psi):
        L=self.size
        n=np.int(k)
        assert n<=L-1 and n>=0, 'Argument must be between 0 and size-1'
        return np.vdot(psi,np.matmul(self.spin_z(n),psi))

    def build_Hamiltonian(self):
        print('Set up the Hamiltonian parameters.')
        L = self.size
        JJ, BB = self.J_vec, self.B_vec
        Jx, Jy, Jz = JJ[0], JJ[1], JJ[2]
        Bx, By, Bz = BB[0], BB[1], BB[2]

        #Initialize empty Hamiltonian
        print("Initialize empty Hamiltonian")
        H = np.zeros([2**L,2**L])

        #Build the Hamiltonian
        print("Filling with the right matrix elements")
        for i in range(L-1):
            H = H + Jx*np.matmul(self.spin_x(i),self.spin_x(i+1))+ Jy*np.matmul(self.spin_y(i),self.spin_y(i+1)) + Jz*np.matmul(self.spin_z(i),self.spin_z(i+1))

        #Take care of boundary conditions
        print("Taking care of the boundary_conditions")
        if self.boundary_conditions.lower()=='closed':
            H = H + Jx*np.matmul(self.spin_x(L-1),self.spin_x(0))+ Jy*np.matmul(self.spin_y(L-1),self.spin_y(0)) + Jz*np.matmul(self.spin_z(L-1),self.spin_z(0))
        print('Hamiltonian built. Returning array')
        return H

    def diagonalize_matrix(self,MM):
        print('Extracting eigenvalues and eigenvectors.')
        e_val, e_vec = LA.eigh(MM)
        #Sort them according to energy
        print('Sorting eigenvectors according to eigenvalues.')
        idx = e_val.argsort()[::-1]
        e_val = e_val[idx]
        e_vec = e_vec[:,idx]
        #This needs to happen so that the eigenvectors are column-vectors and not row-vectors
        e_vec=e_vec.transpose()
        print('Matrix diagonalized and eigensystem sorted. Returning eigenvalues and eigenvectors')
        return e_val, e_vec

class Dynamics:
    def __init__(self,HAM,del_t,T,psi0):
        self.H = HAM
        self.size = np.int(np.log2(HAM.shape[0]))
        self.del_t = del_t
        assert np.isreal(del_t)==True or del_t>0, 'Time delta must be a positive real quantity.'
        self.time_steps = np.int(T)
        self.initial_state = psi0
        assert len(psi0)==2**(self.size), 'Initial state not compatible with the size.'

    def propagator(self,HH):
        print('Building the propagator...')
        L = self.size
        dt = self.del_t
        #Find propagator by exponentiating Hamiltonian
        unit = sLA.expm(-1j*HH*dt)
        print('Propagator built.')
        return unit

    def time_evolution(self,UU):
        T = self.time_steps
        L = self.size
        #Initialize
        psi_all = np.zeros((T,2**L),dtype=complex)
        psi_all[0,:]=self.initial_state
        psi_t = self.initial_state
        print('Running the time evolution...\n')
        for k in range(T-1):
            psi_t = np.matmul(UU,psi_t)
            psi_all[k+1,:]=psi_t

        return psi_all
