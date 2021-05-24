import numpy as np

class geometric_tools:
    def __init__(self,psi,basis_sys,basis_env):
        #basis_sys and basis_env are lists of dS and dE vectors which define the bases of interest for the system and environment
        #Remember, kets should be column arrays while bras should be row arrays, to comply with matrix product.
        #But, in numpy we have that the standard representation is that a vector is a (1,n) array, which is a row array. However, matmul should take care of that.
        self.q_state = psi
        assert np.isclose(self.scalar_product(self.q_state,self.q_state),1,atol=10**(-5))==True, "Global quantum state is not normalized"
        self.basis_sys = basis_sys
        self.basis_env = basis_env
        self.system_size = np.int(np.log2(len(basis_sys)))
        self.env_size = np.int(np.log2(len(basis_env)))
        assert np.int((2**(self.system_size))*(2**(self.env_size)))==len(self.q_state), "Dimension of bases introduced does not match the dimension of the global state"

    def get_attributes(self):
        return print('q_state = quantum state\nbasis_sys = basis for the system\nbasis_env = basis for the environment\nsystem_size = # of qubits of the system\nenv_size = # of qubits of the environment')

    def scalar_product(self,psi1,psi2):
        return np.vdot(psi1,psi2)

    def build_basis(self):
        dS = 2**self.system_size
        dE = 2**self.env_size
        B_Sys = self.basis_sys
        B_Env = self.basis_env
        basis = np.zeros((dS,dE,dS*dE))
        for k in range(dS):
            for n in range(dE):
                basis[k,n,:]=np.kron(B_Sys[k],B_Env[n])
        return basis


    def probabilities(self):
        psi0=self.q_state
        dE = 2**self.env_size
        dS = 2**self.system_size
        x_alpha = np.zeros(dE)
        vec_list = self.build_basis()
        for n in range(dE):
            for k in range(dS):
                vec = np.array(vec_list[k,n,:])
                x_alpha[n]=x_alpha[n]+np.abs(self.scalar_product(vec,psi0))**2
        assert np.isclose(np.sum(x_alpha),1,atol=10**(-5))==True, 'Probabilities do not sum to 1. Sum is {}'.format(np.sum(x_alpha))
        return x_alpha

    def local_vectors(self):
        psi0=self.q_state
        dS = 2**(self.system_size)
        dE = 2**(self.env_size)
        B_Sys = self.basis_sys
        B_Env = self.basis_env
        x_alpha = self.probabilities()
        chi_alpha = []
        chi_alpha_zero = []
        vec_list = self.build_basis()
        for n in range(dE):
            vv = np.zeros(dS)
            if x_alpha[n]!=0:
                for k in range(dS):
                    vec = np.array(vec_list[k,n,:])
                    scal=self.scalar_product(vec,psi0)
                    vv = vv+scal/np.sqrt(x_alpha[n])*B_Sys[k]
                assert np.isclose(self.scalar_product(vv,vv),1)==True, "Norm of the output vector is not 1"
                chi_alpha.append(vv)
            else:
                chi_alpha_zero.append(n)

        #chis = np.array(chi_alpha)
        return chi_alpha, chi_alpha_zero

    def prob_phase(self):
        chis, chis_zero = self.local_vectors()
        dE = 2**self.env_size
        dS = 2**self.system_size
        B_Sys = self.basis_sys
        assert len(chis)+len(chis_zero)==dE, "Total Number of chi_alpha is not equal to the declared environment size"
        p_vec = []
        phi_vec = []
        s=0
        for n in range(dE):
            pp = np.zeros(dS)
            phi = np.zeros(dS)
            if (np.sum(np.array(chis_zero)==n))==1:
                #If the index correspond to a null vector, we enter a null array both for p and for phi.
                p_vec.append(pp)
                phi_vec.append(phi)
            else:
                for k in range(dS):
                    #If the index correspond to a non-null chi, we can identify correctly the (probability,phases) coordinates in CPN of the chi_alphas
                    #Also, if the vector is an element of the basis |0>, |1>, |2> etc, the phases are not defined and we give it a zero value.
                    pp[k]=np.abs(self.scalar_product(B_Sys[k],chis[s]))**2
                    phi[k]=np.angle(self.scalar_product(B_Sys[k],chis[s]))-np.angle(self.scalar_product(B_Sys[0],chis[s]))
                s=s+1
                p_vec.append(pp)
                phi_vec.append(phi)
        return p_vec, phi_vec


class information_transport(geometric_tools):
    #This is reliable only for qubits: systems_size = 2.
    def __init__(self,N_p,N_phi,palpha_dot,phialpha_dot,xalpha_dot,psi,basis_sys,basis_env):
        super().__init__(psi,basis_sys,basis_env)
        self.Np = N_p
        self.Nphi = N_phi
        self.del_p, self.del_phi = 1/self.Np, 2*np.pi/self.Nphi
        self.p_dot, self.phi_dot, self.x_dot = palpha_dot, phialpha_dot, xalpha_dot

    def get_attributes(self):
        print('q_state = quantum state\nbasis_sys = basis for the system\nbasis_env = basis for the environment\nsystem_size = # of qubits of the system\nenv_size = # of qubits of the environment\n')
        print('Np=resolution in p\nNphi=resolution in phi\ndel_p = size of p interval\ndel_phi=size of phi interval\n')
        print('p_dot=list of (env_size) time derivatives of p_alpha\nphi_dot=list of (env_size) time derivatives of phi_alpha\nx_dot=list of (env_size) time derivatives of x_alpha')


    def discretization_properties(self):
        Ip_boundaries = np.linspace(0,1,self.Np)
        Iphi_boundaries = np.linspace(0,2*np.pi,self.Nphi)
        delta_p = Ip_boundaries[1]-Ip_boundaries[0]
        delta_phi = Iphi_boundaries[1]-Iphi_boundaries[0]
        Ip_centers = Ip_boundaries[0:-1]+0.5*delta_p
        Iphi_centers = Iphi_boundaries[0:-1]+0.5*delta_phi
        return Ip_boundaries, Iphi_boundaries, Ip_centers, Iphi_centers

    def get_cell(self,pphi):
        N_p, N_phi = np.int(1/self.del_p), np.int(2*np.pi/self.del_phi)
        Ip_boundaries, Iphi_boundaries, Ip_centers, Iphi_centers = self.discretization_properties()
        p,phi = pphi[0], pphi[1]
        flag = 'N'
        if np.isclose(p*(p-1),0,atol=10**(-8))==False:
            return flag, np.argmin(np.abs(Ip_centers-p)), np.argmin(np.abs(Iphi_centers-phi))
        else:
            return 'Y',p,'aaaa'


    def fluxes_sources(self):
        x_alpha = self.probabilities()
        p_alpha, phi_alpha = self.prob_phase()
        Ip_boundaries, Iphi_boundaries, Ip_centers, Iphi_centers = self.discretization_properties()
        #Initialize fluxes to zero arrays.
        JP, JPHI, SIGMA = np.zeros((len(Ip_centers),len(Iphi_centers))), np.zeros((len(Ip_centers),len(Iphi_centers))), np.zeros((len(Ip_centers),len(Iphi_centers)))
        #Extremal points as |0> and |1> need to be treated in a slightly different way, due to discontinuity in the coordinate map.
        SIGMA_0, SIGMA_1 = 0,0
        JP_0,JP_1 = 0,0
        JPHI_0,JPHI_1 = 0,0
        dE = 2**(self.env_size)
        for n in range(dE):
            #Auxiliary array of the dimension set by the resolution of CP1 that we are using.
            AUX = np.zeros((len(Ip_centers),len(Iphi_centers)))
            #a and b are the two indices that identifies the cell (in the discretization on CP1) on which the specific point (p_alpha,\phi_alpha) can be found at.
            #f is a flag that helps determine if the state is exactly |0> or |1>. f='N' means no. f='Y' means yes. In this last case, the second output is p and
            #it tells if the state is |0> (p=0) or |1> (p=1).
            #print("for n = {num:n} we get".format(num=n))
            #print("p_alpha = {val}".format(val=p_alpha[n][1]))
            f,a,b=self.get_cell(np.array([p_alpha[n][1],phi_alpha[n][1]]))
            #print("f = {fl}; a = {val}".format(fl=f,val=a))
            if f=='N':
                #print('n = {num:n}, is in cell ({a:n},{b:n})'.format(num=n,a=a,b=b))
                #This makes sure that, in the sum that determines the fluxes or sources, the terms are added to the right cell, identified by the position of the point.
                AUX[a,b]=1
                #Then we can add to the previously determined flux/source, and iterate over the environmental label.
                #print('Adding {val} to JP'.format(val=x_alpha[n]*self.p_dot[n][1]))
                JP = JP + x_alpha[n]*self.p_dot[n][1]*AUX
                #print('Adding {val} to JPHI'.format(val=x_alpha[n]*self.phi_dot[n][1]))
                JPHI = JPHI + x_alpha[n]*self.phi_dot[n][1]*AUX
                #print('Adding {val} to SIGMA'.format(val=self.x_dot[n]))
                SIGMA = SIGMA + self.x_dot[n]*AUX
            elif f=='Y':
                if a>0.5:
                    #print('n= {num:n}, is in |0>'.format(num=n))
                    #print('Adding {val} to JP_0'.format(val=x_alpha[n]*self.p_dot[n][1]))
                    JP_0 = JP_0 + x_alpha[n]*self.p_dot[n][1]
                    #Here we already have established that the point is an extremal one: |0>.
                    #As a result of this, any variation of the phase only is pure gauge and should be disregarded. Hence,
                    #we add only the physical terms, associated to a true variation of the state, which involves a non-zero p_dot[n][1]
                    if np.isclose(self.p_dot[n][1],0,atol=10**(-8))==False:
                        JPHI_0 = JPHI_0 + x_alpha[n]*self.phi_dot[n][1]
                        #print('Adding {val} to JPHI_0'.format(val=x_alpha[n]*self.phi_dot[n][1]))
                    #print('Adding {val} to SIGMA_0'.format(val=self.x_dot[n]))
                    SIGMA_0 = SIGMA_0+self.x_dot[n]
                elif a<0.5:
                    #print('n= {num:n}, is in |1>'.format(num=n))
                    #print('Adding {val} to JP_1'.format(val=x_alpha[n]*self.p_dot[n][1]))
                    JP_1 = JP_1 + x_alpha[n]*self.p_dot[n][1]
                    if np.isclose(self.p_dot[n][1],0,atol=10**(-8))==False:
                        JPHI_1 = JPHI_1 + x_alpha[n]*self.phi_dot[n][1]
                        #print('Adding {val} to JPHI_1'.format(val=x_alpha[n]*self.phi_dot[n][1]))
                    #print('Adding {val} to SIGMA_1'.format(val=self.x_dot[n]))
                    SIGMA_1 = SIGMA_1+self.x_dot[n]
            else:
                print('Problem with the flag.')

        return JP, JPHI, SIGMA, JP_0, JP_1, JPHI_0, JPHI_1, SIGMA_0, SIGMA_1
