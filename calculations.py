import numpy as np
import string

class Calculations:
    def __init__(self,psi):
        self.psi_all = psi
        #This allows us to deal, at the same time, with single states and also time-series.
        #time_flag = 0 means it's a single state. time_flag=1 means it'a time series.
        if self.psi_all.ndim==1: #We have passed a one-dimensional array. So, psi it's a single state
            self.size=np.int(np.log2(len(self.psi_all)))
            self.time_steps=1 #define time_steps =1 for consistency
            self.time_flag = 0 #Single state flag
        elif self.psi_all.ndim==2: #We have passed a two-dimensional array. So, psi it's a time series of quantum states
            self.size=np.int(np.log2(len(self.psi_all[0,:])))
            self.time_steps=len(self.psi_all[:,0]) #Extract the number of time_steps.
            self.time_flag=1 #Declare the flag
        else:
            assert self.psi_all.ndim==1 or self.psi_all.ndim==2, "System dimension {val} not identified correctly".format(val=self.size)

    ################BUILDING THE MATHEMATICAL TOOLS NEEDED###############################
    #Representation of local spin operators in the larger Hilbert space of dimension 2**L
    ####################################################################################
    def spin_x(self,k):
        #TESTED: YES
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

    def spin_y(self,k):
        #TESTED: YES
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

    def spin_z(self,k):
        #TESTED: YES
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


    #Useful function that returns the appropriate pauli matrix whether we Use
    #'X','Y','Z' or 1,2,3 to identify them.
    def ordered_pauli(self,n):
        #TESTED: YES
        if n.upper()=='X' or n==1:
            return np.array([[0,1],[1,0]])
        elif n.upper()=='Y' or n==2:
            return np.array([[0, -1j],[1j,0]])
        elif n.upper()=='Z' or n==3:
            return np.array([[1,0],[0,-1]])
        elif n.upper()=='I' or n==0:
            return np.eye(2)
    #Useful function that returns the appropriate local spin operator, specified
    #with the position in the 1D chain and with the tag `x`,`y` or `z`
    def local_spin(self,k,tag):
        #TESTED: YES
        if tag.lower()=='x':
            return self.spin_x(k)
        elif tag.lower()=='y':
            return self.spin_y(k)
        elif tag.lower()=='z':
            return self.spin_z(k)
        else:
            return print('Last argument {val} can only be `x`,`y` or `z`'.format(val=tag))

    ###########################################################################
    ##############CODE TO COMPUTE OBSERVABLE AND/OR USEFUL STUFF###############
    ###########################################################################

    #Returns the appropriate operator, specified with the string SS. For example,
    #if SS = "IIXII", it means the size of the whole system is 5 and we have identified
    #the local spin operator of the third spin, along the X direction. Other example,
    #IXIYI identifies the product of the local spin_x operator for the second spin, times
    #the local spin_y operator for the fourth spin.
    def tagged_operator(self,SS):
        #TESTED: YES
        MM = np.eye(2**self.size)
        for k in range(len(SS)):
            if SS[k].upper()=='I':
                BB = np.eye(2**self.size)
            elif SS[k].upper()=='X':
                BB = self.spin_x(k)
            elif SS[k].upper()=='Y':
                BB = self.spin_y(k)
            elif SS[k].upper()=='Z':
                BB = self.spin_z(k)
            MM = np.matmul(MM,BB)
        return MM

    #This function allows me to avoid rewriting code for the two cases in
    #which we have a single quantum state or a time series of them. Using
    #the time_flag declared at the beginning, if we have a single quantum
    #state, the function `func` with arguments *args will be applied directly
    #to the quantum state. Otherwise, it will iteratively be applied on the
    #time-series of quantum states
    def apply_to_state(self,func,*args):
        #TESTED: YES
        if self.time_flag==0:
            return func(args,psi=self.psi_all)
        elif self.time_flag==1:
            OUT = []
            for t in range(self.time_steps):
                psi_t = self.psi_all[t,:]
                OUT.append(func(*args,psi=psi_t))
            return OUT

    #With the function defined above, all the practical tools I will be
    #defining now will have to be defined in two steps. In the first one I define
    #a function that does something on a state psi, passed as argument. Then,
    #via the apply_to_state function we guarantee it'll work both in the case of
    #a single state and in the case of a time_series, without having to change
    #anything.


    #Function that takes a matrix MM and a quantum state psi as input and
    #returns the expectation value <psi|MM|psi>, checking that it has to be real.
    def state_expectation(self,MM,psi):
        #TESTED: YES
        out = np.vdot(psi,np.matmul(MM,psi))
        assert np.isclose(np.imag(out),0,atol=10**(-8))==True, "Expectation value has a non-negligible imaginary part {val}".format(val=np.imag(out))
        return np.real(out)

    #Computes the expectation value on the tagged_operator, on a specific state
    #psi taken as input.
    def expectation_tagged_operator(self,SS,psi):
        #TESTED: YES
        return self.state_expectation(self.tagged_operator(SS),psi)

    #Final function to use to compute expectation values both on single states
    #and on time-series of quantum states.
    def expectation(self,SS):
        #TESTED: YES
        return self.apply_to_state(self.expectation_tagged_operator,SS)

    #Function which returns the local spin expectation values
    def local_mag(self,k,tag):
        #TESTED: YES
        assert type(tag)==str and len(tag)==1, "Second argument {val} must be a single-character string: `x`, `y` or `z`".format(val=tag)
        L = self.size
        n = np.int(k)
        assert n<=L-1 and n>=0, 'Last argument {val} must be between 0 and {val2}'.format(val=n,val2=L-1)
        return self.apply_to_state(self.state_expectation,self.local_spin(n,tag))

    #Function which returns the one-spin density matrix from the global psi taken in input
    def one_body_rho(self,k,psi):
        #TESTED: YES
        rho = 0.5*np.eye(2)
        for tag in ['x','y','z']:
            rho = rho+0.5*self.state_expectation(self.local_spin(k,tag),psi)*self.ordered_pauli(tag)
        return rho
    #Extend the function above to the whole time-series of quantum states, if needed.
    def one_body_density_matrix(self,k):
        #TESTED: YES
        return self.apply_to_state(self.one_body_rho,k)

    #Function which returns the two-spins density matrix, from the global psi taken in input
    def two_body_rho(self,h,k,psi):
        #TESTED: YES
        assert h!=k, "The two arguments {hh} and {kk} must be different".format(hh=h,kk=k)
        a = np.amin(np.array([h,k]))
        b = np.amax(np.array([h,k]))
        T_Norm = 0.25*np.eye(4)
        T1 = np.zeros((4,4),dtype=complex)
        T2 = np.zeros((4,4),dtype=complex)
        for tag in ['x','y','z']:
            T1 = T1 + 0.25*(self.state_expectation(self.local_spin(a,tag),psi))*np.kron(self.ordered_pauli(tag),np.eye(2))
            T2 = T2 + 0.25*(self.state_expectation(self.local_spin(b,tag),psi))*np.kron(np.eye(2),self.ordered_pauli(tag))
        T3 = np.zeros((4,4),dtype=complex)
        TT = ['I']*self.size
        for i in ['X','Y','Z']:
            for j in ['X','Y','Z']:
                NEW_TT = TT[:]
                NEW_TT[a],NEW_TT[b]=i,j
                T3 = 0.25*(self.expectation_tagged_operator(NEW_TT,psi))*np.kron(self.ordered_pauli(i),self.ordered_pauli(j))
        return T_Norm+T1+T2+T3

    def two_body_density_matrix(self,h,k):
        #TESTED: YES
        return self.apply_to_state(self.two_body_rho,h,k)

    def index_reordering(self,tag,rho):
        #TESTED: YES
        assert type(tag)==int, "This function takes only one spin as first input {val}.".format(val=k)
        rho_size = np.int(np.log2(rho.shape[0]))
        total = [2]*2*rho_size #This is basically [2,2,2,2,...]
        rho_shaped = np.reshape(rho,tuple(total))
        if tag==0:
            new_rho = np.moveaxis(rho_shaped,rho_size,1)
        else:
            new_rho = np.moveaxis(rho_shaped,tag,0)
            new_rho = np.moveaxis(new_rho,tag+rho_size,1)
        return new_rho

    def tracing_out_one_spin(self,tag,rho):
        #TESTED: YES
        #tag is a single number. It's the spin we want to trace out.
        rho_size = np.int(np.log2(rho.shape[0]))
        rho_shaped = self.index_reordering(tag,rho)
        reduced_rho = np.einsum('aa...->...',rho_shaped)
        return np.reshape(reduced_rho,(2**(rho_size-1),2**(rho_size-1)))

    def reduced_rho(self,tag_list,psi):
        #TESTED: YES
        rho = np.outer(np.conj(psi),psi)
        #tag_list is the list of the spins of which we want the reduced density matrix.
        #We need to compute the negative of that. The list of spins which we need to trace over.
        #First we remove duplicates
        res = []
        [res.append(x) for x in tag_list if x not in res]
        #Then we sort the list in ascending order
        new_list = np.sort(res)
        #Now we generate the list of spin to trace over
        trace_list = []
        for k in range(self.size):
            if (k in new_list)==False:
                trace_list.append(k)
        reduced=rho
        counter=0
        for k in trace_list:
            reduced = self.tracing_out_one_spin(k-counter,reduced)
            counter+=1
        return reduced

    def reduced_density_matrix(self,tag_list):
        #TESTED: YES
        return self.apply_to_state(self.reduced_rho,tag_list)

    def xlog2x(self,x):
        #TESTED: YES
        tol=10**(-14)
        assert (np.imag(x)>=-tol).all()==True, "In xlog2x argument {val} must be real".format(val=x)
        assert (x>=-tol).all()==True, "In xlog2x argument {val} must be non-negative".format(val=x)
        y = np.abs(x)
        if np.isclose(y*(1-y),0,atol=tol)==True:
            return 0
        else:
            return -y*np.log2(y)

    def rho_log_rho(self,rho):
        #TESTED: YES
        e_val,e_vec = np.linalg.eigh(rho)
        value = 0
        for s in range(len(e_val)):
            value = value+self.xlog2x(e_val[s])
        return value

    def von_entropy(self,tag_list,psi):
        #TESTED: YES
        return self.rho_log_rho(self.reduced_rho(tag_list,np.outer(np.conj(psi),psi)))

    def von_Neumann_entropy(self,tag_list):
        #TESTED: YES
        return self.apply_to_state(self.von_entropy,tag_list)

#The class  Geometric_QM is a sub-class of calculations. Inheritance is
#straigthforward, as we need everything that has been defined in Calculations.
#In this class we put all the tool about geometric quantum mechanics.

class Geometric_QM(Calculations):
    def __init__(self,psi,sys_list,env_list):
        Calculations.__init__(self,psi)
        self.system = sys_list #List of spins which are part of the system
        self.environment = env_list #List of spins which are part of the environment
        TOT = [n for n in range(self.size)]
        assert sorted(self.system+self.environment)==TOT, "System {val1} or Environment {val2} wrongly specified".format(val1=self.system,val2=self.environment)
        self.system_size = len(self.system)
        self.env_size = len(self.environment)


#############################################################
########SOME USEFUL FUNCTIONS################################
#############################################################

    #A vector |bin=010> corresponds to |num=2>. The number num is the decimal
    #representation of the binary string bin. Here we build the functions to
    #switch from one representation to the other.
    def bin_to_num(self,s_vec):
        #TESTED: YES
        #s_vec is a string of 0s or 1s
        size = len(s_vec)
        return int(str(s_vec),2)

    def num_to_bin(self,num,size):
        #TESTED: YES
        return ('0'*(size-len(bin(num)[2:]))+bin(num)[2:])


    #Here we build the single basis vectors, in num representation
    def comp_basis_vector_num_rep(self,num,size):
        #TESTED: YES
        vec = np.zeros(2**size,dtype=complex)
        vec[num] = 1
        return vec

    #Here we build the single basis vectors, in binary representation
    def comp_basis_vector_bin_rep(self,stringa):
        #TESTED: YES
        vec = np.zeros(2**len(stringa),dtype=complex)
        vec[self.bin_to_num(stringa)] = 1
        return vec

    #List all the binary strings representing a state for a given number
    #of qubit called size
    def listing_states(self,size):
        #TESTED: YES
        AA = []
        for k in range(2**size):
            AA.append(self.num_to_bin(k,size))
        return AA

#############################################################
######## BUILDING THE BASIS ################################
#############################################################

    #Now we take into account the input information about which qubits are
    #part of the system and which ones of the environment. First, we write
    #the generic builder of the string, with the zeros and ones in the right
    #place, dictated by the lists defining system and environment
    def basis_string(self,sys_string,env_string):
        unsorted = self.system+self.environment
        new_list = sorted(zip(unsorted,list(sys_string+env_string)))
        ZZ = [k[1]for k in new_list]
        return "".join(ZZ)
    #Then, we build the vector associated to the string defined above.
    def basis_vector(self,sys_string,env_string):
        return self.comp_basis_vector_bin_rep(self.basis_string(sys_string,env_string))

    #Now we build full basis as a numpy array made by dS * dE vectors, each
    #with dimension dS*dE
    def build_basis(self):
        sys_list = self.listing_states(len(self.system))
        env_list = self.listing_states(len(self.environment))
        dS = len(sys_list)
        dE = len(env_list)
        assert int(np.log2(dS))==self.system_size, "Declared system size {val} and input system size {val1} do not match".format(val=dS,val1=self.system_size)
        assert int(np.log2(dE))==self.env_size, "Declared environment size {val} and input environment size {val1} do not match".format(val=dE,val1=self.env_size)
        basis = np.zeros((len(sys_list),len(env_list),len(sys_list)*len(env_list)),dtype=complex)
        for k in range(dS):
            for n in range(dE):
                basis[k,n,:] = self.basis_vector(sys_list[k],env_list[n])
        return basis

    def scalar_product(self,psi1,psi2):
        return np.vdot(psi1,psi2)

#############################################################
######## GEOMETRIC QUANTUM MECHANICS TOOLS ##################
#############################################################

    #Given a pure state of (system,environment), with finite-dimensional
    #environment, the geometric quantum state of the system is a convex sum
    #of dirac deltas \sum_\alpha x_\alpha \delta (Z-Z_\alpha), with coefficients
    #x_alpha which are probabilities related to the state of the environment.
    #Here we compute the x_alpha.
    def probs(self,psi):
        #TESTED: YES
        np.array(psi)
        dE = 2**self.env_size
        dS = 2**self.system_size
        x_alpha = np.zeros(dE)
        vec_list = self.build_basis() #Build the global basis
        for n in range(dE):
            for k in range(dS):
                vec = np.array(vec_list[k,n,:])
                x_alpha[n]=x_alpha[n]+np.abs(self.scalar_product(vec,psi))**2 #the probability is the modulus square of scalar product.
        assert np.isclose(np.sum(x_alpha),1,atol=10**(-10))==True, 'Probabilities do not sum to 1. Sum is {}'.format(np.sum(x_alpha)) #check that probability sum to 1
        return x_alpha

    #Extend the function to a time-series, when needed, using apply_to_state
    #There is a semi-bug with this function. It does not work properly if I
    # enter as state an arrary as (16,). But it does work with an array as (1,16).
    #It also works with time-series. The quick solution is to use DYN[0:1] instead
    #of DYN[0]. I'll get back to it later.
    def probabilities(self):
        #TESTED: YES. PASSED: 1
        return self.apply_to_state(self.probs)

    #New we build the pure-states which form the support of the geometric quantum
    #state, which is a sum of dirac deltas. Some states are there, but have
    #zero probability of being occupied. We put them in a different list.
    #Takes the global state as input and returns two lists. One with the
    #states that make up the geometric quantum state and another one with the
    #states which have zero probability of being occupied.
    def local_vectors(self,psi):
        dS = 2**(self.system_size)
        dE = 2**(self.env_size)
        x_alpha = self.probs(psi) #x_alpha are the probabilities computed above
        chi_alpha = []
        chi_alpha_zero = []
        vec_list = self.build_basis() #Take the basis we are using for the whole
        #system. Needed for the scalar product
        B_Sys = np.eye(dS) #Initialize basis for the system. It's the
        #computational basis, so it's simply the identity matrix.
        for n in range(dE):
            vv = np.zeros(dS)
            if x_alpha[n]!=0: #States with zero probabilities go into another list.
                for k in range(dS):
                    vec = np.array(vec_list[k,n,:]) #global basis vector.
                    scal=self.scalar_product(vec,psi) #scalar product.
                    vv = vv+scal/np.sqrt(x_alpha[n])*B_Sys[k] #Normalize
                assert np.isclose(self.scalar_product(vv,vv),1)==True, "Norm of the output vector {val} is not 1".format(val=self.scalar_product(vv,vv)) #Check normalization
                chi_alpha.append(vv) #Append to the right list.
            else:
                chi_alpha_zero.append(n) #Append to the zero-list

        return chi_alpha, chi_alpha_zero #Return the two lists, separately.

    #Now extend the function above for time-series.
    def chi_alphas(self):
        return self.apply_to_state(self.local_vectors)

    #Since here we are dealing only with qubit, I decided to fix the Representation
    #to use to be |psi(p,\phi)> = \sqrt{1-p} |0> + \sqrt{p}e^{i\phi}|1>. From
    #this one I can always switch to angle coordinates on the bloch sphere.
    #The function prob_phase(self,psi) takes the state of a qubit psi and
    #returns its (p,\phi) coordinate.
    def pphi(self,psi):
        dS = 2**self.system_size
        assert dS ==len(psi), "Quantum State psi does not have the required dimension dS = {val}".format(val=dS)
        B_Sys = np.eye(dS)
        return np.abs(self.scalar_product(B_Sys[1],psi))**2, np.angle(self.scalar_product(B_Sys[1],psi))-np.angle(self.scalar_product(B_Sys[0],psi))

    #In general, the extraction of (probabilities,phases) coordinates is very
    #similar to what has been defined above.
    def prob_phase(self,psi):
        chis, chis_zero = self.local_vectors(psi)
        dE = 2**self.env_size
        dS = 2**self.system_size
        B_Sys = np.eye(dS)
        assert len(chis)+len(chis_zero)==dE, "Total Number of chi_alpha is not equal to the declared environment size"
        p_vec = []
        phi_vec = []
        s=0
        for n in range(dE): #This ranges over all possible states in the environment (chosen basis)
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
