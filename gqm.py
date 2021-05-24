import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

def maximum_entropy(rho, psi, beta):
    expectation = np.einsum('i...,ij,j...', psi.conj(), np.linalg.inv(rho), psi)
    return np.exp(-beta*expectation).real

class GeomState:
    
    def __init__(self, weights, chis):
        self.weights = weights
        self.chis = chis
    
    def to_pphi(self):
        """
        Convert geometric state to p, phi coords
        Returns: tuple (p, phi) of type (ndarray, ndarray)
        """
        p = np.abs(self.chis[:, 1])**2
        phi = (np.angle(self.chis[:, 1]) - np.angle(self.chis[:, 0])) % (2*np.pi)
        return p, phi
        
    def rho(self):
        """
        Convert geometric state to density matrix
        Returns: ndarray w/ shape (2, 2)
        """
        chi_rhos = np.einsum('ij,ik->ijk', self.chis, self.chis.conj())
        return np.einsum('i,ijk->jk', self.weights, chi_rhos)
        
    def from_psi(psi, i=0):
        """
        Computes local geometric quantum state from global wavefunction.
        Params:
            psi (ndarray): (2^N)-dimensional vector of wavefunction coefficients
            i (int): index of the "local" qubit. Defaults to 0
        Returns:
            tuple(ndarray, ndarray). The first ndarray, shape (M), is a vector of
            weights, which sum to 1. The second ndarray, shape (M, 2), contains
            the corresponding kets $\chi$ in the gqs decomposition.
        """
        # reshape wavefunction as a tensor w/ shape (2, 2, 2, ...)
        n_qubits = np.log2(len(psi))
        assert np.isclose(n_qubits, int(n_qubits)), "# of coeffs must be power of 2"
        tensor_shape = [2]*int(n_qubits)
        psi = psi.reshape(tensor_shape)
        # split the wavefunction over qubit i (i.e. two cols of length 2**(N-1))
        partition = psi.swapaxes(0, i).reshape(2, -1).T
        # compute weights
        weights = np.sum(partition.conj()*partition, axis=1).real
        assert np.isclose(weights.sum(), 1), "wavefunction must be normalized"
        # get rid of components with 0 weight
        partition = partition[weights > 0]
        weights = weights[weights > 0]
        # compute kets, put in homogeneous coords, round to 10 decimal places
        chis = np.einsum('ij,i->ij', partition, 1/np.sqrt(weights))
        inverse_phases = np.exp(-1j*np.angle(chis[:, 0]))
        chis = np.einsum('ij,i->ij', chis, inverse_phases)
        chis = chis.round(10)
        # consolidate kets
        gqs = {}
        for weight, chi in zip(weights, chis):
            state = tuple(chi)
            if state not in gqs:
                gqs[state] = 0
            gqs[state] += weight    
        # organize weights and kets into two lists
        chis = []
        weights = []
        for state, weight in gqs.items():
            chis.append(np.array(state))
            weights.append(weight)
        return GeomState(np.array(weights), np.array(chis))
    
class CP1:
    
    def __init__(self, pdf):
        p = np.linspace(0, 1, 2000)
        phi = np.linspace(0, 2*np.pi, 6000)
        discretized = pdf(*np.meshgrid(p, phi))
        norm = integrate.simps(integrate.simps(discretized, p), phi)        
        self.pdf = lambda p, phi: pdf(p, phi) / norm        
    
    def sample(self, num_samples, resolution=2048):
        
        def get_inverse_cdf(domain, pdf):
            dx = domain[1] - domain[0]
            cdf = np.zeros(pdf.shape)
            cdf[0] = pdf[0]*dx
            for i in range(1, len(cdf)):
                cdf[i] = cdf[i-1] + pdf[i]*dx
            cdf /= cdf[-1]
            cdf_inv = interp1d(cdf, domain, fill_value="extrapolate")
            return cdf_inv
        
        p = np.linspace(0, 1, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        
        discretized = self.pdf(*np.meshgrid(p, phi))
        marginal_on_p = integrate.simps(discretized, phi, axis=0)
        inverse_cdf_p = get_inverse_cdf(p, marginal_on_p)
        
        rng = np.random.default_rng()
        
        count = 0
        while count < num_samples:        
            p_i = inverse_cdf_p(rng.uniform())        
            conditional_phi = self.pdf(np.full(phi.shape, p_i), phi)
            conditional_phi = conditional_phi / integrate.simps(conditional_phi, phi)
            cdf_inv_phi = get_inverse_cdf(phi, conditional_phi)
            phi_i = cdf_inv_phi(rng.uniform())
            yield State(p_i, phi_i)
            count += 1