import numpy as np

class information_transport:
    def __init__(self,p_vec,phi_vec,x_vec):
        self.p_alpha = p_vec
        self.phi_alpha = phi_vec
        self.x_alpha = x_vec
        self.time_steps = len(p_vec)
        assert len(p_vec)==len(phi_vec), "Time steps of p and phi vectors are different"
        assert len(x_vec)==len(phi_vec), "Time steps of x and phi vectors are different"
        self.env_size = len(p_vec[0])
        
