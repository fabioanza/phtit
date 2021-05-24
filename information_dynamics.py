import numpy as np
import geometric_tools as gt


def dot_computation(time_evolution,basis_sys,basis_env):
    x_alpha, p_alpha, phi_alpha = [], [], []
    x_alpha_dot, p_alpha_dot, phi_alpha_dot = [], [], []
    CL = gt.geometric_tools(time_evolution[0],basis_sys,basis_env)
    x_alpha.append(CL.probabilities())
    a,b = CL.prob_phase()
    p_alpha.append(a)
    phi_alpha.append(b)
    T=len(time_evolution)
    co = 10
    for t in range(1,T):
        if t%np.int(T/10)==0:
            print("countdown...",co)
            co=co-1
        CL = gt.geometric_tools(time_evolution[t],basis_sys,basis_env)
        x_alpha.append(CL.probabilities())
        a,b = CL.prob_phase()
        p_alpha.append(a)
        phi_alpha.append(b)
        x_dot = np.array(x_alpha[t])-np.array(x_alpha[t-1])
        p_dot = np.array(p_alpha[t])-np.array(p_alpha[t-1])
        phi_dot = np.array(phi_alpha[t])-np.array(phi_alpha[t-1])
        x_alpha_dot.append(x_dot)
        p_alpha_dot.append(p_dot)
        phi_alpha_dot.append(phi_dot)
    return x_alpha,p_alpha,phi_alpha, x_alpha_dot, p_alpha_dot, phi_alpha_dot
