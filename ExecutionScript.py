import numpy as np
import datagen_Heisenberg as dgen
import datagen_Config as dgen_conf
import calculations as tools
import matplotlib.pyplot as plt
plt.switch_backend('AGG')
import visualization as vis

J_vec = np.array([1,1,1])
B_vec = np.array([1,0,0])
delta_t = 0.01
T=1000
L = 10
psi0 =np.zeros(2**L,dtype=complex)
psi0[0] = 1
psi0_tag = 'zzzzzzzzzz_0000000000'
bc='open'
folder_path = "./Heisenberg/"

datagen_Heisenberg_Test = dgen_conf.DataGen_Config(L,J_vec,B_vec,bc,delta_t,T,psi0,psi0_tag,folder_path)

DYN = datagen_Heisenberg_Test.datagen('dyn')
print('Got the data about the dynamics. Setting up split between system and environment')
final_time = 200
sistema = [5]
ambiente =  [k for k in range(L)]
ambiente.remove(sistema[0])
print("System is qubit {val}".format(val=sistema[0]))
print("Environment is qubits {val}".format(val=ambiente))
GQM = tools.Geometric_QM(DYN[0:final_time],sistema,ambiente)
GQM_IT = tools.information_transport(DYN[0:final_time],delta_t,sistema,ambiente,11,11)
print('Computing Fluxes and Sources')
JP_t, JP0_t, JP1_t, JPHI_t, JPHI0_t, JPHI1_t, SIGMA_t, SIGMA0_t, SIGMA1_t = GQM_IT.fluxes_sources()
print('Now generating the plots')
final_plot_time=100
vis.plot_sigma_time(SIGMA_t,SIGMA0_t,SIGMA1_t,final_plot_time,fsize=20,save_flag='Y',path_to_save=folder_path+'/L={val}/frames/'.format(val=GQM.size),name_of_file='plot_sigma')
vis.plot_flux_time(JP_t,JPHI_t,JP0_t,JP1_t,JPHI0_t,JPHI1_t,final_plot_time,1,20,'Y',path_to_save=folder_path+'/L={val}/frames/'.format(val=GQM.size),name_of_file='plot_flux')
print('DONE')
