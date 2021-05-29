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
L = 9
psi0 =np.zeros(2**L,dtype=complex)
psi0[0] = 1
psi0_tag = 'zzzzzzzzz_000000000'
bc='open'
folder_path = "./Heisenberg/"

datagen_Heisenberg_Test = dgen_conf.DataGen_Config(L,J_vec,B_vec,bc,delta_t,T,psi0,psi0_tag,folder_path)

DYN = datagen_Heisenberg_Test.datagen('dyn')

"""
sistema = [0]
ambiente = [1,2,3]
PROVA = tools.Geometric_QM(DYN[0:10],sistema,ambiente)
PROVA_IT = tools.information_transport(DYN[0:17],delta_t,sistema,ambiente,10,10)

JP_t, JP0_t, JP1_t, JPHI_t, JPHI0_t, JPHI1_t, SIGMA_t, SIGMA0_t, SIGMA1_t = PROVA_IT.fluxes_sources()

vis.plot_sigma_time(SIGMA_t,SIGMA0_t,SIGMA1_t,10,fsize=20,save_flag='Y',path_to_save=folder_path+'/L={val}/frames/'.format(val=PROVA.size),name_of_file='testremote_plot_sigma')
vis.plot_flux_time(JP_t,JPHI_t,JP0_t,JP1_t,JPHI0_t,JPHI1_t,T=10,sc=1,fsize=20,save_flag='Y',path_to_save=folder_path+'/L={val}/frames/'.format(val=PROVA.size),name_of_file='testremote_plot_flux')
"""
