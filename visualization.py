import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

def discretization_properties(Np,Nphi):
    Ip_boundaries = np.linspace(0,1,Np)
    Iphi_boundaries = np.linspace(0,2*np.pi,Nphi)
    delta_p = Ip_boundaries[1]-Ip_boundaries[0]
    delta_phi = Iphi_boundaries[1]-Iphi_boundaries[0]
    Ip_centers = Ip_boundaries[0:-1]+0.5*delta_p
    Iphi_centers = Iphi_boundaries[0:-1]+0.5*delta_phi
    return Ip_boundaries, Iphi_boundaries, Ip_centers, Iphi_centers

def plot_sigma(SIGMA,SIGMA_0,SIGMA_1,scale='N',fsize=20,save_flag='N',path_to_save=None):
    #Extract the information about the discretized CP1, from the shape of SIGMA
    X,Y = SIGMA.shape
    Np, Nphi = X+1,Y+1
    P_Bound, PHI_Bound, P_Center, PHI_Center = discretization_properties(Np,Nphi)
    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(nrows=3, ncols=2)
    fig.suptitle(r'$\sigma_t(\phi,p)$',fontsize=fsize)
    # First subplot
    ax0 = fig.add_subplot(gs[:,0],projection='3d')

    _x = PHI_Bound[:-1]
    _y = P_Bound[:-1]
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = SIGMA.flatten()
    bottom = np.zeros_like(top)
    width = PHI_Bound[1]-PHI_Bound[0]
    depth = P_Bound[1]-P_Bound[0]

    ax0.bar3d(x, y, bottom, width, depth, top, shade=True)
    if scale=='N':
        if np.abs(np.min(SIGMA)-np.max(SIGMA))>10**(-8):
            ax0.set_zlim([np.amin(SIGMA),np.amax(SIGMA)])
    else:
        ax0.set_zlim([scale[0],scale[1]])

    ax0.set_xlabel(r'$\phi$',fontsize=fsize)
    ax0.set_ylabel(r'$p$',fontsize=fsize)

    # Second subplot
    ax1 = fig.add_subplot(gs[1,1])
    #Set the scale.
    x_pos = np.array([0,1,2,3])
    bar_height=0,SIGMA_0,SIGMA_1,0
    if np.abs(np.amin(np.array([SIGMA_0,SIGMA_1]))-np.amax(np.array([SIGMA_0,SIGMA_1])))>0.001:
        ax1.set_ylim([np.amin(np.array([SIGMA_0,SIGMA_1]))*1.1,np.amax(np.array([SIGMA_0,SIGMA_1]))*1.1])
    ax1.bar(x_pos, bar_height,0.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(('','|0>','|1>',''),fontsize=fsize)
    if scale!='N':
        ax1.set_ylim([scale[2],scale[3]])

    if save_flag=='N':
        return fig
    else:
        plt.savefig(path_to_save)

def plot_sigma_time(SIGMA,SIGMA_0,SIGMA_1,T,fsize=20,save_flag='N',path_to_save=None):
    for t in range(T):
        #Define the scale for the right-side plot
        sc_min, sc_max = np.min(np.array([SIGMA_0,SIGMA_1])), np.max(np.array([SIGMA_0,SIGMA_1]))
        #Check that the scale is well behaved
        if np.isclose(np.abs(sc_min-sc_max),0,atol=10**(-5))==True:
            sc_min, sc_max = -0.05,0.05
        #Define the scale for the left side plot
        z_min, z_max = np.min(np.array(SIGMA)),np.max(np.array(SIGMA))
        #Check that the scale is well-behaved
        if np.isclose(np.abs(z_min-z_max),0,atol=10**(-5))==True:
            z_min,z_max=-0.05,0.05
        #Define the name for the file
        name='sigma_t{tt}'.format(tt=t)
        #Collect both scales into a tuple
        scala = (z_min,z_max,sc_min,sc_max)
        print('scala=',scala)
        plot_sigma(SIGMA[t],SIGMA_0[t],SIGMA_1[t],scale=scala,fsize=fsize,save_flag=save_flag,path_to_save=path_to_save+name)


def plot_flux(JP,JPHI,JP_0,JP_1,JPHI_0,JPHI_1,sc=1,fsize=20,save_flag='',path_to_save=None):
    X, Y = JP.shape
    Np, Nphi = X+1,Y+1
    P_Bound, PHI_Bound, P_Center, PHI_Center = discretization_properties(Np,Nphi)
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    fig.suptitle(r'$\vec{J}_t(\phi,p)$',fontsize=fsize)
    # First subplot
    ax0 = fig.add_subplot(gs[:,0])

    ax0.set_ylim([-0.1,1.1])
    ax0.set_xlim([0-0.1*2*np.pi,1.1*2*np.pi])
    for x in range(len(PHI_Bound)):
        ax0.plot(PHI_Bound[x]*np.ones(len(P_Bound)),P_Bound,color='Black',linestyle='dashed',linewidth=0.5)
    for y in range(len(PHI_Bound)):
        ax0.plot(PHI_Bound,P_Bound[y]*np.ones(len(P_Bound)),color='Black',linestyle='dashed',linewidth=0.5)

    ax0.quiver(PHI_Center, P_Center, JPHI, JP,angles='xy',scale_units='xy',scale=sc)
    X,Y = np.meshgrid(PHI_Center,P_Center)
    ax0.scatter(X,Y,s=5,color='red',marker='o')
    ax0.set_xlabel(r'$\phi$',fontsize=fsize)
    ax0.set_ylabel(r'p',fontsize=fsize)

    # Second subplot
    ax1 = fig.add_subplot(gs[:,1])
    for x in [0,len(PHI_Bound)-1]:
        ax1.plot(PHI_Bound[x]*np.ones(len(P_Bound)),P_Bound,color='Black',linestyle='dashed',linewidth=0.5)
    for y in [0,len(P_Bound)-1]:
        ax1.plot(PHI_Bound,P_Bound[y]*np.ones(len(P_Bound)),color='Black',linestyle='dashed',linewidth=0.5)

    ax1.set_ylim([-0.1,1.1])
    ax1.set_xlim([0-0.1*2*np.pi,1.1*2*np.pi])
    ax1.scatter(np.array([0,0]),np.array([0,1]),color='red')
    ax1.quiver(np.array([0,0]), np.array([0,1]), np.array([JPHI_0,JPHI_1]), np.array([JP_0,JP_1]),angles='xy',scale_units='xy',scale=sc)
    ax1.set_yticks(np.array([0,1]))
    ax1.set_yticklabels(('|0>','|1>'),fontsize=fsize)

    if save_flag=='N':
        return fig
    else:
        plt.savefig(path_to_save)

def plot_flux_time(JP,JPHI,JP_0,JP_1,JPHI_0,JPHI_1,T,sc=1,fsize=20,save_flag='N',path_to_save=None):
    for t in range(T):
        name='flux_t{tt}'.format(tt=t)
        plot_flux(JP[t],JPHI[t],JP_0[t],JP_1[t],JPHI_0[t],JPHI_1[t],sc,fsize,save_flag,path_to_save+name)
