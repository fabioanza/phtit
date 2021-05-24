import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_sigma(P_Bound, PHI_Bound, P_Center, PHI_Center,SIGMA,SIGMA_0,SIGMA_1,fsize=20,save_flag='N'):
    # Set up a figure twice as tall as it is wide
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(nrows=3, ncols=2)

    fig.suptitle(r'$\sigma_t(\phi,p)$',fontsize=fsize)
    # First subplot
    ax0 = fig.add_subplot(gs[:,0], projection='3d')

    _x = PHI_Bound[:-1]
    _y = P_Bound[:-1]
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = SIGMA.flatten()
    bottom = np.zeros_like(top)
    width = PHI_Bound[1]-PHI_Bound[0]
    depth = P_Bound[1]-P_Bound[0]

    ax0.bar3d(x, y, bottom, width, depth, top, shade=True)
    if np.abs(np.amin(SIGMA)-np.amax(SIGMA))>10**(-8):
        ax0.set_zlim([np.amin(SIGMA),np.amax(SIGMA)])
    ax0.set_xlabel(r'$\phi$',fontsize=fsize)
    ax0.set_ylabel(r'$p$',fontsize=fsize)

    # Second subplot
    ax1 = fig.add_subplot(gs[1,1])

    x_pos = np.array([0,1,2,3])
    bar_height=0,SIGMA_0,SIGMA_1,0
    if np.abs(np.amin(np.array([SIGMA_0,SIGMA_1]))-np.amax(np.array([SIGMA_0,SIGMA_1])))>0.001:
        ax1.set_ylim([np.amin(np.array([SIGMA_0,SIGMA_1]))*1.1,np.amax(np.array([SIGMA_0,SIGMA_1]))*1.1])
    ax1.bar(x_pos, bar_height,0.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(('','|0>','|1>',''),fontsize=fsize)

    return fig

def plot_flux(P_Bound, PHI_Bound, P_Center, PHI_Center,JP,JPHI,JP_0,JP_1,JPHI_0,JPHI_1,sc=1,fsize=20):
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

    return fig
