import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from pathlib import Path

# hovertext on plots should show coordinates and probability
# return text array isomorphic to coordinate mesh
def make_hovertext(labels, vals):
    text = []
    for i in range(len(vals[0].flat)):
        txt = ""
        for j in range(len(labels)):
            label = labels[j]
            val = vals[j].flat[i].round(4)
            txt += "{}: {}<br>".format(label, val)
        text.append(txt)
    text = np.array(text).reshape(vals[0].shape)
    return text

# colormap
def cmap():
    return [[0, '#0c2c84'],
            [.25, '#1d91c0'],
            [.5, '#7fcdbb'],
            [.75, '#c7e9b4'],
            [1, '#ffffd9']]

# spherical to cartesian
def spher2cart(theta, phi, r=1):
    return [r*np.sin(theta)*np.cos(phi),
            r*np.sin(theta)*np.sin(phi),
            r*np.cos(theta)]

def plot_series(series, title=None):
    fig = go.Figure(layout_title_text=title,
                    layout_xaxis = dict(
                        title_text="φ", title_font=dict(family="Old Standard TT, serif", size=24),
                        range=[0,2*np.pi], ticktext=["0", "π/2", "π", "3π/2", "2π"], tickvals=[0, 1.57, 3.14, 4.71, 6.28],
                        tickfont=dict(family="Old Standard TT, serif", size=20),
                    ),
                    layout_yaxis = dict(
                        title_text="p", title_font=dict(family="Old Standard TT, serif", size=24),
                        range=[0,1], tickvals=[0, 0.25, 0.5, 0.75, 1],
                        tickfont=dict(family="Old Standard TT, serif", size=20),
                    ),
                    layout_width=600,
                    layout_height=600
                   )
    p, phi = np.array([state.vec() for state in series]).T
    text = make_hovertext(["p", "φ"], [p, phi])
    fig.add_trace(go.Scattergl(x=phi, y=p,
                             hoverinfo="text", text=text,
                             mode='markers',
                            ))
    return fig


def plot(pdf, manifold, title=None, resolution=250):
    args = (pdf, title, resolution)
    if manifold == "plane":
        return plot_plane(*args)
    elif manifold == "sphere":
        return plot_sphere(*args)
    elif manifold == "disk":
        return plot_disk(*args)
    elif manifold == "surf":
        return plot_plane_surf(*args)
    else:
        print("Invalid manifold. Quitting.")

def plot_plane_surf(pdf, title, resolution):
    fig = go.Figure(layout_title_text=title,
                    layout_scene_xaxis = dict(
                        title_text="φ", title_font=dict(family="Old Standard TT, serif", size=20),
                        range=[0,2*np.pi], ticktext=["0", "π/2", "π", "3π/2", "2π"], tickvals=[0, 1.57, 3.14, 4.71, 6.28],
                        tickfont=dict(family="Old Standard TT, serif", size=14),
                    ),
                    layout_scene_yaxis = dict(
                        title_text="p", title_font=dict(family="Old Standard TT, serif", size=20),
                        range=[0,1], tickvals=[0, 0.25, 0.5, 0.75, 1],
                        tickfont=dict(family="Old Standard TT, serif", size=14),
                    ),
                    layout_scene_zaxis = dict(
                        title_text="q(p, φ)", title_font=dict(family="Old Standard TT, serif", size=20),
                        tickfont=dict(family="Old Standard TT, serif", size=14),
                    ),
                    layout_scene_camera_up= {'x':0, 'y':0, 'z':-1},
                    layout_scene_camera_eye={'x':-1.5, 'y':-1.5, 'z':1.5},
                    layout_width=600,
                    layout_height=600
                   )

    p, phi = np.linspace(0, 1, resolution), np.linspace(0, 2*np.pi, resolution)
    p, phi = np.meshgrid(p, phi)
    # get continuous distribution
    probs = pdf(p, phi)
    text = make_hovertext(["p", "φ", "PMF"], [p, phi, probs])
    fig.add_trace(go.Surface(x=phi, y=p, z=probs,
                               surfacecolor=probs, colorscale=cmap(), showscale=False, cmax=0.85, cmin=0,
                               contours_z=dict(show=True, width=1, project_z=True),
                               hoverinfo="text", text=text
                              ))
    return fig

def plot_plane(pdf, title, resolution):
    fig = go.Figure(layout_title_text=title,
                    layout_xaxis = dict(
                        title_text="φ", title_font=dict(family="Old Standard TT, serif", size=24),
                        range=[0,2*np.pi], ticktext=["0", "π/2", "π", "3π/2", "2π"], tickvals=[0, 1.57, 3.14, 4.71, 6.28],
                        tickfont=dict(family="Old Standard TT, serif", size=20),
                    ),
                    layout_yaxis = dict(
                        title_text="p", title_font=dict(family="Old Standard TT, serif", size=24),
                        range=[0,1], tickvals=[0, 0.25, 0.5, 0.75, 1],
                        tickfont=dict(family="Old Standard TT, serif", size=20),
                    ),
#                     layout_coloraxis_colorbar_tickfont=dict(family="Old Standard TT, serif", size=20),
                    layout_width=500,
                    layout_height=500
                   )
    # On plane: p, phi = x, y
    x, y = np.linspace(0, 2*np.pi, resolution), np.linspace(0, 1, resolution)
    phi, p = np.meshgrid(x, y)
    # get continuous distribution
    probs = pdf(p, phi)
    text = make_hovertext(["p", "φ", "q(p,φ)"], [p, phi, probs])
    fig.add_trace(go.Contour(x=x, y=y, z=probs,
                             contours_coloring='heatmap', line_width=.5,
                             colorscale=cmap(), showscale=True,
                             colorbar=dict(title_text="q(p,φ)", title_font=dict(family="Old Standard TT, serif", size=16),
                                           tickfont=dict(family="Old Standard TT, serif", size=16)),
                             hoverinfo="text", text=text
                            ))
    return fig

def plot_sphere(pdf, title, resolution):
    fig = go.Figure(layout_title_text=title,
                    layout_scene_xaxis = dict(
                        title_text="x = cos(φ)", title_font_family="Old Standard TT, serif",
                    ),
                    layout_scene_yaxis = dict(
                        title_text="y = sin(φ)", title_font_family="Old Standard TT, serif",
                    ),
                    layout_scene_zaxis = dict(
                        title_text="z = 1-2p", title_font_family="Old Standard TT, serif",
                    ),
                    layout_width=700,
                    layout_height=700
                   )

    # Use spherical coordinates for even coverage
    theta, phi = np.linspace(0, np.pi, resolution), np.linspace(0, 2*np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    x, y, z = spher2cart(theta, phi)
    # get p from spherical coordinates
    p = (1-np.cos(theta)) / 2
    # get continuous distribution
    probs = pdf(p, phi)
    text = make_hovertext(["p", "φ", "q(p,φ)"], [p, phi, probs])
    fig.add_trace(go.Surface(x=x, y=y, z=z,
                             surfacecolor=probs, opacity=1, showscale=True,
                             colorscale=cmap(),
                             hoverinfo="text", hovertext=text))
    return fig

def plot_disk(pdf, title, resolution):
    fig = go.Figure(layout_title_text=title,
                    layout_scene_xaxis_visible=False,
                    layout_scene_yaxis_visible=False,
                    layout_scene_zaxis_visible=False,
                    layout_width=500,
                    layout_height=500,
                    layout_dragmode=False,
                    layout_scene_camera_up= {'x':0, 'y':1, 'z':0},
                    layout_scene_camera_eye={'x':0, 'y':0, 'z':1.5},
                   )

    p, phi = np.linspace(0, 1, resolution), np.linspace(0, 2*np.pi, resolution)
    p, phi = np.meshgrid(p, phi)
    # get continuous distribution
    probs = pdf(p, phi)
    # convert to x,y coords
    b = np.sqrt(p) * np.exp(1j*phi)
    x, y = b.real, b.imag
    text = make_hovertext(["p", "φ", "z"], [p, phi, b.round(2)])
    fig.add_trace(go.Surface(x=x, y=y, z=0*x,
                             surfacecolor=probs, colorscale=cmap(), showscale=True,
                             hoverinfo="text", text=text
                            ))
    return fig

def plot_2_simplex(points=None, title=None, resolution=250):
    cam_r = 0.7
    z_offset = 0.2
    color="white"
    fig = go.Figure(layout_title_text=title,
                    layout_scene_xaxis_visible=False,
                    layout_scene_yaxis_visible=False,
                    layout_scene_zaxis_visible=False,
                    layout_width=500,
                    layout_height=500,
                    layout_dragmode=False,
                    layout_scene_camera_center_z=z_offset,
                    layout_scene_camera_eye={'x':cam_r, 'y':cam_r, 'z':cam_r+z_offset},
                   )

    theta, phi = np.linspace(0, np.pi/2, resolution), np.linspace(0, np.pi/2, resolution)
    theta, phi = np.meshgrid(theta, phi)
    x = (np.cos(phi)*np.sin(theta))**2
    y = (np.sin(phi)*np.sin(theta))**2
    z = np.cos(theta)**2
    fig.add_trace(go.Surface(x=x, y=y, z=z,
                               surfacecolor=0*x, colorscale=[color, color], showscale=False,
                               hoverinfo="none"
                              ))
    x_data, y_data, z_data = [], [], []
    if points is not None:
        x_data, y_data, z_data = points.T
    fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data,
                               mode="markers", marker={"size":2}))
    return fig


##################################
#####                       ######
#####   ANIMATION RENDERS   ######
#####                       ######
##################################


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$%s/%s$'%(latex,den)
            elif num==-1:
                return r'$-%s/%s$'%(latex,den)
            else:
                return r'$%s%s/%s$'%(num,latex,den)
    return _multiple_formatter


def print_dot(i, nstep):
    """
    Helper function to print progress bar for long renders
    """
    if i == 0:
        print('.', end='')
    elif int(i/nstep*100) > int((i-1)/nstep*100):
        print('.', end='')
    if i == nstep-1:
        print()


def animate_time_series(series, marker='.'):
    from ipywidgets import interact, Layout
    from ipywidgets.widgets import IntSlider, fixed

    def create_plot(time_series, t):
        ensemble = time_series[t]
        p = [state.p[0] for state in ensemble]
        phi = [state.phi[0] for state in ensemble]
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.plot(phi, p, marker)
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax.set_xlim(0, 2*np.pi)
        ax.set_xlabel("φ")
        ax.set_ylim(0, 1)
        ax.set_ylabel("p")
        plt.show()

    interact(create_plot, time_series=fixed(series), t=IntSlider(value=0, min=0, max=len(series)-1))


def gqs_animation_frames(gqs_series, dirpath, num_frames, tstep, show_sizes=True, dpi=120):

    tstep = int(tstep)
    num_frames = int(num_frames)

    def create_plot():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), dpi=dpi, sharey=True)
        ax_gqs, ax_maxent = axes

        ax_gqs.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax_gqs.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax_gqs.set_xlabel("φ")
        ax_gqs.set_xlim(0, 2*np.pi)
        ax_gqs.set_ylabel("p")
        ax_gqs.set_ylim(0, 1)
        gqs = ax_gqs.scatter([], [], s=.1,
                             color=(0, .4, 0.6, 0.5), edgecolor='none')

        ax_maxent.set_title("Max entropy state (insets: Bloch vector)")
        ax_maxent.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax_maxent.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax_maxent.set_xlabel("φ")
        ax_maxent.set_xlim(0, 2*np.pi)
        ax_maxent.set_ylim(0, 1)
        p_ = np.linspace(0, 1, 250)
        phi_ = np.linspace(0, 2*np.pi, 250)
        phi, p = np.meshgrid(phi_, p_)
        maxent = ax_maxent.pcolormesh(phi, p, p, cmap='bone')

        ax_bloch = ax_maxent.inset_axes([0.02, 0.02, 0.25, 0.25])
        ax_bloch.axhline(c='black', lw=0.5)
        ax_bloch.axhline(y=-1, ls='--', c='black', lw=0.5)
        ax_bloch.axhline(y=1, ls='--', c='black', lw=0.5)
        ax_bloch.set_ylim(-1.1, 1.1)
        ax_bloch.set_xticklabels('')
        ax_bloch.set_yticklabels('')
        bx = ax_bloch.plot([], [], label='x')
        by = ax_bloch.plot([], [], label='y')
        bz = ax_bloch.plot([], [], label='z')
        ax_bloch.legend(handlelength=.5, borderaxespad=0.2, handletextpad=0.3,
                  loc=4, prop={'size': 6})

        ax_pur = ax_maxent.inset_axes([0.02, 0.29, 0.25, 0.25])
        ax_pur.axhline(y=0.5, ls='--', c='black', lw=0.5)
        ax_pur.axhline(y=1, ls='--', c='black', lw=0.5)
        ax_pur.set_ylim(0.25, 1.05)
        ax_pur.set_xticklabels('')
        ax_pur.set_yticklabels('')
        pur = ax_pur.plot([], [], label='purity')
        ax_pur.legend(handlelength=.5, borderaxespad=0.2, handletextpad=0.3,
                  loc=4, prop={'size': 6})

        plt.tight_layout()

        return (ax_gqs, ax_maxent, ax_bloch, ax_pur), (gqs, maxent, bx[0], by[0], bz[0], pur[0])

    def rho_to_bloch(rho):
        """
        Computes bloch vector from density matrix
        """
        x = 2*rho[1, 0].real
        y = 2*rho[1, 0].imag
        z = 2*rho[0, 0].real - 1
        return np.array([x, y, z])

    from gqm import maximum_entropy

    rhos = np.array([gqs.rho() for gqs in gqs_series])
    purity = [np.trace(rho@rho).real for rho in rhos]
    bloch = np.array([rho_to_bloch(rho) for rho in rhos])

    if (num_frames-1)*tstep >= len(gqs_series):
        print("Time series not long enough. Aborting.")
        return

    Path(dirpath).mkdir(parents=True, exist_ok=True)
    print("Rendering frames at {}:".format(dirpath))
    axes, plots = create_plot()
    ax_gqs, ax_maxent, ax_bloch, ax_pur = axes
    plot_gqs, plot_maxent, plot_bx, plot_by, plot_bz, plot_pur = plots
    for i in range(1, num_frames):
        print_dot(i, num_frames)
        t = i * tstep

        gqs = gqs_series[t]
        p, phi = gqs.to_pphi()
        plot_gqs.set_offsets(np.array([phi, p]).T)
        size_factor = 1000 if show_sizes else 0
        min_size = 5
        plot_gqs.set_sizes(gqs.weights*size_factor + 3)

        rho = rhos[t]
        p_ = np.linspace(0, 1, 250)
        phi_ = np.linspace(0, 2*np.pi, 250)
        phi, p = np.meshgrid(phi_, p_)
        q = maximum_entropy(rho, np.array([np.sqrt(1-p), np.sqrt(p)*np.exp(1j*phi)]), 1/2)
        q = q[:-1,:-1]
        plot_maxent.set_array(q.ravel())
        plot_maxent.set_clim(vmin=q.min(), vmax=q.max())

        interval = 25
        start = t - interval if t - interval > 0 else 0
        end = start + 2*interval
        times = np.arange(start, t)
        pur = purity[start:t]
        bx, by, bz = bloch[start:t].T
        plot_bx.set_data(times, bx)
        plot_by.set_data(times, by)
        plot_bz.set_data(times, bz)
        plot_pur.set_data(times, pur)
        ax_bloch.set_xlim(start, end)
        ax_pur.set_xlim(start, end)

        ax_gqs.set_title("Geometric state, t={}".format(t))

        plt.savefig(dirpath+'{:04d}.png'.format(i))
    print("Done.")
    print()

    plt.cla()
    plt.close()


def gqs_animation_3D_frames(gqs_series, dirpath, num_frames, tstep, sphere_res=300,
                            camera={'x':1.5, 'y':-1.5, 'z':0.25}, camera_up={'x':0, 'y':0, 'z':1}):
    def cmap():
        return [[0, '#F9FCF5'],
                [0.005, '#CCD6EB'],
                [.33, '#A3C3D9'],
                [.66, '#6E4BAA'],
                [1, '#392759']]

    if (num_frames-1)*tstep >= len(gqs_series):
        print("Time series not long enough. Aborting.")
        return

    fig = go.Figure(
                    layout_scene_xaxis = dict(
                        title_text="x = cos(φ)", title_font_family="Old Standard TT, serif",
                    ),
                    layout_scene_yaxis = dict(
                        title_text="y = sin(φ)", title_font_family="Old Standard TT, serif",
                    ),
                    layout_scene_zaxis = dict(
                        title_text="z = 1-2p", title_font_family="Old Standard TT, serif",
                    ),
                    layout_scene_camera_up=camera_up,
                    layout_scene_camera_eye=camera,
                    layout_width=700,
                    layout_height=700
                   )
    theta, phi = np.linspace(0, np.pi, sphere_res), np.linspace(0, 2*np.pi, 2*sphere_res)
    theta, phi = np.meshgrid(theta, phi)
    x, y, z = spher2cart(theta, phi)
    distr = phi.copy()
    fig.add_trace(go.Surface(x=x, y=y, z=z,
                             surfacecolor=distr, showscale=False,
                             colorscale=cmap()))

    Path(dirpath).mkdir(parents=True, exist_ok=True)

    print("Rendering frames at {}:".format(dirpath))
    for i in range(1, num_frames):
        print_dot(i, num_frames)
        t = i * tstep
        gqs = gqs_series[t]

        distr *= 0
        bump = distr.copy()
        for weight, chi in zip(gqs.weights, gqs.chis):
            theta_ = 2*np.arccos(chi[0]).real
            phi_ = np.angle(chi[1])
            dist_FS = 0.5 * np.arccos(np.cos(theta)*np.cos(theta_) + np.sin(theta)*np.sin(theta_)*np.cos(phi_-phi))
            bump *= 0
            bump += weight
            bump[0.008 - dist_FS < 0] = 0
            distr += bump
            del dist_FS
        del bump
        fig.update_layout(title="t={}".format(t))
        fig.update_traces(surfacecolor=distr, lightposition=dict(x=-12, y=0, z=0),
                          lighting=dict(ambient=0.6, diffuse=0.5))
        fig.write_image(dirpath+'{:04d}.png'.format(i), engine="kaleido")


def gqs_animation_3D_html(gqs_series, dirpath, num_frames, tstep,
                          show_sizes=True, offset=0,
                          camera={'x':1.5, 'y':-1.5, 'z':0.25}, camera_up={'x':0, 'y':0, 'z':1}):
    def cmap():
        return [[0, '#F9FCF5'],
                [1, '#F9FCF5']]

    if (num_frames-1)*tstep >= len(gqs_series):
        print("Time series not long enough. Aborting.")
        return

    theta, phi = np.linspace(0, np.pi, 200), np.linspace(0, 2*np.pi, 400)
    theta, phi = np.meshgrid(theta, phi)
    x, y, z = spher2cart(theta, phi)
    distr = phi.copy()

    Path(dirpath).mkdir(parents=True, exist_ok=True)

    print("Rendering interactive model ...")
    scatters = []
    for i in range(1, num_frames):
        print_dot(i, num_frames)
        t = i * tstep
        gqs = gqs_series[t]

        theta_ = 2*np.arccos(gqs.chis[:, 0]).real
        phi_ = np.angle(gqs.chis[:, 1])
        x_, y_, z_ = spher2cart(theta_, phi_, r=1+offset)
        wts = 400*gqs.weights if show_sizes else 2
        scatters.append((x_, y_, z_, wts))

    sphere = go.Surface(x=x, y=y, z=z,
                        surfacecolor=0*x, showscale=False,
                        colorscale=cmap(), lighting=dict(diffuse=0.9),
                    )
    frames = []
    for i, scatter in enumerate(scatters):
        sx, sy, sz, wts = scatter
        frame = go.Frame(data=[sphere,
                               go.Scatter3d(x=sx, y=sy, z=sz,
                                            mode='markers',
                                            marker_size=wts,marker_color=gqs.weights,
                                            marker_colorscale='ylorrd_r',
                                            marker_colorbar=dict(tickmode='auto',nticks=5,title='Probability Mass')
                                           ),
                              ],
                         name=str(i))
        frames.append(frame)
    fig = go.Figure(frames=frames)
    fig.add_trace(sphere)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))

    # animation stuff
    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    # Layout
    fig.update_layout(
            scene_xaxis = dict(
                title_text="x = cos(φ)", title_font_family="Old Standard TT, serif",
            ),
            scene_yaxis = dict(
                title_text="y = sin(φ)", title_font_family="Old Standard TT, serif",
            ),
            scene_zaxis = dict(
                title_text="z = 1-2p", title_font_family="Old Standard TT, serif",
            ),
            scene_camera_up=camera_up,
            scene_camera_eye=camera,
            width=700,
            height=700,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=[
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]
    )
    print("Saving interactive model at {}interactive.html:".format(dirpath))
    fig.write_html(dirpath+'interactive.html')
