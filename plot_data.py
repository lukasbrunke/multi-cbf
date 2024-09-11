from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import wandb
import json
import os
import tikzplotlib


from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)


def get_file_path_from_run(wandb_project, run_name=None, file_name=None, use_latest=False):
    """Get the file path from the specified run name or file name."""
    # Use the latest run if no run name or file name is provided
    if run_name is None and file_name is None:
        use_latest = True

    wandb_api = wandb.Api()
    runs = wandb_api.runs(wandb_project)
    
    if use_latest:
        # Get the latest run
        run = runs[0]
    elif run_name is not None:
        # Get the run with the specified name
        run = None
        for r in runs:
            if r.name == run_name:
                run = r
                break
        if run is None:
            raise ValueError("Run with name {} not found".format(run_name))
    elif file_name is not None:
        # Get the run with the specified file name
        run = None
        for r in runs:
            r_json = json.loads(r.json_config)
            run_file_name = r_json['dir']['value'].rsplit('/')[-1]
            print(run_file_name, file_name)
            if run_file_name == file_name:
                run = r
                break
        if run is None:
            raise ValueError("Run with file name {} not found".format(file_name))
    else:
        raise ValueError("No run name or file name provided")
    
    run_json = json.loads(run.json_config)
    print("Run name: ", run.name)

    file_path = run_json['dir']['value']

    return file_path, run_json


# Define a function to plot a 2d ellipsoid
def plot_ellipsoid_cs(P, c, offset=0.0, color='gray', alpha=0.2, ax=None, fill=True):
    # Make sure that c is in the correct shape
    if len(c.shape) == 2:
        c = c.flatten()

    # Find the eigenvalues and eigenvectors of the matrix P
    eig_val, eig_vec = np.linalg.eig(P)
    # Find the rotation angle of the ellipse
    theta = np.arctan2(eig_vec[1, 0], eig_vec[0, 0])
    # Find the length of the major and minor axes
    a = 1 / np.sqrt(eig_val[0]) * np.sqrt(1 - offset)
    b = 1 / np.sqrt(eig_val[1]) * np.sqrt(1 - offset)
    # Create an ellipse with the given parameters
    ellipse = Ellipse(xy=c, width=2 * a, height=2 * b, angle=np.rad2deg(theta), color=color,
                                         alpha=alpha, fill=fill)
    
    if ax is None:
        ax = plt.gca()
    # Add the ellipse to the plot
    ax.add_artist(ellipse)


def plot_ellipsoids(ax, P_list, c_list, kappa_list, B, xlims, ylims, fill=True):

    for i in range(len(c_list)):
        c = c_list[i]
        P = P_list[i]
        offset = kappa_list[i]["offset"]

        # plot the ellipsoid
        plot_ellipsoid_cs(P, c, offset=offset, alpha=0.4, ax=ax, fill=fill)
        
        # Plot the vector resulting from P @ B in both directions
        PB = (P @ B).flatten()

        scale = 0.0
        if PB[0] >= 1e-6:
            scale = max(scale, - (ylims[0] - c[0]) / PB[0],
                        (ylims[1] - c[0]) / PB[0])
        if PB[1] >= 1e-6:
            scale = max(scale, - (xlims[0] - c[1]) / PB[1],
                        (xlims[1] - c[1]) / PB[1])

        # Plot the vector normal to PB in both directions
        ax.plot([c[0], c[0] - scale * PB[1]], [c[1], c[1] + scale * PB[0]], 'g', label='L{}'.format(i))
        ax.plot([c[0], c[0] + scale * PB[1]], [c[1], c[1] - scale * PB[0]], 'g')


def plot_colormesh_and_bar(fig, ax, X1, X2, Z, cmap='RdBu'):
    # plot the heatmap using divided colormap
    ax.pcolormesh(X1, X2, Z, cmap=cmap, vmin=-np.max(abs(Z)), vmax=np.max(abs(Z)))

    # add colorbar to ax
    fig.colorbar(ax.pcolormesh(X1, X2, Z, cmap=cmap, vmin=-np.max(abs(Z)), vmax=np.max(abs(Z))), ax=ax, )


def plot_data(data_dir, run_json):     
    # Set the number of steps to skip when plotting the time-based data
    skip_steps = 1

    # Load the data from the run json
    kappa_list = run_json['kappas']['value']
    P_list = [np.array(P) for P in run_json['P_list']['value']]
    c_list = [np.array(c) for c in run_json['c_list']['value']]
    A = np.array(run_json['A']['value'])
    B = np.array(run_json['B']['value'])
    dt = run_json['dt']['value']
    N_list = run_json['N']['value']
    x1_min = run_json['x1_min']['value']
    x1_max = run_json['x1_max']['value']
    x2_min = run_json['x2_min']['value']
    x2_max = run_json['x2_max']['value']
    
    # Load npz data file
    data = np.load(os.path.join(data_dir, 'data.npz'))
    # Extract the data
    x = data['x']
    u_safe_traj = data['u_safe_traj']
    u_unsafe_traj = data['u_unsafe_traj']
    u_max_traj = data['u_max_traj']
    u_min_traj = data['u_min_traj']
    infeasible_points = data['infeasible_points']
    x1 = data['x1']
    x2 = data['x2']
    U_filtered = data['U_filtered']
    inside_safe_set = data['inside_safe_set']
    u_max = data['u_max']
    U_max = data['U_max']
    U_min = data['U_min']

    # First set of subplots: U_max, U_min, U_max - U_min over the state space
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Find the indices where U_max is inf
    u_max_infeasible = np.where(np.isinf(U_max))
    # Find the indices where U_min is -inf
    u_min_infeasible = np.where(np.isinf(-U_min))

    # Find the indices where Y_max - Y_min is less than zero
    u_infeasible = np.where(U_max - U_min < 0)

    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack((X1.flatten(), X2.flatten()))

    U_max = U_max.reshape(X1.shape)
    U_min = U_min.reshape(X1.shape)
    U_filtered = U_filtered.reshape(X1.shape)
    inside_safe_set = inside_safe_set.reshape(X1.shape)
    
    # plot the heatmap using divided colormap
    plot_colormesh_and_bar(fig, ax1, X1, X2, U_max, cmap='RdBu')
    plot_colormesh_and_bar(fig, ax2, X1, X2, U_min, cmap='RdBu')
    plot_colormesh_and_bar(fig, ax3, X1, X2, U_max - U_min, cmap='RdBu')

    # plot the unconstrained inputs
    ax1.plot(X[0, u_max_infeasible], X[1, u_max_infeasible], 'rx')
    ax2.plot(X[0, u_min_infeasible], X[1, u_min_infeasible], 'rx')
    ax3.plot(X[0, u_infeasible], X[1, u_infeasible], 'rx')

    axes = [ax1, ax2, ax3]

    xlims = [x1_min, x1_max]
    ylims = [x2_min, x2_max]

    for ax in axes:
        plot_ellipsoids(ax, P_list, c_list, kappa_list, B, xlims, ylims, fill=False)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    # Save the plots as tikz
    tikzplotlib.save(os.path.join(data_dir, "U_max_min.tex"))

    # Save the plots as png
    plt.savefig(os.path.join(data_dir, "U_max_min.png"))

    # Second set of subplots: U_filtered over the state space
    fig, ax = plt.subplots(1, 1)

    plot_colormesh_and_bar(fig, ax, X1, X2, U_filtered, cmap='RdBu')

    plot_ellipsoids(ax, P_list, c_list, kappa_list, B, xlims, ylims, fill=False)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # Save the plots as tikz
    tikzplotlib.save(os.path.join(data_dir, "U_filtered.tex"))

    # Save the plots as png
    plt.savefig(os.path.join(data_dir, "U_filtered.png"))
    
    # Third set of subplots: x(t) and u(t) over time
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    xlims = [-2, 2]
    ylims = [-2, 2]

    plot_ellipsoids(ax1, P_list, c_list, kappa_list, B, xlims, ylims)

    # Plot the results
    ax1.plot(x[0, ::skip_steps], x[1, ::skip_steps], 'rx-', label='x(t)')

    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()

    # plot the input
    time_steps = np.arange(sum(N_list) - 1) * dt
    ax2.plot(time_steps[::skip_steps], u_unsafe_traj.flatten()[::skip_steps], 'b', label='$\pi(x)$')
    ax2.plot(time_steps[::skip_steps], u_safe_traj.flatten()[::skip_steps], 'r', label='u_s(x)')

    if len(P_list) > 1:
        ax2.plot(time_steps[::skip_steps], u_max_traj.flatten()[::skip_steps], 'k--', label='u_max')
        ax2.plot(time_steps[::skip_steps], u_min_traj.flatten()[::skip_steps], 'k--', label='u_min')

    if len(infeasible_points) > 0:
        print("Infeasible points detected")
        infeasible_points = np.array(infeasible_points).T
        ax2.plot(time_steps[infeasible_points], u_safe_traj.flatten()[infeasible_points], 'bx', label='infeasible points')

    ax2.set_ylim([-1.1, 1.1])
    ax2.set_xlabel('$t$ [s]')
    ax2.set_ylabel('$u$')
    ax2.legend()

    time_steps = np.arange(sum(N_list)) * dt
    ax3.plot(time_steps[::skip_steps], x[0, ::skip_steps], 'r', label='$x_1(t)$')
    ax3.plot(time_steps[::skip_steps], x[1, ::skip_steps], 'b', label='$x_2(t)$')
    ax3.set_xlabel('$t$ [s]')
    ax3.set_ylabel('$x$')
    ax3.legend()

    # Save the plots as tikz
    tikzplotlib.save(os.path.join(data_dir, "x_u_t.tex"))

    # Save the plots as png
    plt.savefig(os.path.join(data_dir, "x_u_t.png"))

    plt.show()


if __name__ == "__main__":
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Read parameters from config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract parameters from config dictionary
    run_name = config['run_name']
    file_name = config['file_name']
    use_latest = config['use_latest']

    # Read WandB project name from separate config file
    with open ('configs/config.json', 'r') as f:
        config = json.load(f)

    wandb_project = config['wandb_project']

    if file_name is None and run_name is None:
        use_latest = True
    
    file_path, run_json = get_file_path_from_run(wandb_project, run_name, file_name, use_latest)
    print("Plotting data from: ", file_path)

    plot_data(file_path, run_json)
