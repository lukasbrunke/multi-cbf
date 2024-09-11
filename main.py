import os
import numpy as np
import casadi as cs
import time


from safety_filter import SafetyFilterQuadratic


# Define a linear casadi system as a function
def system_cs(x_cs, u_cs, A, B):
    # Define the system
    x_dot = A @ x_cs + B @ u_cs
    x_dot_func = cs.Function('x_next_func', [x_cs, u_cs], [x_dot])
    return x_dot_func


# Create Runge-Kutta 4 integrator using CasADi
def rk4_cs(x_cs, u_cs, x_dot_func, dt):
    k1 = x_dot_func(x_cs, u_cs)
    k2 = x_dot_func(x_cs + dt / 2 * k1, u_cs)
    k3 = x_dot_func(x_cs + dt / 2 * k2, u_cs)
    k4 = x_dot_func(x_cs + dt * k3, u_cs)
    x_next = x_cs + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x_next_func = cs.Function('x_next_func', [x_cs, u_cs], [x_next])
    return x_next_func


def calc_d(kappa, lipschitz, M, f_max, g_max, u_max, lipschitz_sys, dt):
    return - 1.0 / kappa *  (lipschitz * M * (f_max + g_max * u_max) / lipschitz_sys * 
        (1 - np.exp(lipschitz_sys * dt)))


def calc_f_max(A, P_list, c_list):
    opti = cs.Opti()

    state_dim = A.shape[0]

    x_opti = opti.variable(state_dim, 1)
    opti.set_initial(x_opti, 0.1 * np.ones((state_dim, 1)))

    obj = cs.norm_2(A @ x_opti) **2

    for i in range(len(P_list)):
        Pi = P_list[i]
        ci = c_list[i]

        opti.subject_to((x_opti - ci).T @ Pi @ (x_opti - ci) <= 1.0)

    opti.minimize(-obj)
    opti.solver('ipopt')

    sol = opti.solve()
    f_max = np.linalg.norm(sol.value(A @ x_opti))

    return f_max


def calc_Mi(B, P_list, c_list):
    Mi = []
    for i in range(len(P_list)):
        Pi = P_list[i]
        ci = c_list[i]

        opti = cs.Opti()

        state_dim = B.shape[0]

        x_opti = opti.variable(state_dim, 1)
        # opti.set_initial(x_opti, 0.1 * np.ones((state_dim, 1)))

        obj = - 2 * B.T @ Pi @ (x_opti - ci)

        for i in range(len(P_list)):
            Pi = P_list[i]
            ci = c_list[i]

            opti.subject_to((x_opti - ci).T @ Pi @ (x_opti - ci) <= 1.0)

        opti.minimize(-obj)
        opti.solver('ipopt')

        sol = opti.solve()
        M = abs(sol.value(obj))

        Mi.append(M)

    return Mi


def calc_lipschitz_const_policy(U_filtered, x1, x2, inside_safe_set):
    # Estimate the Lipschitz constant for the closed-loop policy
     
    lipschitz = 0.0
    
    x_id_offset = [1, 0, -1]
    y_id_offset = [1, 0, -1]
    
    # Go through all points that are in the safe set and calculate the Lipschitz constant
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = np.array([x1[i], x2[j]])
            ux = np.reshape(U_filtered[i, j], (-1, 1))
            if inside_safe_set[i, j] == 1:
                for x_offset in x_id_offset:
                    for y_offset in y_id_offset:
                        x_id = i + x_offset
                        y_id = j + y_offset
                        if x_id >= 0 and x_id < len(x1) and y_id >= 0 and y_id < len(x2) and not (x_offset == 0 and y_offset == 0):
                            if inside_safe_set[x_id, y_id] == 1:
                                if x_id == i and y_id == j:
                                    print("Indices are the same: {} - {} and {} - {}".format(i, x_id, j, y_id))
                                z = np.array([x1[x_id], x2[y_id]])
                                u_z = np.reshape(U_filtered[x_id, y_id], (-1, 1))
                                # Calculate the difference in states
                                state_diff = np.linalg.norm(x - z, 2)
                                input_diff = np.linalg.norm(ux - u_z, 2)

                                lipschitz_tmp = input_diff / state_diff
                                if lipschitz_tmp > lipschitz:
                                    lipschitz = lipschitz_tmp

    print("The policy's Lipschitz constant is {}".format(lipschitz))

    return lipschitz


def simulate_certified_sys(safety_filter, x_next_func, N, x0, u):
    print("Simulating the system in closed-loop with the safety filter")

    state_dim = x0.shape[0]
    input_dim = u.shape[0]

    # Simulate the system
    num_infeasible_points = 0
    infeasible_points = []
    x = np.zeros((state_dim, N))
    u_safe_traj = np.zeros((input_dim, N-1))
    u_unsafe_traj = np.zeros((input_dim, N-1)) 
    u_max_traj = np.zeros((input_dim, N-1))
    u_min_traj = np.zeros((input_dim, N-1))
    x[:, 0] = x0[:, 0]

    u_max_infeasible = []
    u_min_infeasible = []

    for i in range(N - 1):
        u_unsafe_traj[:, i] = u

        # Get input bounds
        input_bounds = safety_filter.input_bounds(x[:, [i]])
        u_max = input_bounds[0]
        u_min = input_bounds[1]
        u_max_traj[:, i] = u_max
        u_min_traj[:, i] = u_min

        if u_max == np.inf:
            u_max_infeasible.append(i)
        if u_min == -np.inf:
            u_min_infeasible.append(i)

        # Filter the input
        try:
            u_filtered = safety_filter.filter_input(x[:, [i]], u)
        except:
            print("Infeasible point detected")
            u_filtered = u
            num_infeasible_points += 1
            infeasible_points.append(i)
        # u_filtered = u
        u_safe_traj[:, i] = u_filtered
        x[:, i + 1] = x_next_func(x[:, i], u_filtered).full().flatten()

    print("Number of infeasible points: {}".format(num_infeasible_points))

    return x, u_safe_traj, u_unsafe_traj, u_max_traj, u_min_traj, infeasible_points


def determine_input_bounds(safety_filter, P_list, c_list, u, x1, x2):
    print("Determining input bounds")

    def cbf_params(P, c):
        def cbf(x):
            return 1 - (x - c).T @ P @ (x - c)
        
        return cbf
    
    cbfs = []
    for i in range(len(c_list)):
        cbfs.append(cbf_params(P_list[i], c_list[i]))

    u_max = - np.inf

    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack((X1.flatten(), X2.flatten()))
    U_min = np.zeros(X.shape[1])
    U_max = np.zeros(X.shape[1])
    U_filtered = np.zeros(X.shape[1])
    inside_safe_set = np.zeros(X.shape[1])
    infeasible_points = []
    num_infeasible_points = 0
    for i in range(X.shape[1]):
        input_bounds = safety_filter.input_bounds(X[:, [i]])
        U_max[i] = input_bounds[0]
        U_min[i] = input_bounds[1]

        try:
            u_filtered = safety_filter.filter_input(X[:, [i]], u)
        except:
            print("Infeasible point detected")
            u_filtered = u
            num_infeasible_points += 1
            infeasible_points.append(i)
        U_filtered[i] = u_filtered

        # Check if the point is inside the safe set: h_i(x) >= 0
        in_all_safe_sets = True
        for cbf in cbfs:
            if cbf(X[:, [i]]) <= -1e-6:   
                in_all_safe_sets = False
            
        if in_all_safe_sets:
            # Take the maximum input over all safe states
            u_max = max(u_max, max(abs(U_max[i]), abs(U_min[i])))
            inside_safe_set[i] = 1

    print("Number of infeasible points: {}".format(num_infeasible_points))
    print("u_max = {}".format(u_max))

    U_filtered = U_filtered.reshape(X1.shape)
    inside_safe_set = inside_safe_set.reshape(X1.shape)

    return u_max, U_max, U_min, U_filtered, inside_safe_set


def main(u_list, data_dir, N_list=[1000], dt=0.01, d_offsets=None, multi=True, opti_type="conic", wandb_project=""):
    # Initialize quadrotor motion
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, data_dir)
    # create file name with date and time stamp
    dir_name = time.strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(data_dir, dir_name)
    # create directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)
    print("Data will be saved to: ", data_dir)

    # Define the state and input dimensions
    state_dim = 2
    input_dim = 1

    # Define the casadi state and input variables
    x_cs = cs.SX.sym('x_cs', (state_dim, 1))
    u_cs = cs.SX.sym('u_cs', (input_dim, 1))

    # Define the system matrices
    A = np.array([[0, 1], [0.0, 0.0]])  # used for evaluation experiment data!
    B = np.array([[0], [1.0]])  # used for evaluatin experiment data!

    # Define the system
    x_dot_func = system_cs(x_cs, u_cs, A, B)

    # Quadratic CBF: Define the ellipsoid matrices in numpy
    P = np.array([[1.0, 0.0], [0.0, 2.0]])
    P_large = np.array([[0.5, 0.0], [0.0, 0.3]])
    # P_large = np.array([[0.4, 0.0], [0.0, 0.2]])
    
    # Quadratic CBF: Define the ellipsoid center
    c = np.array([[0], [0]])
    c_large_up = np.array([[0.0], [1.3]])
    c_large_down = np.array([[0.0], [-1.3]])

    # Quadratic CBF: Define the class K functions
    kappa_large_down = {"family": "piecewise", "slope_pos": 2.0, "slope_neg": 2.0, "inner_slope_pos": 1.0, "inner_slope_neg": 1.0, "offset": 0.0}
    kappa_large_up = {"family": "piecewise", "slope_pos": 2.0, "slope_neg": 2.0, "inner_slope_pos": 1.0, "inner_slope_neg": 1.0, "offset": 0.0}
    kappa = {"family": "piecewise", "slope_pos": 2.0, "slope_neg": 2.0, "inner_slope_pos": 1.0, "inner_slope_neg": 1.0, "offset": 0.0}

    # Select whether to use multiple CBFs or only a single one
    if multi:
        P_list = [P_large, P_large]
        c_list = [c_large_down, c_large_up]
        kappa_list = [kappa_large_down, kappa_large_up]

        if not d_offsets is None:
            kappa_large_down["offset"] = d_offsets[0]
            kappa_large_up["offset"] = d_offsets[1]
            if len(kappa_list) == 3:
                kappa["offset"] = d_offsets[2]
    else:
        P_list = [P]
        c_list = [c]
        kappa_list = [kappa]

    # Calculate f_max
    f_max = calc_f_max(A, P_list, c_list)

    # Calculate Mi
    Mi = calc_Mi(B, P_list, c_list)

    # Define the integrator
    x_next_func = rk4_cs(x_cs, u_cs, x_dot_func, dt)

    # Define the initial state
    x0 = np.array([[0.5], [0.3]])

    # Set up a grid for the input bounds
    x1_min, x1_max = -1.0, 1.0
    x2_min, x2_max = -0.7, 0.7
    num_points_x1 = 200
    num_points_x2 = 200
    x1 = np.linspace(x1_min, x1_max, num_points_x1)
    x2 = np.linspace(x2_min, x2_max, num_points_x2)

    if wandb_project != "":
        import wandb
        wandb.init(project=wandb_project, 
                config={'dir': data_dir, 
                        'is_real': False,
                        'N': N_list,
                        'dt': dt,
                        'multi': multi,
                        'kappas': kappa_list,
                        'P_list': P_list,
                        'c_list': c_list,
                        'A': A,
                        'B': B,
                        'x0': x0,
                        'u': u_list,
                        'x1_min': x1_min,
                        'x1_max': x1_max,
                        'x2_min': x2_min,
                        'x2_max': x2_max,
                        'num_points_x1': num_points_x1,
                        'num_points_x2': num_points_x2,
                        'd_offsets': d_offsets,})

    # Define the safety filter
    f_x = cs.Function('f_x', [x_cs], [A @ x_cs])
    g_x = cs.Function('g_x', [x_cs], [B])
    safety_filter = SafetyFilterQuadratic(input_dim, kappa_list, P_list, c_list, f_x, g_x, opti_type=opti_type)
    
    # Simulate the system using the safety filter
    if len(N_list) == 1:
        N = N_list[0]
        u = u_list[0]
        x, u_safe_traj, u_unsafe_traj, u_max_traj, u_min_traj, infeasible_points = simulate_certified_sys(safety_filter, x_next_func, N, x0, u)
    else:
        x = np.zeros((state_dim, sum(N_list)))
        u_safe_traj = np.zeros((input_dim, sum(N_list)-1))
        u_unsafe_traj = np.zeros((input_dim, sum(N_list)-1)) 
        u_max_traj = np.zeros((input_dim, sum(N_list)-1))
        u_min_traj = np.zeros((input_dim, sum(N_list)-1))
        infeasible_points = []
        
        # Initialize the state
        start_id = 0
        x[:, start_id] = x0[:, 0]

        for i in range(len(N_list)):
            N = N_list[i]
            
            if i != 1:
                if i == 0:
                    u = u_list[i]
                elif i == 2:
                    u = u_list[i - 1]
                    N += 1
                
                x_tmp, u_safe_traj[:, start_id:start_id + N - 1], u_unsafe_traj[:, start_id:start_id + N - 1], u_max_traj[:, start_id:start_id + N - 1], u_min_traj[:, start_id:start_id + N - 1], infeasible_points_i = simulate_certified_sys(safety_filter, x_next_func, N, x0, u)
                infeasible_points.extend([start_id + infeasible_point for infeasible_point in infeasible_points_i])
                x[:, start_id + 1 :start_id + N] = x_tmp[:, 1:]
                start_id += N - 1

                if i != len(N_list) - 1:
                    x0 = x[:, start_id] 

            else:
                for j in range(N):
                    print("Interpolating between the two control inputs")
                    N_interpolate = 2
                    u_interpolate = (N - j) / N * u_list[0] + j / N * u_list[1]
                    x_tmp, u_safe_traj[:, start_id + j], u_unsafe_traj[:, start_id + j], u_max_traj[:, start_id + j], u_min_traj[:, start_id + j], infeasible_points_i = simulate_certified_sys(safety_filter, x_next_func, N_interpolate, x0, u_interpolate)
                    infeasible_points.extend([start_id + j + infeasible_point for infeasible_point in infeasible_points_i])
                    x[:, start_id + j + 1] = x_tmp[:, -1]
                    x0 = x[:, start_id + j + 1]

                start_id += N

    # Determine the input bounds and the filtered control inputs over a grid
    u_max, U_max, U_min, U_filtered, inside_safe_set = determine_input_bounds(safety_filter, P_list, c_list, u, x1, x2)

    # Calculate the tightening of the constraints based on the sampling time
    lipschitz_sys = np.linalg.norm(A, 2)
    g_max = np.linalg.norm(B, 2)
    lipschitz = calc_lipschitz_const_policy(U_filtered, x1, x2, inside_safe_set)

    print("f_max = {}".format(f_max))
    print("g_max = {}".format(g_max))
    print("M_i = {}".format(Mi))
    print("L_pi = {}".format(lipschitz))
    print("L(u_max) = {}".format(lipschitz_sys))
    print("u_max = {}".format(u_max))
    print("dt = {}".format(dt))
    print("gamma_i = {}".format(kappa["slope_pos"]))

    d_offsets = []
    for M in Mi:
        d_offsets.append(- 1.0 / (kappa["slope_pos"]) * 
                         (lipschitz * M * (f_max + g_max * u_max) / lipschitz_sys * 
                          (1 - np.exp(lipschitz_sys * dt))))
    print("d_offsets = {}".format(d_offsets))

    if wandb_project != "":
        wandb.config.update({'f_max': f_max, 
                            'g_max': g_max, 
                            'Mi': Mi, 
                            'lipschitz': lipschitz, 
                            'lipschitz_sys': lipschitz_sys, 
                            'u_max': u_max, 
                            'd_offsets_result': d_offsets})
        
        wandb.finish()
    
    # Save the data
    np.savez(os.path.join(data_dir, 'data.npz'), 
             x=x, 
             u_safe_traj=u_safe_traj, 
             u_unsafe_traj=u_unsafe_traj, 
             u_max_traj=u_max_traj, 
             u_min_traj=u_min_traj, 
             infeasible_points=infeasible_points, 
             x1=x1, 
             x2=x2, 
             U_filtered=U_filtered, 
             inside_safe_set=inside_safe_set, 
             u_max=u_max, 
             U_max=U_max, 
             U_min=U_min)


if __name__ == '__main__':
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
    multi = config['multi']
    use_const_input = config['use_const_input']
    opti_type = config['opti_type']
    dt = config['dt']
    T = config['T']
    run_name = config['run_name']

    # Read WandB project name from separate config file
    with open ('configs/config.json', 'r') as f:
        config = json.load(f)

    wandb_project = config['wandb_project']
    data_dir = config['data_dir']

    # Define the number of steps to simulate
    N = int(T / dt)

    # Define the uncertified control input
    u = np.array([-0.1])
    if use_const_input:
        N_list = [N]
        u_list = [u]
    else:
        N_1 = int(T / 2 / dt)
        interpolation_time = 1.0
        N_interpolate = int(interpolation_time / dt)
        N_list = [N_1, N_interpolate, N - N_1 - N_interpolate]
        u_list = [u, -u]

    if run_name is None:
        d_offsets = None
        print("No offsets applied.")
    else:
        from plot_data import get_file_path_from_run

        _, run_json = get_file_path_from_run(wandb_project, run_name, file_name=None, use_latest=False)

        f_max = run_json['f_max']['value']
        g_max = run_json['g_max']['value']
        Mi = run_json['Mi']['value']
        lipschitz = run_json['lipschitz']['value']
        lipschitz_sys = run_json['lipschitz_sys']['value']
        u_max = run_json['u_max']['value']
        kappa_list = run_json['kappas']['value']

        # TODO assert that the CBFs match

        d_offsets = []
        for i in range(len(Mi)):
            M = Mi[i]
            kappa_slope = kappa_list[i]["slope_pos"]
            d_offsets.append(calc_d(kappa_slope, lipschitz, M, f_max, g_max, u_max, lipschitz_sys, dt))
        print(d_offsets)

    main(u_list=u_list, data_dir=data_dir, N_list=N_list, dt=dt, d_offsets=d_offsets, multi=multi, opti_type=opti_type, 
         wandb_project=wandb_project)
