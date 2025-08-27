import sys
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import os


def random_orthogonal_matrix(n):
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    # Ensure Q is uniformly random
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


def sample_eigenvalues(n, lam_min=1.0, lam_max=10.0):
    return np.random.uniform(lam_min, lam_max, size=n)


def sample_pd_matrix(n, lam_min=1.0, lam_max=10.0, diag_only=False):
    Q = random_orthogonal_matrix(n)
    lambdas = sample_eigenvalues(n, lam_min, lam_max)
    Lambda = np.diag(lambdas)
    if diag_only:
        return Lambda
    return Q @ Lambda @ Q.T


def sample_bounded_vector(n, limits):
    rand_vec = np.random.rand(n)
    for i in range(n):
        # scale the i-th element of the vector to be within the limits
        rand_vec[i] = rand_vec[i] * (limits[i][1] - limits[i][0]) + limits[i][0]
    return rand_vec


def sample_cbf_params(state_dim, state_lim, eig_lim, diag_only=False):
    P = sample_pd_matrix(state_dim, eig_lim[0], eig_lim[1], diag_only)
    c = sample_bounded_vector(state_dim, state_lim)
    return P, c


def plot_cbf_level_sets(P, c, P_list, c_list, state_lim, box=None, x_samples=None, infeasible_indices=None):
    x = np.linspace(state_lim[0][0], state_lim[0][1], 100)
    y = np.linspace(state_lim[1][0], state_lim[1][1], 100)
    X, Y = np.meshgrid(x, y)
    # Random colors
    colors = ['red', 'blue', 'green', 'orange']
    Z = 1 - (X - c[0])**2 * P[0, 0] - (Y - c[1])**2 * P[1, 1] - 2 * (X - c[0]) * (Y - c[1]) * P[0, 1]
    plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2, linestyles='dashed')

    for i in range(len(P_list)):
        Z = 1 - (X - c_list[i][0])**2 * P_list[i][0, 0] - (Y - c_list[i][1])**2 * P_list[i][1, 1] - 2 * (X - c_list[i][0]) * (Y - c_list[i][1]) * P_list[i][0, 1]
        plt.contour(X, Y, Z, levels=[0], colors=colors[i], linewidths=2)

    if box is not None:
        bx_min = box[0][0]
        bx_max = box[0][1]
        by_min = box[1][0]
        by_max = box[1][1]
        plt.plot([bx_min, bx_max], [by_min, by_min], 'k-', linewidth=2)
        plt.plot([bx_min, bx_min], [by_min, by_max], 'k-', linewidth=2)
        plt.plot([bx_max, bx_max], [by_min, by_max], 'k-', linewidth=2)
        plt.plot([bx_min, bx_max], [by_max, by_max], 'k-', linewidth=2)

    if x_samples is not None:
        plt.scatter(x_samples[:, 0], x_samples[:, 1], s=1, c='black')

        if infeasible_indices is not None:
            plt.scatter(x_samples[infeasible_indices, 0], x_samples[infeasible_indices, 1], s=1, c='red')

    plt.xlim(state_lim[0])
    plt.ylim(state_lim[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Level sets of CBFs')
    plt.show()


class SubsetChecker:
    def __init__(self, P, c, num_cbfs, opti_type='', solver='ipopt'):
        self.P = P
        self.c = c
        self.state_dim = P.shape[0]
        self.num_cbfs = num_cbfs
        self._setup_solver(opti_type, solver)
        self._setup_optimizer()

    def _setup_solver(self, opti_type, solver):
        """Set up solver type and options."""
        self.opti_type = opti_type
        self.solver = solver
        
        # Configure solver options based on optimization type
        if self.opti_type == 'conic':
            self.opts = {'printLevel': 'low'}
        else:
            self.solver = 'ipopt'
            self.opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

        # Create optimizer object based on type
        if self.opti_type == 'conic':
            self.opti = cs.Opti(self.opti_type)
        else:
            self.opti = cs.Opti()

        # Configure solver
        self.opti.solver(self.solver, self.opts)

    def _setup_optimizer(self):
        self.x_opti = self.opti.variable(self.state_dim)

        self.P_list = [None] * self.num_cbfs
        self.c_list = [None] * self.num_cbfs
        for i in range(self.num_cbfs):
            self.P_list[i] = self.opti.parameter(self.state_dim, self.state_dim)
            self.c_list[i] = self.opti.parameter(self.state_dim)

        self.cost = (self.x_opti - self.c).T @ self.P @ (self.x_opti - self.c)

        self.opti.minimize(- self.cost)

        self.constraints = []

        for i in range(self.num_cbfs):
            self.constraints.append((self.x_opti - self.c_list[i]).T @ self.P_list[i] @ (self.x_opti - self.c_list[i]) <= 1.0)

        self.opti.subject_to(self.constraints)

    def _set_parameters(self, P_list, c_list):
        for i in range(self.num_cbfs):
            self.opti.set_value(self.P_list[i], P_list[i])
            self.opti.set_value(self.c_list[i], c_list[i])

    def solve(self, P_list, c_list):
        self._set_parameters(P_list, c_list)
        try:
            self.opti.solve()
            max_value = self.opti.value(self.cost)
            return max_value
        except Exception as e:
            # print(e)
            return None
        
    def verify_subset(self, P_list, c_list, tolerance=1e-6):
        """Check if subset relationship holds"""
        try:
            max_value = self.solve(P_list, c_list)
            if max_value is not None:
                return max_value <= 1.0 + tolerance
            else:
                return False
        except RuntimeError:
            return False  # Conservative: assume verification failed


class BoundingBoxFinder:
    def __init__(self, state_dim, num_cbfs, opti_type='', solver='ipopt'):
        self.state_dim = state_dim
        self.num_cbfs = num_cbfs
        self._setup_solver(opti_type, solver)
        self._setup_optimizer()
        self.directions = np.zeros((self.state_dim * 2, self.state_dim))
        for i in range(self.state_dim):
            self.directions[i, i] = -1.0
            self.directions[i + self.state_dim, i] = 1.0

        # print(self.directions)
        self.limits = np.zeros((self.state_dim * 2))

    def limits_to_box(self, limits):
        box = []
        # print(len(box))
        # print(limits.shape)
        for i in range(self.state_dim):
            limit_i = [limits[i], limits[i + self.state_dim]]
            box.append([min(limit_i), max(limit_i)])
        return box

    def _setup_solver(self, opti_type, solver):
        """Set up solver type and options."""
        self.opti_type = opti_type
        self.solver = solver
        
        # Configure solver options based on optimization type
        if self.opti_type == 'conic':
            self.opts = {'printLevel': 'low'}
        else:
            self.solver = 'ipopt'
            self.opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

        # Create optimizer object based on type
        if self.opti_type == 'conic':
            self.opti = cs.Opti(self.opti_type)
        else:
            self.opti = cs.Opti()

        # Configure solver
        self.opti.solver(self.solver, self.opts)

    def _setup_optimizer(self):
        self._setup_solver('', 'ipopt')

        self.x_opti = self.opti.variable(self.state_dim)

        self.direction = self.opti.parameter(self.state_dim)

        self.P_list = [None] * self.num_cbfs
        self.c_list = [None] * self.num_cbfs
        for i in range(self.num_cbfs):
            self.P_list[i] = self.opti.parameter(self.state_dim, self.state_dim)
            self.c_list[i] = self.opti.parameter(self.state_dim)

        self.cost = self.direction.T @ self.x_opti

        self.opti.minimize(- self.cost)

        self.constraints = []

        for i in range(self.num_cbfs):
            self.constraints.append((self.x_opti - self.c_list[i]).T @ self.P_list[i] @ (self.x_opti - self.c_list[i]) <= 1.0)

        self.opti.subject_to(self.constraints)

    def _set_parameters(self, direction, P_list, c_list):
        self.opti.set_value(self.direction, direction)
        for i in range(self.num_cbfs):
            self.opti.set_value(self.P_list[i], P_list[i])
            self.opti.set_value(self.c_list[i], c_list[i])

    def solve(self, P_list, c_list):
        for i, direction in enumerate(self.directions):
            # print(f"direction: {direction}")
            self._set_parameters(direction, P_list, c_list)
            try:
                self.opti.solve()
                limit = np.abs(direction).T @ self.opti.value(self.x_opti)
                self.limits[i] = limit
            except Exception as e:
                print(e)
                return None
            
        return self.limits

class RelativeDegreeChecker:

    def __init__(self, B, P, c, num_cbfs, opti_type='', solver='ipopt'):
        self.B = B
        self.P = P
        self.c = c
        self.state_dim = P.shape[0]
        self.num_cbfs = num_cbfs
        self._setup_solver(opti_type, solver)
        self._setup_optimizer()

    def _setup_solver(self, opti_type, solver):
        """Set up solver type and options."""
        self.opti_type = opti_type
        self.solver = solver
        
        # Configure solver options based on optimization type
        if self.opti_type == 'conic':
            self.opts = {'printLevel': 'low'}
        else:
            self.solver = 'ipopt'
            self.opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
            
        # Create optimizer object based on type
        if self.opti_type == 'conic':
            self.opti = cs.Opti(self.opti_type)
        else:
            self.opti = cs.Opti()

        # Configure solver
        self.opti.solver(self.solver, self.opts)

    def _setup_optimizer(self):
        self._setup_solver('', 'ipopt')

        self.x_opti = self.opti.variable(self.state_dim)

        self.P_i = self.opti.parameter(self.state_dim, self.state_dim)
        self.c_i = self.opti.parameter(self.state_dim)

        self.P_list = [None] * self.num_cbfs
        self.c_list = [None] * self.num_cbfs
        for i in range(self.num_cbfs):
            self.P_list[i] = self.opti.parameter(self.state_dim, self.state_dim)
            self.c_list[i] = self.opti.parameter(self.state_dim)
        
        L = ((self.x_opti - self.c_i).T @ self.P_i @ self.B).T @ (self.x_opti - self.c_i)
        self.cost = L.T @ L

        self.opti.minimize(self.cost)

        self.constraints = []

        for i in range(self.num_cbfs):
            self.constraints.append((self.x_opti - self.c_list[i]).T @ self.P_list[i] @ (self.x_opti - self.c_list[i]) <= 1.0)

        self.opti.subject_to(self.constraints)

    def _set_parameters(self, P_i, c_i, P_list, c_list):
        self.opti.set_value(self.P_i, P_i)
        self.opti.set_value(self.c_i, c_i)
        for i in range(self.num_cbfs):
            self.opti.set_value(self.P_list[i], P_list[i])
            self.opti.set_value(self.c_list[i], c_list[i])

    def solve(self, P_list, c_list):
        L_min = np.inf
        for i in range(self.num_cbfs):
            P_i = P_list[i]
            c_i = c_list[i]
            self._set_parameters(P_i, c_i, P_list, c_list)

            try:
                self.opti.solve()
                L_value = self.opti.value(self.cost)
                L_min = min(L_min, L_value)
                # print(f"L_value: {L_value}")
            except Exception as e:
                print(e)
            
        return L_min
    

class StateSampler:

    def __init__(self, state_lim):
        self.state_dim = len(state_lim)
        self.state_lim = state_lim

    def sample(self):
        x_sample = np.random.rand(self.state_dim)
        for i in range(self.state_dim):
            # scale the i-th element of the vector to be within the limits
            x_sample[i] = x_sample[i] * (self.state_lim[i][1] - self.state_lim[i][0]) + self.state_lim[i][0]
        return x_sample
    
    def sample_feasible(self, P_list, c_list):
        x_sample = self.sample()
        for i in range(len(P_list)):
            if (x_sample - c_list[i]).T @ P_list[i] @ (x_sample - c_list[i]) >= 1.0:
                return self.sample_feasible(P_list, c_list)
        return x_sample.reshape((self.state_dim, 1))
    

class FeasibilityChecker:
    def __init__(self, A, B, gamma, num_cbfs, opti_type='', solver='ipopt'):
        self.A = A
        self.B = B
        self.gamma = gamma
        self.state_dim = A.shape[0]
        self.input_dim = B.shape[1]
        self.num_cbfs = num_cbfs
        self._setup_solver(opti_type, solver)
        self._setup_optimizer()

    def _setup_solver(self, opti_type, solver):
        """Set up solver type and options."""
        self.opti_type = opti_type
        self.solver = solver

        # Configure solver options based on optimization type
        if self.opti_type == 'conic':
            self.opts = {'printLevel': 'low'}
        else:
            self.solver = 'ipopt'
            self.opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

        self.opti = cs.Opti()
        self.opti.solver(self.solver, self.opts)

    def _setup_optimizer(self):
        self.u_opti = self.opti.variable(self.input_dim)

        self.x_sample = self.opti.parameter(self.state_dim)

        self.P_list = [None] * self.num_cbfs
        self.c_list = [None] * self.num_cbfs
        for i in range(self.num_cbfs):
            self.P_list[i] = self.opti.parameter(self.state_dim, self.state_dim)
            self.c_list[i] = self.opti.parameter(self.state_dim)

        self.cost = self.u_opti.T @ self.u_opti

        self.opti.minimize(self.cost)

        self.constraints = []

        for i in range(self.num_cbfs):
            lhs = - 2.0 * (self.x_sample - self.c_list[i]).T @ self.P_list[i] @ (self.A @ self.x_sample + self.B @ self.u_opti)
            rhs = - self.gamma * (1.0 - (self.x_sample - self.c_list[i]).T @ self.P_list[i] @ (self.x_sample - self.c_list[i]))
            self.constraints.append(lhs >=rhs)

        self.opti.subject_to(self.constraints)

    def _set_parameters(self, x_sample, P_list, c_list):
        self.opti.set_value(self.x_sample, x_sample)
        for i in range(self.num_cbfs):
            self.opti.set_value(self.P_list[i], P_list[i])
            self.opti.set_value(self.c_list[i], c_list[i])

    def solve(self, x_sample, P_list, c_list):
        self._set_parameters(x_sample, P_list, c_list)
        try:
            self.opti.solve()
            sol = self.opti.value(self.cost)
            return sol
        except Exception as e:
            # print(e)
            return None
            
def process_sample_batch(args):
    """
    Worker function to process a batch of samples in parallel.
    Each worker gets its own instances of the optimization classes to avoid conflicts.
    """
    (start_idx, end_idx, num_cbfs, state_dim, x_lims, y_lims, eig_min, eig_max, 
     diag_only, A, B, P, c, gamma, eps, num_state_samples, min_bbox_volume, worker_id) = args
    
    # Set random seed for this worker to ensure reproducibility while maintaining independence
    np.random.seed(int(time.time() * 1000) % 2**32 + worker_id)
    
    # Create worker-specific instances of optimization classes
    subset_checker = SubsetChecker(P, c, num_cbfs)
    relative_degree_checker = RelativeDegreeChecker(B, P, c, num_cbfs)
    bounding_box_finder = BoundingBoxFinder(state_dim, num_cbfs)
    feasibility_checker = FeasibilityChecker(A, B, gamma, num_cbfs)
    
    P_list = [None] * num_cbfs
    c_list = [None] * num_cbfs
    
    worker_successful_samples = {}
    
    for sample_idx in range(start_idx, end_idx):
        if sample_idx % 1000 == 0:
            print(f"Worker {worker_id}: Processing sample {sample_idx}")
            
        # Sample CBF parameters
        for i in range(num_cbfs):
            P_list[i], c_list[i] = sample_cbf_params(state_dim, [x_lims[i], y_lims[i]], [eig_min, eig_max], diag_only)
            # check that P is positive definite
            if not np.all(np.linalg.eigvals(P_list[i]) >= 1e-6):
                break  # Skip this sample if P is not positive definite
        
        # Check subset condition
        is_subset = subset_checker.verify_subset(P_list, c_list)
        
        if is_subset:

            # Check relative degree condition
            L_min = relative_degree_checker.solve(P_list, c_list)
            if L_min >= eps:
                # Find bounding box
                limits = bounding_box_finder.solve(P_list, c_list)
                if limits is not None:

                    box = bounding_box_finder.limits_to_box(limits)
                    bbox_volume = np.prod([box[i][1] - box[i][0] for i in range(state_dim)])
                    
                    if bbox_volume >= min_bbox_volume:
                        
                        # Check feasibility for state samples
                        state_sampler = StateSampler(box)
                        feasible_samples = 0
                        
                        for i in range(num_state_samples):
                            x_sample = state_sampler.sample_feasible(P_list, c_list)
                            sol = feasibility_checker.solve(x_sample, P_list, c_list)
                            h_value = 1 - (x_sample - c).T @ P @ (x_sample - c)

                            if h_value < 0:
                                break
                                
                            if sol is not None:
                                feasible_samples += 1
                            else:
                                break
                                
                        if feasible_samples == num_state_samples:
                            print(f"Worker {worker_id}: Sample {sample_idx} is successful")
                            worker_successful_samples[sample_idx] = {
                                'P_list': [P.copy() for P in P_list], 
                                'c_list': [c.copy() for c in c_list], 
                                'bbox_volume': bbox_volume
                            }
    
    return worker_successful_samples


def save_results_safely(successful_samples, date_str, num_cbfs, lock):
    """Thread-safe function to save results to numpy file."""
    with lock:
        filename = f"synthesized_cbfs/successful_samples_{date_str}_{num_cbfs}.npy"
        np.save(filename, successful_samples)


def main():
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Read parameters from config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    with open ('configs/config.json', 'r') as f:
        config.update(json.load(f))

    max_num_workers = config['max_num_workers']
    min_num_workers = config['min_num_workers']
    min_untouched_cpus = config['min_untouched_cpus']
    seed = config['seed']
    num_samples = config['num_samples']
    num_state_samples = config['num_state_samples']
    state_dim = config['state_dim']
    eig_min = config['eig_min']
    eig_max = config['eig_max']
    min_bbox_volume = config['min_bbox_volume']
    eps = config['eps']
    gamma = config['gamma']
    diag_only = config['diag_only']
    x_lim = config['x_lim']
    x_lims = config['x_lims']
    y_lim = config['y_lim']
    y_lims = config['y_lims']
    num_cbfs = config['num_cbfs']

    # Define the system matrices
    A = np.array(config['A'])  # used for evaluation experiment data!
    B = np.array(config['B'])  # used for evaluation experiment data!

    # Define the state and input dimensions
    state_dim = A.shape[0]
    input_dim = B.shape[1]

    B = B.reshape((state_dim, input_dim))

    # Quadratic CBF: Define the ellipsoid matrices in numpy
    P = np.diag(config['P_diag'])
    
    # Quadratic CBF: Define the ellipsoid center
    c = np.array(config['c']).reshape((state_dim, 1))

    # Calculate the volume of the ellipsoid
    init_volume = np.pi ** (state_dim / 2) / np.prod(np.sqrt(np.linalg.eigvals(P)))
    print(f"Volume of the initial ellipsoid: {init_volume}")

    # Parallelization setup
    num_workers = max(min_num_workers, min(mp.cpu_count() - min_untouched_cpus, max_num_workers))  # Set the number of workers
    batch_size = max(1000, num_samples // (num_workers * 10))  # Adaptive batch size
    if num_workers == 1:
        batch_size = num_samples
    
    print(f"Using {num_workers} workers with batch size {batch_size}")
    
    successful_samples = {}
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_lock = threading.Lock()
    
    # Create batches for parallel processing
    batches = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        worker_id = len(batches)
        batch_args = (
            i, end_idx, num_cbfs, state_dim, x_lims, y_lims, eig_min, eig_max,
            diag_only, A, B, P, c, gamma, eps, num_state_samples, min_bbox_volume, worker_id
        )
        batches.append(batch_args)
    
    print(f"Created {len(batches)} batches for processing")
    
    # Process batches in parallel
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_sample_batch, batch): batch for batch in batches}
        
        completed_batches = 0
        total_batches = len(batches)
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            completed_batches += 1
            
            try:
                worker_results = future.result()
                
                # Merge worker results into main dictionary
                successful_samples.update(worker_results)
                
                # Save results periodically (thread-safe)
                if len(worker_results) > 0:
                    save_results_safely(successful_samples, date_str, num_cbfs, save_lock)
                
                # Progress update
                elapsed_time = time.time() - start_time
                progress = completed_batches / total_batches
                
                print(f"Completed batch {completed_batches}/{total_batches} "
                      f"({progress*100:.1f}%) - "
                      f"Found {len(worker_results)} new successful samples. "
                      f"Total successful: {len(successful_samples)}. "
                      f"Time taken so far: {elapsed_time/60:.1f} minutes")
                      
            except Exception as exc:
                print(f"Batch {batch[0]}-{batch[1]} generated an exception: {exc}")
    
    # Final save
    save_results_safely(successful_samples, date_str, num_cbfs, save_lock)
    
    total_time = time.time() - start_time
    print(f"\nCompleted processing {num_samples} samples in {total_time/60:.2f} minutes")
    print(f"Found {len(successful_samples)} successful samples")
    print(f"Success rate: {len(successful_samples)/num_samples*100:.4f}%")
    # print(successful_samples)

if __name__ == "__main__":
    main()