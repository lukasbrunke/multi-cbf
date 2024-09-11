import numpy as np
import casadi as cs


class SafetyFilter:

    def __init__(self, input_dim, kappa_list, opti_type='conic', solver='qpoases'):
        self.opti_type = opti_type
        self.solver = solver
        self.opts = {'printLevel': 'low'}
        if not self.opti_type == 'conic':
            self.solver = 'ipopt'
            self.opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}

        self.input_dim = input_dim

        self.kappa_list = kappa_list
        self.num_cbf_constraints = len(kappa_list)

        self.setup_cbf_filter_optimizer()

    def setup_cbf_filter_optimizer(self):
        if self.opti_type == 'conic':
            self.opti = cs.Opti(self.opti_type)
        else:
            self.opti = cs.Opti()
        self.opti.solver(self.solver, self.opts)

        self.u_opti = self.opti.variable(self.input_dim, 1)

        # set up Lie derivative as a parameter
        self.Lfh = self.opti.parameter(self.num_cbf_constraints, 1)
        self.Lgh = self.opti.parameter(self.num_cbf_constraints, self.input_dim)

        # set up uncertified control input as parameter
        self.u_unsafe = self.opti.parameter(self.input_dim, 1)

        # set up cbf value as parameter
        self.kappa_at_hx = self.opti.parameter(self.num_cbf_constraints, 1)

        # set up safety filtering objective
        self.cost = (self.u_unsafe - self.u_opti).T @ (self.u_unsafe - self.u_opti)

        # set up constraints
        self.constraints = []
        for i in range(self.num_cbf_constraints):
            Lfh = self.Lfh[i]
            Lgh = self.Lgh[i, :]
            kappa_at_hx = self.kappa_at_hx[i]
            self.constraints.append(self.opti.subject_to(Lfh + Lgh @ self.u_opti >= - kappa_at_hx))

        # Set up minimization problem
        self.opti.minimize(self.cost)

    def solve_cbf_filter_problem(self, Lfh_value, Lgh_value, u_unsafe, kappa_at_hx):
        # plug in value for Lfh and Lgh
        self.opti.set_value(self.Lfh, Lfh_value)
        self.opti.set_value(self.Lgh, Lgh_value)
        
        # plug in value for u_unsafe
        self.opti.set_value(self.u_unsafe, u_unsafe)

        # plug in value for h_at_x
        self.opti.set_value(self.kappa_at_hx, kappa_at_hx)

        sol = self.opti.solve()
        u_sol = sol.value(self.u_opti)
        cost_sol = sol.value(self.cost)

        return u_sol, cost_sol
    

class SafetyFilterQuadratic(SafetyFilter):

    def __init__(self, input_dim, kappa_list, P_list, c_list, fx_func, gx_func, opti_type='conic', solver='qpoases'):
        # CBF of the form h_i(x) = offset_i - (x - c_i)^T P_i (x - c_i) >= 0
        self.P_list = P_list
        self.c_list = c_list

        # Control affine system of the form x_dot = f(x) + g(x)u
        self.f = fx_func
        self.g = gx_func
        
        super().__init__(input_dim, kappa_list, opti_type, solver)

    def kappa_at_x(self, kappa, P, x, c):
        rhs = (1 - kappa["offset"]) - (x - c).T @ P @ (x - c) 
        # print("h_val offsetted = {}".format(rhs))   

        if rhs >= 0:
            if kappa["family"] == "tanh":
                rhs = np.tanh(kappa["inner_slope_pos"] * (rhs - kappa["offset"]))
            rhs = kappa["slope_pos"] * rhs
        else:
            if kappa["family"] == "tanh":
                rhs = np.tanh(kappa["inner_slope_neg"] * rhs)
            rhs = kappa["slope_neg"] * rhs

        return rhs
    
    def setup_cbf_conditions(self, x_init):
        Lfh_value = np.zeros((self.num_cbf_constraints, 1))
        Lgh_value = np.zeros((self.num_cbf_constraints, self.input_dim))
        kappa_at_x = np.zeros((self.num_cbf_constraints, 1))

        for i in range(self.num_cbf_constraints):
            Pi = self.P_list[i]
            ci = self.c_list[i]
            kappai = self.kappa_list[i]
            
            Lfh_value[i] = - 2 * (x_init - ci).T @ Pi @ self.f(x_init)
            Lgh_value[i, :] = - 2 * (x_init - ci).T @ Pi @ self.g(x_init)
            kappa_at_x[i] = self.kappa_at_x(kappai, Pi, x_init, ci)

        return Lfh_value, Lgh_value, kappa_at_x
    
    def input_bounds(self, x_init):
        Lfh_value, Lgh_value, kappa_at_x = self.setup_cbf_conditions(x_init)

        u_max = np.inf
        u_min = -np.inf
        for i in range(self.num_cbf_constraints):
            u_max_i = np.inf
            u_min_i = -np.inf

            if Lgh_value[i, :] < 0:
                u_max_i = (- kappa_at_x[i] - Lfh_value[i]) / Lgh_value[i, :]
            else:
                u_min_i = (- kappa_at_x[i] - Lfh_value[i]) / Lgh_value[i, :]

            u_max = min(u_max, u_max_i)
            u_min = max(u_min, u_min_i)

        return u_max, u_min

    def filter_input(self, x_init, u_unsafe):
        Lfh_value, Lgh_value, kappa_at_x = self.setup_cbf_conditions(x_init)

        u_sol, _ = self.solve_cbf_filter_problem(Lfh_value, Lgh_value, u_unsafe, kappa_at_x)

        return u_sol
    