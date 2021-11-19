import numpy as np
import geometry as gm
import math
import qpfunc as qp
from matplotlib import pyplot as plt
from scipy import io as scio


class SphereWorld:
    def __init__(self):
        data = scio.loadmat('sphereworld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = np.asscalar(sphere_args[1])
            sphere_args[2] = np.asscalar(sphere_args[2])
            self.world.append(gm.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def plot(self,ax=None):
        for sphere in self.world:
            ax = sphere.plot('r',ax)
        
        x_goal = self.x_goal[:,1]
        x_goal = np.reshape(x_goal, (2,1))
        ax.scatter(x_goal[0, :], x_goal[1, :], c='g', marker='*',zorder = 100)
        
        return ax


class RepulsiveSphere:
    def __init__(self, sphere):
        self.sphere = sphere

    def eval(self, x_eval):
        distance = self.sphere.distance(x_eval)
        
        distance_influence = self.sphere.distance_influence
        if(distance > distance_influence):
            u_rep = 0
        elif(distance_influence>distance>0):
            u_rep = ((distance**-1 - distance_influence**-1)**2)/2
            u_rep = u_rep.item()
        else:
            u_rep = math.nan
        return u_rep

    def grad(self, x_eval):
        distance = self.sphere.distance(x_eval)
        distance_influence = self.sphere.distance_influence
        if(distance > distance_influence):
            grad_u_rep = np.zeros((2,1))
        elif(distance_influence>distance>0):
            distance_grad = self.sphere.distance_grad(x_eval)
            grad_u_rep = -(distance**-1 - distance_influence**-1)*(distance**-2)*distance_grad
        else:
            grad_u_rep = np.zeros((2,1))
            grad_u_rep[:] = np.nan
        return grad_u_rep


class Attractive:
    def __init__(self, potential):
        self.potential = potential

    def eval(self, x_eval):
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if(shape == 'conic'):
            expo = 1
        else:
            expo = 2
        u_attr = np.linalg.norm(x_eval - x_goal)**expo
        return u_attr

    def grad(self, x_eval):
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if(shape == 'conic'):
            expo = 1
        else:
            expo = 2
        grad_u_attr = expo*(x_eval-x_goal)*np.linalg.norm(x_eval - x_goal)**(expo-2)
        return grad_u_attr


class Total:
    def __init__(self, world, potential):
        self.world = world
        self.potential = potential

    def eval(self, x_eval):
        rep_weight = self.potential['repulsive_weight']
        attractive = Attractive(self.potential)
        attr_term = attractive.eval(x_eval)
        rep_term = 0
        worlds = self.world.world
        for sphere in worlds:
            repulsive = RepulsiveSphere(sphere)
            rep_term += repulsive.eval(x_eval)
        u_eval = attr_term + rep_weight*rep_term
        return u_eval

    def grad(self, x_eval):
        rep_weight = self.potential['repulsive_weight']
        attractive = Attractive(self.potential)
        attr_grad_term = attractive.grad(x_eval)
        rep_grad_term = 0
        worlds = self.world.world
        for sphere in worlds:
            repulsive = RepulsiveSphere(sphere)
            rep_grad_term += repulsive.grad(x_eval)
        grad_u_eval = attr_grad_term + rep_weight*rep_grad_term
        return grad_u_eval

class Navigation:
    def __init__(self, world):
        self.world = world

    def eval(self, x_eval, k_param):
        sphere_world = self.world
        x_goal = sphere_world.x_goal[:,1]
        x_goal = np.reshape(x_goal, (2,1))
        distance_squared = np.linalg.norm(x_eval - x_goal)**2
        beta_total = 1
        for sphere in sphere_world.world:
            beta_total *=  sphere.beta(x_eval)
        phi_k = distance_squared/((distance_squared**k_param + beta_total)**(1/k_param))
        phi_k = phi_k.item()
        if np.isnan(phi_k):
            phi_k = 1
        if phi_k > 1:
            phi_k = 1
        return phi_k

    def grad(self, x_eval, k_param):
        sphere_world = self.world
        x_goal = sphere_world.x_goal[:,1]
        x_goal = np.reshape(x_goal, (2,1))

        distance_squared = np.linalg.norm(x_eval - x_goal)**2
        distance = np.linalg.norm(x_eval - x_goal)
        
        beta_total = 1
        for sphere in sphere_world.world:
            beta_total *=  sphere.beta(x_eval)
        
        nb_obstacles = len(sphere_world.world)
        beta_grad_total = np.zeros((2,1))
        for i in range(0,nb_obstacles):
            beta_grad_multiplier = sphere_world.world[i].beta_grad(x_eval)
            for j in range(0,nb_obstacles):
                if i != j:
                    beta_grad_multiplier = beta_grad_multiplier * sphere_world.world[j].beta(x_eval)
            beta_grad_total +=beta_grad_multiplier

        
        term1 = 2*(x_eval - x_goal)*((distance_squared**k_param + beta_total)**(1/k_param))
        term2 = (distance_squared/k_param)*((distance_squared**k_param + beta_total)**(1/k_param - 1))
        term3 = 2*k_param*(distance**(2*k_param-2))*(x_eval - x_goal) + beta_grad_total
        term4 = ((distance_squared**k_param + beta_total)**(2/k_param))

        phi_grad = (term1 - term2*term3)/term4
        
        #phi_grad = np.array([[1],[1]])
        return phi_grad

def clfcbf_control(x_eval, world, potential):
    """
    Compute u^* according to      (  eq:clfcbf-qp  ).
    """
    nb_worlds = len(world)
    a_barrier = np.zeros((nb_worlds,2))
    b_barrier = np.zeros((nb_worlds,1))

    repul = potential['repulsive_weight']
    attr = Attractive(potential)
    
    u_ref = attr.grad(x_eval)
    #u_ref = x_eval
    
    iter = 0
    for sphere in world:
        dist = sphere.distance(x_eval)
        if dist < sphere.distance_influence:
            a_barrier[iter,:] = sphere.distance_grad(x_eval).T
            b_barrier[iter,:] = sphere.distance(x_eval)
        iter = iter+1
    b_barrier *= -repul
    u_opt = qp.qp_supervisor(a_barrier,b_barrier,u_ref)
    


    return u_opt


class Planner:
    def run(self, x_start, planned_parameters, stop_condition = 5e-3):
        total_eval = planned_parameters['U']
        total_grad = planned_parameters['control']
        epsilon = planned_parameters['epsilon']
        nb_steps = planned_parameters['nb_steps']
        
        x_path = np.zeros((2,nb_steps))
        u_path = np.zeros((1,nb_steps))

        x_path[:] = np.nan
        u_path[:] = np.nan
        
        x_path[:,0] = x_start
        u_path[:,0] = total_eval(np.reshape(x_path[:,0],(2,1)))
        
        for iter in range(1,nb_steps):
            change = -total_grad(np.reshape(x_path[:,iter-1],(2,1)))
            change = np.reshape(change,(2,))
            x_path[:,iter] = x_path[:,iter-1] + epsilon * change
            
            u_path[:,iter] = total_eval(np.reshape(x_path[:,iter],(2,1)))
            norm = np.linalg.norm(total_grad(np.reshape(x_path[:,iter],(2,1))),axis = 0)
            #if norm < stop_condition:
            #    break
            if np.linalg.norm(x_path[:,iter])>8:
                epsilon = .1

        return x_path, u_path

    def run_plot(self, planned_parameters, sphere_world, stop_condition = 5e-3):
        x_start = sphere_world.x_start
        nb_starts = len(x_start[0,:])
        x_paths = []
        u_paths = []
        for start in range(0,nb_starts):
            x_start = sphere_world.x_start[:,start]
            x_path, u_path = self.run(x_start,planned_parameters)
            x_paths.append(x_path)
            u_paths.append(u_path)

        
        fig, ax = plt.subplots(ncols = 2)
        fig.set_size_inches(12, 5)
        ax[0].set_aspect('equal', adjustable = 'box')
        ax[0] = sphere_world.plot(ax[0])
        ax[0].set_xlabel('x1')
        ax[0].set_ylabel('x2')
        ax[1].set_xlabel('nb_steps')
        ax[1].set_ylabel('U_total')
        
        for start in range(0,nb_starts):
            x_path = x_paths[start]
            ax[0].plot(x_path[0,:],x_path[1,:])
            u_path = u_paths[start]
            ax[1].plot(u_path.T)

        plt.show()

def test_runPlanner():
    planner = Planner()
    planner.run_plot()

if __name__ == "__main__":
    test_runPlanner()