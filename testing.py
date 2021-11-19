
import numpy as np
import scipy as sp
import potential as pot
import geometry as gm
import math
from matplotlib import cm, pyplot as plt

def world_plotTesting():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    world = pot.SphereWorld()
    world.plot(ax)
    x_goal = world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])
    plt.show()

def test_positiveField():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    x_goal = sphere_world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    potential = {'x_goal': x_goal, 'repulsive_weight': .01, 'shape': 'quadratic'}
    attr = pot.Attractive(potential)
    gm.field_plot_threshold(attr.eval, threshold=1000)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.show()

def test_positiveGrad():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    x_goal = sphere_world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    potential = {'x_goal': x_goal, 'repulsive_weight': .01, 'shape': 'quadratic'}
    attr = pot.Attractive(potential)
    gm.field_plot_threshold(attr.grad,threshold = 100,nb_grid = 20)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.show()

def test_repusliveField():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    
    for sphere in sphere_world.world:
        repul = pot.RepulsiveSphere(sphere)
        gm.field_plot_threshold(repul.eval,threshold = 100,nb_grid=150)
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.show()

def test_repulsiveGrad():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    
    for sphere in sphere_world.world:
        repul = pot.RepulsiveSphere(sphere)
        gm.field_plot_threshold(repul.grad,threshold = 100, nb_grid=80)
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.show()

def test_totalField():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    x_goal = sphere_world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    potential = {'x_goal': x_goal, 'repulsive_weight': 1, 'shape': 'quadratic'}
    tot = pot.Total(sphere_world, potential)
    gm.field_plot_threshold(tot.eval,threshold=500)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.show()

def test_totalGrad():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    x_goal = sphere_world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    potential = {'x_goal': x_goal, 'repulsive_weight': 1, 'shape': 'quadratic'}
    tot = pot.Total(sphere_world, potential)
    gm.field_plot_threshold(tot.grad,threshold=100)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.show()

def test_navigationField():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    k_param = 10
    nav = pot.Navigation(sphere_world)
    nav_handle = lambda x_eval : nav.eval(x_eval, k_param)
    gm.field_plot_threshold(nav_handle, threshold= 1)
    plt.show()

def test_navigationGrad():
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal', adjustable = 'box')

    sphere_world = pot.SphereWorld()
    k_param = 10
    nav = pot.Navigation(sphere_world)
    nav_grad_handle = lambda x_eval : nav.grad(x_eval, k_param)
    gm.field_plot_threshold(nav_grad_handle, threshold = 10, nb_grid = 100)
    plt.show()


def test_run_potentialPlanner():
    planner = pot.Planner()
    sphere_world = pot.SphereWorld()

    x_goal = sphere_world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    potential = {'x_goal': x_goal, 'repulsive_weight': 1, 'shape': 'quadratic'}
    total = pot.Total(sphere_world,potential)

    nb_steps = 250
    planned_parameters ={'U' : total.eval, 'control' : total.grad, 'epsilon' : .01, 'nb_steps' : nb_steps}
    
    planner.run_plot(planned_parameters, sphere_world)

def test_run_navigationPlanner():
    planner = pot.Planner()
    sphere_world = pot.SphereWorld()

    x_goal = sphere_world.x_goal[:,1]
    x_goal = np.reshape(x_goal, (2,1))
    
    potential = {'x_goal': x_goal, 'repulsive_weight': 1, 'shape': 'quadratic'}
    total = pot.Total(sphere_world,potential)

    k_param = 4
    nav = pot.Navigation(sphere_world)
    nav_handle = lambda x_eval : nav.eval(x_eval, k_param)
    nav_grad_handle = lambda x_eval : nav.grad(x_eval, k_param)

    nb_steps = 20000
    planned_parameters ={'U' : nav_handle, 'control' : nav_grad_handle, 'epsilon' : 20, 'nb_steps' : nb_steps}
    
    planner.run_plot(planned_parameters, sphere_world,stop_condition=5e-20)


if __name__ == "__main__":
    #world_plotTesting()
    #test_positiveField()
    #test_positiveGrad()
    #test_repusliveField()
    #test_repulsiveGrad()
    #test_totalField()
    #test_totalGrad()
    #test_run_potentialPlanner()
    #test_navigationField()
    #test_navigationGrad()
    test_run_navigationPlanner()