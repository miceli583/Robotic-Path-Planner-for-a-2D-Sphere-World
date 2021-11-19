# Robotic Path Planner for a 2D Sphere World
This repository contains code implementing a robotic path planner in a 2D sphere world with obstacles.
The sphere world is intilized with 5 starting locations and one goal location.
The control paramter for the planner can be generated in 1 of 3 ways:
 using a potential based planner
 using a navigation function
 using a CLF-CBF fomrulation

The simulation results are contained in the file "output.pdf"
Note, the CLF-CBF formulation requires a Quadratic Programming solver.
Code is conatined implementing a QP solver using the cvxopt package
 
<img width="459" alt="Screen Shot 2021-11-19 at 2 29 25 PM" src="https://user-images.githubusercontent.com/55464981/142680588-ee8cb8a1-18ac-4d92-906b-87a8d522c513.png">
