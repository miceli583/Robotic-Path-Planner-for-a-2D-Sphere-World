# Robotic Path Planner for a 2D Sphere World
This repository contains code implementing a robotic path planner in a 2D sphere world with obstacles.
The sphere world is intilized with 5 starting locations and one goal location.
The control paramter for the planner can be generated in 1 of 3 ways:
 using a potential based planner,
 using a navigation function,
 or using a CLF-CBF formulation

The simulation results are contained in the file "output.pdf"
Note, the CLF-CBF formulation requires a Quadratic Programming solver.
Code is conatined implementing a QP solver using the cvxopt package

 <img width="479" alt="Screen Shot 2021-11-19 at 2 30 15 PM" src="https://user-images.githubusercontent.com/55464981/142680720-a95a385e-74e8-4450-9a22-46b589536bc0.png">
<img width="459" alt="Screen Shot 2021-11-19 at 2 29 25 PM" src="https://user-images.githubusercontent.com/55464981/142680588-ee8cb8a1-18ac-4d92-906b-87a8d522c513.png">
<img width="678" alt="Screen Shot 2021-11-19 at 2 31 17 PM" src="https://user-images.githubusercontent.com/55464981/142680852-236d06ed-5254-4f2d-9b33-bb27d7d17129.png">
<img width="471" alt="Screen Shot 2021-11-19 at 2 31 02 PM" src="https://user-images.githubusercontent.com/55464981/142680854-083b1aad-a37d-4348-8438-db07ec0e59a8.png">
