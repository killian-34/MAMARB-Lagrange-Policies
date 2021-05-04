# MAMARB-Lagrange-Policies
Code for paper: Beyond "To Act or Not to Act": Fast Lagrangian Approaches to General Multi-Action Restless Bandits in AAMAS'21

Example Usage: python3 adherence_simulation.py -N 5 -n 200 -b 0.1 -s 0 -ws 0 -l 25 -d healthcare -S 4
- N: Number of trials to average over
- n: Number of processes 
- b: Budget as a fraction of n
- s, ws: Seeds for random generator streams
- l: Length of simulation 
- d: simulation environment
- S: number of states for the arms in the simulation (see specific simulation environments)


Note, the above example command should take about 10 minutes to run.


### Note: The above example command will run the simulation for all policies. To change the policies that are run, edit the `policies` dict on line 1173 of adherence_simulation.py or pass the -pc option to run one policy at a time: 

For a specific policy python3 adherence_simulation.py -N 5 -n 200 -b 0.1 -s 0 -ws 0 -l 25 -d healthcare -S 4 -pc 37
- -pc 0: No actions 
- -pc 21: Hawkins 
- -pc 24: VfNc
- -pc 27: SampleLam
- -pc 37: BLam0.1
- -pc 38: BLam0.2
- -pc 39: BLam0.3
- -pc 40: BLam0.5
