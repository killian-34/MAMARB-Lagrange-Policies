from gurobipy import *
import numpy as np 
import sys
import time



# Transition matrix, reward vector, action cost vector
def action_knapsack(values, C, B):


	m = Model("Knapsack")
	m.setParam( 'OutputFlag', False )

	c = C

	x = np.zeros(values.shape, dtype=object)

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] = m.addVar(vtype=GRB.BINARY, name='x_%i_%i'%(i,j))



	m.modelSense=GRB.MAXIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	# minimze the value function
	m.setObjectiveN((x*values).sum(), 0, 1)

	# set constraints
	m.addConstr( x.dot(C).sum() <= B )
	for i in range(values.shape[0]):
		# m.addConstr( x[i].sum() <= 1 )
		m.addConstr( x[i].sum() == 1 )


	# Optimize model

	m.optimize()

	x_out = np.zeros(x.shape)

	for v in m.getVars():
		if 'x' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			x_out[i,j] = v.x

		else:
			pass
			# print((v.varName, v.x))

	obj = m.getObjective()
	


	return x_out, obj.getValue()



# Transition matrix, reward vector, action cost vector
def action_knapsack_dp(values, C, B):
	n = len(values)

	m = np.zeros((n, B+1))
	for j in range(B):
		m[0, j] = 0

	for i in range(n):
		for j in range(B+1):
			eligible_actions = C <= j

			m[i, j] = m[i-1, j]
			for k in range(1, len(eligible_actions)):
				if eligible_actions[k]:
					m[i, j] = max(m[i, j], m[i-1, j-C[k]] + values[i,k])		

			

	return m


def get_knapsack_actions(m, values, C, i, j):
	tol = 1e-4
	if i == 0:
		for k in range(len(C)):
			if abs(m[i, j] - values[i, k]) <= tol:
				return set([(i,k)])
	if m[i, j] > m[i-1, j]:
		for k in range(len(C)):
			if abs(m[i, j] - m[i-1, j-C[k]] - values[i, k]) <= tol:
				return set([(i,k)]) | get_knapsack_actions(m, values, C, i-1, j-C[k])
	else:
		return set([(i,0)]) | get_knapsack_actions(m, values, C, i-1, j)


def convert_knapsack_actions(action_set, n, num_actions):
	actions = np.zeros((n, num_actions), dtype=int)
	# actions[:,0] = 1
	for i,k in action_set:
		actions[i,k] = 1
	return actions




NUM_TRIALS = 500
N_ITEMS = 50
NUM_ACTIONS = 5

dp_timings = np.zeros(NUM_TRIALS)
g_timings = np.zeros(NUM_TRIALS)
tol = 0.01

for i in range(NUM_TRIALS):
	# print(i)

	# values = np.sort(np.random.randint(low=0, high=5, size=(5,2)), axis=1)
	values = np.sort(np.random.rand(N_ITEMS,NUM_ACTIONS-1), axis=1)
	values = np.concatenate([np.zeros((N_ITEMS,1)), values], axis=1)

	costs = np.sort(np.random.randint(low=1, high=5, size=NUM_ACTIONS-1))
	costs = np.concatenate([[0], costs])
	budget = np.random.randint(N_ITEMS*2)

	start = time.time()
	m = action_knapsack_dp(values, costs, budget)
	dp_timings[i] = time.time() - start

	# print("DP solution")
	# print(m[-1, -1])
	start = time.time()
	x_out, objval = action_knapsack(values, costs, budget)
	g_timings[i] = time.time() - start

	# print("gurobi solution")
	# print(objval)
	if (abs(m[-1,-1] - objval) > tol):
		print(values)
		print(costs)
		print(budget)
		print(m[-1, -1])
		print(objval)
		raise ValueError


	# if i == 0:
	dp_actions = get_knapsack_actions(m, values, costs, N_ITEMS-1, budget)
	dp_actions = convert_knapsack_actions(dp_actions, N_ITEMS, NUM_ACTIONS)
	# print(dp_actions)
	# print(x_out.astype(int))
	if not np.array_equal(dp_actions, x_out.astype(int)):
		print(dp_actions)
		print(x_out.astype(int))
		print(dp_actions - x_out.astype(int))
		dif_matrix = dp_actions - x_out.astype(int)
		for i,row in enumerate(dif_matrix):
			if (row!=0).any():
				print(i,row)
				print(values[i])
				print(costs)
				print(objval)
				print(m[-1, -1])

		raise ValueError

print("Num trials completed:",i)
print("No errors")

print(dp_timings.mean())
print(g_timings.mean())


