import numpy as np

# Loss function (calculated to make sure that algorithm is converging)
def loss_function(indicator, X, centroids):
	K = centroids.shape[0]
	N = X.shape[0]
	cost = 0
	for i in range(N):
		for j in range(K):
			cost += indicator[j][i] * (np.linalg.norm(X[i]-centroids[j]) ** 2)
	return cost


# Step that fixes centroids, solves for allocation of points to clusters
def fix_u_solve_r(indicator, X, centroids):
	K = centroids.shape[0]
	N = X.shape[0]
	for i in range(N):
		arg_min = 0
		for j in range(K):
			if np.linalg.norm(X[i]-centroids[j]) < np.linalg.norm(X[i]-centroids[arg_min]):
				arg_min = j
		for j in range(K):
			if j == arg_min:
				indicator[j][i] = 1
			else:
				indicator[j][i] = 0


# Step that fixes allocations, solves for centroids
def fix_r_solve_u(indicator, X, centroids):
	N = indicator.shape[1]
	K = indicator.shape[0]
	for centroid in range(K):
		denominator = indicator[centroid].sum()
		feature_space = X.shape[1]
		numerator = []
		for i in range(feature_space):
			temp_numerator = 0
			for j in range(N):
				temp_numerator += indicator[centroid][j] * X[j][i]
			numerator.append(temp_numerator/denominator)
		centroids[centroid] = numerator


# Main k-means function
def kmeans(X, initial_centroids):
	"""Runs k-means for given data and computes some metrics.
	Input parameters:
	X:- Data set matrix where each row of X is represents a single training example.
	initial_centroids:- Matrix storing initial centroid position.
	max_iters:- Maximum number of iterations that K means should run for.

	Return values:
	newCentroid:- Matrix storing final cluster centroids.
	evaluationMatrix:- Array that returns MI, AMI, RI, ARI.
	loss:- value of loss function
	"""
	main_data = X.astype(float)
	K = initial_centroids.shape[0]
	N = main_data.shape[0]
	indicator = np.zeros((K, N))
	newCentroid = initial_centroids.astype(float)
	prev_cost = np.inf
	# Train k-means
	while True:
		fix_u_solve_r(indicator, main_data, newCentroid)
		fix_r_solve_u(indicator, main_data, newCentroid)
		loss = loss_function(indicator, main_data, newCentroid)
		if loss >= prev_cost:
			break
		prev_cost = loss
		print("Loss function: %f" % loss)
	return loss

