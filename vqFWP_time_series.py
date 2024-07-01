# Datetime
from datetime import datetime
import time


import matplotlib.pyplot as plt
from pandas import DataFrame

import warnings


import torch
import torch.nn as nn
import torch.optim as optim 

import pennylane as qml
from pennylane import numpy as np

from functools import *

# Saving
import pickle
import os
import copy

# sklearn
from sklearn.preprocessing import StandardScaler

# Dataset

from data.damped_shm import get_damped_shm_data



### VQC

def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates.
	"""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)

def RX_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RX(element, wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RY(element, wires=idx)

def RZ_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RZ(element, wires=idx)




def entangling_layer(nqubits):
	"""Layer of CNOTs followed by another shifted layer of CNOT.
	"""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])
	

def cycle_entangling_layer(nqubits):
	for i in range(0, nqubits):  
		qml.CNOT(wires=[i, (i + 1) % nqubits])



##############



### VQC function

def quantum_net(inputs, q_weights, n_outputs):
	n_dep = q_weights.shape[0]
	n_qub = q_weights.shape[1]

	H_layer(n_qub)

	RY_layer(inputs)

	for k in range(n_dep):
		entangling_layer(n_qub)
		RY_layer(q_weights[k])


	return [qml.expval(qml.PauliZ(position)) for position in range(n_outputs)]


##############

class HideSignature:
	def __init__(self, partial_func):
		self.partial_func = partial_func

	def __call__(self, inputs, q_weights):
		return self.partial_func(inputs, q_weights)

##############

### VQC batch wrapper

class BatchVQC:
	def __init__(self, q_func):
		self.q_func = q_func

	def __call__(self, inputs, q_weights):

		res_all = []
		for input_item, q_weight_item in zip(inputs, q_weights):
			res = self.q_func(input_item, q_weight_item) 
			res_all.append(torch.stack(res)) 

		return torch.stack(res_all)


##############



### FWP Cell Module


class FWPCell(nn.Module):
	def __init__(self, s_dim, a_dim):
		super().__init__()

		latent_dim = 8
		self.n_qubits = latent_dim
		self.q_depth = 2

		dev = qml.device("default.qubit", wires = self.n_qubits)
		# self.q_func = BatchVQC(VQCWrapper(qml.QNode(quantum_net, dev, interface = "torch"), a_dim))
		# self.q_func = BatchVQC(qml.QNode(quantum_net, dev, interface = "torch"))
		self.q_func = BatchVQC(qml.QNode(HideSignature(partial(quantum_net, n_outputs = a_dim)), dev, interface = "torch"))

		self.slow_program_encoder = torch.nn.Linear(s_dim, latent_dim)
		self.slow_program_layer_idx = torch.nn.Linear(latent_dim, self.q_depth)
		self.slow_program_qubit_idx = torch.nn.Linear(latent_dim, self.n_qubits)

		self.post_processing = torch.nn.Linear(a_dim, 1) 


	def forward(self, batch_item, previous_circuit_param):
		res = self.slow_program_encoder(batch_item)

		res_layer_idx = self.slow_program_layer_idx(res)
		res_qubit_idx = self.slow_program_qubit_idx(res)

		# Calculate the VQC params
		out_circuit_params = []
		for layer_idx, qubit_idx in zip(res_layer_idx, res_qubit_idx):
			outer_product = torch.outer(layer_idx, qubit_idx)
			# print("outer_product: {}".format(outer_product))
			out_circuit_params.append(outer_product)
		out_circuit_params = torch.stack(out_circuit_params)

		# Add of previous_circuit_param and out_circuit_params
		out_circuit_params = torch.add(out_circuit_params, previous_circuit_param)

		# Go through the VQC
		res = self.q_func(batch_item, out_circuit_params)

		# Post-processing 

		res = self.post_processing(res)


		return res, out_circuit_params

	def initial_fast_params(self, batch_size):

		return torch.zeros(batch_size, self.q_depth, self.n_qubits)

### FWP Module


class FWP(nn.Module):
	# FWP module: processes the whole sequence
	# (N, T, F)
	# N: number of batch
	# T: number of time-step
	# F: number of features
	def __init__(self, s_dim, a_dim):
		super().__init__()
		self.fwp_cell = FWPCell(s_dim = s_dim, a_dim = a_dim)

		

	def forward(self, batch_item):
		batch_size = batch_item.shape[0]
		initial_fast_params = self.fwp_cell.initial_fast_params(batch_size)
		time_length = batch_item.shape[1]

		output_collection_list = []

		for t in range(time_length):
			out_batch, initial_fast_params = self.fwp_cell(batch_item[:, t, :], initial_fast_params)
			output_collection_list.append(out_batch)

		res = torch.stack(output_collection_list)

		return res 



### Training routine

def train_epoch_full(opt, model, X, Y, batch_size):
	losses = []

	for beg_i in range(0, X.shape[0], batch_size):
		X_train_batch = X[beg_i:beg_i + batch_size]
		# print(x_batch.shape)
		Y_train_batch = Y[beg_i:beg_i + batch_size]

		# opt.step(closure)
		since_batch = time.time()
		opt.zero_grad()
		# print("CALCULATING LOSS...")
		model_res = model(X_train_batch)
		loss = nn.MSELoss()
		loss_val = loss(model_res[-1], Y_train_batch)
		# print("BACKWARD..")
		loss_val.backward()
		losses.append(loss_val.data.cpu().numpy())
		opt.step()
		# print("LOSS IN BATCH: ", loss_val)
		# print("FINISHED OPT.")
		# print("Batch time: ", time.time() - since_batch)
		# print("CALCULATING PREDICTION.")
	losses = np.array(losses)
	return losses.mean()

##############

### Plotting and Saving

def saving(exp_name, exp_index, train_len, iteration_list, train_loss_list, test_loss_list, model, simulation_result, ground_truth):
	# Generate file name
	file_name = exp_name + "_NO_" + str(exp_index) + "_Epoch_" + str(iteration_list[-1])
	saved_simulation_truth = {
	"simulation_result" : simulation_result,
	"ground_truth" : ground_truth
	}

	if not os.path.exists(exp_name):
		os.makedirs(exp_name)

	# Save the train loss list
	with open(exp_name + "/" + file_name + "_TRAINING_LOST" + ".txt", "wb") as fp:
		pickle.dump(train_loss_list, fp)

	# Save the test loss list
	with open(exp_name + "/" + file_name + "_TESTING_LOST" + ".txt", "wb") as fp:
		pickle.dump(test_loss_list, fp)

	# Save the simulation result
	with open(exp_name + "/" + file_name + "_SIMULATION_RESULT" + ".txt", "wb") as fp:
		pickle.dump(saved_simulation_truth, fp)

	# Save the model parameters
	torch.save(model.state_dict(), exp_name + "/" +  file_name + "_torch_model.pth")

	# Plot
	plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list)
	plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth)

	return


def plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list):
	# Plot train and test loss
	fig, ax = plt.subplots()
	# plt.yscale('log')
	ax.plot(iteration_list, train_loss_list, '-b', label='Training Loss')
	ax.plot(iteration_list, test_loss_list, '-r', label='Testing Loss')
	leg = ax.legend();

	ax.set(xlabel='Epoch', 
		   title=exp_name)
	fig.savefig(exp_name + "/" + file_name + "_" + "loss" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
	plt.clf()

	return

def plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth):
	# Plot the simulation
	plt.axvline(x=train_len, c='r', linestyle='--')
	plt.plot(simulation_result, '-')
	plt.plot(ground_truth.detach().numpy(), '--')
	plt.suptitle(exp_name)
	# savfig can only be placed BEFORE show()
	plt.savefig(exp_name + "/" + file_name + "_" + "simulation" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
	return

##############

def main():
	dtype = torch.DoubleTensor

	x, y = get_damped_shm_data()

	num_for_train_set = int(0.67 * len(x))

	x_train = x[:num_for_train_set].type(dtype)
	y_train = y[:num_for_train_set].type(dtype)

	x_test = x[num_for_train_set:].type(dtype)
	y_test = y[num_for_train_set:].type(dtype)

	print("x_train: ", x_train)
	print("x_test: ", x_test)
	print("x_train.shape: ", x_train.shape)
	print("x_test.shape: ", x_test.shape)

	x_train_transformed = x_train.unsqueeze(2)
	x_test_transformed = x_test.unsqueeze(2)

	print("x_train: ", x_train_transformed)
	print("x_test: ", x_test_transformed)
	print("x_train.shape: ", x_train_transformed.shape)
	print("x_test.shape: ", x_test_transformed.shape)

	print(x_train[0])
	print(x_train_transformed[0])

	print("y.shape: {}".format(y.shape))




	# create FWP

	model = FWP(s_dim = 1, a_dim = 4).double()

	test_input = torch.rand(3, 4, 1).double()
	print("test input: {}".format(test_input))

	model_res = model(test_input)
	# print("model_res: {}".format(model_res))
	print("model_res.shape: {}".format(model_res.shape))
	# print("model_res[-1]: {}".format(model_res[-1]))
	print("model_res[-1].shape: {}".format(model_res[-1].shape))

	# test the training set

	# model_res = model(x_train_transformed)

	# # print("model_res: {}".format(model_res))
	# print("model_res.shape: {}".format(model_res.shape))
	# print("model_res[-1]: {}".format(model_res[-1]))
	# print("model_res[-1].shape: {}".format(model_res[-1].shape))

	# Batch = 4
	# batch_size = 4
	# for beg_i in range(0, x_train_transformed.shape[0], batch_size):
	# 	X_train_batch = x_train_transformed[beg_i:beg_i + batch_size]
	# 	Y_train_batch = y_train[beg_i:beg_i + batch_size]

	# 	model_res = model(X_train_batch)

	# 	print("X_train_batch: {}".format(X_train_batch))
	# 	print("X_train_batch.shape: {}".format(X_train_batch.shape))

	# 	print("model_res.shape: {}".format(model_res.shape))
	# 	print("model_res[-1]: {}".format(model_res[-1]))
	# 	print("model_res[-1].shape: {}".format(model_res[-1].shape))

	# 	print("Y_train_batch: {}".format(Y_train_batch))
	# 	print("Y_train_batch.shape: {}".format(Y_train_batch.shape))

	# 	loss = nn.MSELoss()
	# 	loss_val = loss(model_res[-1], Y_train_batch)
	# 	print("loss_val: {}".format(loss_val))
	# 	loss_val.backward()

	exp_name = "VQ_FWP_TS_MODEL_DAMPED_SHM"
	exp_index = 1
	train_len = len(x_train_transformed)


	opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	
	train_loss_for_all_epoch = []
	test_loss_for_all_epoch = []
	iteration_list = []

	for i in range(100):
		iteration_list.append(i + 1)
		train_loss_epoch = train_epoch_full(opt = opt, model = model, X = x_train_transformed, Y = y_train, batch_size = 10)


		# Calculate test loss
		test_loss = nn.MSELoss()
		model_res_test = model(x_test_transformed)
		test_loss_val = test_loss(model_res_test[-1], y_test).detach().numpy()
		print("TEST LOSS at {}-th epoch: {}".format(i, test_loss_val))

		train_loss_for_all_epoch.append(train_loss_epoch)
		test_loss_for_all_epoch.append(test_loss_val)

		# Run the test
		total_res = model(x.type(dtype).unsqueeze(2))[-1].detach().cpu().numpy()
		ground_truth_y = y.clone().detach().cpu()

		saving(
				exp_name = exp_name, 
				exp_index = exp_index, 
				train_len = train_len, 
				iteration_list = iteration_list, 
				train_loss_list = train_loss_for_all_epoch, 
				test_loss_list = test_loss_for_all_epoch, 
				model = model, 
				simulation_result = total_res, 
				ground_truth = ground_truth_y)
		




if __name__ == '__main__':
	main()























