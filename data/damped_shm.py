# From https://skill-lync.com/projects/Solving-2nd-order-ODE-for-a-simple-pendulum-using-python-40080

import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

def system(theta,t,b,g,l,m):
	theta1 = theta[0]
	theta2 = theta[1]
	dtheta1_dt = theta2
	dtheta2_dt = -(b/m)*theta2-g*math.sin(theta1)
	dtheta_dt=[dtheta1_dt,dtheta2_dt]

	return dtheta_dt



b=0.15
g=9.81
l=1
m=1


theta_0 = [0,3]


t = np.linspace(0,20,240)


theta = odeint(system,theta_0,t,args = (b,g,l,m))



# f=1
# for i in range(0,240):
# 	filename = str(f)+'.png'
# 	f= f+1
# 	plt.figure()
# 	plt.plot([10,l*math.sin(theta[i,0])+10],[10,10-l*math.cos(theta[i,0])],marker="o")
# 	plt.xlim([0,20])
# 	plt.ylim([0,20])

	
# 	plt.savefig(filename)
	
# Plotting	
# plt.plot(t,theta[:,0],'b-')
# plt.plot(t,theta[:,1],'r--')
# plt.show()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(theta[:,1].reshape(-1, 1))


def plotting_test(x, data_src):
	# create a figure window
	fig = plt.figure(1, figsize=(9,8))
	ax1 = fig.add_subplot()
	ax1.plot(x,data_src)
	ax1.axhline(color="grey", ls="--", zorder=-1)
	ax1.set_ylim(-1,1)
	ax1.text(0.5, 0.95,'Damped SHM', ha='center', va='top',
		 transform = ax1.transAxes)

	plt.show()

def transform_data_single_predict(data, seq_length):
	x = []
	y = []

	for i in range(len(data)-seq_length-1):
		_x = data[i:(i+seq_length)]
		_y = data[i+seq_length]
		x.append(_x)
		y.append(_y)
	x_var = Variable(torch.from_numpy(np.array(x).reshape(-1, seq_length)).float())
	y_var = Variable(torch.from_numpy(np.array(y)).float())

	return x_var, y_var



def get_damped_shm_data(data = dataset, seq_len = 4):
	return transform_data_single_predict(data = data, seq_length = seq_len)


def main():
	x, y = get_damped_shm_data()
	plotting_test(t, dataset)

	print(x.size())
	print(y.size())



if __name__ == '__main__':
	main()