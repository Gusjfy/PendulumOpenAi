import gym
import gym.spaces 
import gym.wrappers
import numpy
import matplotlib.pyplot
import tensorflow as tf
from collections import deque
from keras.layers import Dense
from keras.models import Sequential

def create_action_bins(z):
	return numpy.linspace(-2.0, 2.0, z)

def find_actionbin(action, actionbins):
    return (numpy.abs(actionbins - action)).argmin()

def build_model(num_output_nodes):
	model = Sequential()
	model.add(Dense(128, input_shape = (3,), activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(num_output_nodes, activation = 'linear')) 
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	return model

def train_model(memory, gamma = 0.9):
	for state, actionbin, reward, state_new in memory:
		flat_state_new = numpy.reshape(state_new, [1,3])
		flat_state = numpy.reshape(state, [1,3])
		target = reward + gamma * numpy.amax(model.predict(flat_state_new))
		targetfull = model.predict(flat_state)
		targetfull[0][actionbin] = target
		model.fit(flat_state, targetfull, epochs = 1, verbose = 0) 
		
def run_episodes(eps = 0.999, iters = 100):
	eps_decay = 0.9999
	eps_min = 0.02
	for i in range(iters):
		state = env.reset()
		totalreward = 0
		memory = deque()
		cnt = 0	
		if eps>eps_min:
			eps = eps * eps_decay
		while cnt < 250:
			cnt += 1
			if numpy.random.uniform() < eps:
				action = env.action_space.sample()
			else:
				flat_state = numpy.reshape(state, [1,3])
				action = numpy.amax(model.predict(flat_state))
			actionbin = find_actionbin(action, actionbinslist)
			action = actionbinslist[actionbin]
			action = numpy.array([action])
			observation, reward, done, _ = env.step(action)  
			totalreward += reward
			state_new = observation 
			memory.append((state, actionbin, reward, state_new))
			state = state_new
		train_model(memory, gamma = 0.9)
	return eps

def start():
	state = env.reset()
	totalreward = 0
	cnt = 0
	eps = 0
	while cnt < 200:
		cnt += 1
		if numpy.random.uniform() < eps:
			action = env.action_space.sample()
			actionbin = find_actionbin(action, actionbinslist)
		else:
			flat_state = numpy.reshape(state, [1,3])
			actionbin = numpy.argmax(model.predict(flat_state))
		action = actionbinslist[actionbin]
		action = numpy.array([action])
		observation, reward, done, _ = env.step(action)  
		totalreward += reward
		state_new = observation 
		state = state_new
	return totalreward

if __name__ == '__main__':
	env = gym.make('Pendulum-v1')
	eps = 1
	num_action_bins = 10
	actionbinslist = create_action_bins(num_action_bins)
	model = build_model(num_action_bins)
	a_total = []
	a_counter = []
	totaliters = 1000
	test_interval = 25	
	numeps = int(totaliters)
	print('numeps = ', numeps)
	cnt = 0
	while cnt < totaliters:
		eps = run_episodes(eps = eps, iters = test_interval)
		cnt += test_interval
		trarray = []
		for i in range(20):
			trarray.append(start())
		print(cnt, 'iterations. Average test reward = ', numpy.average(trarray))
		a_counter.append(cnt)
		a_total.append(numpy.average(trarray))
	matplotlib.pyplot.plot(a_counter, a_total)
	matplotlib.pyplot.xlabel('Iterations')
	matplotlib.pyplot.ylabel('Reward')
	matplotlib.pyplot.show()
	model.save('model.h5')
	print('process finished')
