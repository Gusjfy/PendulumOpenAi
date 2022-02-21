import gym
import gym.spaces 
import gym.wrappers
import numpy
from keras.models import load_model

def create_action_bins(z):
	return numpy.linspace(-2.0, 2.0, z)

def find_actionbin(action, actionbins):
    return (numpy.abs(actionbins - action)).argmin()

def start(counter = 0):
	state = env.reset()
	while counter < 200:
		counter += 1
		env.render()
		if numpy.random.uniform() < 0:
			action = env.action_space.sample()
			actionbin = find_actionbin(action, actionbinslist)
		else:
			flat_state = numpy.reshape(state, [1,3])
			actionbin = numpy.argmax(model.predict(flat_state))	
		action = actionbinslist[actionbin]
		action = numpy.array([action])
		observation, reward, done, _ = env.step(action)  
		state_new = observation 
		state = state_new

if __name__ == '__main__':
	env = gym.make('Pendulum-v1')
	num_action_bins = 10
	actionbinslist = create_action_bins(num_action_bins)
	model = load_model('model.h5')
	for i in range(1):
		start()