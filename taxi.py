import gym
import numpy as np
import random

from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v2").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])
frames = []

def print_frames(frames):
	for i, frame in enumerate(frames):
		clear_output(wait=True)
		if frame['episode'] != 0:
			print("\nEpisode: {}".format(frame['episode']))
		print(frame['frame'].getvalue())
		print("Timestep: {}".format(i + 1))
#		print("State: {}".format(frame['state']))
#		print("Action: {}".format(frame['action']))
#		print("Reward: {}".format(frame['reward']))
		if frame['record'] != 0:
			print("Record Time: {}".format(frame['record']))
		
		sleep(.05)

def training(max_range):
	alpha = 0.25
	gamma = 0.5
	epsilon = 0.1

	all_epochs = []
	all_penalties = []

	record = 999999

	for i in range(1, max_range +1):
		state = env.reset()

		epochs, penalties, rewards = 0, 0, 0
		done = False

		while not done:
			if random.uniform(0, 1) < epsilon:
				action = env.action_space.sample()
			else:
				action = np.argmax(q_table[state])

			next_state, reward, done, info = env.step(action)

			old_value = q_table[state, action]
			next_max = np.max(q_table[next_state])

			#frames.append({
			#	'frame': env.render(mode='ansi'),
			#	'state': state,
			#	'action': action,
			#	'reward': reward,
			#	'episode': i,
			#	'record': record
			#	}
			#)

			new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

			q_table[state, action] = new_value

			if reward == -10:
				penalties += 1

			state = next_state
			epochs += 1

		if epochs < record:
			record = epochs

		#print_frames(frames)
		#global frames
		#frames = []

		if i % 100 == 0:
			clear_output(wait=True)
			print("Episode: {}".format(i))

	print("Training Finished.\n")

def run_game():
	
	total_epochs, total_penalties = 0, 0
	episodes = 100

	record = 99999

	for i in range(1, episodes + 1):
		state = env.reset()

		epochs, penalties, reward = 0, 0, 0

		done = False

		while not done:
			action = np.argmax(q_table[state])
			state, reward, done, info = env.step(action)

			if reward == -10:
				penalties += 1
	
			frames.append({
				'frame': env.render(mode='ansi'),
				'state': state,
				'action': action,
				'reward': reward,
				'episode': i,
				'record': record
				}
			)

			epochs += 1

		if epochs < record:
			record = epochs

		print_frames(frames)
		frames.clear()
		
		total_penalties += penalties
		total_epochs += epochs

	print("Results after {} episodes".format(episodes))
	print("Avarage timesteps per episode: {}".format(total_epochs / episodes))
	print("Avarage penalties por episode: {}".format(total_penalties / episodes))

training(10000)
run_game()
#print_frames(frames)

