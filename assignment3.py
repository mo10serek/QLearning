#import numpy
import sys
import math

# This program is

def calculatingReward(state):
    # This function is use to calculate the reward based on the number of dirty
    # This function is static because it does not use any other parameters in the class

    reward = 0
    # it loop though the string that represents the state and subtract the reward by 1 if the character is 1
    for counter in range(1, len(state)):
        if int(state[counter]) == 1:
            reward = reward - 1

    return reward

class td_qlearning:
    alpha = 0.1
    gamma = 0.5

    # this is a dictionary that represents the q Table. It stores keys as states and values as another dictionary that
    # represents keys as possible actions and what value each action has
    qTable = {}
    # this represents an array that contains the lists in order from the trajectory.
    allIterationsInOrder = []
    # this represent the index of where is the current state is located
    counter = 0

    # this is the constructor the is used to trained the agent
    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return nothing

        # this opens the trajectory file and convert it to an array
        trajectory_filename_readable = open(trajectory_filepath, "r")

        self.allIterationsInOrder = trajectory_filename_readable.read().split("\n")
        # go though each of the states in the array to set up the q table
        for stateAndAction in self.allIterationsInOrder:
            # if the array reaches the end or reaches '', it terminate to prevent an error
            if stateAndAction == '':
                break

            # get the state of the list
            state = stateAndAction.split(",")[0]

            if stateAndAction.split(",")[0] in self.qTable.values():
                break

            # get the location from the state at the first character
            location = int(state[0])

            # set up the q table with the possible states in the file with the appropriate actions
            if location == 1:
                self.qTable[state] = {"C": 0, "D": 0}
            elif location == 2:
                self.qTable[state] = {"C": 0, "R": 0}
            elif location == 3:
                self.qTable[state] = {"C": 0, "L": 0, "R": 0, "U": 0, "D": 0}
            elif location == 4:
                self.qTable[state] = {"C": 0, "L": 0}
            elif location == 5:
                self.qTable[state] = {"C": 0, "U": 0}

        # go though each of the states in the array to train the agent
        for stateAndAction in self.allIterationsInOrder:

            if stateAndAction == '':
                break

            # get the state and action from each iteration in the array
            state = stateAndAction.split(",")[0]
            action = stateAndAction.split(",")[1]

            # calculate the q value using the current state and action and using the calculating function
            if self.allIterationsInOrder[self.counter + 1] != '':
                q = self.calculatingQvalue(state, action)

            # update the q table
            self.qTable[state][action] = q
            # go to the next state in the list
            self.counter = self.counter + 1

    # this function updates the q value of the current state and action
    def calculatingQvalue(self, state, action):
        # state is a string representation of a state
        # action is a string representation of an action
        # to calculate the q value,
        # first, it need to get the reward of the state
        reward = calculatingReward(state)

        # Second, it need to get the current q value by taking it from the q table
        currentQValue = self.qTable[state][action]

        # third, it need to get the max value of the next state-action pair
        maxQValue = self.findingMaxQValue()

        # last, it uses the Temporal Difference Q-Learning function to calculated the updated q value of the current
        # state
        qV = currentQValue + (self.alpha * (reward + (self.gamma * maxQValue) - currentQValue))

        # Return the q-value for the state-action pair
        return qV

    # This function finds the max value of the next state
    def findingMaxQValue(self):

        # First it gets the state that is after the current state
        S_t1 = self.allIterationsInOrder[self.counter + 1].split(",")[0]

        # Then it receives the dictionary that represents keys as actions and q values as values
        stateWithActionsQValues = self.qTable[S_t1]

        # Last it find the max q value in the dictonay values
        maxQValue = max(stateWithActionsQValues.values())

        return maxQValue

    # This is a getter to get the q value depends on the state and action
    def qvalue(self, state, action):
        # state is a string representation of a state
        # action is a string representation of an action

        # it fetches the q value of the current state and action that is passed in the function
        q = self.qTable[state][action]

        # Return the q-value for the state-action pair
        return q

    # this is a getter to get the actions that has the highest q value depending on the state
    def policy(self, state):
        # state is a string representation of a state

        # it go to the qTable to receive all of the different values that are based on the action
        actionDictionary = self.qTable[state]

        action = ""

        maxQValue = -1000000000.0
        # it loop on each key in the action dictionary to find which action has the highest q value
        for keyState in actionDictionary.keys():
            if float(maxQValue) < float(actionDictionary[keyState]):
                action = keyState
                maxQValue = actionDictionary[keyState]

        # Return the optimal action under the learned policy
        return action