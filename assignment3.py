import numpy
import sys
import math

class td_qlearning:
    alpha = 0.1
    gamma = 0.5

    allTrajectoryIterations = {}
    allIterationsInOrder = []

    reward = 0

    def findingMaxQValue(self, state, action):

        S_t_index = self.allIterationsInOrder.index(str(state) + "," + action)
        S_t1 = self.allIterationsInOrder[S_t_index + 1].split(",")[0]

        stateWithActionsQValues = self.allTrajectoryIterations[S_t1]

        maxQValue = -1000000000
        for qValue in stateWithActionsQValues.values():
            if maxQValue < int(qValue):
                maxQValue = qValue

        return maxQValue

    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return nothing

        trajectory_filename_readable = open(trajectory_filepath, "r")

        stateInAList = trajectory_filename_readable.read().split("\n")

        for stateAndAction in stateInAList:

            if stateAndAction == '': break

            state = stateAndAction.split(",")[0]

            location = int(state[0])

            if location == 1:
                self.allTrajectoryIterations[state] = {"C": 0, "D": 0}
            elif location == 2:
                self.allTrajectoryIterations[state] = {"C": 0, "R": 0}
            elif location == 3:
                self.allTrajectoryIterations[state] = {"C": 0, "D": 0, "R": 0, "L": 0, "U": 0}
            elif location == 4:
                self.allTrajectoryIterations[state] = {"C": 0, "L": 0}
            elif location == 5:
                self.allTrajectoryIterations[state] = {"C": 0, "U": 0}

            self.allIterationsInOrder.append(stateAndAction)

    def qvalue(self, state, action):
        # state is a string representation of a state
        # action is a string representation of an action

        self.reward = 0

        for counter in range(1, len(state)):
            if int(state[counter]) == 1:
                self.reward = self.reward - 1

        currentQValue = self.allTrajectoryIterations[state][action]

        maxQValue = self.findingMaxQValue(state, action)

        q = currentQValue + self.alpha * (self.reward + self.gamma * maxQValue - currentQValue)

        self.allTrajectoryIterations[state][action] = q

        # Return the q-value for the state-action pair
        return round(q, 6)

    def policy(self, state):
        # state is a string representation of a state

        actionDictionary = self.allTrajectoryIterations[state]

        action = "C"

        maxQValue = -1000000000
        for keyState in actionDictionary.keys():
            if maxQValue < actionDictionary[keyState]:
                action = keyState
                maxQValue = actionDictionary[keyState]

        # Return the optimal action under the learned policy
        return action

#td_qlearning = td_qlearning("Example1/trajectory.csv")
