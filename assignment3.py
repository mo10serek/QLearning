import numpy
import sys
import math


class td_qlearning:
    alpha = 0.1
    gamma = 0.5

    qValuesWithActions = {"C": [0, 0, 0, 0, 0],
                          "L": [0, 0, 0, 0, 0],
                          "R": [0, 0, 0, 0, 0],
                          "U": [0, 0, 0, 0, 0],
                          "D": [0, 0, 0, 0, 0]}

    reward = 0

    def findingMaxValue(self, state):
        nextStatesValue = []
        nextStatesAction = []

        currentLocation = int(state[0])
        index = currentLocation-1

        if currentLocation == 1:
            nextStatesAction.append('C')
            nextStatesValue.append(self.qValuesWithActions['C'][index])

        if 3 <= currentLocation <= 4:
            nextStatesValue.append(self.qValuesWithActions['L'][index - 1])
            nextStatesAction.append('L')

        if 2 <= currentLocation <= 3:
            nextStatesValue.append(self.qValuesWithActions['R'][index + 1])
            nextStatesAction.append('R')

        if currentLocation == 3 or currentLocation == 5:
            nextStatesValue.append(self.qValuesWithActions['U'][index - 2])
            nextStatesAction.append('U')

        if currentLocation == 1 or currentLocation == 3:
            nextStatesValue.append(self.qValuesWithActions['D'][index + 2])
            nextStatesAction.append('D')

        indexMax = nextStatesValue.index(max(nextStatesValue))

        return max(nextStatesValue), nextStatesAction[indexMax]

    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return nothing

        trajectory_filepath_readable = open(trajectory_filepath, "r")

        stateInAList = trajectory_filepath_readable.read().split("\n")

        for stateAndAction in stateInAList:

            if stateAndAction == '':
                break

            state = stateAndAction.split(",")[0]
            action = stateAndAction.split(",")[1]
            squareLocation = int(state[0]) - 1

            q = self.qvalue(state, action)
            a = self.policy(state)

            self.qValuesWithActions[a][squareLocation] = q

            print(state + ": " + str(q) + " " + a)

    def qvalue(self, state, action):
        # state is a string representation of a state
        # action is a string representation of an action

        self.reward = 0

        for counter in range(1, len(state)):
            if int(state[counter]) == 1:
                self.reward = self.reward - 1

        currentLocation = int(state[0])
        currentQValue = self.qValuesWithActions[action][currentLocation - 1]

        maxQValue = self.findingMaxValue(state)[0]

        q = currentQValue + self.alpha * (self.reward + self.gamma * maxQValue - currentQValue)

        # Return the q-value for the state-action pair
        return round(q, 6)

    def policy(self, state):
        # state is a string representation of a state

        a = self.findingMaxValue(state)[1]

        # Return the optimal action under the learned policy
        return a

td_qlearning = td_qlearning("Example2/trajectory.csv")
