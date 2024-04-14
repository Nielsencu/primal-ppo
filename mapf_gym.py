import random
import numpy as np

from map_generator import *
from alg_parameters import EnvParameters, NetParameters, TrainingParameters
import copy
from util import getFreeCell, returnAsType, renderWorld, Sequence

class Human(object):
    def __init__(self, world=np.zeros((1,1))):
        self.world = np.copy(world)
        self.position = Human.getEntrance(world)
        self.world[self.position] = 1
        self.goal = getFreeCell(self.world)
        self.getAstarPath()
        self.step = 0
        
    @staticmethod
    def getEntrance(world : np.ndarray):
        entrance = (-1,-1)
        while entrance[0]==-1 or not (entrance[0]==0 or entrance[1]==0):
            entrance = getFreeCell(world)
        return entrance
    
    def nextStep(self):
        if(self.step >= len(self.path)-1):
            self.getNextGoal()
            self.step = 0
        else:
            self.step+=1
        self.position = self.path[self.step]
    
    def getAstarPath(self):
        path = astar_4(self.world, self.position, self.goal)[0]
        self.path =  path[::-1]
        # Human goes from start -> goal -> start
        self.path += path[1:]
    
    def getPos(self, type='np'):
        return returnAsType(self.position, type)
    
    def getNextGoal(self):
        self.goal = getFreeCell(self.world)
        self.getAstarPath()

    def getNextPos(self, type='np'):
        if(self.step >= len(self.path)-1):
            return returnAsType(self.path[-1], type)
        else:
            return returnAsType(self.path[self.step+1], type)

class LoopingHuman(Human):
    def __init__(self, world=np.zeros((1,1)), startPos : tuple[int, int] | None = None, goalPos: tuple[int, int] | None = None):        
        if startPos is None:
            startPos = getFreeCell(self.world)
        if goalPos is None:
            goalPos = getFreeCell(self.world)
        self.world = np.copy(world)
        self.position : tuple[int, int] = startPos
        self.world[self.position] = 1
        self.goal : tuple[int, int] = goalPos
        super().getAstarPath()
        self.step = 0
        
    def getNextGoal(self):
        """
        Does nothing, since basic human goes from start -> goal -> start,
        will keep reusing the previous computed path
        """
        return

class Agent():
    dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 
               4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1), 8: (-1, 1)}  # x,y operation for corresponding action
    
    oppositeAction = {0:0, 1:3, 2:4, 3:1, 4:2}

    def __init__(self):
        self.__position = np.array([-1,-1])
        self.__goal = np.array([-1,-1])
        self.__emulatedStep = np.array([-1,-1])

        self.invalidActions = [[],[],[]]
        # array 0: static invalid Actions
        # array 1: human invalid Actions
        # array 2: action which takes agent to previous position (repetition)

        self.restrictedAction = dict()
        # otherAgent restricted Actions (represented as {x:[[a, y], ...]} meaning x action is invalid if agent 'a' takes y action simultaneously)

        self.unconditionallyGoodActions = list()

        self.actionsAlreadyGrouped = False

        self.bfsMap = []

    def setGoodActions(self, actions):
        self.unconditionallyGoodActions = actions
        self.actionsAlreadyGrouped = True

    def setInvalidActions(self, listToSet, actions):
        self.invalidActions[listToSet] = actions

    def updateRestrictedPosition(self, action, newRestriction):
        if(action in self.restrictedAction):
            self.restrictedAction[action].append(newRestriction)
        else:
            self.restrictedAction[action] = [newRestriction]

    def setPos(self, pos):
        self.__position = np.array(pos)
        self.invalidActions = [[],[],[]]
        self.restrictedAction = dict()
        self.unconditionallyGoodActions = list()
        self.actionsAlreadyGrouped = False


    def getPos(self, type='np'):
        return returnAsType(self.__position, type)

    def setGoal(self, goal):
        self.__goal = np.array(goal)

    def getGoal(self, type='np'):
        return returnAsType(self.__goal, type)
    
    def getEmulatedStep(self, type='np'):
        return returnAsType(self.__emulatedStep, type)

    def emulateStep(self, action):
        step = np.array(self.dirDict[action])
        self.__emulatedStep = np.add(self.getPos(), step)
    
    def takeStep(self, action):
        step = np.array(self.dirDict[action])
        self.setPos(np.add(self.getPos(), step))
        self.setInvalidActions(2, [self.oppositeAction[action]])

class MapfGym():
    def __init__(self, num_agents=EnvParameters.N_AGENTS, size=EnvParameters.WORLD_SIZE):
        self.agentList = [Agent() for i in range(num_agents)]
        self.obstacleMap = generateWarehouse(num_block=size) # this is random
        self.human = Human(world=self.obstacleMap) # this is random
        self.populateMap()
        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)
        self.num_channel = NetParameters.NUM_CHANNEL
        
        self.use_hp = False
        self.use_da = False
        
    def populateMap(self):
        tempMap = np.copy(self.obstacleMap)
        tempMap[returnAsType(self.human.position,'mat')] = 1
        for agentId, agent in enumerate(self.agentList):
            agent.setPos(self.getAgentStart(tempMap, agentId))
            tempMap[agent.getPos(type='mat')] = 2
            
            agent.setGoal(self.getNextGoal(tempMap, agentId))
            self.makeBfsMap(agent)
            tempMap[agent.getGoal(type='mat')] = 3
            
    def getAgentStart(self, worldMap : np.ndarray, agentId : int | None = None):
        return getFreeCell(worldMap)

    def getNextGoal(self, worldMap : np.ndarray, agentId : int | None = None):
        return getFreeCell(worldMap)

    def worldWithAgents(self):
        world = np.copy(self.obstacleMap)
        for i,agent in enumerate(self.agentList):
            if not np.any(agent.getPos()<0):
                world[agent.getPos(type='mat')] = i+1

        return world

    def worldWithAgentsAndGoals(self):
        world = np.copy(self.obstacleMap)
        
        for i,agent in enumerate(self.agentList):
            if not np.any(agent.getPos()<0):
                world[agent.getPos(type='mat')] = i+1
            if not np.any(agent.getGoal()<0):
                world[agent.getGoal(type='mat')] = i+1

        return world
    
    def makeBfsMap(self, agent):

        bfsMap = np.copy(self.obstacleMap)

        bfsMap[bfsMap==0] = -2
        size = bfsMap.shape
        curr, end = 0,0
        openedList = list()

        value = -1
        node = (-1,-1)

        openedList.append(agent.getGoal('mat'))

        while end<len(openedList):
            end = len(openedList)
            value+=1

            while curr<end:
                node = openedList[curr]
                # print(node)
                curr+=1
                bfsMap[node] = value
                
                if node[0]>0 and (bfsMap[node[0]-1, node[1]]==-2) and ((node[0]-1, node[1]) not in openedList):
                    openedList.append((node[0]-1, node[1]))
                if (node[0]+1)<size[0] and (bfsMap[node[0]+1, node[1]]==-2) and ((node[0]+1, node[1]) not in openedList):
                    openedList.append((node[0]+1, node[1]))
                if node[1]>0 and (bfsMap[node[0], node[1]-1]==-2) and ((node[0], node[1]-1) not in openedList):
                    openedList.append((node[0], node[1]-1))
                if (node[1]+1)<size[1] and (bfsMap[node[0], node[1]+1]==-2) and ((node[0], node[1]+1) not in openedList):
                    openedList.append((node[0], node[1]+1))

        agent.bfsMap = bfsMap

    def observe(self, indexOfAgent=-1):
        agent = self.agentList[indexOfAgent]

        #PART 1: FOV Observations

        top_left = (agent.getPos()[0] - EnvParameters.FOV_SIZE // 2, agent.getPos()[1] - EnvParameters.FOV_SIZE // 2)  # (top, left)
        # print(top_left)
        
        observations = np.zeros((self.num_channel, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))  #observations per parameters and FOV Size
        # 0: obs map
        # 1: other Agents
        # 2: own goal
        # 3: agents' in Fov goals
        # 4: human in Fov
        # 5: dangerous area
        
        world = self.worldWithAgents()
        size = world.shape

        visibleAgents = list()

        for i in range(top_left[0], top_left[0] + EnvParameters.FOV_SIZE):  # top and bottom
            for j in range(top_left[1], top_left[1] + EnvParameters.FOV_SIZE):  # left and right
                
                if i >= size[0] or i < 0 or j >= size[1] or j < 0:
                    # out of boundaries (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1
                    continue
                elif world[i,j] == -1:
                    #obstacle (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1

                elif world[i,j] >0 and world[i,j]== indexOfAgent+1:
                    #self Position (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1
                
                elif world[i,j]>0:
                    # other agents in FOV (in agent Map)
                    visibleAgents.append(world[i,j])
                    observations[1,i - top_left[0], j - top_left[1]] = 1

                human_next_pos = np.array(self.human.getNextPos())
                # TODO: Fix this, only used for evaluation
                if self.use_da and TrainingParameters.USE_INFLATED_HUMAN and np.linalg.norm(human_next_pos - np.array([i,j])) <= EnvParameters.PENALTY_RADIUS:
                    observations[4, i - top_left[0], j - top_left[1]] = 1
                    
                # TODO: Fix this, only used for evaluation
                if self.use_hp and self.num_channel == 6 and TrainingParameters.USE_HUMAN_TRAJECTORY_PREDICTION:
                    human_astar_path = self.human.path[1:TrainingParameters.K_TIMESTEP_PREDICT+1]
                    for astar_pt in human_astar_path:
                        if np.linalg.norm(np.array(astar_pt) - np.array([i,j])) <= 0.01:
                            observations[5, i - top_left[0], j - top_left[1]] = 1
        if(top_left[0]<= agent.getGoal()[0]<top_left[0] + EnvParameters.FOV_SIZE and top_left[1]<= agent.getGoal()[1]<top_left[1] + EnvParameters.FOV_SIZE):
            # own goal in FOV (in own goal frame)
            observations[2,agent.getGoal()[0] - top_left[0], agent.getGoal()[1] - top_left[1]] = 1

        for agentIndex in visibleAgents:
            # print(agentIndex)
            x, y = self.agentList[agentIndex-1].getGoal()
            # projection of visible agents' goal in FOV (in others' goal frame)
            min_node = (max(top_left[0], min(top_left[0] + EnvParameters.FOV_SIZE - 1, x)),
                        max(top_left[1], min(top_left[1] + EnvParameters.FOV_SIZE - 1, y)))
            observations[3,min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        if(top_left[0]<= self.human.getNextPos()[0] <top_left[0] + EnvParameters.FOV_SIZE and top_left[1]<= self.human.getNextPos()[1]<top_left[1] + EnvParameters.FOV_SIZE):
            # human in FOV (in human frame)
            observations[4,self.human.getNextPos()[0] - top_left[0], self.human.getNextPos()[1] - top_left[1]] = 1

        #PART2: Vector

        vector = np.zeros(NetParameters.VECTOR_LEN)

        vector[0] = agent.getGoal()[0] - agent.getPos()[0]  # distance on x axes
        vector[1] = agent.getGoal()[1] - agent.getPos()[1]  # distance on y axes
        vector[2] = (vector[0] ** 2 + vector[1] ** 2) ** .5  # total distance
        if vector[2] != 0:  # normalized
            vector[0] = vector[0] / vector[2]
            vector[1] = vector[1] / vector[2]
        
        return observations, vector

    def getAllObservations(self):
        allObs = np.zeros((1, EnvParameters.N_AGENTS, self.num_channel , EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
        allVec = np.zeros((1, EnvParameters.N_AGENTS, NetParameters.VECTOR_LEN), dtype=np.float32)

        for i in range(0, EnvParameters.N_AGENTS):
            obs,vec = self.observe(i)
            allObs[:, i, :, :, :] = obs
            allVec[:, i, : ] = vec
        
        return allObs, allVec


    def getInvalidActions(self):

        for agent in self.agentList:

            staticInvalidAction = list()
            humanInvalidAction = list()

            for i in range(0, EnvParameters.N_ACTIONS):
                agent.emulateStep(i)
                pos = agent.getEmulatedStep('mat')
                if not ((0<= pos[0] < self.obstacleMap.shape[0]) and (0<= pos[1] <self.obstacleMap.shape[1])): ## Falling out of map
                    staticInvalidAction.append(i)
                elif(self.obstacleMap[pos] !=0): ## Running into walls
                    staticInvalidAction.append(i)

                elif(np.array_equal(agent.getEmulatedStep(), self.human.getNextPos())): ## Vertex collision with Human
                    humanInvalidAction.append(i)
                elif(np.array_equal(agent.getPos(), self.human.getNextPos()) and np.array_equal(agent.getEmulatedStep(), self.human.getPos())): ##Swapping Collision with Human
                    humanInvalidAction.append(i)

            agent.setInvalidActions(0, staticInvalidAction)
            agent.setInvalidActions(1, humanInvalidAction)
        
    
    def getRestrictedActions(self):
        np.zeros((EnvParameters.N_AGENTS, EnvParameters.N_AGENTS, EnvParameters.N_ACTIONS, EnvParameters.N_ACTIONS))
        # Get set of codependent restricted actions

        #Part1: get possible agents that can collide

        agentsAtRisk = list()

        for i in range(EnvParameters.N_AGENTS):
            for j in range(i+1, EnvParameters.N_AGENTS):
                if(np.sum(np.square(self.agentList[i].getPos() - self.agentList[j].getPos()))) <= 4:
                    agentsAtRisk.append([i,j])

        #Part2: get simultaneous actions which cause collision
        for agentOneIndex,agentTwoIndex in agentsAtRisk:

            agentOne = self.agentList[agentOneIndex]
            agentTwo = self.agentList[agentTwoIndex]
            currentDistance = np.sum(np.square(agentOne.getPos() - agentTwo.getPos()))

            for i in range(EnvParameters.N_ACTIONS):
                agentOne.emulateStep(i)

                #Collision is only possible if the agents get closer or atleast stay at the same distance
                if np.sum(np.square(agentOne.getEmulatedStep() - agentTwo.getPos()))<=currentDistance: 
                    
                    #Now check which corressponding action(if any) of agentTwo causes a vertex collision
                    for j in range(EnvParameters.N_ACTIONS):
                        agentTwo.emulateStep(j)

                        if(np.array_equal(agentOne.getEmulatedStep(), agentTwo.getEmulatedStep())):
                            
                            # Add the to the lists
                            agentOne.updateRestrictedPosition(i, [agentTwoIndex, j])
                            agentTwo.updateRestrictedPosition(j, [agentOneIndex, i])

                    # Also account for swapping collision
                    if(np.array_equal(agentOne.getEmulatedStep(), agentTwo.getPos())):
                        agentOne.updateRestrictedPosition(i, [agentTwoIndex, Agent.oppositeAction[i]])
                        agentTwo.updateRestrictedPosition(Agent.oppositeAction[i], [agentOneIndex, i])
        
    def getUnconditionallyGoodActions(self, returnIsNeeded = False):
        # First get bad actions
        self.getInvalidActions()
        self.getRestrictedActions()

        if(returnIsNeeded):
            allGoodActions = list()

        for agent in self.agentList:
            badActions = list()

            badActions += agent.invalidActions[0]
            badActions += agent.invalidActions[1]
            badActions += agent.invalidActions[2]

            for i in agent.restrictedAction:
                badActions.append(i)
            

            goodActions = np.setdiff1d(np.arange(EnvParameters.N_ACTIONS),badActions)
            agent.setGoodActions(goodActions)

            if(returnIsNeeded):
                allGoodActions.append(goodActions)
        
        if(returnIsNeeded):
            return allGoodActions

        

    def getActionStatus(self, actions):
        ## Lets check the consequence of each action [pun intended]
        
        assert(len(actions)==EnvParameters.N_AGENTS)

        actionStatus = np.zeros(shape=EnvParameters.N_AGENTS)
        # -1 for static collision
        # -2 for human collision
        # -3 for agent collision
        # -4 for repeating action
        # 1 for valid actions


        for indexOfAgent, agent in enumerate(self.agentList):
            if(actionStatus[indexOfAgent]!=0):
                # print(indexOfAgent, actionStatus[indexOfAgent])
                continue
            action  = actions[indexOfAgent]
            try:
                assert(action in range(0, EnvParameters.N_ACTIONS+1))
            except:
                print(actions, indexOfAgent, action)
                raise Exception("Well, Shit")
            if action in agent.invalidActions[0]: ##This caluses a static collision
                actionStatus[indexOfAgent] = -1

            elif action in agent.invalidActions[1]: ##This causes a human collision
                actionStatus[indexOfAgent] = -2
            
            elif action in agent.unconditionallyGoodActions: ##This action is unconditionally good
                actionStatus[indexOfAgent] = 1
            
            else:
                if(action in agent.restrictedAction):
                    for fellowAgent,agentAction in agent.restrictedAction[action]: ##Check if this is a restricted action and a collision is being caused due to it
                        if(actions[fellowAgent]==agentAction):
                            #Set status -3 for both agents
                            actionStatus[indexOfAgent] = -3
                            actionStatus[fellowAgent] = -3

                if(actionStatus[indexOfAgent]==0 and action in agent.invalidActions[2]): ## Check it this is a repetition
                    actionStatus[indexOfAgent] = -4
                
                elif(actionStatus[indexOfAgent]==0): ## This means this is a valid action. It might have been restricted but the other agent might be performing some other action, hence it is valid.
                    actionStatus[indexOfAgent] = 1

        return actionStatus
    

    def calculateActionReward(self, actions, actionStatus):  
        shadowGoal = 0      
        rewards = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)

        for i, status in enumerate(actionStatus):
            if(status == -1):
                rewards[:, i]=EnvParameters.COLLISION_COST
            
            elif(status == -2):
                rewards[:, i]=EnvParameters.HUMAN_COLLISION_COST
            elif(status==-3):
                rewards[:, i]=EnvParameters.COLLISION_COST

            elif(status==-4):
                rewards[:, i]=EnvParameters.REPEAT_POS
            
            elif(status==1):
                self.agentList[i].emulateStep(actions[i])
                if(np.array_equal(self.agentList[i].getEmulatedStep(), self.agentList[i].getGoal())):
                    # rewards[:, i] = EnvParameters.GOAL_REWARD
                    rewards[:, i]=EnvParameters.ACTION_COST
                    shadowGoal+=1
                else:
                    rewards[:, i]=EnvParameters.ACTION_COST
            else:
                raise Exception("How did this even happen")
            # self.agentList[i].emulateStep(actions[i])
            # rewards[:, i] += self.calculateRadialConstraintCost(self.human.getNextPos(), self.agentList[i].getEmulatedStep()) * EnvParameters.CONSTRAINT_VIOLATION_COST    
        return rewards, shadowGoal
    
    def calculateRadialConstraintCost(self, human_pos, robot_pos):
        """
        Returns the cost of the radial constraint between human and agent
        Cost 0 means safe
        """
        
        return max(EnvParameters.PENALTY_RADIUS - np.linalg.norm(human_pos - robot_pos), 0.0)

    def calculateRadialConstraintCostNormalized(self, human_pos, robot_pos):
        """
        Returns normalized radial constraint cost to [0,1].
        0 denotes safe distance, 1 denotes collision.
        """
        return self.calculateRadialConstraintCost(human_pos, robot_pos) / EnvParameters.PENALTY_RADIUS
        
    def calculateCostReward(self, actions):
        costRewards = np.zeros((1, EnvParameters.N_AGENTS), dtype=np.float32)
        for i in range(EnvParameters.N_AGENTS):
            self.agentList[i].emulateStep(actions[i])
            costRewards[:, i] = self.calculateRadialConstraintCostNormalized(self.human.getNextPos(), self.agentList[i].getEmulatedStep())
        return costRewards

    def getTrainValid(self, actions):
        trainValid = np.zeros((EnvParameters.N_AGENTS, EnvParameters.N_ACTIONS), dtype=np.float32)

        for idx, agent in enumerate(self.agentList):
            for action in range(EnvParameters.N_ACTIONS):
                if action in agent.unconditionallyGoodActions:
                    trainValid[idx,action] = 1

                elif action in agent.restrictedAction:
                    trainValid[idx,action] = 1
                    for fellowAgent, simAction in agent.restrictedAction[action]:
                        if(actions[fellowAgent]==simAction):
                            trainValid[idx,action] = 0
                            break

        return trainValid
    
    def fixActions(self, actions, actionStatus):
        agentActionPairs =  np.full(shape=(EnvParameters.N_AGENTS, 2), fill_value=-1) 
        ## Using this in such a way so that I can later easily check if this pair of actions is in the array corresponding to a restricted action for some action


        problemAgents = np.where(actionStatus < 0 )[0].tolist()

        for idx in np.where(actionStatus==1)[0]:
            agentActionPairs[idx] = [idx, actions[idx]]


        while(len(problemAgents)!=0):
            agentIdx = problemAgents[0]
            
            agent = self.agentList[agentIdx]
            
            if(len(agent.unconditionallyGoodActions)!=0):
            
                agentActionPairs[agentIdx] = [agentIdx, agent.unconditionallyGoodActions[0]]
                problemAgents.remove(agentIdx)
            
            else:

                viableActions = np.setdiff1d(np.arange(EnvParameters.N_ACTIONS), agent.invalidActions[0]+agent.invalidActions[1])

                for tryAction in viableActions: ##
                    # print(agentIdx, tryAction)
                    # print([x for x in set(tuple(x) for x in agentActionPairs) & set(tuple(x) for x in agent.restrictedAction[tryAction])], len([x for x in set(tuple(x) for x in agentActionPairs) & set(tuple(x) for x in agent.restrictedAction[tryAction])]))
                    if(tryAction not in agent.restrictedAction or len([x for x in set(tuple(x) for x in agentActionPairs) & set(tuple(x) for x in agent.restrictedAction[tryAction])])==0):

                        agentActionPairs[agentIdx] = [agentIdx, tryAction]
                        problemAgents.remove(agentIdx)
                        break


                if(agentActionPairs[agentIdx][1]==-1):
                    randomChoiceOfAction = random.choice(viableActions)
                    
                    if(randomChoiceOfAction in agent.restrictedAction):
                        conflicts = [x for x in set(tuple(x) for x in agentActionPairs) & set(tuple(x) for x in np.array(agent.restrictedAction[randomChoiceOfAction]))]
                    
                        for conflict in conflicts:

                            agentActionPairs[conflict[0]] = [-1,-1]
                            problemAgents.append(conflict[0])

                    agentActionPairs[agentIdx] = [agentIdx, randomChoiceOfAction]
                    problemAgents.remove(agentIdx)
        try:
            temp = self.getActionStatus(agentActionPairs[:,1]) 
            assert(np.all((temp>0) | (temp<=-4)))
        except:
            temp = self.getActionStatus(agentActionPairs[:,1]) 
            print((temp>0) | (temp<=-4))
            print(actions)
            print(self.getActionStatus(actions))
            print(agentActionPairs[:,1])
            print(self.getActionStatus(agentActionPairs[:,1]))
            raise Exception("lets see")
            
        return agentActionPairs[:,1]

    def jointStep(self, actions, actionStatus):
        goalsReached = np.zeros(EnvParameters.N_AGENTS)
        constraintsViolated = np.zeros(EnvParameters.N_AGENTS)
        if not np.all((actionStatus>0) | (actionStatus<=-4)):
            actions = self.fixActions(actions, actionStatus)

        for agentIdx, agent in enumerate(self.agentList):
            agent.takeStep(actions[agentIdx])
            
            if(EnvParameters.LIFELONG):
                if(np.array_equal(agent.getPos(), agent.getGoal())):
                    goalsReached[agentIdx]+=1
                    agent.setGoal(self.getNextGoal(self.worldWithAgentsAndGoals(), agentIdx))
                    self.makeBfsMap(agent)

        self.human.nextStep()
        
        humanPos = self.human.getPos()
        for agentIdx, agent in enumerate(self.agentList):
            constraintsViolated[agentIdx] += int(self.calculateRadialConstraintCostNormalized(humanPos, agent.getPos()) >= 0.01)

        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)

        return goalsReached, constraintsViolated

    def _render(self):
        goals = []
        agents = []
        for i in self.agentList:
            agents.append(i.getPos('mat'))
            goals.append(i.getGoal('mat'))
        return renderWorld(world=self.obstacleMap, agents=agents,goals=goals, human=self.human.getPos('mat'),\
                            humanPath=self.human.path, humanStep=self.human.step)
        
class FixedMapfGym(MapfGym):
    def __init__(self, obstaclesMap : np.ndarray, agentsSequence : list[Sequence], humanStart : tuple[int, int], humanGoal : tuple[int, int], numChannel = None, useDA = False, useHP = False):
        self.agentList = [Agent() for _ in range(EnvParameters.N_AGENTS)]
        self.obstacleMap = copy.deepcopy(obstaclesMap)
        self.human : Human = LoopingHuman(world=self.obstacleMap, startPos=humanStart, goalPos=humanGoal)
        self.agentsSequence : list[Sequence] = copy.deepcopy(agentsSequence)
        self.populateMap()
        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)
        if numChannel is None:
            numChannel = NetParameters.NUM_CHANNEL
        self.num_channel = numChannel
        self.use_da = useDA
        self.use_hp = useHP
        
    def getAgentStart(self, worldMap: np.ndarray, agentId: int | None = None):
        return self.agentsSequence[agentId].getNext()
    
    def getNextGoal(self, worldMap: np.ndarray, agentId: int | None = None) -> tuple[int, int]:
        return self.agentsSequence[agentId].getNext()
        