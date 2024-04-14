import random

import imageio
import numpy as np
import torch
import wandb

from matplotlib.colors import hsv_to_rgb
import math
import cv2

from alg_parameters import *

class Sequence:
    def __init__(self, itemsIn : list[tuple[int,int]] | None = None):
        if itemsIn is None:
            itemsIn = list()
        self.items : list[tuple[int,int]] = itemsIn
        self.curIdx = 0
        
    def add(self, item):
        self.items.append(item)
        
    def getAtPos(self, pos : int) -> tuple[int, int] | None:
        if len(self.items) == 0:
            print("Empty items!!")
            return None
        if pos >= len(self.items):
            print("Invalid pos ", pos)
            return None
        return self.items[pos]
        
    def getNext(self) -> tuple[int, int]:
        if self.curIdx == len(self.items):
            print("No more goals to retrieve!! Returning last goal...")
            return self.items[-1]
        goal = self.items[self.curIdx]
        self.curIdx +=1
        return goal

class BatchValues:
    def __init__(self):
        self.observations = list()
        self.vectors = list()
        self.rewards = list()
        self.values = list()
        self.ps = list()
        self.actions = list()
        self.hiddenState = list()
        self.returns = list()
        self.trainValid = list()
        self.costRewards = list()
        self.costValues = list()
        self.costReturns = list()

class OneEpPerformance():
    def __init__(self):
        self.totalGoals = 0
        self.shadowGoals = 0
        self.episodeReward = 0
        self.staticCollide = 0
        self.humanCollide = 0
        self.agentCollide = 0
        self.episodeCostReward = 0
        self.constraintViolations = 0

def getFreeCell(world):
    size = world.shape

    i = -1
    j = -1
    while(i==-1 or world[i][j]!=0):
        i = np.random.randint(0,size[0])
        j = np.random.randint(0,size[1])

    return (i,j)


def returnAsType(arr, type):
    if(type=='np'): # numpy array
        return arr
    elif(type=='mat'): # to be used directly as a cell of matrix
        return (arr[0], arr[1])
    else:
        raise Exception("Invalid Type as input")

    
def init_colors():
    """the colors of agents and goals"""
    c = {a + 1: hsv_to_rgb(np.array([a / float(EnvParameters.N_AGENTS), 1, 1])) for a in range(EnvParameters.N_AGENTS)}
    c[0] = [1,1,1]
    c[-1] = [0,0,0]
    c[-2] = [0.5,0.5,0.5]
    return c

def getArrowPoints(direction, coord, scale, tailWidth, headWidth):

    if(np.array_equal(direction, np.array([0,1]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        
        arrow = [
                [center[0], center[1]-tailWidth],
                [center[0]-tailHeight, center[1]-tailWidth], 
                [center[0]-tailHeight, center[1]+tailWidth],
                [center[0], center[1]+tailWidth],
                [center[0], center[1]+headWidth], 
                [center[0]+headWidth, center[1]], 
                [center[0], center[1]-headWidth]
                ]
        
    elif(np.array_equal(direction, np.array([1,0]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        arrow = [
                [center[0]-tailWidth, center[1]],
                [center[0]-tailWidth, center[1]-tailHeight], 
                [center[0]+tailWidth, center[1]-tailHeight], 
                [center[0]+tailWidth, center[1]],
                [center[0]+headWidth, center[1]], 
                [center[0], center[1]+headWidth], 
                [center[0]-headWidth, center[1]]
                ]
        
    elif(np.array_equal(direction, np.array([0,-1]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        arrow = [
                [center[0], center[1]+tailWidth],
                [center[0]+tailHeight, center[1]+tailWidth], 
                [center[0]+tailHeight, center[1]-tailWidth],
                [center[0], center[1]-tailWidth],
                [center[0], center[1]-headWidth], 
                [center[0]-headWidth, center[1]], 
                [center[0], center[1]+headWidth]
                ]
        
    elif(np.array_equal(direction, np.array([-1,0]))):
        halfScale = int(scale/2)-1
        center = [coord[1]*scale+halfScale, coord[0]*scale+halfScale]
        tailHeight = halfScale-2
        arrow = [
                [center[0]+tailWidth, center[1]],
                [center[0]+tailWidth, center[1]+tailHeight], 
                [center[0]-tailWidth, center[1]+tailHeight], 
                [center[0]-tailWidth, center[1]],
                [center[0]-headWidth, center[1]], 
                [center[0], center[1]-headWidth], 
                [center[0]+headWidth, center[1]]
                ]
        
    return np.array(arrow, dtype='int64')

def drawStar(coord, scale, diameter, numPoints):
    halfScale = int(scale/2)-1
    centerX, centerY = coord[1]*scale+halfScale, coord[0]*scale+halfScale
    outerRad=diameter//2
    innerRad=int(outerRad*3/8)
    #fill the center of the star
    angleBetween=2*math.pi/numPoints#angle between star points in radians
    points = list()
    for i in range(numPoints):
        #p1 and p3 are on the inner radius, and p2 is the point
        pointAngle=math.pi/2+i*angleBetween
        p1X=centerX+innerRad*math.cos(pointAngle-angleBetween/2)
        p1Y=centerY-innerRad*math.sin(pointAngle-angleBetween/2)
        p2X=centerX+outerRad*math.cos(pointAngle)
        p2Y=centerY-outerRad*math.sin(pointAngle)
        p3X=centerX+innerRad*math.cos(pointAngle+angleBetween/2)
        p3Y=centerY-innerRad*math.sin(pointAngle+angleBetween/2)
        points+=(p1X,p1Y),(p2X,p2Y),(p3X,p3Y)
    return np.array(points, dtype='int64')

def getRectPoints(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return np.array([base, [base[0]+scale-1, base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def getCenter(coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return [int(math.floor(base[0]+scale/2)), int(math.floor(base[1]+scale/2))]

def getTriPoints( coord, scale):
    base = [coord[1]*scale, coord[0]*scale]
    return  np.array([[int(math.floor(base[0]+scale/2)), base[1]], [base[0]+scale-1,base[1]+scale-1], [base[0], base[1]+scale-1]])    

def renderWorld(scale=20, world = np.zeros(1),agents=[], goals=[], human=(-1,-1), humanPath=list(), humanStep=0):
    size = world.shape

    half = int(len(humanPath)/2)

    if(humanStep<half):
        path = humanPath[humanStep+1:half+1]
    else:
        path = humanPath[humanStep+1:]

    scale = 20

    size = world.shape

    screen_height = scale*size[0]
    screen_width = scale*size[1]

    colours = init_colors()

    scene = np.zeros([screen_height, screen_width, 3])

    for coord,val in np.ndenumerate(world):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=colours[val])

    for idx, val in enumerate(path):
        if(idx==len(path)-1):
            cv2.fillPoly(scene, pts=[drawStar(coord=val, scale=scale, diameter=scale, numPoints=5)], color=colours[-2])
        else:
            direction = np.subtract(path[idx+1],val)
            cv2.fillPoly(scene, pts=[getArrowPoints(direction=direction, coord=val, scale=scale,tailWidth=scale/10,headWidth=scale/2-2)], color=colours[-2])


    for val,coord in enumerate(agents):
        cv2.fillPoly(scene, pts=[getRectPoints(coord=coord, scale=scale)], color=colours[val+1])


    for val,coord in enumerate(goals):
        cv2.circle(scene, getCenter(coord=coord, scale=scale), math.floor(scale/2)-1, colours[val+1], -1)

    cv2.fillPoly(scene, pts=[getTriPoints(coord=human, scale=scale)], color=colours[-2])

    scene = scene*255
    scene = scene.astype(dtype='uint8')
    return scene


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def write_to_wandb(step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True, greedy=True):
    """record performance using wandb"""
    if imitation_loss is not None:
        wandb.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        wandb.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
    if evaluate:
        if greedy:
            for i in dir(performance_dict):
                if not i.startswith('__'):
                    wandb.log({'Perf_greedy_eval/'+i: getattr(performance_dict, i)}, step=step)

        else:
            for i in dir(performance_dict):
                if not i.startswith('__'):
                    wandb.log({'Perf_random_eval/'+i: getattr(performance_dict, i)}, step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        for i in dir(performance_dict):
            if not i.startswith('__'):
                wandb.log({'Perf/'+i: getattr(performance_dict, i)}, step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                wandb.log({'Grad/' + name: val}, step=step)
            else:
                wandb.log({'Loss/' + name: val}, step=step)
                
def write_to_wandb_with_run(run, step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True, greedy=True):
    """record performance using wandb"""
    if imitation_loss is not None:
        run.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        run.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
    if evaluate:
        if greedy:
            for i in dir(performance_dict):
                if not i.startswith('__'):
                    run.log({'Perf_greedy_eval/'+i: getattr(performance_dict, i)}, step=step)

        else:
            for i in dir(performance_dict):
                if not i.startswith('__'):
                    run.log({'Perf_random_eval/'+i: getattr(performance_dict, i)}, step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        for i in dir(performance_dict):
            if not i.startswith('__'):
                run.log({'Perf/'+i: getattr(performance_dict, i)}, step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                run.log({'Grad/' + name: val}, step=step)
            else:
                run.log({'Loss/' + name: val}, step=step)


def make_gif(images, file_name):
    """record gif"""
    print("writing gif to ", file_name)
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("wrote gif")