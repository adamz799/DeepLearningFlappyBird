#!/usr/bin/env python
from __future__ import print_function

import torch
from torch import nn, optim
import torch.nn.init as init

import cv2
import sys
sys.path.append("game/")

import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1370001. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 64 # size of minibatch
FRAME_PER_ACTION = 1

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda')
# batch_size = 64 
# image_size = 80

# #Create transform
# transform = transforms.Compose(
#     [transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),]
# )

# #Load dataset
# dataset = datasets.CIFAR10(root = 'F:/t/python/pytorch-practice/cifar-10', download=False,transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=2)
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 1.0)
        m.bias.data.fill_(0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0,1.0)
        m.bias.data.fill_(0.1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, bias=True, stride=4,padding=2),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,64,4, bias= True,stride=2,padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2,padding=1),
            nn.Conv2d(64,64,3,bias=True,stride=1, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=1)
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 2,bias=True)
            #nn.Linear(512,ACTIONS,bias=True), 
        )
        for layer in self.conv.children() :
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean = 0, std=0.01)
                init.constant_(layer.bias, 0.1)

        for layer in self.fc.children():
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight,std=0.01)
                init.constant_(layer.bias, 0.1)

        #torch.autograd.set_grad_enabled(True)

    def forward(self, inputs):
        return self.fc(self.conv(inputs).view(-1,256))


# define the cost function
def cost(a, readout, y):
    readout_action = (readout* a).sum(dim = 1)
    t = y.detach()- readout_action
    cost = torch.mean(t*t)
    return cost

def trainNetwork(net):
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-6)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    current_frame = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret,current_frame = cv2.threshold(current_frame,1,255,cv2.THRESH_BINARY)
    current_frame.astype(np.float32)
    frame_data = np.stack((current_frame, current_frame, current_frame, current_frame), axis=0)
    input_img = torch.from_numpy(frame_data).float()

    # start training
    epsilon = 0.0001
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        Q_value1 = net(input_img.cuda().unsqueeze(0))
        a_t = np.zeros([ACTIONS])      
        
        if t % FRAME_PER_ACTION == 0 :
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(Q_value1.cpu().detach().numpy())
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        frame = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret,frame = cv2.threshold(frame,1,255,cv2.THRESH_BINARY)
        frame.astype(np.float32)
        frame = torch.Tensor(frame).cpu()
        new_input = torch.stack([frame, input_img[0,:,:],input_img[1,:,:],input_img[2,:,:]])   
        #new_frame_data = np.stack((frame, current_frame, current_frame, current_frame), axis=0)
        #new_input = torch.from_numpy(new_frame_data).float()

        # store the transition in D
        D.append(
            (input_img, 
            torch.Tensor(a_t, device = device), 
            torch.Tensor([r_t], device = device),
            new_input, 
            terminal)
        )
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            orginal_input_batch =  torch.stack([d[0] for d in minibatch]).cuda()#s_t
            a_batch = torch.stack([d[1] for d in minibatch]).cpu()#a_t
            reward_batch = torch.stack([d[2] for d in minibatch]).cuda()#r_t
            current_input_batch = torch.stack([d[3] for d in minibatch]).cuda()#s_t1

            action_batch = a_batch.numpy()
            index = np.argmax(action_batch, axis=1)
            index=np.reshape(index,[BATCH,1])
            index = torch.Tensor(index).cuda()
            y_batch = []
            Q_value_batch = net(current_input_batch)
            # for inputs in current_input_batch:
            #     readout_j1_batch.append(net(inputs.cuda()))#GPU
           
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * torch.max(Q_value_batch[i].detach()))

            y_batch = torch.stack(y_batch).cuda()
            y_predict = net(orginal_input_batch).gather(1,index.long())
            #y_predict.requires_grad_()
            # perform gradient step

            # for i in range(0, len(y_batch)):
            loss = loss_func(y_predict,y_batch)
            #print( 'loss: ',loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update the old values
        input_img = new_input
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            torch.save(net,'dqn-3.pkl')
            #saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ''
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(Q_value1.cpu().detach().numpy()))

if __name__ == "__main__":
    #net = Net().cuda()
    net = torch.load('dqn-2.pkl')
    #torch.autograd.set_grad_enabled(True)
    #net = net.apply(weight_init)
    trainNetwork(net)
