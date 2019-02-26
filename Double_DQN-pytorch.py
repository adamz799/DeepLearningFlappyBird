#!/usr/bin/env python
from __future__ import print_function

import torch
from torch import nn, optim
import torch.nn.init as init
from Net import Net

import cv2
import sys
sys.path.append("game/")
from tensorboardX import SummaryWriter

import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import copy


GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1260000+10000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
INITIAL_EPSILON = 0.099*(200-126)/200  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 64  # size of minibatch
FRAME_PER_ACTION = 1

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda')
writer = SummaryWriter()

def trainNetwork(target_net, evaluate_net):
    loss_func = nn.MSELoss().cuda()
    optimizer_target = optim.Adam(target_net.parameters(), lr=1e-6)
    #optimizer_evaluate = optim.Adam(evaluate_net.parameters(), lr=1e-6)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    current_frame = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, current_frame = cv2.threshold(
        current_frame, 1, 255, cv2.THRESH_BINARY)
    current_frame = current_frame.astype(np.float32)
    frame_data = np.stack((current_frame, current_frame,
                           current_frame, current_frame), axis=0)
    #input_img = torch.from_numpy(frame_data).float()
    #input_img = frame_data
    input_img = torch.from_numpy(frame_data).float()

    # start training
    epsilon = INITIAL_EPSILON
    t = 1260000
    while "flappy bird" != "angry bird":

        # choose an action epsilon greedily
        Q_value1 = evaluate_net(input_img.cuda().unsqueeze(0))#To generate action
        actions = np.zeros([ACTIONS])  # [0,0]

        if t % FRAME_PER_ACTION == 0:
            Q_value1 = Q_value1.cpu().detach().numpy()[0]
            #action_index = np.argmax(Q_value1)

            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                actions[action_index] = 1
            else:               
                action_index = np.argmax(Q_value1)
                #action_index = np.argmax(Q_value1.cpu().detach().numpy())
                actions[action_index] = 1
        else:
            actions[0] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        frame, reward, terminal = game_state.frame_step(actions)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (80, 80))
        _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = frame.astype(np.float32)

        frame = torch.Tensor(frame).cpu()
        new_input = torch.stack(
            [frame, input_img[0,:,:], input_img[1,:,:], input_img[2,:,:]]
        )

        #frame = torch.Tensor(frame).cpu()
        # new_input = torch.stack(
        #     [frame, input_img[0, :, :], input_img[1, :, :], input_img[2, :, :]])
    
        # store the transition in D
        current_pair = (input_img,
            actions,
            # np.array([reward]).astype(np.float32),
            #  torch.Tensor(actions, device=device),
            torch.Tensor([reward], device=device),
            new_input,
            terminal)
        D.append(current_pair)

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH-1)
            minibatch.append(current_pair)

            # get the batch variables
            input_batch = torch.stack([d[0] for d in minibatch]).cuda()
            action_batch = np.stack([d[1] for d in minibatch])
            reward_batch = torch.stack([d[2] for d in minibatch]).cuda()
            new_input_batch = torch.stack([d[3] for d in minibatch]).cuda()


            #input_batch = torch.stack(input_batch).cuda()  # s_t
            #action_batch = torch.stack([d[1] for d in minibatch]).cpu()  # a_t
            #reward_batch = torch.from_numpy(reward_batch).cuda()  #reward should be a tensor 
            #new_input_batch = torch.stack(new_input_batch).cuda()  # s_t1

            #action_batch = a_batch.numpy()
            
            y_batch = []
            Q_apostrophe_batch = target_net(new_input_batch) #Q'

            #Get y_batch
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    value = torch.max(Q_apostrophe_batch[i])
                    y_batch.append(reward_batch[i] + GAMMA * value)#Y = Reward+gamma*Q'

            y_batch = torch.stack(y_batch).cuda()

            index = np.argmax(action_batch, axis=1)
            index = np.reshape(index, [BATCH, 1])
            index = torch.Tensor(index).cuda()
            Q_batch = target_net(input_batch).gather(1, index.long()) #Q
            
            #Q-learning: Q = Q +alpha*(Reward+gamma*max(Q')-Q)
            loss = loss_func(Q_batch, y_batch)# loss = (Y-Q)^2
            #print( 'loss: ',loss)
            optimizer_target.zero_grad()
            loss.backward()
            optimizer_target.step()

            writer.add_scalar('data/loss', loss, t)
            writer.add_scalar('data/reward', reward, t)

        # update the old values
        input_img = new_input
        t += 1
        
        if t> OBSERVE and t% 10000 == 0:
            evaluate_net = copy.deepcopy(target_net)
        # save progress every 10000 iterations
        if t % 10000 == 0:
            #torch.save(target_net, 'target_net.pkl')
            torch.save(evaluate_net, 'evaluate_net.pkl')

        # print info
        state = ''
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print(t, "/ EPSILON", epsilon, "/ ACTION", 1-actions[0], "/ REWARD", reward,
              "/ Q_MAX %e" % Q_value1[action_index])


if __name__ == "__main__":
    # net1 = Net().cuda()
    # net2 = Net().cuda()
    net1 = torch.load('evaluate_net.pkl')
    net2 = torch.load('evaluate_net.pkl')
    #torch.autograd.set_grad_enabled(True)
    #net = net.apply(weight_init)
    trainNetwork(net1, net2)
    writer.export_scalars_to_json("Double_DQN_test.json")
    writer.close()
