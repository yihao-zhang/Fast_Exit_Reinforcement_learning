import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from collections import deque
import os

H = 15
W = 15
A_list = { (H//3 ,W//2),\
          (H//3*2 , W//3),\
          (H//3*2 , W//3*2) }

P = [H //2 , 0]

A_list = { *[(i,W//2) for i in range(1,H)] , *[(i,W//3*2) for i in range(0,H-1)], *[(i,W//3) for i in range(0,H-1)]}

A_list.discard((10, W//2))
A_list.discard((8, W//3*2))

class Plate:
    
    def __init__(self, height = 15, width = 15):
        self.height = height
        self.width = width
        self.grid = np.ones((height, width), dtype = 'float')
        self.grid[P[0], P[1]] = 0.
        self.grid[:, -1] = -1.0
        self.grid_color = np.full((height, width, 3), 255, dtype = 'int')
        self.cmap = {0. : [0,0,0], 0.5 : [0, 0, 255], 1.0: [255,255,255], -1.0 : [255,0,0]}
        self.action = np.array([[-1,0], [1, 0], [0,1], [0,-1]], dtype = 'int') ###up, down, right, left
        self.policy = np.array([0.25,0.25,0.25,0.25])  
        self.state = np.array([P[0], P[1]], dtype = 'int')

        self.a_list = A_list
        self.reward = -1
        self.end_reward = 10
        self.blue_reward = 8
        self.out_reward = -5
                
    
    def choose_action(self):
        
        return np.argmax(np.random.multinomial(1, self.policy))
    
    def out_boundary(self, position):
        if position[0]< 0 or position[0] >= self.height or position[1]<0 or position[1]>=self.width:
            return True
        else:
            return False
    
    def step(self, action):
        
        done = False
        reward = self.reward
        
        new_state = self.state + self.action[action]
        
        #######reach the end
        if new_state[1] == self.width-1:
            done = True
            reward = self.end_reward
            
        #######
        if (new_state[0], new_state[1]) in self.a_list:
            if self.grid[new_state[0], new_state[1]] == 0.5:
                reward = self.blue_reward
        
        ####### out of boundary
        if self.out_boundary(new_state):
            new_state = self.state        
            reward = self.out_reward
 
        self.grid[new_state[0], new_state[1]] = 0.
        
        return new_state, reward, done

    def reset(self):
        self.grid[:,:] = 1.0     
        self.grid[P[0], P[1]] = 0.
        self.grid[:, -1] = -1.0
        self.state = np.array([P[0], P[1]])
        
        ###
        for i,j in self.a_list: 
            self.grid[i, j] = 0.5


        
    def show(self):
        
        for i in range(self.height):
            for j in range(self.width):
                self.grid_color[i,j,:] = self.cmap[ self.grid[i,j] ]
        
        self.grid_color[self.state[0], self.state[1], :] = [0,255,0]
        
        im = plt.imshow(self.grid_color, cmap ='viridis', animated = True)
        return [im]
    
        

class Agent:
    
    def __init__(self):

        self.gamma = 0.95
        self.alpha = 0.5
        self.build_QA_table()
        self.max_iteration = 10000
        self.theta = 1e-4
        self.action = np.array([[-1,0], [1, 0], [0,1], [0,-1]], dtype = 'int') ###up, down, right, left

    def build_QA_table(self):
        self.QA = np.zeros((H, W, 4))
        
    def value_iteration(self):
        
        for i in range(self.max_iteration):
            
            QA_new = np.zeros_like(self.QA)
            num = 0.
            
            for r in range(H):
                for c in range(W-1):
                    for a in range(4):
                        reward = -1
                        new_state = [r + self.action[a,0], c + self.action[a,1] ]
                        
                        if (new_state[0],new_state[1]) in A_list:
                            reward = -50
                            
                        if new_state[1] == W-1:
                            reward = 10
                        
                        if new_state[0] < 0 or new_state[0]>=H or new_state[1]<0 or new_state[1]>=W:
                            reward = -5
                            new_state = [r, c]
                        
                        tmp = reward + self.gamma * np.max(self.QA[new_state[0],new_state[1],:])
             
                        QA_new[r,c,a] = tmp
    
            
            num = np.sum( np.abs(QA_new - self.QA) )
            self.QA = QA_new
            if num < self.theta:
                print(self.QA[0,W-2,:], num, i)
                break


if __name__ == '__main__':
    
    env = Plate(H,W)
    state = env.reset()
    
    
    #################### value iteration shortest path
    agent = Agent()
    agent.value_iteration()
    
    G = 0
    step = 100
    ims = []
    fig = plt.figure()
    
    
    for n in range(100):
        print("step:{}".format(n))
        action = np.argmax( agent.QA[env.state[0], env.state[1], :] )
        
        new_state, reward, done = env.step(action)
        G += reward
        env.state = new_state
        ims.append(env.show())
        
        if done:
            print("Reach the end")
            break
     
    print("Total reward:{}".format(G))
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=100)
    ani.save('Fast_Exit.gif', writer='pillow')       
    
