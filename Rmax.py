import numpy as np

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4
def create_matrix(width,height,liste):
    return [[liste for i in range(width)] for j in range(height)]

class Rmax_Agent:

    def __init__(self,environment, gamma, max_visits_per_state, epsilon = 0.2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_visits_per_state = max_visits_per_state
        self.Q =np.array(create_matrix(environment.width, environment.height,[0,0,0,0,0])) 
        self.R = np.zeros((environment.height, environment.width))
        self.nSA = np.zeros((environment.height, environment.width))
        self.nSAS = np.zeros((environment.height, environment.width,environment.height))
        self.val1 = []
        self.val2 = []  #This is for keeping track of rewards over time and for plotting purposes  
        print(int( np.ceil(np.log(1 / (self.epsilon * (1-self.gamma))) / (1-self.gamma))))

    def estimate_transition_probablities(self):

        for episode in range(self.max_episodes):

            obs = env.reset()
            if(episode % 20 == 0):
                self.val1.append(self.mean_rewards_per_500())
                self.val2.append(episode)            
            

            for step in range(self.max_steps):

                best_action = self.choose_action(obs)
                new_obs, reward, done, _ = env.step(best_action)
                #print(obs)
                if self.nSA[obs][best_action] < self.max_visits_per_state :

                    self.nSA[obs][best_action] +=1
                    self.R[obs][best_action] += reward
                    self.nSAS[obs][best_action][new_obs] += 1

                    if self.nSA[obs][best_action] == self.max_visits_per_state:

                        for i in range(int( np.ceil(np.log(1 / (self.epsilon * (1-self.gamma))) / (1-self.gamma)) )):

                            for state in range(env.nS):
                                
                                for action in range(env.nA):

                                    if self.nSA[state][action] >= self.max_visits_per_state:
                                        
                                        #In the cited paper it is given that reward[s,a]= summation of rewards / nSA[s,a]
                                        #We have already calculated the summation of rewards in line 28
                                        q = (self.R[state][action]/self.nSA[state][action])

                                        
                                        for next_state in range(env.nS):
                                            
                                            #In the cited paper it is given that transition[s,a] = nSAS'[s,a,s']/nSA[s,a]

                                            transition = self.nSAS[state][action][next_state]/self.nSA[state][action]
                                            q += (transition * np.max(self.Q[next_state,:]))

                                        self.Q[state][action] = q 
                                        #print(q + self.gamma*(self.R[state][action]/self.nSA[state][action]))
                                        #In the cited paper it is given that reward[s,a]= summation of rewards / nSA[s,a]
                                        #We have already calculated the summation of rewards in line 28
                
               
                if done:
                    if not(reward==1):
                        self.R[obs][best_action]=-10

                    break

                obs = new_obs




    def choose_action(self,observation):
        if np.random.random() > (1-self.epsilon):
            action = np.random.choice([UP,DOWN,LEFT,RIGHT,STAY])
        else:
            action = np.argmax(self.Q[observation])
        return action