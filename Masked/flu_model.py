from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid, Grid
from mesa.datacollection import DataCollector
import numpy as np
import enum
import random

class FluAgent(Agent):

    def __init__(self, unique_id, model):

        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE
        self.infection_time = 0

        # age (currently not used)
        self.age = self.random.normalvariate(40, 20)

        # "base probability" of getting tested. (mean=1, st. dev=0.2)
        self.p_test = self.random.normalvariate(1, 0.1)

        
    def step(self):

        self.status()
        self.move()
        self.contact()

    def move(self):

        # get possible movements (including not moving)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True)
        
        # Social Distancing
        if self.model.socialDistancing == True:
            #print("Social Distancing In Effect")
            p_badmv = 0.025
            mv_probs = [0] * len(possible_steps)
            #print('Number of mvs:',len(mv_probs))
            count = 0
            # Check if adjacent cells for each move contain agents, if they do, 
            # place a penalty on moving to that cell. Over time, this should prioritze movements 
            # toward locations that are spread apart from one another
            for i, mv in enumerate(possible_steps):
                #print(i, mv)
                # Should get neighbors here and check if each of those neighbors is empty
                adj_cells = self.model.grid.get_neighborhood(
                mv, moore=True, include_center=False)
               # mv_bool
                for adj in adj_cells:
                    tmp = True
                    # If any adjacent cell for a move has an agent, then reduce the probability
                    if self.model.grid.is_cell_empty(adj) == False:
                        #mv_probs[i] = 0.075
                        tmp = False
                        #continue # break this loop
                if tmp == False:
                   # print('I:', i);
                    mv_probs[i] = p_badmv
                    count += 1
                        
            # Original (checking adj moves)
            # if self.model.grid.is_cell_empty(mv) == False:
                #    mv_probs[i] = 0.075
                 #   count += 1
           # print(mv_probs)
            indexes = [i for i, x in enumerate(mv_probs) if x == 0]
          #  print('Number of empties:', len(indexes))
          #  print('Empties: ',indexes)
          #  print('Num Bad Moves: ', count)
            if count != 0 and len(indexes) > 0:
                replacements = [(1-(p_badmv * count))/(len(indexes))] * len(indexes)

             ##   print('Replacements: ', replacements)                               

                for j, index in enumerate(indexes):
                    mv_probs[index] = replacements[j]
               # print(mv_probs)
        
                
        
        # pick randomly from list of spaces & move there
        #new_position = self.random.choice(possible_steps, p = mv_probs)
            vals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #new_position = np.random.choice(possible_steps, len(possible_steps), p = mv_probs)
            if count != 0 and len(indexes) > 0:
                new_position = possible_steps[np.random.choice(vals,p=mv_probs)]
            else:
                new_position = self.random.choice(possible_steps)
        # With no social distancing
        else:
            new_position = self.random.choice(possible_steps)
       
        # Make move
        self.model.grid.move_agent(self, new_position)

    def status(self):
        if self.state == State.INFECTED:

            # probability to get tested increased amount of symptoms.
            # (Approximately doubled from normal probability, with some var)

            # BIASED
            if self.model.biased:
                self.p_test = self.random.normalvariate(2, 0.25)

            # generate random number and compare to death rate
            drate = self.model.death_rate
            alive = np.random.choice([0,1], p=[drate, 1-drate])

            # remove agent from simulation and increment deceased count
            if alive == 0:
                self.model.deceased += 1
                self.model.schedule.remove(self)

            # check to see if recovered
            t = self.model.schedule.time - self.infection_time

            if t >= self.recovery_time:

                # set new state and change p of being sampled to 0.5 * original
                self.state = State.RECOVERED

                # BIASED
                if self.model.biased:
                    self.p_test = self.random.normalvariate(0.5, 0.05)

    def contact(self):

        # Find agent neighbors in grid
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        # If there are any neighboring agents
        if len(cellmates) > 1:

            for other in cellmates:

                # If random p generated is larger than p_transmission,
                # that agent gains the disease. Else, continue.
                
                # transmissionP changes depending on if the agent is wearing a mask
                if self.random.random() < self.model.p_mask:
                    transmissionP = self.model.mask_inf
                else:
                    transmissionP = self.model.ptrans
                    
                if self.random.random() > transmissionP: #changed back
                    continue

                # If current agent is Infected & other agent is Susceptibke
                if self.state is State.INFECTED and other.state is State.SUSCEPTIBLE:
                    other.state = State.INFECTED
                    other.infection_time = self.model.schedule.time
                    other.recovery_time = self.model.get_recovery_time()

class FluModel(Model):

    def __init__(self, N, width=10, height=10, death_rate=0.006, ptrans=0.5,
                            recovery_days=24, recovery_sd=6,
                            init_inf = 1.5, biased=False, pmask = 0.5, maskEffect = 0.9, socialDistancing = False):

        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        self.death_rate = death_rate
        self.ptrans = ptrans
        self.socialDistancing = socialDistancing
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.running = False
        self.deceased = 0
        self.init_inf = init_inf / 100
        self.biased = biased
        self.p_mask = pmask # probability of agents wearing mask
        self.mask_inf = self.ptrans * (1-float(maskEffect)) # effect mask has on the infection rate.         
        for i in range(self.num_agents):

            a = FluAgent(i, self)
            self.schedule.add(a)

            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
                    
            infected = np.random.choice([0,1], p=[1-self.init_inf, self.init_inf])

            if (infected == 1):
                a.state = State.INFECTED
                a.recovery_time = self.get_recovery_time()

        # if initial infection percent is very small, pick a single agent to infect
        if (self.init_inf * self.num_agents < 1):
            a = self.random.choice(self.schedule.agents)
            a.state = State.INFECTED
            a.recovery_time = self.get_recovery_time()

        self.datacollector = DataCollector(agent_reporters={"State": "state", 'p': 'p_test'})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def get_recovery_time(self):
        return int(self.random.normalvariate(self.recovery_days,self.recovery_sd))

    def sample(self, percent, i):

        # access DataCollector (states, weights, deceased agents)
        df = self.datacollector.get_agent_vars_dataframe()
        weights = df.p / sum(df.p)

        # select only current step
        df = df.iloc[df.index.get_level_values('Step') == i]

        # sample defined percent for State
        df2 = df.sample(frac=percent/100, random_state=1,
                weights=weights).drop(columns='p')

        return int(df2[df2.State == 1].count())

# State enum class
class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
