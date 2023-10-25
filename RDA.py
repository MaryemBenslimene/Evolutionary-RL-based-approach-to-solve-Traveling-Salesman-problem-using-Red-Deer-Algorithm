import numpy as np
import matplotlib.pyplot as plt
import random, math
import torch
import copy
import torch.nn.functional as F
import torch
import torch.optim as optim

from model import DRL4TSP
from tasks import tsp
import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RDA_Optim (optim.Optimizer):

    def __init__(self, population_size, num_generations, upper_bound, lower_bound, gamma, alpha, beta):
        self.population_size = population_size
        self.num_generations = num_generations
        self.UB = upper_bound
        self.LB = lower_bound
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.num_males = int(0.25 * self.population_size)
        self.num_hinds = self.population_size - self.num_males
        self.num_coms = int(self.num_males * gamma)  
        self.num_stags = self.num_males - self.num_coms 
        self.deer_population = []
        self.males_deer = []
        self.hinds_deer = []
        self.commanders_deer = []
        self.stags_deer =[] 


    def create_deer(self, static_size, dynamic_size, args, update_fn):
        return DRL4TSP(static_size,
                    dynamic_size,
                    args.hidden_size,
                    update_fn,
                    tsp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    def initialize_deer_population (self, static_size, dynamic_size, args, update_fn):
        return [[self.create_deer(static_size, dynamic_size, args, update_fn),0] for _ in range(self.population_size)]


    def fitness_fn(self, deer, critic, test_loader, test_dir, num_plot=5, **kwargs):
        trainer.train(deer, critic, **kwargs)
        return trainer.validate(test_loader, deer, tsp.reward, tsp.render, test_dir, num_plot)

    def fitness_population(self, population, critic, test_loader, test_dir, num_plot, **kwargs) :
        for deer in population :
            deer[1] = self.fitness_fn(deer[0],critic, test_loader, test_dir, num_plot, **kwargs)
        return sorted(population, key=lambda x: x[1])

    """""
    def sort_agents (self, population, fitness):
        idx = list(np.argsort(fitness))
        print("idx", idx)
        sorted_population = [0 for _ in range(len(population))]
        sorted_fitness = [0 for _ in range(len(population))]
        for id in idx :
            sorted_population[id] = population[id]
            sorted_fitness[id] = fitness[id]
        return sorted_population, sorted_fitness
    """""

    def males_population(self):
        self.males_deer = self.deer_population[:self.num_males]
        
    def hinds_population(self):
        self.hinds_deer = self.deer_population[self.num_males:]

    def commanders_population(self):
        self.commanders_deer = self.males_deer[:self.num_coms]

    def stags_population(self):
        self.stags_deer = self.males_deer[self.num_coms:]


    
    def males_roaring(self, critic, test_loader, test_dir, num_plot, **kwargs):
        for i in range(self.num_males):
            r1 = np.random.random() # r1 is a random number in [0, 1]
            r2 = np.random.random() # r2 is a random number in [0, 1]
            r3 = np.random.random() # r3 is a random number in [0, 1]
            new_male = copy.deepcopy(self.males_deer[i][0])
            for name, param in new_male.named_parameters():
                r1_tensor = r1 * torch.ones_like(param)
                r2_tensor = r2 * torch.ones_like(param)
                ub_tensor = self.UB * torch.ones_like(param)
                lb_tensor = self.LB * torch.ones_like(param)
                if r3 >= 0.5:
                    new_param = param + r1_tensor * (((ub_tensor - lb_tensor) * r2_tensor) + lb_tensor) #Eq (3)
                else:
                    new_param = param - r1_tensor * (((ub_tensor - lb_tensor) * r2_tensor) + lb_tensor) #Eq (3)

                new_male.state_dict()[name].copy_(new_param)
            
            new_male_fitness = self.fitness_fn(new_male, critic, test_loader, test_dir, num_plot, **kwargs)

            if new_male_fitness < self.males_deer[i][1] :
                self.males_deer[i] = [new_male, new_male_fitness]




    def commanders_stags_fight(self, critic, test_loader, test_dir, num_plot, **kwargs):
        for i in range(self.num_coms):
            com = self.commanders_deer[i]
            chosen_com = com[0]
            random_stag = random.choice(self.stags_deer)
            chosen_stag = random_stag[0]

            r1 = np.random.random()
            r2 = np.random.random()

            new_male_1 = copy.deepcopy(chosen_com)
            new_male_2 = copy.deepcopy(chosen_com)

            for name_com, param_com in chosen_com.named_parameters():
                for name_stag, param_stag in chosen_stag.named_parameters():
                    r1_tensor = r1 * torch.ones_like(param_com)
                    r2_tensor = r2 * torch.ones_like(param_com)
                    ub_tensor = self.UB * torch.ones_like(param_com)
                    lb_tensor = self.LB * torch.ones_like(param_com)
                    #r1_tensor = torch.full(param_com.shape, r1, dtype=param_com.dtype, device=param_com.device)
                    #ub_tensor = torch.full(param_com.shape, self.UB, dtype=param_com.dtype, device=param_com.device)
                    #lb_tensor = torch.full(param_com.shape, self.LB, dtype=param_com.dtype, device=param_com.device)
                    #r2_tensor = torch.full(param_com.shape, r2, dtype=param_com.dtype, device=param_com.device)
                    if name_com == name_stag:
                        new_male_1_param = (param_com + param_stag) / 2 + (r1_tensor * (((ub_tensor - lb_tensor) * r2_tensor) + lb_tensor)) #Eq (6)
                        new_male_2_param = (param_com + param_stag) / 2 - (r1_tensor * (((ub_tensor - lb_tensor) * r2_tensor) + lb_tensor)) #Eq (7)
                        new_male_1.state_dict()[name_com].copy_(new_male_1_param)
                        new_male_2.state_dict()[name_com].copy_(new_male_2_param)
            
            fitness_male1 = self.fitness_fn(new_male_1, critic, test_loader, test_dir, num_plot, **kwargs)
            fitness_male2 = self.fitness_fn(new_male_2, critic, test_loader, test_dir, num_plot, **kwargs)
            
            fitness_males = [com[1], random_stag[1], fitness_male1, fitness_male2]
            bestfit = np.min(fitness_males)
            if fitness_males[0] > fitness_males[1] and fitness_males[1] == bestfit:
                self.commanders_deer[i] = random_stag
            elif fitness_males[0] > fitness_males[2] and fitness_males[2] == bestfit:
                self.commanders_deer[i] = [new_male_1, fitness_male1]
            elif fitness_males[0] > fitness_males[3] and fitness_males[3] == bestfit:
                self.commanders_deer[i] = [new_male_2, fitness_male2]

        
    def harems_formation(self):
        print("harems formation")
        print("len comm deer", len(self.commanders_deer))
        sorted_pop, fitness = zip(*self.deer_population)
        norm = np.linalg.norm(fitness)
        print("norm", norm)
        normal_fit = fitness / norm
        print("normal fit", normal_fit)
        total = np.sum(normal_fit)
        print("total", total)
        power = normal_fit / total # Eq. (9)
        print("power", power)
        num_harems = [int(x * self.num_hinds) for x in power] # Eq.(10)
        print("num harems", num_harems)
        max_harem_size = np.max(num_harems)
        print("max harem size", max_harem_size)
        harem = []
        random.shuffle(self.hinds_deer)
        itr = 0
        for i in range(self.num_coms):
            harem_size = num_harems[i]
            harem_com = []
            for _ in range(harem_size):
                print("hind deer", self.hinds_deer[itr])
                harem_com.append(self.hinds_deer[itr])
                itr += 1
            harem.append(harem_com)
        print("harem = ", harem)
        return harem, num_harems


    def commanders_harem_mating(self, harem, num_harems, critic, test_loader, test_dir, num_plot, **kwargs):
        offsprings = []
        num_harem_mate = [int(x * self.alpha) for x in num_harems] # Eq. (11)
        for i in range(self.num_coms):
            random.shuffle(harem[i])
            for j in range(num_harem_mate[i]):
                offspring = copy.deepcopy(self.commanders_deer[i][0])
                r = np.random.random() # r is a random number in [0, 1]
                for name_com, param_com in self.commanders_deer[i][0].named_parameters():
                    for name_harem, param_harem in harem[i][j][0].named_parameters():
                        r_tensor = r * torch.ones_like(param_com)
                        ub_tensor = self.UB * torch.ones_like(param_com)
                        lb_tensor = self.LB * torch.ones_like(param_com)
                        if name_com == name_harem:
                            offspring_param = (param_com + param_harem) / 2 + (ub_tensor - lb_tensor) * r_tensor # Eq. (12)
                            offspring.state_dict()[name_com].copy_(offspring_param)
                offsprings.append([offspring, self.fitness_fn(offspring, critic, test_loader, test_dir, num_plot, **kwargs)])
                # if number of commanders is greater than 1, inter-harem mating takes place
                if self.num_coms > 1:
                    # mating of commander with hinds in another harem
                    k = i 
                    while k == i:
                        k = random.choice(range(self.num_coms))

                    num_mate = int(num_harems[k] * self.beta) # Eq. (13)

                    np.random.shuffle(harem[k])
                    offspring = copy.deepcopy(self.commanders_deer[i][0])
                    for j in range(num_mate):
                        r = np.random.random() # r is a random number in [0, 1]

                        for name_com, param_com in self.commanders_deer[i][0].named_parameters():
                            r_tensor = r * torch.ones_like(param_com)
                            ub_tensor = self.UB * torch.ones_like(param_com)
                            lb_tensor = self.LB * torch.ones_like(param_com)
                            for name_harem, param_harem in harem[i][j][0].named_parameters():
                                if name_com == name_harem:

                                    offspring_param = (param_com + param_harem) / 2 + (ub_tensor - lb_tensor) * r_tensor # Eq. (12)

                                    offspring.state_dict()[name_com].copy_(offspring_param)
                        offsprings.append([offspring, self.fitness_fn(offspring, critic, test_loader, test_dir, num_plot, **kwargs)])


        return offsprings
    
    def cosine_similarity(self, model1, model2):
        w1 = torch.cat([p.flatten() for p in model1.parameters()])
        w2 = torch.cat([p.flatten() for p in model2.parameters()])
        return F.cosine_similarity(w1, w2, dim=0).item()

    def stag_hind_mating(self, critic, test_loader, test_dir, num_plot, **kwargs):
        offsprings = []
        for stag in self.stags_deer:
            dist = np.zeros(self.num_hinds)
            for i in range(self.num_hinds):
                dist[i] = self.cosine_similarity(stag[0], self.hinds_deer[i][0])
            min_dist = np.max(dist)
            for i in range(self.num_hinds):
                distance = self.cosine_similarity(stag[0], self.hinds_deer[i][0]) # Eq. (14)
                if(distance == min_dist):
                    offspring = copy.deepcopy(stag[0])
                    r = np.random.random() # r is a random number in [0, 1]
                    for name_stag, param_stag in stag[0].named_parameters():
                            for name_hind, param_hind in self.hinds_deer[i][0].named_parameters():
                                r_tensor = r * torch.ones_like(param_stag)
                                ub_tensor = self.UB * torch.ones_like(param_stag)
                                lb_tensor = self.LB * torch.ones_like(param_stag)
                                if name_stag == name_hind:
                                    offspring_param = (param_stag + param_hind)/2 + ((ub_tensor - lb_tensor) * r_tensor)
                                    offspring.state_dict()[name_stag].copy_(offspring_param)
                    offsprings.append([offspring, self.fitness_fn(offspring, critic, test_loader, test_dir, num_plot, **kwargs)])
        return offsprings

        