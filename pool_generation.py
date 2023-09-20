from sklearn.model_selection import train_test_split


from deap import algorithms
from deap import base
from deap import creator
from deap import tools


import collections

import Cpx

import numpy as np
import random



from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects.packages as rpackages

ecol = rpackages.importr("ECoL")

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

class poolGeneration:
    def __init__(
        self,
        method_disperse=True,
        fit_value=[1.0, 1.0, -1.0],
        nr_generation=20,
        nr_individual=100,
        nr_pop=100,
        proba_crossover=0.9,
        proba_mutation=0.1,
        nr_child=100,
        iteration = 20,
        stop_criteria = "maxdistance",
        classifier = "tree",
        tam_bags = 0.5,
        nr_bags = 100,
        group = ["overlapping", "neighborhood"],
        types = None,

    ):

        # tipo de avaliação Disperção/ Acc
        self.method_disperse = method_disperse

        # Função de fitnes, aproximação ou distanciamento dos dados
        self.fit_value1 = fit_value[0]
        self.fit_value2 = fit_value[1]
        self.fit_value3 = fit_value[2]
        # numero de gerações de bags
        self.nr_generation = nr_generation
        # n de bags resultantes de uma geração
        self.nr_individual = nr_individual
        # população inicial
        self.nr_pop = nr_pop

        # Probab de cossover ou mutation
        self.proba_crossover = proba_crossover
        self.proba_mutation = proba_mutation

        # Numero de bags filhos criados
        self.nr_child = nr_child
        # controle nao por no doc
        self.cont_crossover = 1

        self.iteration = iteration  # numero de variações de bags
        # verificar
        self.dist_temp = 0

        # nao usa mais
        self.jobs = 8
        # Escolha de melhor bag por dist ou acc
        self.stop_criteria = stop_criteria # maxacc
        # escolha do classificador
        self.classifier = classifier  # tree,perc
        # Acho que nao usa
        self.save_info = False
        # var d econtrole
        self.seq = -1
        # nao usa mais -> retirar
        self.base_name = "Base1"
        # self.file_out = "maxdistanceree"

        self.tem2 = []

        self.acc_temp = 0
        # Divisão da base em treino e val
        self.tam_bags = tam_bags
        # numero de bags
        self.nr_bags = nr_bags
        # nao usa
        self.file_out = "isto_e_um_teste"

        self.local = "saida"

        # salva complex
        self.c = []
        # salva os bags gerados
        self.bags_saved = []

        self.pool_classificators = []
        self.group = group
        self.types = types


    def generate_bags(self, X_train, y_train):
        indices = np.arange(len(X_train))
        bags = dict()
        bags["name"] = list()
        bags["inst"] = list()
        for i in range(0, self.nr_bags):
            X_bag, X_temp, y_bag, y_temp, id_bag, id_temp = train_test_split(
                X_train, y_train, indices, test_size=self.tam_bags, stratify=y_train
            )
            bags["name"].append(i)
            bags["inst"].append(id_bag.tolist())
            del X_bag, X_temp, y_bag, y_temp, id_bag, id_temp

        return bags

    def sequencia(self):
        self.seq += 1
        return self.seq

    def split_data(self, X_data, y_data):
        self.X = X_data
        self.y = y_data

        indices = np.arange(len(X_data))
        (
            self.X_train,
            X_temp,
            self.y_train,
            y_temp,
            self.id_train,
            id_temp,
        ) = train_test_split(X_data, y_data, indices, test_size=0.5, stratify=y_data)
        (
            self.X_test,
            self.X_vali,
            self.y_test,
            self.y_vali,
            self.id_test,
            self.id_vali,
        ) = train_test_split(X_temp, y_temp, id_temp, test_size=0.5, stratify=y_temp)

        return (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.X_vali,
            self.y_vali,
            self.id_train,
            self.id_test,
            self.id_vali,
        )

    def get_complexity(self, first_evaluate=False, population=None):

        dist = dict()
        dist["name"] = list()
        dist["dist"] = list()
        dist["diver"] = list()
        dist["score"] = list()
        dist["score_g"] = list()

        if first_evaluate == True and self.generation == 0:
            dist["name"] = self.pop
            r = []
            # r = Parallel(n_jobs=self.jobs)(delayed(self.parallel_distance2)(i, self.bags, self.group, self.types) for i in range(len(dist['name'])))
            for i in range(len(dist["name"])):
                r.append(self.parallel_distance2(i, self.bags, self.group, self.types))

            c, score, pred, pool = zip(*r)

            self.c = c
            

        elif first_evaluate == False and population == None:
            begin = self.name_individual - self.nr_individual
            for i in range(begin, self.name_individual):
                x = []
                x.append(i)
                dist["name"].append(x)
            # r = Parallel(n_jobs=self.jobs)(
            #     delayed(self.parallel_distance2)(j, self.bags, self.group, self.types) for j in range(100, self.nr_individual + 100))

            r = []
            for j in range(100, self.nr_individual + 100):
                r.append(self.parallel_distance2(j, self.bags, self.group, self.types))

            c, score, pred, pool = zip(*r)
            self.c = c

        elif population != None:
            dist["name"] = population
            indices = []
            for i in population:
                indices.append(self.bags["name"].index(i[0]))
            # r = Parallel(n_jobs=self.jobs)(delayed(self.parallel_distance2)(i,self.bags, self.group, self.types) for i in indices)

            r = []
            for i in indices:
                r.append(self.parallel_distance2(i, self.bags, self.group, self.types))
            c, score, pred, pool = zip(*r)
            self.c = c

        dist["dist"] = Cpx.dispersion_linear(c)
        dist["score"] = score
        d = self.diversity_ga(pred, self.y_val)
        dist["diver"] = Cpx.min_max_norm(d)
        dist["score_g"] = Cpx.voting_classifier(pool, self.X_val, self.y_val)

        self.dist = dist
        return

    def diversity_ga(self, pred, y):
        pred = np.array(pred)
        d = Cpx.diversitys(y, pred)
        return d

    def parallel_distance2(self, i, bags, group, types):
        """

        :param i: lista de indices do bag a ser testado
        :param bags: lista com todos os bags
        :param grupo: lista com o nome dos grupos de complexidades ex: [overllaping,,,,,]
        :param tipos: lista com o nome da complexidade
        :return: listas de complexidade, score do prorpio bag, score sobre a validacao, e a predicao sobre a validacao
        """
        indx_bag1 = bags["inst"][i]
        X_bag, y_bag = self.build_bags(indx_bag1)
        cpx = Cpx.complexity_data3(X_bag, y_bag, group, types)

        if self.classifier == "perc":
            estimator, score, pred = Cpx.biuld_classifier(
                X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val
            )
        elif self.classifier == "tree":
            estimator, score, pred = Cpx.biuld_classifier_tree(
                X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val
            )
        return cpx, score, pred, estimator

    def build_bags(self, indx_bag):

        """
        Recebe o indice de instancias de um bag
        :param indx_bag:
        :param vet_classes: false, retorna o vetor de classes
        :return: X_data, y_data
        """
        X_data = []
        y_data = []
        for i in indx_bag:
            X_data.append(self.X_train[int(i)])
            y_data.append(self.y_train[int(i)])
        return X_data, y_data

    def evaluate_linear_dispersion(self, ind1):
        dist = self.dist
        for i in range(len(dist["name"])):
            if dist["name"][i][0] == ind1[0]:
                dst1 = dist["dist"][i][0]
                dist2 = dist["dist"][i][1]
                diver = dist["diver"][i]
                break
        return (
            dst1,
            dist2,
            diver,
        )

    def crossover(self, ind1, ind2):
        """
        Para funcionar os bags devem ter o mesmo tamanho
        :param ind1:
        :param ind2:
        :return:
        """

        individual = False
        indx = self.bags["name"].index(ind1[0])
        indx2 = self.bags["name"].index(ind2[0])
        indx_bag1 = self.bags["inst"][indx]
        indx_bag2 = self.bags["inst"][indx2]
        _, y_data = self.build_bags(indx_bag1)
        cont = 0

        while individual != True:
            ind_out1 = self.short_cross(y_data, indx_bag1, indx_bag2)
            individual = self.verify_bag(ind_out1)
            cont = cont + 1
            if cont == 30:
                print("Stratification error")
                exit(0)

        ind1[0] = self.name_individual
        ind2[0] = self.name_individual

        self.bags["name"].append(self.name_individual)
        self.bags["inst"].append(ind_out1)
        self.name_individual += 1

        if self.method_disperse == True:
            self.cont_crossover = self.cont_crossover + 1
            if self.cont_crossover == self.nr_individual + 1:
                self.cont_crossover = 1
                self.get_complexity(first_evaluate=False, population=None)

        return creator.Individual(ind1), creator.Individual(ind2)

    def short_cross(self, y_data, indx_bag1, indx_bag2):
        beginning = finish = 0
        ind_out1 = []
        while y_data[beginning] == y_data[finish]:
            beginning = random.randint(0, len(y_data) - 1)
            finish = random.randint(beginning, len(y_data) - 1)
        for i in range(len(y_data)):
            if i <= beginning or i >= finish:
                ind_out1.append(indx_bag1[i])
            else:
                ind_out1.append(indx_bag2[i])
        return ind_out1

    def verify_bag(self, ind_out):

        classes = collections.Counter(self.y_train)
        _, y = self.build_bags(ind_out)
        counter = collections.Counter(y)
        if len(counter.values()) == len(classes) and min(counter.values()) >= 2:
            return True
        else:
            return False

    def mutation(self, ind):
        ind_out = []
        indx = self.bags["name"].index(ind[0])
        indx_bag1 = self.bags["inst"][indx]
        X, y_data = self.build_bags(indx_bag1)
        inst = 0
        inst2 = len(y_data)

        if self.generation == 0 and self.off == []:
            ind2 = random.randint(0, 99)
        else:
            ind2 = random.sample(self.off, 1)
            ind2 = ind2[0]
        indx2 = self.bags["name"].index(ind2)
        indx_bag2 = self.bags["inst"][indx2]
        X2, y2_data = self.build_bags(indx_bag2)

        while y_data[inst] != y2_data[inst2 - 1]:
            inst = random.randint(0, len(y_data) - 1)
        for i in range(len(indx_bag1)):
            if i == inst:
                ind_out.append(indx_bag2[i])
            else:
                ind_out.append(indx_bag1[i])

        self.bags["name"].append(self.name_individual)
        self.bags["inst"].append(ind_out)
        ind[0] = self.name_individual
        self.name_individual += 1
        if self.method_disperse == True:
            self.cont_crossover = self.cont_crossover + 1
            if self.cont_crossover == self.nr_individual + 1:
                self.cont_crossover = 1
                self.get_complexity(first_evaluate=False, population=None)

        return (ind,)

    def the_function(self, population, gen, fitness):

        generation = gen
        self.off = []
        bags_ant = self.bags
        bags = dict()
        bags["name"] = list()
        bags["inst"] = list()
        for j in population:
            # print(bags_ant)
            indx = bags_ant["name"].index(j[0])
            bags["name"].append(bags_ant["name"][indx])
            bags["inst"].append(bags_ant["inst"][indx])
        del bags_ant
        for i in range(len(population)):
            self.off.append(population[i][0])
        if self.stop_criteria == "maxdistance":
            self.max_distance(
                fitness, generation=self.generation, population=self.off, bags=bags
            )
        elif self.stop_criteria == "maxacc":
            self.max_acc(
                self.dist["score_g"],
                generation=self.generation,
                population=self.off,
                bags=bags,
            )
        if generation == self.nr_generation:
            if self.stop_criteria == "maxdistance" or self.stop_criteria == "maxacc":
                self.save_bags(
                    self.pop_temp, self.bags_temp, self.gen_temp, self.base_name
                )
            else:
                self.save_bags(self.off, bags, base_name=self.base_name)
        if self.method_disperse == True and generation != self.nr_generation:
            self.get_complexity(population=population)
        return population

    def save_bags(
        self,
        pop_temp,
        bags_temp,
        gen_temp=None,
        base_name=None,
        type=0,
        generations_escolhida="x",
    ):

        if type == 0:

            for j in pop_temp:
                name = []
                indx = self.bags["name"].index(j)
                nm = self.bags["inst"][indx]
                name.append(self.bags["name"][indx])
                name.extend(nm)
                self.bags_saved.append(name)
                # Cpx.save_bag(name, 'bags', self.local + "/Bags", base_name + self.file_out, self.iteration)

        elif type == 1:
            x = open(generations_escolhida, "a")
            x.write(base_name + ";" + str(gen_temp) + "\n")
            x.close()
            for j in pop_temp:
                name = []
                indx = bags_temp["name"].index(str(j))
                nm = bags_temp["inst"][indx]
                name.append(bags_temp["name"][indx])
                name.extend(nm)
                # if self.classifier=="perc":
                #     Cpx.save_bag(name, 'bags', self.local + "/Bags", base_name + self.file_out, self.iteration)
                # elif self.classifier=="tree":
                #     Cpx.save_bag(name, 'bags', self.local + "/tree/Bags", base_name + self.file_out, self.iteration)

        elif type == 2:
            x = open(generations_escolhida, "a")
            x.write(base_name + ";" + str(gen_temp) + "\n")
            x.close()
            for j in pop_temp:
                name = []
                indx = bags_temp["name"].index(str(j))
                nm = bags_temp["inst"][indx]
                name.append(bags_temp["name"][indx])
                name.extend(nm)
                # Cpx.save_bag(name, 'bags', self.local + "tree/Bags", base_name + self.file_out, self.iteration)

    def max_distance(self, fitness, generation=None, population=None, bags=None):
        """
        :param fit1: fittnes 1
        :param fit2:
        :param fit3:
        ideal para distancia linear
        """
        if fitness[2]:
            dist_dist_media = np.mean(
                Cpx.dispersion(np.column_stack([fitness[0], fitness[1], fitness[2]]))
            )
        else:
            dist_dist_media = np.mean(
                Cpx.dispersion(np.column_stack([fitness[0], fitness[1]]))
            )
        if dist_dist_media > self.dist_temp:
            self.dist_temp = dist_dist_media
            self.pop_temp = population
            self.gen_temp = generation
            self.bags_temp = bags

    def max_acc(self, acc, generation=None, population=None, bags=None):
        """
        :param fit1: fittnes 1
        :param fit2:
        :param fit3:
        :param population: popoluacao atual geralmente o off
        :param bags: bags atuais
        :return:
        ideal para distancia linear
        """

        if acc > self.acc_temp:
            self.acc_temp = acc
            self.pop_temp = population
            self.gen_temp = generation
            self.bags_temp = bags

    def complexities(self, X_train, y_train,grupos):
        
        _ ,X_bag ,_ ,y_bag = train_test_split(X_train, y_train, test_size=self.tam_bags)
        cpx = Cpx.complexity_data3(X_bag, y_bag,grupos)
        return cpx
    

        
    def vote_complexity(self, X_data,y_data,grupos):
        voto = [0] * 11
        for i in range(1,12):
            
            stad = []
            comp = []
            cp=[]
            
            for  j in range(100):
                cp.append(self.complexities(X_data, y_data, grupos))
            # print("passou cpx")
            # np.set_printoptions(threshold=np.nan)
            comp.append(cp)
            comp=np.array(comp)
            cpx = np.squeeze(comp)
            cpx = cpx.T
            for k in cpx:
                # print(k)
                norm = Cpx.min_max_norm(k)
                
                std = np.std(norm)
                # print(std)
                std = std.tolist()
                # print(std)
                stad.append(std)
                # exit(0)

            max = np.argsort(stad)
            stad = np.array(stad)
            max = max[::-1]
            del cpx
            overlapping = stad[0:5]
            neighborhood = stad[5:11]

            o = np.argmax(overlapping)
            nei = np.argmax(neighborhood)

            voto[o] = voto[o] + 1
            voto[nei + 5] = voto[nei + 5] + 1

            text = ''
            for carro, cor in zip(Cpx.header, voto):
                text += '{} {}, '.format(carro, cor)
            # print(text)
            # print("\n", voto)
        return voto, text, max, stad


    def get_best_types(self, X_train, y_train, group, n=2):
        print("votting complex...")
        res = self.vote_complexity(X_train, y_train, group)
        votes = res[0]
        ordened = np.argsort(votes)
        types = []
        for i in range(1,n+1):
            feature = Cpx.header[ordened[-i]]
            types.append(feature.split('.')[1])
        
        return types


    def generate(self, X_train, y_train, X_val, y_val, iteration=20):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.iteration = iteration
        
        if self.types is None:
            self.types = self.get_best_types( X_train, y_train, self.group)


        for t in range(0, self.iteration):
            print("Interation - ", t)

            self.name_individual = 100
            self.off = []
            self.seq = -1
            self.repetition = t

            self.generation = 0
            self.bags = self.generate_bags(
                self.X_train, self.y_train
            )
            creator.create(
                "FitnessMult",
                base.Fitness,
                weights=(self.fit_value1, self.fit_value2, self.fit_value3),
            )
            creator.create("Individual", list, fitness=creator.FitnessMult)
            toolbox = base.Toolbox()
            toolbox.register("attr_item", self.sequencia)
            toolbox.register(
                "individual", tools.initRepeat, creator.Individual, toolbox.attr_item, 1
            )
            population = toolbox.register(
                "population", tools.initRepeat, list, toolbox.individual
            )
            self.pop = toolbox.population(n=self.nr_pop)
            if self.method_disperse == True:
                self.get_complexity(first_evaluate=True)

            toolbox.register("evaluate", self.evaluate_linear_dispersion)
            toolbox.register("mate", self.crossover)
            toolbox.register("mutate", self.mutation)
            toolbox.register("select", tools.selNSGA2)
            self.pop = algorithms.eaMuPlusLambda(
                self.pop,
                toolbox,
                self.nr_child,
                self.nr_individual,
                self.proba_crossover,
                self.proba_mutation,
                self.nr_generation,
                generation_function=self.the_function,
            )

    def get_bags(self):
        bags = []
        for bag in self.bags_saved:
            bags.append(self.build_bags(bag[1:]))
        return bags

    def get_pool(self):

        bags = self.get_bags()
        pool = []

        if self.classifier == "tree":

            for bag in bags:

                tree = DecisionTreeClassifier()
                X_bag =bag[0]
                y_bag = bag[1]
                
                pool.append(tree.fit(X_bag, y_bag))
        else:
            for bag in bags:
                percP = Perceptron(tol=1.0)
                X_bag =bag[0]
                y_bag = bag[1]
                
                pool.append(percP.fit(X_bag, y_bag))
        self.pool_classificators = pool
        return pool