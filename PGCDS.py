import collections
import numpy as np
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from joblib import  Parallel, delayed
import Util

def distance(first=False, population=None):
    global pop, name_individual, bags, group, types
    dist = dict()
    dist['name'] = list()
    dist['dist'] = list()
    dist['diver'] = list()
    dist['score'] = list()
    dist['score_g'] = list()
    if first == True and generation == 0:
        dist['name'] = pop
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance)(i) for i in range(len(dist['name'])))
        c, score,  pred, pool = zip(*r)
    elif first == False and population == None:
        start_ = name_individual - nr_individual
        for i in range(start_, name_individual):
            x = []
            x.append(i)
            dist['name'].append(x)
        r = Parallel(n_jobs=jobs)(
            delayed(parallel_distance)(j) for j in range(100, nr_individual + 100))
        c, score, pred, pool = zip(*r)
    elif population != None:
        dist['name'] = population
        indices = []
        for i in population:
            indices.append(bags['name'].index(str(i[0])))
        r = Parallel(n_jobs=jobs)(delayed(parallel_distance)(i) for i in indices)
        c, score, pred, pool = zip(*r)
    dist['dist'] = Util.dispersion_line(c)
    dist['score'] = score
    d = diversity(pred, y_vali)
    dist['diver'] = Util.min_max_norm(d)
    dist['score_g']=Util.voting_classifier(pool, X_vali, y_vali)
    return

def diversity(pred, y):
    """
    Calculate diversity measure
    @param pred: prediction of classifiers
    @param y: original labels
    @return:
    """
    pred=np.array(pred)
    d =Util.diversitys(y, pred)
    return d

def parallel_distance(i):
    """
          @param i: list of indices of the bag to be tested
          :return: complexity lists, bag score, validation score, and validation prediction
      """
    global classifier, bags, group, types

    pred_=[]
    indx_bag1 = bags['inst'][i]
    X_bag, y_bag = build_original_bag(indx_bag1)
    cpx = (Util.complexity_data(X_bag, y_bag, group, types))

    if classifier=="perceptron":
        classifier, score_, pred_ = Util.biuld_classifier(X_bag, y_bag, X_bag, y_bag, X_vali, y_vali)
    elif classifier=="tree":
        classifier, score_, pred_ = Util.biuld_classifier_tree(X_bag, y_bag, X_bag, y_bag, X_vali, y_vali)

    return cpx,  score_,  pred_, classifier

def build_original_bag(indx_bag):
    """
    Receives the index of instances of a bag
    @param indx_bag:
    @return: X_data, y_data
    """
    global X, y
    X_data = []
    y_data = []
    for i in indx_bag:
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    return X_data, y_data

def cross(ind1, ind2):
    """
    Crossover two individuals
    @param ind1: individual
    @param ind2: individual
    @return: new individual
    """
    global name_individual, cont_crossover, nr_individual, bags, dispersion

    individual = False
    indx = bags['name'].index(str(ind1[0]))
    indx2 = bags['name'].index(str(ind2[0]))
    indx_bag1 = bags['inst'][indx]
    indx_bag2 = bags['inst'][indx2]
    _, y_data = build_original_bag(indx_bag1)
    cont = 0

    while individual != True:

        ind_out1 = short_cross(y_data, indx_bag1, indx_bag2)
        individual = check_data(ind_out1)
        cont = cont + 1
        if cont == 30:
            print("error")
            exit(0)

    ind1[0] = name_individual
    ind2[0] = name_individual

    bags['name'].append(str(name_individual))
    bags['inst'].append(ind_out1)
    name_individual += 1

    if dispersion:
        cont_crossover = cont_crossover + 1

        if cont_crossover == nr_individual + 1:
            cont_crossover = 1
            distance(first=False, population=None)

    return creator.Individual(ind1), creator.Individual(ind2)

def short_cross(y_data, indx_bag1, indx_bag2):
    """
    complementary function for crossover
    @param y_data: labels
    @param indx_bag1: indices of bag
    @param indx_bag2: indices of bag
    @return:
    """
    start_ = fim = 0
    ind_out1 = []
    while y_data[start_] == y_data[fim]:
        start_ = random.randint(0, len(y_data) - 1)
        fim = random.randint(start_, len(y_data) - 1)
    for i in range(len(y_data)):
        if i <= start_ or i >= fim:
            ind_out1.append(indx_bag1[i])
        else:
            ind_out1.append(indx_bag2[i])
    return ind_out1

def check_data(ind_out):
    """
    Checks if the number of classes for each label is greater than 2
    @param ind_out:
    @return: true or false
    """
    global classes, y, X
    _, y = build_original_bag(ind_out)
    counter = collections.Counter(y)
    if len(counter.values()) == len(classes) and min(counter.values()) >= 2:
        return True
    else:
        return False

def mutate(ind):
    """
    Mutate a individual
    @param ind: individual
    @return: new individual
    """
    global off, name_individual, cont_crossover, nr_individual, bags, dispersion, X
    print("mutate")
    ind_out = []
    indx = bags['name'].index(str(ind[0]))
    indx_bag1 = bags['inst'][indx]
    X, y_data = build_original_bag(indx_bag1)
    inst = 0
    inst2 = len(y_data)
    if generation == 0 and off == []:
        ind2 = random.randint(0, 99)
    else:
        ind2 = random.sample(off, 1)
        ind2 = ind2[0]
    indx2 = bags['name'].index(str(ind2))
    indx_bag2 = bags['inst'][indx2]
    X2, y2_data = build_original_bag(indx_bag2)

    while y_data[inst] != y2_data[inst2 - 1]:
        inst = random.randint(0, len(y_data) - 1)

    for i in range(len(indx_bag1)):
        if i == inst:
            ind_out.append(indx_bag2[i])
        else:
            ind_out.append(indx_bag1[i])

    bags['name'].append(str(name_individual))
    bags['inst'].append(ind_out)

    ind[0] = name_individual
    name_individual += 1
    if dispersion:
        cont_crossover = cont_crossover + 1
        if cont_crossover == nr_individual + 1:
            cont_crossover = 1
            distance(first=False, population=None)

    return ind,


def fitness_dispercion_line(ind1):
    """
    evaluate fitness
    @param ind1: individual
    @return: dist between
    """
    global dist
    dist1 = None
    dist2 = None
    diver = None
    score = None
    for i in range(len(dist['name'])):
        if dist['name'][i][0] == ind1[0]:
            dist1 = dist['dist'][i][0]
            dist2 = dist['dist'][i][1]
            ###########################
            score = dist["score"][i]
            diver=dist['diver'][i]
            break
    ###############################
    return dist1, dist2, diver, score,

def continus_():
    global seq
    seq += 1
    return seq

def the_function(population, gen, fitness):

    """
    Callback of eaMuPlusLambda DEAP library. Responsible for changing the generation, as well as resetting variables,
    changing populations, and copying files
    @param population:
    @param gen:
    @param fitness:
    @return:
    """
    global generation, off, dispersion, nr_generation, bags, \
        dist_temp, generations_final, base_name, fitness_temp

    generation = gen

    off = []
    base_name = base_name + str(generation)
    bags_ant = bags
    bags = dict()
    bags['name'] = list()
    bags['inst'] = list()
    for j in population:
        indx = bags_ant['name'].index(str(j[0]))
        bags['name'].append(bags_ant['name'][indx])
        bags['inst'].append(bags_ant['inst'][indx])
    del bags_ant
    for i in range(len(population)):
        off.append(population[i][0])
    if stop=="maxdistance":
        max_distance(fitness, generation=generation, population=off, bags=bags)
    elif stop =="maxacc":
        max_acc(fitness, dist['score_g'], generation=generation, population=off, bags=bags)
    if generation == nr_generation:
        if stop=="maxdistance":
            save_bags( gen_temp, base_name, type=1, generations_final=generations_final)
        elif stop=="maxacc":
            save_bags( gen_temp, base_name, type=2, generations_final=generations_final)
        else:
            save_bags(off,bags,base_name=base_name,type=0)
    if dispersion == True and generation != nr_generation:
        distance(population=population)
    return population

def save_bags(gen_temp=None, base_name=None, type=0,generations_final="x"):
    '''

    :param gen_temp: current generations, or the chosen generations (best generations) this adds up to the final file name
    :param base_name: in this case the number of generations next to the base name (this adds up to the output file (final file name))
    :@param name_arq_generations: the name of the file that writes the chosen generation REQUIRED in types 1 and 2
    :param type: first type (0) the traditional final population, type (1) the distance average population, type (2) the global accuracy population
    :return:
    '''
    global file_out, iteration, local, pop_temp, bags_temp
    if type==0:
        for j in pop_temp:
            name = []
            indx = bags['name'].index(str(j))
            nm = bags['inst'][indx]
            name.append(bags['name'][indx])
            name.extend(nm)
            Util.save_bag(name, 'bags', local + "/Bags", base_name + file_out, iteration)
    elif type==1:
        x = open(generations_final, "a")
        x.write(base_name + ";" + str(gen_temp) + "\n")
        x.close()
        for j in pop_temp:
            name = []
            indx = bags_temp['name'].index(str(j))
            nm = bags_temp['inst'][indx]
            name.append(bags_temp['name'][indx])
            name.extend(nm)
            if classifier=="perc":
                Util.save_bag(name, 'bags', local + "/Bags", base_name + file_out, iteration)
            elif classifier=="tree":
                Util.save_bag(name, 'bags', local + "/tree/Bags", base_name + file_out, iteration)

    elif type==2:
        x = open(generations_final, "a")
        x.write(base_name + ";" + str(gen_temp) + "\n")
        x.close()
        for j in pop_temp:
            name = []
            indx = bags_temp['name'].index(str(j))
            nm = bags_temp['inst'][indx]
            name.append(bags_temp['name'][indx])
            name.extend(nm)
            Util.save_bag(name, 'bags', local + "tree/Bags", base_name + file_out, iteration)

def max_distance(fitness, generation=None, population=None, bags=None):
    """
    Max distance between individuals, compare the 3 fitness result
    @param fitness:
    @param generation:
    @param population:
    @param bags:
    @return:
    """
    global dist_temp, fitness_temp, bags_temp,pop_temp
    fitness_temp[0] = fitness[0]
    fitness_temp[1] = fitness[1]
    if fitness[2]:
        fitness_temp[2] = fitness[2]
        dist_dist_average = np.mean(Util.dispersion(np.column_stack([fitness[0], fitness[1], fitness[2]])))

    else:
        dist_dist_average = np.mean(Util.dispersion(np.column_stack([fitness[0], fitness[1]])))
    if dist_dist_average > dist_temp:
        dist_temp = dist_dist_average
        pop_temp = population
        gen_temp = generation
        bags_temp = bags

def max_acc(fitness, acc,generation=None, population=None, bags=None):
    """
    Find the max accuracy in the generation (especific version)
    @param fitness:
    @param acc:
    @param generation:
    @param population:
    @param bags:
    @return:
    """
    global pop_temp, gen_temp, bags_temp, acc_temp, fitness_temp
    print(fitness[0])
    fitness_temp[0] = fitness[0]
    fitness_temp[1] = fitness[1]
    if fitness[2]:
        fitness_temp[2] = fitness[2]
        if acc > acc_temp:
            acc_temp = acc
            pop_temp = population
            gen_temp = generation
            bags_temp = bags

tem2=[]
acc_temp=0
group = ["overlapping", 'neighborhood', '', '', '', '']
types = ["F3", 'N3', '', '', '', '']
dispersion = True
#fit_values
fit_value1 = 1.0
fit_value2 = 1.0
fit_value3 = -1.0
#number of generation
nr_generation = 19
#number of individuals in generations
nr_individual = 100
#population size
nr_pop=100
#probability crossover and mutation
proba_crossover = 0.99
proba_mutation = 0.01
#number of children
nr_child=100
#counter corssover
cont_crossover = 1
#number of iteration


dist_temp=0
#number of jobs
jobs = 7
#type method "maxacc maxdistance"
stop="maxacc"
#classifieres "tree" or "perc"
classifier="tree"#tree,perc
#name do arquivo de generation (qual generations foi escolhida)
generations_final="generations_final"
#name do arquivo com os bags finais
#file_out = "graficos_tese"+base_name
fitness_=fitness_dispercion_line
#variavel para salvar o fitnees
fitness_temp=[]
X=#original instances
y=#original labels
X_vali, y_vali=#validationset
X_train, y_train=#trainset
bags=#indices of instaces for each bag
iteration=21
classes=#lables [0,1,2,3...]
local="c:/"#where the final bags are saving
for t in range(1, iteration):

    off = []
    name_individual = 100
    iteration = t
    generation = 0
    seq = -1

    #type de fitness e os fit values
    creator.create("FitnessMult", base.Fitness, weights=(fit_value1, fit_value2, fit_value3))#, fit_value2, fit_value3))#verificar
    creator.create("Individual", list, fitness=creator.FitnessMult)
    toolbox = base.Toolbox()
    toolbox.register("attr_item", continus_)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                      toolbox.attr_item, 1)
    population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=nr_pop)
    if dispersion:
         distance(first=True)
    #funcao de fitness: fitness_andre - DSOC, fitness_dispercion_diver _ diversity e distance, dispercion_line - dist dist diver
    toolbox.register("evaluate", fitness_)
    toolbox.register("mate", cross)
    toolbox.register("mutate", mutate)
    #funcao de selecao
    toolbox.register("select", tools.selNSGA2)

    #loop GA
    pop = algorithms.eaMuPlusLambda(pop, toolbox, nr_child, nr_individual, proba_crossover, proba_mutation,
                                          nr_generation,
                                             generation_function=the_function)
