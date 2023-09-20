from sklearn.linear_model import Perceptron
from deslib.util import diversity

from sklearn.metrics import pairwise_distances

from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.tree import DecisionTreeClassifier


import numpy as np
import csv, os

import pandas as pd
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects.packages as rpackages

ecol = rpackages.importr('ECoL')
import rpy2.robjects as robjects

header=['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4', 'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1', 'neighborhood.LSC', 'linearity.L1', 'linearity.L2', 'linearity.L3', 'dimensionality.T2', 'dimensionality.T3', 'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs']

def dispersion_linear(complexity):
    # print(complexity)

    """
    :param complexity: listas de complexidades
    :return: media das distancias feitas de forma manual a-b normalizadas
    """
    result = []
    result1 = []

    complexity = list(complexity)
    complexity = np.array(complexity)
    # print((complexity))
    n = (len(complexity)) - 1
    complexity = complexity.T

    # exit(0)
    for i in complexity:
        dist = []
        for j in range(len(i)):
            dista = 0
            for l in range(len(i)):
                if (j == l):
                    continue
                else:
                    dista += abs(i[j] - i[l])
            dist.append((dista) / n)
        result.append(dist)
    result = np.array(result)
    # print(result)
    for i in result:
        r = min_max_norm(i)

        result1.append(r)
    result1 = np.array(result1)
    del result, r, dist, complexity
    result1 = result1.T
    result1 = result1.tolist()
    #print(result1)

    return result1

    
def diversitys(y_test, predicts):
    q_test = []
    double_faults = []
    for i in range(len(predicts)):
        db = []
        for j in range(len(predicts)):
            if i == j:
                continue
            else:
                #  q.append(diversity.Q_statistic(y_test,predicts[i],predicts[j]))
                db.append(diversity.double_fault(y_test, predicts[i], predicts[j]))
                #coloquei um paramentro novo na funcao de retorno _process_predictions
        double_faults.append(np.mean(db))

    return double_faults

def min_max_norm(dataset):
    """
    :param dataset: dataset [samples,features]
    :return: dataset normalizado por minmax
    """

    norm_list = list()
    min_value = np.min(dataset)
    max_value = np.max(dataset)

    if min_value == max_value:
        # print("entrei")
        for i in dataset:
            norm_list.append(0)
        return norm_list

    for value in dataset:
        tmp = (value - min_value) / (max_value - min_value)
        norm_list.append(tmp)

    return norm_list

def voting_classifier(pool, X_val, y_val):
    #voting = EnsembleVoteClassifier(clfs=pool, voting='hard', refit=False)
    voting = EnsembleVoteClassifier(clfs=pool, voting='hard')

    voting.fit(X_val, y_val)
    result = voting.score(X_val, y_val)
    return result

def complexity_data3(X_data, y_data, group, types=None):
    
    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    complex = np.array([])
    for i in range(0, len(group)):
        if types:
            measures = types[i]
        else:
            measures= "all"
        
        if group[i] == 'overlapping':
            over = ecol.overlapping(dfx, dfy, measures=measures,summary='mean')
            
            over = np.asarray(over)
            
            complex = np.append(complex, over[:,0])
            
        if group[i] == "neighborhood":
            nei = ecol.neighborhood(dfx, dfy, measures=measures,summary='mean')
            nei = np.asarray(nei)
            complex = np.append(complex, nei[:,0])
        if group[i] == "linearity":
            line = ecol.linearity(dfx, dfy, measures=measures,summary='mean')
            line = np.asarray(line)
            complex = np.append(complex, line[:,0])
        if group[i] == "dimensionality":
            dim = ecol.dimensionality(dfx, dfy, measures=measures,summary='mean')
            dim = np.asarray(dim)
            complex = np.append(complex, dim[:,0])
        if group[i] == "balance":
            bal = ecol.balance(dfx, dfy, measures=measures,summary='mean')
            bal = np.asarray(bal)
            complex = np.append(complex, bal[:,0])
        if group[i] == "network":
            net = ecol.network(dfx, dfy, measures=measures,summary='mean')
            net = np.asarray(net)
            complex = np.append(complex, net[:,0])
    complex = complex.tolist()
    del dfx, dfy
    return complex

def biuld_classifier_tree(X_train, y_train, X_val, y_val, X_test=None, y_test=None, score_train=False):
    '''
    se score_train false, retorna o score sobre o proprio treino, e a predicao sobre a validacao, caso contrario,
    retorna duas acuracias, treino e validacao mais a predicao sobre a validacao
    retorna um perceptron com sua acuracia e com a lista de predicao
    :param X_train: X do treino
    :param y_train: y do treino
    :param X_val: X valida ou teste
    :param y_val: y valida, ou teste
    :param X_test: retorna o predict e o segundo score
    :return: classificador, accuracia, lista de predicao
    '''
    # constroi os classificadores, e retorna classificador, score e predict
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    score = tree.score(X_val, y_val)
    if X_test.all()!=None and y_test.all()!=None and score_train==False:

        predict = tree.predict(X_test)

        return tree, score, predict

    elif (score_train):
        score2 = tree.score(X_test, y_test)
        predict = tree.predict(X_test)
        return tree, score, score2, predict

    else:

        return tree, score

def biuld_classifier(X_train, y_train, X_val, y_val, X_test=None, y_test=None, score_train=False):
    '''
    se score_train false, retorna o score sobre o proprio treino, e a predicao sobre a validacao, caso contrario,
    retorna duas acuracias, treino e validacao mais a predicao sobre a validacao
    retorna um perceptron com sua acuracia e com a lista de predicao
    :param X_train: X do treino
    :param y_train: y do treino
    :param X_val: X valida ou teste
    :param y_val: y valida, ou teste
    :param X_test: retorna o predict e o segundo score
    :return: classificador, accuracia, lista de predicao
    '''
    # constroi os classificadores, e retorna classificador, score e predict
    perc = Perceptron(n_jobs=4, max_iter=100, tol=1.0)
    perc.fit(X_train, y_train)
    score = perc.score(X_val, y_val)
    if X_test!=None and y_test!=None and score_train==False:

        predict = perc.predict(X_test)

        return perc, score, predict

    elif (score_train):
        score2 = perc.score(X_test, y_test)
        predict = perc.predict(X_test)
        return perc, score, score2, predict

    else:

        return perc, score


def save_bag(inds, types, local, base_name, iteration):
    if types == 'validation':
        # print('entreivali')
        if (os.path.exists(local + "Validacao/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration)+ "/" + base_name + ".csv")
        with open(base_name + ".csv", 'w') as f:
            # print('entreivali')
            w = csv.writer(f)
            w.writerow(inds)

    if types == "test":
        if (os.path.exists(local + "Teste/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration)+ "/" + base_name + ".csv")
        with open(base_name + ".csv", 'w') as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "train":
        if (os.path.exists(local + "Treino/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration)+ "/" + base_name + ".csv")
        with open(base_name + ".csv", 'w') as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "bags":
        if (os.path.exists(local + "Bags/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration)+ "/" + base_name + ".csv")
        with open(base_name + ".csv", 'a') as f:
            w = csv.writer(f)
            w.writerow(inds)


def dispersion(complexity):
    """
    :param complexity: listtas de complexidades
    :return: distancia media par a par das complexidades(biblioteca)
    """
    result = []
    dista = pairwise_distances(complexity, n_jobs=6)
    dista = dista.tolist()

    for i in dista:
        result.append(np.mean(i))
    return result

