#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
random.seed(10)
import time
import datetime
import multiprocessing
from multiprocessing import Lock
import logging
import logging.handlers
import numpy as np
import pandas as pd
import hvplot.pandas

from deap import base
from deap import creator
from deap import tools, algorithms
import pickle

import os

import config
from evaluationEP import evaluation

from path import Path

import logzero
from logzero import logger
logzero.loglevel(logging.DEBUG)
logzero.logfile("./activity.log", maxBytes=2e6, backupCount=30)
logger.propagate = False

#     Problem definition
NPARAM = config.NPARAM
NDIM = config.IND

BOUNDS = config.BOUNDS

toolbox = base.Toolbox()


def init_indp(icls, ranges, genome=list()):
    """
    Initialization of individuals: this function initializes with integer the
    individuals with respect to the bounds given in input

    Args:
        icls (creator): class created for Individuals
        ranges (list): Bounds for individuals

    Returns:
        icls (creator): Individuals created initialized with random genome

    """
    genome = list()
    if genome == list():
        logger.debug(ranges)
        nparam = len(ranges)
        for p in ranges[0:nparam-1]:
            genome.append(random.randrange(*p,5))
        genome.append(np.random.randint(*ranges[nparam-1]))#origine random.randint
    return icls(genome)


def init_opti():
    """
    creation of the toolboxes objects that defines
    - the type of population and the creator of individuals
    - the difference (or ot) in the initialization of individuals
    - the evaluation function
    - the crossing operator
    - the mutation operator
    - the algorithm used to select best individuals

    Args:
        None

    """
    # Creation des objects liés à l'optimisation
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0)) # weights negative ==> minimize 
                                                                        # Creates a new class named "FitnessMin" inheriting from "base.Fitness" with attrebute "weights=(-1.0,)"
                                                                        # The fitness is a measure of quality of a solution.
    creator.create("Individual",
                   list,
                   typecode="d",
                   fitness=creator.FitnessMin)

    # toolbox.register("attr_int", random.randint, BOUND_LOW, UP_WALLS)
    toolbox.register("individual", init_indp, icls=creator.Individual,
                     ranges=BOUNDS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)#deap.tools.initRepeat(container, func, n) 
                                                                            #Call the function func n times and return the results in a container type container

    toolbox.register("evaluate", evaluation)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("matep", tools.cxUniformPartialyMatched, indpb=0)
    toolbox.register("mutate", tools.mutUniformInt, low=[x[0] for x in BOUNDS],
                     up=[x[1] for x in BOUNDS], indpb=1 / NPARAM)
    toolbox.register("mutatep", tools.mutShuffleIndexes, indpb=1 / NPARAM)

    toolbox.register("select", tools.selNSGA2)



def parameters(GEN=config.NGEN, NBIND=config.IND, CX=config.CX, MX=config.MX):
    """
    Parameters definition for a genetic algorithm. That can apply to most of
    evolutionary optimization algorithm.

    Args:
        GEN (int): Number of generation of GA
        NBIND (int): Number of Individual of GA
        CX (float): crossing probability for each individual
    """
    logger.info("NGEN: " + str(GEN) + ", Individuals: " + str(NBIND))

    return GEN, NBIND, CX, MX


def main():
    """
    Everything else to launch the optimization.

    ToDo:
        Refactor the stat modules
        Refactor the selection process in a function

    """
    NGEN, MU, CXPB, MXPB = parameters()

    pareto = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    #initialisation LB
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    graph = []
    data = []
    df = []
    logger.debug(pop)

    logger.debug("Evaluate the individuals with an invalid fitness")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    logger.debug("Evaluation finished")

    data.append([ind.fitness.values for ind in pop])

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        graph.append(ind.fitness.values)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    #update LB
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), pop=list(pop), halloffame=np.array([(ind.fitness.values, ind) for ind in pareto]),pareto=pareto,**record)
    logger.info(logbook.stream)

    logger.debug("Begin the generational process")
    for gen in range(1, NGEN):
        logger.debug("Generation " + str(gen) + " out of " + str(NGEN))
        logger.debug("Vary the population")
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values
        for mutant in offspring:
            if random.random() < MXPB:
                logger.debug("mutation start")
                toolbox.mutate(mutant)
                del mutant.fitness.values


        logger.debug(pop)

        logger.debug("Evaluate the individuals with an invalid fitness")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        logger.debug("Evaluation finished")

        data.extend([ind.fitness.values for ind in pop])
        config.indiv.extend([ind for ind in pop])

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            graph.append(ind.fitness.values)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        #update LB
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind),pop=list(pop), halloffame=np.array([(ind.fitness.values, ind) for ind in pareto]),pareto=pareto, **record)
        logger.info(logbook)

        pareto.update(pop)

        hypervolume, wobj = calcul_hypervolume(pareto)
        df.append(hypervolume)
        
        #sauvegarde LB
        fname = './monitoring/opti_log.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(logbook, f)
        
        # Dump population into csv file
        monit = pd.DataFrame([dict(vals=pop, fitness=pop.fitness.values)
                          for pop in pop])
        with open("monitoring.csv", "a") as f:
            monit.to_csv(f, header=False)
        logger.debug("population saved")

        save_pareto_csv(gen, wobj, pareto)

    plot_hypervolume (df)
    return pop, logbook, pareto, graph, data


def calcul_hypervolume(pareto):
    """
    Plots the hypervolume of the population to monitor the optimization

    Args:
        pareto (pareto): contains population of Pareto front
        df: population

    Returns:

    """
    logger.debug("calcul hypervolume")
    try:
        # try importing the C version
        from deap.tools._hypervolume import hv
    except ImportError:
        # fallback on python version
        from deap.tools._hypervolume import pyhv as hv

    # Must use wvalues * -1 since hypervolume use implicit minimization
    # And minimization in deap use max on -obj
    wobj = np.array([ind.fitness.wvalues for ind in pareto]) * -1 #origine wvalues
    ref = np.max(wobj, axis=0) * 1.5
    
    hypervolume=hv.hypervolume(wobj, ref)

    logger.debug("calculation of HV done")

    return hypervolume, wobj
 
def plot_hypervolume (df):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.ioff()
        xaxis = [i for i in range(len(df))]
        plt.scatter(xaxis, df, c="b", marker="v")
        plt.ylabel("Hypervolume de la population")
        plt.xlabel("Génération")
        plt.title("Convergence de l\"optimisation")
        plt.savefig("monitoring_hypervolume.png")
        logger.debug("plot done")
    except Exception as e:
        logger.error("Monitoring plot error")
        logger.exception(e)

def write_pareto(pareto, everyindiv, data):
    """
    Write pareto front in txt file to enable further exploitation

    Args:
        pareto (pareto): pareto object from DEAP containing best individuals

    Returns:
        pareto.items (list) : list of fitnesses of pareto individuals
    """
    from datetime import date
    logger.debug("writing results in files")
    s=str(datetime.datetime.now())
    s_replace_point=s.replace(".","")
    s_replace=s_replace_point.replace(":","")[:17]
    #s=date.today()
    with open("./results/pareto_decision_parametres" + s_replace + ".txt", "w") as front:
        for line in pareto.items:
            front.write(str(line) + "\n")
    fct_objectif = pd.DataFrame([dict(vals=ind, fitness=ind.fitness.values)
                          for ind in pareto])
    with open("fct_objectif.csv", "a") as f:
        fct_objectif.to_csv(f, header=False)
    with open("./results/data" + s_replace + ".txt", "w") as front:
        for ind in data:
            front.write(str(ind) + "\n")
    pareto_fitnesses=[]
    with open("./results/pareto_fitnesses" + s_replace + ".txt", "w") as resultats:
        for ind in pareto:
            pareto_fitnesses.append(ind.fitness)
            resultats.write(str(ind.fitness) + "\n")

    with open("./results/graph_data" + s_replace + ".txt", "w") as every:
        for ind in everyindiv:
            every.write(str(ind) + "\n")

    return pareto_fitnesses


def save_pareto_csv(gen, wobj, pareto):
    # Dump pareto into csv file
    with open("./monitoring/pareto_obj_gen"+str(gen)+".csv", "w") as f:
        np.savetxt(f, wobj, delimiter=",")
    with open("./monitoring/pareto_param_gen"+str(gen)+".csv", "w") as f:
        np.savetxt(f, pareto.items, delimiter=",")
    logger.debug("pareto saved")

def plots(population, plot2D = True, plot3D = False, interactive = False): #optimal_indiv, data
    """
    Plots the summary of optimization
    Saves all plots in files

    Args:
        all the data provided by DEAP optimization

    Returns:
        None

    """

    logger.debug("plotting")
    import matplotlib.pyplot as plt
    Pareto_decision_parametres= np.array([ind for ind in population])
    Pareto_objective_functions = np.array([ind.fitness.values for ind in population])
    plt.figure()
    if plot2D == True : 
        """Only works for problems with 2 functions to optimise"""
        plt.scatter(Pareto_objective_functions[:, 0], Pareto_objective_functions[:, 2], c=Pareto_objective_functions[:, 1])
        plt.xlabel("besoins de chauffage kWh")
        plt.ylabel('Cout global actualisé en euros')
        plt.title('Front de Pareto')
        plt.colorbar(label="heures d'inconfort (T>Tconf+2°C)")
        plt.savefig('Front de Pareto_objectifs.png')
        plt.clf()
    if plot3D == True:
        """Only works for problems with 3 functions to optimise"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Pareto_objective_functions[:, 0], Pareto_objective_functions[:, 1], Pareto_objective_functions[:, 2])
        ax.xlabel("f1")
        ax.ylabel('f2')
        ax.zlabel('f3')
        ax.title('Front de Pareto')
        ax.savefig('Front de Pareto.png')
    if interactive == True:
        df = pd.DataFrame({"besoins chauffage" : Pareto_objective_functions[:, 0], 
                        "couts" :  Pareto_objective_functions[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres[:, 2],
                        "type_fenetre" : Pareto_decision_parametres[:, 3]
                        })
        front=df.hvplot.scatter(x='besoins chauffage', 
                                y='couts', 
                                hover_cols=['ep_murs_ext',"ep_plancher_haut","ep_plancher_bas","type_fenetre"])
        hvplot.show(front)
        from bokeh.resources import INLINE
        hvplot.save(front, 'fronthvplot.html', resources=INLINE)

'''
        plt.scatter(Pareto_decision_parametres[:, 0], Pareto_decision_parametres[:, 1])
        plt.xlabel("x1")
        plt.ylabel('x2')
        plt.title('Front de Pareto')
        plt.savefig('Front de Pareto_parametres.png')

    #optimal_indiv = np.array(optimal_indiv)

    #plt.scatter(optimal_indiv[:, 0], optimal_indiv[:, 1],
                #c="c", alpha=0.7, marker="+")
    #print(front)
    fig, ax = plt.subplots()
    ax.plot(front[:, 0], front[:, 1])
    #plt.scatter(front[:, 0], front[:, 1])
    #plt.axis("tight")
    #plt.ylabel("f2")#Inconfort de l\"occupant \n(°C.h en dehors de 18-28°C)
    #plt.xlabel("f1")#Consommation énergétique (kWh)
    #plt.title("Front de Pareto (en rouge) et \n individus optimaux (en bleu)")
    plt.savefig("front.png")

    x, y = zip(*data)
    c = np.array([i // 96 for i in range(len(x))])
    plt.ylabel("Inconfort de l\"occupant \n(°C.h en dehors de 18-28°C)")
    plt.xlabel("Consommation énergétique (kWh)")
    plt.title("Individus par génération")
    plt.scatter(x, y, c=c)
    plt.savefig("front2.png")'''
'''
https://github.com/daydrill/ga_pycon_2016_apac/blob/master/Decision_Making_with_GA_using_DEAP.ipynb
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh import palettes
output_notebook()
def viz_front(fronts):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize"
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("individual", "@ind"),
            ]
        )
    front_colors = []
    p = figure(plot_width=700, plot_height=700, tools=[TOOLS,hover], title="NSGAii Test")

    for i,inds in enumerate(fronts):
        par = [(ind, toolbox.evaluate(ind)) for ind in inds]
        source = ColumnDataSource(
                data=dict(
                    x= [p[1][0] for p in par],
                    y= [p[1][1] for p in par],
                    ind= [p[0] for p in par]
                )
            )
        p.circle('x', 'y', size=10, source=source, alpha=0.7, fill_color=palettes.YlGnBu9[i], legend='Front %s'%(i+1), line_color="#ffffff")
    show(p)
viz_front(fronts)'''

lock = Lock()
logger.debug("initializing")
init_opti()
if __name__ == "__main__":
    start_time = time.time()
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if os.path.exists("monitoring.csv"):
        os.remove("monitoring.csv")
    if os.path.exists("pareto_monitoring.csv"):
        os.remove("pareto_monitoring.csv")
    #   Multiprocessing pool
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    logger.debug("optimizing")
    indiv, statslog, optimal_front, graph_data, allindiv = main() #pop, logbook, pareto, graph, data
    logger.debug("out of main")


#    except Exception as e:
#        import pdb; pdb.set_trace()
#        logger.error(e)


    logger.info("Exiting program")
    write_pareto(optimal_front, graph_data, allindiv)
    plots(optimal_front)
    # Print running time
    print("--- %s seconds ---" % (time.time() - start_time))