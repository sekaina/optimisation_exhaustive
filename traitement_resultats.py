from re import T
from matplotlib.pyplot import clabel
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
def plots(df, Pareto_objective_functions, economic_solution=None, efficient_solution=None, comfortable_solution=None,base_solution=None, plot2D = True, plot3D = False, interactive = False): 
        
    x = Pareto_objective_functions[:, 0]/97.5 #chauffage
    y = Pareto_objective_functions[:, 2]/97.5 #cout
    z = Pareto_objective_functions[:, 1] #inconfort
        
    if plot2D == True : 
        """to plot 2 by 2"""

        '''
        fig = plt.figure()
        plt.scatter(x,y,c=z)
        plt.xlabel("Besoins de chauffage kWh")
        plt.ylabel('Cout global actualisé en euros')
        plt.title('Front de Pareto')
        plt.colorbar(label="Heures d'inconfort (T>Tconf+2°C)")
        plt.savefig('Front de Pareto_objectifs.png')
        '''
        
        fig = plt.figure()
        fig.set_size_inches(15,10)
        
        axe1 = plt.subplot2grid((2,2),(0,0))
        axe1.set_ylabel('Cout global actualisé en euros/m2', fontsize=15)
        plot1=axe1.scatter(x, y, c=z)
        axe1.scatter(economic_solution[0][0]/97.5, economic_solution[0][2]/97.5, marker="s", color="red", label="la solution la moins chère")
        axe1.scatter(efficient_solution[0][0]/97.5, efficient_solution[0][2]/97.5, marker="s", color="k", label="la solution qui consomme le moins de chauffage")
        axe1.scatter(comfortable_solution[0][0]/97.5, comfortable_solution[0][2]/97.5, marker="s", color="y", label="la solution la plus confortable")
        plt.colorbar(plot1,ax=axe1,label="Heures d'inconfort (T>Tconf+2°C)")


        axe2 = plt.subplot2grid((2,2),(1,0))
        axe2.set_ylabel("Heures d'inconfort (T>Tconf+2°C)", fontsize=15)
        axe2.set_xlabel("Besoins de chauffage kWh/m2", fontsize=15)
        plot2=axe2.scatter(x, z, c=y)
        axe2.scatter(economic_solution[0][0]/97.5, economic_solution[0][1], marker="s", color="red")
        axe2.scatter(efficient_solution[0][0]/97.5, efficient_solution[0][1], marker="s", color="k")
        axe2.scatter(comfortable_solution[0][0]/97.5, comfortable_solution[0][1], marker="s", color="y")
        plt.colorbar(plot2,ax=axe2,label="Cout global actualisé en euros")

        axe3 = plt.subplot2grid((2,2),(1,1))
        axe3.set_xlabel("Cout global actualisé en euros/m2", fontsize=15)
        plot3 = axe3.scatter(y, z, c=x)
        axe3.scatter(economic_solution[0][2]/97.5, economic_solution[0][1], marker="s", color="red")
        axe3.scatter(efficient_solution[0][2]/97.5, efficient_solution[0][1], marker="s", color="k")
        axe3.scatter(comfortable_solution[0][2]/97.5, comfortable_solution[0][1], marker="s", color="y")
        plt.colorbar(plot3,ax=axe3,label="Besoins de chauffage kWh/m2")


        if base_solution != None: #.all()
            axe1.scatter(base_solution[0]/97.5, base_solution[2]/97.5, marker="s", color="blue", label="le cas de référence")
            axe2.scatter(base_solution[0]/97.5, base_solution[1], marker="s", color="blue")
            axe3.scatter(base_solution[2]/97.5, base_solution[1], marker="s", color="blue")

        fig.legend(loc="right")
        plt.savefig('Front de Pareto_objectifs_deterministic.png')
        plt.show()
        
    if plot3D == True:
        """to plot the tree functions"""
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(Pareto_objective_functions[:, 0], Pareto_objective_functions[:, 2], Pareto_objective_functions[:, 1])
        ax.set_xlabel("besoins de chauffage kWh/m2")
        ax.set_ylabel("Cout global actualisé en euros/m2")
        ax.set_zlabel("heures d'inconfort (T>Tconf+2°C)")
        ax.set_title('Front de Pareto')
        ax.figure.savefig('Front de Pareto.png')
        fig.show()
    if interactive == True:
        """To interactively analyze the results of the optimization"""
        import pandas as pd
        import hvplot.pandas
        front=df.hvplot.scatter(x="besoins de chauffage kWh/m2", 
                                y="Cout global actualisé en euros/m2", 
                                c="heures d'inconfort (T>Tconf+2°C)",
                                clabel="heures d'inconfort (T>Tconf+2°C)",
                                hover_cols=["ep_murs_ext","ep_plancher_haut","ep_plancher_bas","type_fenetre"])
        hvplot.show(front)
        from bokeh.resources import INLINE
        hvplot.save(front, 'fronthvplot.html', resources=INLINE)


def comparaison_objectifs_deterministic_nomass(Pareto_objective_functions_deterministic,  Pareto_objective_functions_nomass):
    """trace les fonctions objectifs des deux fronts de pareto sur la même figure"""
    x_deterministic = Pareto_objective_functions_deterministic[:, 0]/97.5 #chauffage
    y_deterministic = Pareto_objective_functions_deterministic[:, 2]/97.5 #cout
    z_deterministic = Pareto_objective_functions_deterministic[:, 1] #inconfort

    x_nomass = Pareto_objective_functions_nomass[:, 0]/97.5 #chauffage
    y_nomass = Pareto_objective_functions_nomass[:, 2]/97.5 #cout
    z_nomass = Pareto_objective_functions_nomass[:, 1] #inconfort

    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((2,2),(0,0))
    axe1.set_ylabel('Cout global actualisé en euros/m2', fontsize=15)
    plot1=axe1.scatter(x_deterministic, y_deterministic, c='b')
    axe1.scatter(x_nomass, y_nomass, c='c')
    #axe1.scatter(economic_solution[0][0]/97.5, economic_solution[0][2]/97.5, marker="s", color="red", label="la solution la moins chère")
    #axe1.scatter(efficient_solution[0][0]/97.5, efficient_solution[0][2]/97.5, marker="s", color="k", label="la solution qui consomme le moins de chauffage")
    #axe1.scatter(comfortable_solution[0][0]/97.5, comfortable_solution[0][2]/97.5, marker="s", color="y", label="la solution la plus confortable")
    #plt.colorbar(plot1,ax=axe1,label="Heures d'inconfort (T>Tconf+2°C)")


    axe2 = plt.subplot2grid((2,2),(1,0))
    axe2.set_ylabel("Heures d'inconfort (T>Tconf+2°C)", fontsize=15)
    axe2.set_xlabel("Besoins de chauffage kWh/m2", fontsize=15)
    plot2=axe2.scatter(x_deterministic, z_deterministic, c='b')
    axe2.scatter(x_nomass, z_nomass, c='c')
    #axe2.scatter(economic_solution[0][0]/97.5, economic_solution[0][1], marker="s", color="red")
    #axe2.scatter(efficient_solution[0][0]/97.5, efficient_solution[0][1], marker="s", color="k")
    #axe2.scatter(comfortable_solution[0][0]/97.5, comfortable_solution[0][1], marker="s", color="y")
    #plt.colorbar(plot2,ax=axe2,label="Cout global actualisé en euros")

    axe3 = plt.subplot2grid((2,2),(1,1))
    axe3.set_xlabel("Cout global actualisé en euros/m2", fontsize=15)
    plot3 = axe3.scatter(y_deterministic, z_deterministic, c='b', label='deterministic')
    axe3.scatter(y_nomass, z_nomass, c='c', label='nomass')
    #axe3.scatter(economic_solution[0][2]/97.5, economic_solution[0][1], marker="s", color="red")
    #axe3.scatter(efficient_solution[0][2]/97.5, efficient_solution[0][1], marker="s", color="k")
    #axe3.scatter(comfortable_solution[0][2]/97.5, comfortable_solution[0][1], marker="s", color="y")
    #plt.colorbar(plot3,ax=axe3,label="Besoins de chauffage kWh/m2")

    '''
    if base_solution != None: #.all()
        axe1.scatter(base_solution[0]/97.5, base_solution[2]/97.5, marker="s", color="blue", label="le cas de référence")
        axe2.scatter(base_solution[0]/97.5, base_solution[1], marker="s", color="blue")
        axe3.scatter(base_solution[2]/97.5, base_solution[1], marker="s", color="blue")
    '''
    fig.legend(loc="right")
    plt.savefig('comparaison Front de Pareto_objectifs.png')
    plt.show()

def comparaison_parametres_deterministic_nomass(Pareto_parametres_deterministic,  Pareto_parametres_nomass):
    """trace les paramètres de décision des deux fronts de pareto sur la même figure"""
    x_deterministic = Pareto_parametres_deterministic[:, 0] #epaisseur murs exterieurs
    y_deterministic = Pareto_parametres_deterministic[:, 1] #epaisseur plancher haut
    z_deterministic = Pareto_parametres_deterministic[:, 2] #epaisseur plancher bas
    u_deterministic = Pareto_parametres_deterministic[:, 3] #type fenetre
    
    x_nomass = Pareto_parametres_nomass[:, 0] #epaisseur murs exterieurs
    y_nomass = Pareto_parametres_nomass[:, 1] #epaisseur plancher haut
    z_nomass = Pareto_parametres_nomass[:, 2] #epaisseur plancher bas
    u_nomass = Pareto_parametres_nomass[:, 3] #type fenetre

    fig = plt.figure()
    fig.set_size_inches(15,10)
        
    axe1 = plt.subplot2grid((2,2),(0,0))
    axe1.set_ylabel('epaisseur plancher haut en cm', fontsize=15)
    axe1.scatter(x_deterministic, y_deterministic, c='b')
    axe1.scatter(x_nomass, y_nomass, c='c')

    axe2 = plt.subplot2grid((2,2),(1,0))
    axe2.set_ylabel("epaisseur plancher bas en cm", fontsize=15)
    axe2.set_xlabel("epaisseur murs exterieurs en cm", fontsize=15)
    axe2.scatter(x_deterministic, z_deterministic, c='b')
    axe2.scatter(x_nomass, z_nomass, c='c')

    axe3 = plt.subplot2grid((2,2),(1,1))
    axe3.set_xlabel("epaisseur plancher haut en cm", fontsize=15)
    axe3.scatter(y_deterministic, z_deterministic, c='b', label='deterministic')
    axe3.scatter(y_nomass, z_nomass, c='c', label='nomass')

    fig.legend(loc="right")
    plt.savefig('comparaison Front de Pareto_parametres.png')
    plt.show()
def best_solution(Pareto_objective_functions,Pareto_decision_parametres, economic=True, confortable=False, efficient=False):
    """To find the best economical, efficient(energetic) or comfortable solution"""
    df = pd.DataFrame({"Besoins de chauffage kWh/m2" : Pareto_objective_functions[:, 0], 
                        "Heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres[:, 2],
                        "type_fenetre" : Pareto_decision_parametres[:, 3]
                        })
    if economic==True:
        cout_min=df["Cout global actualisé en euros/m2"].min()
        index_cout_min=df[df["Cout global actualisé en euros/m2"]==cout_min].index.values
        solution=df.iloc[index_cout_min]
        solution=solution.values.tolist()
        print("la solution la moins chère", cout_min, "est\n", solution)

    if confortable==True:
        inconfort_min=df["Heures d'inconfort (T>Tconf+2°C)"].min()
        index_inconfort_min=df[df["Heures d'inconfort (T>Tconf+2°C)"]==inconfort_min].index.values
        solution=df.iloc[index_inconfort_min]
        solution=solution.values.tolist()
        print("la solution la plus confortable", inconfort_min, "est\n", solution)

    if efficient==True:   
        chauffage_min=df["Besoins de chauffage kWh/m2"].min()
        #df_chauff_min=df[df["besoins de chauffage kWh"].isin([chauffage_min])]
        index_chauff_min=df[df["Besoins de chauffage kWh/m2"]==chauffage_min].index.values
        solution=df.iloc[index_chauff_min]
        solution=solution.values.tolist()
    return solution

def same_cost_min_heating (Pareto_objective_functions,Pareto_decision_parametres, base_solution):
    df = pd.DataFrame({"Besoins de chauffage kWh/m2" : Pareto_objective_functions[:, 0], 
                        "Heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions[:, 1],
                        "Cout global actualisé en euros" :  Pareto_objective_functions[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres[:, 2],
                        "type_fenetre" : Pareto_decision_parametres[:, 3]
                        })    
    index_cout=df[df["Cout global actualisé en euros/m2"]==base_solution[2]/97.5].index.values
    solution=df.iloc[index_cout]
    solution=solution.values.tolist()


if __name__=="__main__":
#    Pareto_decision_parametres= np.array([ind for ind in population])
#    Pareto_objective_functions = np.array([ind.fitness.values for ind in population])
    
    #deterministic
    with open('./Results_To_Plot/pareto_obj_gen99.csv', 'r') as f:
        Pareto_objective_functions_deterministic=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_objective_functions_deterministic=Pareto_objective_functions_deterministic.astype('float64')
    with open('./Results_To_Plot/pareto_param_gen99.csv', 'r') as f:
        Pareto_decision_parametres_deterministic=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_decision_parametres_deterministic=Pareto_decision_parametres_deterministic.astype('float64')
    
    #nomass
    with open('./Results_To_Plot/pareto_obj_gen72.csv', 'r') as f:
        Pareto_objective_functions_nomass=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_objective_functions_nomass=Pareto_objective_functions_nomass.astype('float64')
    with open('./Results_To_Plot/pareto_param_gen72.csv', 'r') as f:
        Pareto_decision_parametres_nomass=np.array(list(csv.reader (f, delimiter=',')))
    Pareto_decision_parametres_nomass=Pareto_decision_parametres_nomass.astype('float64')

    #mettre les resultats dans dataframe
    df_deterministic = pd.DataFrame({"besoins de chauffage kWh/m2" : Pareto_objective_functions_deterministic[:, 0], 
                        "heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions_deterministic[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions_deterministic[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres_deterministic[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres_deterministic[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres_deterministic[:, 2],
                        "type_fenetre" : Pareto_decision_parametres_deterministic[:, 3]
                        })
    df_nomass = pd.DataFrame({"besoins de chauffage kWh/m2" : Pareto_objective_functions_nomass[:, 0], 
                        "heures d'inconfort (T>Tconf+2°C)" :  Pareto_objective_functions_nomass[:, 1],
                        "Cout global actualisé en euros/m2" :  Pareto_objective_functions_nomass[:, 2],
                        "ep_murs_ext" : Pareto_decision_parametres_nomass[:, 0],
                        "ep_plancher_haut" : Pareto_decision_parametres_nomass[:, 1],
                        "ep_plancher_bas" : Pareto_decision_parametres_nomass[:, 2],
                        "type_fenetre" : Pareto_decision_parametres_nomass[:, 3]
                        })

from fitnessesEP import evaluate
#ref_solution=evaluate([20,40,20,2])
ref_solution=[396.99, 174.93, 16201.53]
'''
economic_solution_nomass=best_solution(Pareto_objective_functions_nomass,Pareto_decision_parametres_nomass, economic=True, confortable=False, efficient=False)
comfortable_solution_nomass=best_solution(Pareto_objective_functions_nomass,Pareto_decision_parametres_nomass, economic=False, confortable=True, efficient=False)
efficient_solution_nomass=best_solution(Pareto_objective_functions_nomass,Pareto_decision_parametres_nomass, economic=False, confortable=False, efficient=True)

economic_solution_deterministic=best_solution(Pareto_objective_functions_deterministic,Pareto_decision_parametres_deterministic, economic=True, confortable=False, efficient=False)
comfortable_solution_deterministic=best_solution(Pareto_objective_functions_deterministic,Pareto_decision_parametres_deterministic, economic=False, confortable=True, efficient=False)
efficient_solution_deterministic=best_solution(Pareto_objective_functions_deterministic,Pareto_decision_parametres_deterministic, economic=False, confortable=False, efficient=True)
'''
#same_cost_min_heating (Pareto_objective_functions,Pareto_decision_parametres, ref_solution)
#plots(Pareto_objective_functions, Pareto_decision_parametres,economic_solution,efficient_solution, comfortable_solution, plot2D = True, plot3D = False, interactive = False)
#print("la solution ayant le même cout mais moins de chauffage est\n", solution)
#comparaison_objectifs_deterministic_nomass(Pareto_objective_functions_deterministic,  Pareto_objective_functions_nomass)
comparaison_parametres_deterministic_nomass(Pareto_decision_parametres_deterministic,  Pareto_decision_parametres_nomass)