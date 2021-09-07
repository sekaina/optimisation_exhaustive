#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soukaina

"""
import pandas as pd
import numpy as np
import logging
import logging.handlers
import ast
import time

from eppy.modeleditor import IDF
from energyplus_wrapper import EPlusRunner
import config

import logzero
from logzero import logger
from logzero import setup_logger

import zipfile
from path import Path
archive_folder = Path("./archive/").abspath()

log_format = '%(message)s'
monitfmt = logzero.LogFormatter(fmt=log_format)

monitbuilding = setup_logger(logfile="monitbuilding.log", 
                             level=logging.INFO, 
                             formatter=monitfmt)

logzero.loglevel(logging.DEBUG)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())
logzero.logger.addHandler(log)
# logger.propagate = False

IDDPATH = config.IDDPATH
EPLUSPATH = config.EPLUSPATH

IDFPATH = "./model/"
#LIBFILE = "./model/material.idf"
LIBWINDOW = "./model/windows.idf"

EPWFILE = IDFPATH + "CHAMBERY.epw"


def initialize(idf):
    """
    Creation of Eppy IDF object for one building

    args:
        idf (str): path to the idf file

    returns:
        model (IDF): Eppy building model

    """
    logger.debug("Eppy initialization")
    IDF.setiddname(IDDPATH)
    model = IDF(idf, EPWFILE)

    return model

def build_library(LIBWINDOW, EPWFILE):
    """
    Loading of Material and building Construction in the optimized IDF.

    args:
        #model (IDF) : future optimized IDF
        libfile (str) : path to IDF containing materials to add to model
        epwfile (EPW) : any EPW file, not used for simulation but needed to
                        load libfile
    returns:
        None

    Example:
        >>> IDF.setiddname(/usr/local/EnergyPlus-8-6-0/Energy+.idd)
        >>> model = IDF("./tests/test_model", "./tests/test_weather.epw")
        >>> build_library(model, "./model/test_mat.idf", "./tests/test_weather.epw")

    """
    config.IDF_WINDOWS = IDF(LIBWINDOW, EPWFILE)
    for glazing in config.IDF_WINDOWS.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]:
        config.IDF_WINDOWS.newidfobject(
            "CONSTRUCTION", Name="window_" + str(glazing.Name), Outside_Layer=glazing.Name)

def modify(ind, model):
    """
    Launch modifications for classes in EP Model and

    args:
        ind (list): individual of the optimization associated with this
                    building

    returns:
        model (Eppy IDF): modified building model
        surface_mat (list) : surfaces and type of modified parts of model
    """
    logger.debug("modifying model %s with ind %s" % (model.idfname, ind))
    surface_mat = []

    surface_mat.append(modify_thickness_insulation_wall(ind[0], model))
    surface_mat.append(modify_thickness_insulation_ceiling(ind[1], model))
    surface_mat.append(modify_thickness_insulation_floor(ind[2], model))
    surface_mat.append(modify_window(ind[3], model))
    #model.saveas("modelIDM"+str(ind[0])+'_'+str(ind[1])+'_'+str(ind[2])+'_'+str(ind[3])+".idf")
    logger.debug("surface_mat %s " % (surface_mat))
    return model, surface_mat  

def modify_window(ind_window, model):
    """
    Modification of any floor connected to the outside by checking boundary
    condition

    args:
        ind (int): window solution from optimization

    returns:
        amount (int) : number of modified windows

    """
    logger.debug("modify windows")
    area = 0
    window_ind = config.IDF_WINDOWS.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"][ind_window]
    window_name=window_ind.Name
    windows_construction=["pf_S_R+1","pf_S_RDC","pf_O_RDC","f_E_R+1","f_R+2_E-RDC","f_O_RDC","f_N_RDC","f_N_R+1","pf_O_R+1"]
    windows_shades=["VR_pf_S_R+1","VR_pf_S_RDC","VR_pf_O_RDC","VR_f_E_R+1","VR_f_R+2_E-RDC","VR_f_O_RDC","VR_f_N_RDC","VR_f_N_R+1","VR_pf_O_R+1"]#F_R+2_E-RDC
    model.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"].append(window_ind)
    for window in model.idfobjects["CONSTRUCTION"]:
        if window.Name in windows_construction:
            window.Outside_Layer=window_name
        if window.Name in windows_shades:
            window.Layer_2=window_name
    windows=[object for object in model.idfobjects["FENESTRATIONSURFACE:DETAILED"] if object.Surface_Type=="Window"]
    for window in windows:
        area += window.area

    return area, window_name, ind_window

def modify_thickness_insulation_ceiling(ep, model):
    """
    modification of the thickness of insulation in all walls

    args : 
        ep(int): insulation thickness from optimization

    returns:
        surface.area (str) : surface of modified constructions
        surface.Construction_Name (str) : Name of modified construction
    """
    logger.debug("modify ceiling")
    ep_metre=ep*0.01
    area = 0
    construction_Name = "PB_COMBLES_isole"
    materials=model.idfobjects["Material"]
    for material in materials:
        if material.Name=="LDV35_40cm":
            material.Thickness=ep_metre
    for surface in model.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if (surface.Construction_Name == "PB_COMBLES_isole"):
            area += surface.area
    return area, construction_Name, ep

def modify_thickness_insulation_floor(ep, model):
    """
    modification of the thickness of insulation in all walls

    args : 
        ep(int): insulation thickness from optimization

    returns:
        surface.area (str) : surface of modified constructions
        surface.Construction_Name (str) : Name of modified construction
    """
    logger.debug("modify floor")
    ep_metre=ep*0.01
    area = 0
    construction_Name = "PB_RDC_isole"
    materials=model.idfobjects["Material"]
    for material in materials:
        if material.Name=="PolystyreneXtrude30":
            material.Thickness=ep_metre
    for surface in model.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if (surface.Construction_Name == construction_Name):
            area += surface.area
    return area, construction_Name, ep

def modify_thickness_insulation_wall(ep, model):
    """
    modification of the thickness of insulation in all walls

    args : 
        ep(int): insulation thickness from optimization en cm

    returns:
        surface.area (str) : surface of modified constructions
        surface.Construction_Name (str) : Name of modified construction
    """
    logger.debug("modify walls")
    ep_metre=ep*0.01
    area = 0
    construction_Name = "MurExt_isole"
    materials=model.idfobjects["Material"]
    for material in materials:    
        if material.Name=="LDV35_20_MurExt":
            material.Thickness=ep_metre
    for surface in model.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if (surface.Construction_Name == construction_Name):
            area += surface.area

    return area, construction_Name, ep
       
def heating_needs(result):
    logger.debug("computing heating needs")
    import csv
    for data in result:
        if "table" in data:
            logger.debug("found table")
            table=result['table']
            #print(table[49])
            Chauffage = float(table[49][13])# [49][6] in EP v9.3   [49][13] in EP v9.5 pour simulation annuelle
        return Chauffage

def heating_needs_modified(result):
    logger.debug("computing heating needs")
    import csv
    for data in result:
        #try:
            if "table" in data:
                logger.debug("found eplus-table.csv")
                with open('table.csv', 'w') as f:
                    f.write(result[data])
                f.close()
                with open('table.csv', 'rt') as f:
                    file=csv.reader(f, delimiter=",")
                    for row in file:
                        for column in row:
                            if "Heating" in column:
                                i = row.index(column) + 1
                                meter = row[i]
                                logger.debug("In eplus-table.csv, %s kWh" % (meter))
                                return meter

def parse_csv(lines):
    for row in list(lines):
        for column in row:
            if "Total Site Energy" in column:
                i = row.index(column) + 1
                meter = row[i]
                logger.debug("In eplus-table.csv, %s kWh" % (meter))
                return pd.DataFrame([column, meter])

def save_result(simulation):
    import shutil
    for file in simulation.working_dir.files("eplus*.csv"):
        filename = simulation.name + file.basename()
        print (filename)
        shutil.copy(file, "./results/"+ filename)

def process_table(filename, working_dir, simulname):
    import csv
    with open(filename, "rt") as f:
        file = csv.reader(f, delimiter=",")
        return parse_csv(file)

def process_table_modified(filename):
    import csv
    with open(filename, "rt") as f:
        file = csv.reader(f, delimiter=",")
        return parse_csv(file)


def evaluate_model(model, indb, surfacemat):
    """
    Simulation of a building model and post processing results to return
    comfort and heating demand

    args:
        model (IDF): model to simulation

    returns:
        results (tuple): heating demand and comfort

    ToDo:
        Windows
        overheating
    """
    start_time = time.time()
    logger.debug("running energyplus")
    runner = EPlusRunner(EPLUSPATH)

    simulation = runner.run_one(model, EPWFILE) #extra_file dans le cas de fmu NoMASS
    result=simulation.time_series

    logger.debug("Evaluation of %s with %s in %s s" %
                     (model.idfname, indb, time.time() - start_time))

    logger.debug("Computing objectives from result dataframes")
    heating = float(heating_needs(result))
    heating_m2=heating/config.building_area
    logger.debug("In table.csv, %s kWh, %s kWh/m2" % (heating, heating_m2)) # kwh par m2
    comfort = float(overheating(result))

    logger.debug("%s hours of discomfort (where temperature is above Tconf+2°C) " % (comfort))

    logger.debug("computing investment price")
    investment = np.array(economy_investment(surfacemat)).sum()
    logger.debug("Investment Price %s " % (investment))
    logger.debug("computing operation price")
    operation = economy_operation(heating)
    logger.debug("Operation Price %s " % (operation))
    total_price = investment + operation
    total_price_m2 = total_price/config.building_area # euros par m2
    logger.debug("Total Price %s euros/m2" % (total_price))
    return heating_m2, comfort, total_price_m2


def evaluate(ind):
    """
    Evaluation function called by optimization. Deals with all the building
    of the building stock.

    args:
        ind (list): individual from genetic algorithm optimization

    returns:
        results (tuple): objective function

    ToDo
        put result in tuple
    """
    #ind = unconstrain(ind)
    #indb = [ind[i:i + 4] for i in range(0, len(ind), 4)]
    surfacemat = []
    
    epmodel = initialize(config.building)
    logger.info(config.building)
    build_library(LIBWINDOW, EPWFILE)
    logger.debug("modifying model %s" % (epmodel.idfname))
    epmodel, surfacemat = modify(
        ind, epmodel)
    logger.debug("launching evaluation function")
    fitness = np.array(evaluate_model(epmodel, ind, surfacemat))
    logger.info("fitness for %s : %s" % (epmodel.idfname, fitness))
    monitbuilding.info((ind, epmodel.idfname, fitness))
    logger.debug("returning fitness")

    return fitness

def economy_investment(surface_mat):
    """
    Compute the price of a given retrofit plan given by the construction content and
    the window performance

    args:
        surface_mat (list): surface and composition of each type of wall (ext, ceiling, floor)

    returns:
        price (float) : total price of the retrofit measure
    """
    material = {"Polystyrene": "polystyrene_price",
                #"Rockwool": "rockwool_price",
                "Glasswool": "glasswool_price",
                #"Polyurethane": "polystyrene_price",
                "Window": "window_price"
                }
    price_mat = []
    
    for paroi in surface_mat[:-1]:
        e=paroi[2]
        #try:
        price_this_mat = 0
        if (paroi[1]=="MurExt_isole" or paroi[1]=="PB_COMBLES_isole"):
                mat="Glasswool"
        elif paroi[1]=="PB_RDC_isole":
                mat = "Polystyrene"
        price_this_mat = globals()[material[mat]](e)
        price_mat.append(paroi[0] * (price_this_mat + 10))  # +10 for coating
            
        #except AttributeError:
        #    logger.error("eco : No price for %s" % (mat))
        #    pass
    window=surface_mat[-1]
    price_window = window_price(window)
    price_mat.append(price_window)
    return price_mat


def window_price(window):
    #cout de la forme a*surface_fenetre+b
    a = {0: 460.45, 1: 454.16, 2: 390.85, 3:349.35 }
    b= {0: 34.45, 1: 36.62, 2: 29.37, 3: 28.17}

    price = a[window[2]]*window[0]+b[window[2]]

    #price = price + 49  # 49€ is flat price for arranging walls for construction

    return price

def glasswool_price(e):
    return 0.39 * e + 0.17 # price_this_mat

def polystyrene_price(e):
    return 1.25 * e + 1

def moyenne_glissante(valeurs, intervalle):
    indice_debut = (intervalle - 1) // 2
    liste_moyennes=valeurs[:intervalle-1]
    liste_moyennes += [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
def moyenne_glissante_norme (valeurs, intervalle):
    indice_debut=(intervalle - 1) // 2
    liste_moyennes=valeurs[1:intervalle]
    liste_moyennes += [(0.2*valeurs[i - indice_debut]+0.3*valeurs[i - indice_debut+1]+0.4*valeurs[i - indice_debut+2]+
                    0.5*valeurs[i - indice_debut+3]+0.6*valeurs[i - indice_debut+4]+0.8*valeurs[i - indice_debut+5]+
                    valeurs[i - indice_debut+6]) / 3.8 for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
def overheating(result):
    logger.debug("computing overheating")
    indoor = None
    out = None
    heures_inconfort=[]
    oh = []
    for data in result:
            if "eplus" in data:
                logger.debug("found eplus.csv")
                indoor = result[data].iloc[:, [
                    "Mean Air Temperature" in col for col in result[data].columns]]
                out = result[data].iloc[:,[
                    "Outdoor Air Drybulb Temperature" in col for col in result[data].columns]]
                Text_moy_jour=[float(out[i:289+i].mean()) for i in range(0,len(out),288)]
                Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)#moyenne glissante sur 7 jours selon la norme NF EN 16798-1
                Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes] # temperature de confort adaptatif selon la norme NF EN 16798-1
                for zone, area in config.zones_areas.items():
                    oh_zone=0
                    heures_inconfort_zone=0
                    indoor_zone=indoor.iloc[:,[zone in col for col in indoor.columns]]
                    T_moy_jour=[float(indoor_zone[i:289+i].mean()) for i in range(0,len(indoor_zone),288)]
                    for i in range(len(T_moy_jour)):
                        if T_moy_jour[i]>(Tconfort[i]+2):
                            oh_zone+=T_moy_jour[i]-(Tconfort[i]+2)
                            heures_inconfort_zone+=1
                    oh.append(oh_zone)
                    heures_inconfort.append(heures_inconfort_zone)
    area_tot=config.building_area
    areas=[]
    for zone,area in config.zones_areas.items():
        areas.append(area)
    oh_tot=sum([x*y for x,y in zip(areas,oh)])/area_tot  #somme pondérée par les surfaces
    heures_inconfort_tot=sum([x*y for x,y in zip(areas,heures_inconfort)])/area_tot  
    logger.debug("overheating = %s °C/h" % (oh_tot))
    logger.debug("heures inconfort = %s " % (heures_inconfort_tot))
    return heures_inconfort_tot





def unconstrain(ind):
    """
    Constraints study by decision space limitation
    """
    new_ind = []
    logger.debug("unconstraining")
    for idx, item in enumerate(ind):
        if idx % 3 == 0:
            new_wall, new_window = change_bit(item)
            new_ind.append(new_wall)
            new_ind.append(ind[idx +1])
            new_ind.append(ind[idx +2])
            new_ind.append(new_window)
    logger.debug("unconstrained ind %s" % (new_ind))
    return new_ind

def change_bit(comb_bit):
    """
    From on gene for ext walls AND windows to 2 bits: one for walls one for windows
    """
    logger.debug("changing bit %s" % (comb_bit))
    if comb_bit % 43 == 0:
        inter = comb_bit + 6
        window = inter % 5
        wall = inter // 5 + 10 * (comb_bit // 43)

    elif comb_bit % 43 <= 2:
        window = comb_bit % 43
        wall = comb_bit // 43 * 10

    elif comb_bit % 43 <= 6:
        inter = comb_bit % 43 + 2
        window = inter - 5
        wall = comb_bit // 43 * 10 + 1

    elif comb_bit % 43 <= 36:
        inter = comb_bit % 43 + 3
        window = inter % 5
        wall = inter // 5 + 10 * (comb_bit // 43)
    
    elif comb_bit % 43 <= 40:
        inter = comb_bit % 43 + 4
        window = inter % 5
        wall = inter // 5 + 10 * (comb_bit // 43)

    elif comb_bit % 43 <= 43:
        inter = comb_bit % 43 + 6
        window = inter % 5
        wall = inter // 5 + 10 * (comb_bit // 43)

    logger.debug("returning new bits %s and %s" % (wall, window))
    return wall, window

def save_and_compress(simulation):
    with zipfile.ZipFile(archive_folder / f"{simulation.name}.gz", "w", zipfile.ZIP_DEFLATED) as zip_handler:
        for file in simulation.working_dir.files("*"):
            zip_handler.write(file)

#cd = Path('C:/Users/elkhatts/Desktop/model/')

def save_and_compress_flattened(simulation, res_dir=Path('C:/Users/elkhatts/Desktop/model/')):
    """
    save all simulation files in a zipped folder
    """
    with zipfile.ZipFile(res_dir / f"{simulation.name}.gz", "w",
                         zipfile.ZIP_DEFLATED) as zip_handler:
        for file in simulation.working_dir.files("eplus*"):
            # là on demande d'enregistrer toutes les sorties dans le zip, sauf le .eso qui est trop lourd
            if '.eso' not in file:
                zip_handler.write(file, arcname=file.basename())
    return

def economy_operation(Echauffage):
    cost=config.K*config.Pelec*(Echauffage/config.COP) # PAC de COP=4
    return cost

if __name__ == "__main__": #pour tester
    #building="./model/IDM.idf"
    #epmodel = initialize(building)
    ind=[20,40,20,2]
    fitness= evaluate(ind)
    chauffage=fitness[0]
    print(chauffage, type(chauffage))


    

