import logzero
from logzero import logger
import config
from energyplus_wrapper import EPlusRunner
from eppy import modeleditor
from eppy.modeleditor import IDF
def overheating_modified(result):
    logger.debug("computing overheating")
    indoor = None
    out = None

    for data in result:
        try:
            if "eplus" in data:
                logger.debug("found eplus.csv")
                out = result[data].iloc[:,
                                ["Outdoor Air Drybulb Temperature" in col for col in result[data].columns]]
                indoor = result[data].iloc[:, [
                    "Mean Air Temperature" in col for col in result[data].columns]]

                oh = 0
                for index in indoor.columns:
                    for temp in indoor[index]:
                        if temp > 26:
                            oh += temp - 0.31 * out.iat[indoor[index].values.tolist().index(temp), 0] + 17.8
                           
        except IndexError:#Raised when a sequence subscript is out of range. (Slice indices are silently truncated to fall in the allowed range; if an index is not an integer, TypeError is raised.)
            if "eplusout" in data:
                logger.debug("found eplusout.csv")
                out = result[data].iloc[:,
                                ["Outdoor Air Drybulb Temperature" in col for col in result[data].columns]]
                indoor = result[data].iloc[:, [
                    "Mean Air Temperature" in col for col in result[data].columns]]

                oh = 0
                for index in indoor.columns:
                    for temp in indoor[index]:
                        if temp > 26:
                            oh += temp - 0.31 * out.iat[indoor[index].values.tolist().index(temp), 0] + 17.8

    logger.debug("overheating = %s °C/h" % (oh))
    return oh / config.TIMESTEP # /2 car demi heure sur l'échantillonage

EPWFILE = "./model/in.epw"
building="./model/IDM.idf"
IDDPATH = "C:\EnergyPlusV9-3-0\Energy+.idd"
EPLUSPATH = "C:\EnergyPlusV9-3-0"
simu_name = 'Maison' 
IDF.setiddname(IDDPATH)
epmodel = IDF(building, EPWFILE)

logger.debug("running energyplus")
runner = EPlusRunner(EPLUSPATH)

simulation = runner.run_one(epmodel, EPWFILE)
                            
result=simulation.time_series

print(overheating_modified(result))