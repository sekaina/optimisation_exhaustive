from energyplus_wrapper import EPlusRunner
from eppy import modeleditor
from eppy.modeleditor import IDF
import zipfile
from path import Path
import warnings
import csv
import pandas as pd
import logzero
from logzero import logger
# ============== SAVE AND COMPRESS FUNCTION ===================
archive_folder = Path("./archive/").abspath()

def save_and_compress(simulation):
    with zipfile.ZipFile(archive_folder / f"{simulation.name}.zip", "w", zipfile.ZIP_DEFLATED) as zip_handler:
        for file in simulation.working_dir.files("*"):
            zip_handler.write(file)
            

cd = Path('C:/Users/elkhatts/Desktop/model/')

def save_and_compress_flattened(simulation, res_dir=cd):
    """
    save all simulation files in a zipped folder
    """
    with zipfile.ZipFile(res_dir / f"{simulation.name}.gz", "w",
                         zipfile.ZIP_DEFLATED) as zip_handler:
        for file in simulation.working_dir.files("eplus*"):
            # là on demande d'enregistrer toutes les sorties dans le zip, sauf le .eso qui est trop lourd
            if '.eso' not in file:
                zip_handler.write(file, arcname=file.basename())
                return file.basename() 

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
        shutil.copy(file, "C:/Users/elkhatts/Desktop/results/"+ filename)

def process_table(simulation):
    import csv
    for file in simulation.working_dir.files("eplus*.csv"):
        if "table" in file:
            with open(file, "rt") as f:
                file = csv.reader(f, delimiter=",")
                df_heating = parse_csv(file)

EPWFILE = "C:/Users/elkhatts/Desktop/model/in.epw"
building="C:/Users/elkhatts/Desktop/model/BarreMontreau.idf"
IDDPATH = "C:\EnergyPlusV9-3-0\Energy+.idd"
EPLUSPATH = "C:\EnergyPlusV9-3-0"
backup_dir = "C:/Users/elkhatts/Desktop/failed"
simu_name = 'Maison' 
IDF.setiddname(IDDPATH)
epmodel = IDF(building, EPWFILE)
runner = EPlusRunner(EPLUSPATH)
simulation = runner.run_one(building, EPWFILE,
                     version_mismatch_action="ignore",
                     simulation_name=simu_name,
                     backup_strategy='on_error',
                     backup_dir=backup_dir,
                     )

def heating_needs_modified(result):
    logger.debug("computing heating needs")
    import csv
    for data in result:
        #try:
            if "table" in data:
                logger.debug("found eplus-table.csv")
                print(type(result[data]))
                with open('table.csv', 'w') as f:
                    f.write(result[data])
                f.close()
                with open('table.csv', 'rt') as f:
                    file=csv.reader(f, delimiter=",")
                    for row in file:
                        for column in row:
                            if "Total Site Energy" in column:
                                i = row.index(column) + 1
                                meter = row[i]
                                print(type(meter))
                                logger.debug("In eplus-table.csv, %s kWh" % (meter))
                                return meter
                
        #except Exception:
            #pass    
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
                           
        except IndexError:
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
    return oh / 2 # /2 car demi heure sur l'échantillonage

#print(simulation.time_series)
result=simulation.time_series
heating=heating_needs_modified(result)
overheating=overheating_modified(result)
print(heating)
print(overheating)
#report=simulation.reports



