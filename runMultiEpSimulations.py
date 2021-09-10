#version qui marche pour modification du seed dans SimulationConfig.xml

from path import Path
from shutil import copyfile, copytree, rmtree
import random
from time import sleep
import os
import time
import xml.etree.ElementTree as ET

from eppy import modeleditor
from eppy.modeleditor import IDF

from energyplus_wrapper import EPlusRunner

import zipfile
import tempfile

import config

def updateZip(zipname, filename, xmlFile):
    # generate a temp file
    tmpfd, tmpname = tempfile.mkstemp(dir=os.path.dirname(zipname))
    os.close(tmpfd)

    # create a temp copy of the archive without filename            
    with zipfile.ZipFile(zipname, 'r') as zin:
        with zipfile.ZipFile(tmpname, 'w') as zout:
            #zout.comment = zin.comment # preserve the comment
            for item in zin.infolist():
                if item.filename != filename:
                    zout.writestr(item, zin.read(item.filename))

    # replace with the temp archive
    os.remove(zipname)
    os.rename(tmpname, zipname)

    # now add filename with its new data
    with zipfile.ZipFile(zipname, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
        f = open(xmlFile,'r')
        zf.writestr(filename,f.read())

def modify_SimulationConfig(IDFPATH, run):
    xmlFile = IDFPATH+'SimulationConfig.xml'#IDFPATH + 'tmp-fmus/agentFMU.fmu_FMI/SimulationConfig.xml'
    if os.path.exists(xmlFile):
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        root.find('seed').text = str(run * 989)
        tree.write(xmlFile)
    copyfile(xmlFile, IDFPATH + "/results/SimulationConfig" + str(run)+'.xml')
    updateZip(IDFPATH+'agentFMU.fmu', 'SimulationConfig.xml', xmlFile)
def save_result(simulation):
    #import shutil
    for file in simulation.working_dir.files("*"):
        print (file.basename())
        if ".dat" in file :
            copyfile(file, IDFPATH + file.basename())#shutil.copy
        if ".out" in file :
            filename= os.path.splitext(file.basename())[0] #enlever l'extension du fichier
            copyfile(file, ResultsPATH + filename + str(i) + ".out")
        if "eplus.csv" in file :
            filename= os.path.splitext(file.basename())[0]
            copyfile(file, ResultsPATH + filename + str(i) + ".csv")
        if "table.csv" in file :
            filename= os.path.splitext(file.basename())[0]
            copyfile(file, ResultsPATH + filename + str(i) + ".csv")
def heating_needs(results):
    import csv
    print("Computing heating needs from result dataframes")
    for data in results:
        if "table" in data:
            print("found table")
            table=results['table']
            print(table[49])
            Chauffage =float(table[49][2])+float(table[49][3]) #float(table[49][13])
            print("In table.csv, %s kWh" % (Chauffage))
    return Chauffage

def runmulti(IDFPATH, EPWFILE, IDDPATH, EPLUSPATH, model_name, fmu_name, numberOfSimulations):
    idf = IDFPATH + model_name
    
    #define extra-files to be added to the temporary folder of running the idf model
    fmu_file = IDFPATH + fmu_name#"C:/Users/se266887/No-MASS/Configuration/Officelearn/agentFMU.fmu"#
    learning_files=[fmu_file]
    if os.path.exists(IDFPATH+"eplus.rvi"):
        learning_files.append(IDFPATH+"eplus.rvi")
    if os.path.exists(IDFPATH+"Weekday-1-0.dat"):
        learning_files.append(IDFPATH+"Weekday-1-0.dat")
    if os.path.exists(IDFPATH+"Weekday-1-1.dat"):
        learning_files.append(IDFPATH+"Weekday-1-1.dat")
    if os.path.exists(IDFPATH+"Weekend-1-0.dat"):
        learning_files.append(IDFPATH+"Weekend-1-0.dat")
    if os.path.exists(IDFPATH+"Weekend-1-1.dat"):
        learning_files.append(IDFPATH+"Weekend-1-1.dat")
        
    #Eppy initialization
    IDF.setiddname(IDDPATH)
    model = IDF(idf, EPWFILE)

    heating_liste=[]
    global i
    for run in range(0,numberOfSimulations):       
        print ("running simulation" + str(run))
        start_time = time.time()
        #modify_SimulationConfig(IDFPATH,run)
        runner = EPlusRunner(EPLUSPATH)
        simulation = runner.run_one(model, EPWFILE, extra_files=learning_files, custom_process=save_result)
        i+=1
        #result=simulation.time_series # doesn't work if we define custom_process (to be included in the save_result function)
        #heating = float(heating_needs(result))
        #print(heating)
        #heating_liste.append(heating)
        #p = subprocess.Popen(['C:/EnergyPlusV9-5-0/EnergyPlus.exe','-w', "CHAMBERY.epw", 'IDM_NoMASS.idf'], cwd=IDFPATH)
        #p.communicate()
        #copyfile(IDFPATH+'/eplustbl.htm', IDFPATH + "/results/eplustbl" + str(proc)+'.htm')
        #copyfile(location+'\\NoMASS.out', location + "\\results\\NoMASS" + str(run)+'.out')
        print("run %s in %s s" %(run, time.time() - start_time))
    print ("%s simulations done" %numberOfSimulations)
    print (heating_liste)
    return (heating_liste)  
    

if __name__ == "__main__":
    IDDPATH = config.IDDPATH
    EPLUSPATH = config.EPLUSPATH
    IDFPATH = "C:/Users/se266887/No-MASS/Configuration/Officelearn/"
    ResultsPATH = IDFPATH + "results/"
    if not os.path.exists(ResultsPATH):
        os.makedirs(ResultsPATH)
    #IDFPATH = r"C:\Users\\se266887\No-MASS\Configuration\OfficeLearn\\" #"C:/Users/se266887/Desktop/modelNoMASS/"
    #EPWFILE = IDFPATH + "CHAMBERY.epw"
    EPWFILE = IDFPATH + "in.epw"
    #model_name = 'IDM_NoMASS.idf'
    model_name = 'in.idf'
    fmu_name = "agentFMU.fmu" 
    i=0
    numberOfSimulations=1
    runmulti(IDFPATH, EPWFILE, IDDPATH, EPLUSPATH, model_name, fmu_name, numberOfSimulations)
    
