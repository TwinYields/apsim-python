import pythonnet
if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
import clr
import sys
import numpy as np
import pandas as pd
import shutil
import os
import pathlib

import shutil
apsim_path = shutil.which("Models")
if apsim_path is not None:
    apsim_path = os.path.split(os.path.realpath(apsim_path))[0]
    sys.path.append(apsim_path)
clr.AddReference("Models")
clr.AddReference("System")

# C# imports
import Models
import Models.Core
import Models.Core.ApsimFile
import Models.Core.Run;
import Models.PMF
import System.IO
import System.Linq
from System.Collections.Generic import *
from System import *

#from Models.Core import Zone, Simulations
from Models.PMF import Cultivar
from Models.Core.ApsimFile import FileFormat
from Models.Climate import Weather
from Models.Soils import Soil, Physical, SoilCrop

class APSIMX():

    def __init__(self, apsimx_file, copy=True):
        name, ext = os.path.splitext(apsimx_file)
        if copy:
            copy_path = f"{name}_py{ext}"
            shutil.copy(apsimx_file, copy_path)
            pathlib.Path(f"{name}.db").unlink(missing_ok=True)
            self.path = copy_path
        else:
            self.path = apsimx_file

        self.results = None
        self._Simulation = None #TopLevel Simulations object
        self.simulations = None # List of Simulation object
        self.py_simulations = None
        self.datastore = None
        self.harvest_date = None

        self.load(self.path)
        plant = self._Simulation.FindDescendant[Models.Core.Zone]().Plants[0]
        cultivar = plant.FindChild[Cultivar]()
        self.cultivar_command = self._cultivar_params(cultivar)

    def load(self, path):
        self._Simulation = FileFormat.ReadFromFile[Models.Core.Simulations](path, None, False)
        self.simulations = list(self._Simulation.FindAllChildren[Models.Core.Simulation]())
        self.py_simulations = [Simulation(s) for s in self.simulations]
        self.datastore = self._Simulation.FindChild[Models.Storage.DataStore]().FileName
        self._DataStore = self._Simulation.FindChild[Models.Storage.DataStore]()

    def save(self, out_path=None):
        if out_path is None:
            out_path = self.path
        json = Models.Core.ApsimFile.FileFormat.WriteToString(self._Simulation)
        with open(out_path, "w") as f:
            f.write(json)

    def run(self, simulations=None):
        # Clear old data before running
        self.results=None
        self._DataStore.Dispose()
        pathlib.Path(self._DataStore.FileName).unlink(missing_ok=True)
        self._DataStore.Open()
        if simulations is None:
            r = Models.Core.Run.Runner(self._Simulation)
        else:
            sims = self.find_simulations(simulations)
            # Runner needs C# list
            cs_sims = List[Models.Core.Simulation]()
            for s in sims:
                cs_sims.Add(s)
            r = Models.Core.Run.Runner(cs_sims)
        e = r.Run()
        if (len(e) > 0):
            print(e[0].ToString())
        self.results = self._read_results()

        try:
            self.harvest_date = self.results.loc[self.results.WheatPhenologyCurrentStageName  == 'HarvestRipe',
                                    ["Zone", "ClockToday"]]
        except:
            self.harvest_date = None

    def _read_results(self):
        df = pd.read_sql_table("Report", "sqlite:///" + self.datastore)
        df = df.rename(mapper=lambda x: x.replace(".", ""), axis=1)
        return df

    """Convert cultivar command to dict"""
    def _cultivar_params(self, cultivar):
        cmd = cultivar.Command
        params = {}
        for c in cmd:
            if c:
                p, v = c.split("=")
                params[p.strip()] = v.strip()
        return params

    """Update cultivar parameters"""
    def update_cultivar(self, parameters, clear=False):
        for sim in self.simulations:
            zone = sim.FindChild[Models.Core.Zone]()
            cultivar = zone.Plants[0].FindChild[Models.PMF.Cultivar]()
            if clear:
                params = parameters
            else:
                params = self._cultivar_params(cultivar)
                params.update(parameters)
            cultivar.Command = [f"{k}={v}" for k,v in params.items()]
            self.cultivar_command = params

    def show_management(self):
        for sim in self.simulations:
            zone = sim.FindChild[Models.Core.Zone]()
            print("Zone:", zone.Name)
            for action in zone.FindAllChildren[Models.Manager]():
                print("\t", action.Name, ":")
                for param in action.Parameters:
                    print("\t\t", param.Key,":", param.Value)

    def update_management(self, management, zones=None, reload=True):
        for sim in self.simulations:
            zone = sim.FindChild[Models.Core.Zone]()
            if zones is not None and zone.Name.lower() not in zones:
                #print("Not in zones", zone.Name)
                continue
            for action in zone.FindAllChildren[Models.Manager]():
                if action.Name in management:
                    #print("Updating", action.Name)
                    values = management[action.Name]
                    for i in range(len(action.Parameters)):
                        param = action.Parameters[i].Key
                        if param in values:
                            action.Parameters[i] = KeyValuePair[String, String](param, f"{values[param]}")
        # Saved and restored the model to recompile the scripts
        # haven't figured out another way to make it work
        if reload:
            self.save()
            self.load(self.path)

    # Convert CS KeyValuePair to dictionary
    def kvtodict(self, kv):
        return {kv[i].Key : kv[i].Value for i in range(kv.Count)}

    def get_management(self):
        res = []
        for sim in self.simulations:
            actions = sim.FindAllDescendants[Models.Manager]()
            out = {}
            out["simulation"] = sim.Name
            for action in actions:
                params = self.kvtodict(action.Parameters)
                if "FertiliserType" in params:
                    out[params["FertiliserType"]] = float(params["Amount"])
                if "CultivarName" in params:
                    out["crop"] = params["Crop"]
                    out["cultivar"] = params["CultivarName"]
                    out["plant_population"] = params["Population"]

            if len(out) > 1:
                res.append(out)
        return pd.DataFrame(res)

    def set_dates(self, start_time=None, end_time=None):
        for sim in self.simulations:
            clock = sim.FindChild[Models.Clock]()
            if start_time is not None:
                #clock.End = DateTime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
                clock.End = DateTime.Parse(start_time)
            if end_time is not None:
                #clock.End = DateTime(end_time.year, end_time.month, end_time.day, 0, 0, 0)
                clock.End = DateTime.Parse(end_time)

    def set_weather(self, weather_file):
        for weather in self._Simulation.FindAllDescendants[Weather]():
            weather.FileName = weather_file

    def show_weather(self):
        for weather in self._Simulation.FindAllDescendants[Weather]():
            print(weather.FileName)

    def set_report(self, report, simulations = None):
        simulations = self.find_simulations(simulations)
        for sim in simulations:
            r = sim.FindDescendant[Models.Report]()
            r.set_VariableNames(report.strip().splitlines())

    def get_report(self, simulation = None):
        sim = self.find_simulation(simulation)
        report = list(sim.FindAllDescendants[Models.Report]())[0]
        return list(report.get_VariableNames())

    def find_physical_soil(self, simulation = None):
        sim = self.find_simulation(simulation)
        soil = sim.FindDescendant[Soil]()
        psoil = soil.FindDescendant[Physical]()
        return psoil

    # Find a list of simulations by name
    def find_simulations(self, simulations = None):
        if simulations is None:
            return self.simulations
        if type(simulations) == str:
            simulations = [simulations]
        sims = []
        for s in self.simulations:
            if s.Name in simulations:
                sims.append(s)
        if len(sims) == 0:
            print("Not found!")
        else:
            return sims

    # Find a single simulation by name
    def find_simulation(self, simulation = None):
        if simulation is None:
            return self.simulations[0]
        sim = None
        for s in self.simulations:
            if s.Name == simulation:
                sim = s
                break
        if sim is None:
            print("Not found!")
        else:
            return sim

    def get_dul(self, simulation=None):
        psoil = self.find_physical_soil(simulation)
        return np.array(psoil.DUL)

    def set_dul(self, dul, simulation=None):
        psoil = self.find_physical_soil(simulation)
        psoil.DUL = dul

    def set_sat(self, sat, simulation=None):
        psoil = self.find_physical_soil(simulation)
        psoil.SAT = sat
        psoil.SW = psoil.DUL

    def get_sat(self, simulation=None):
        psoil = self.find_physical_soil(simulation)
        return np.array(psoil.SAT)

    def get_ll15(self, simulation=None):
        psoil = self.find_physical_soil(simulation)
        return np.array(psoil.LL15)

    def set_ll15(self, ll15, simulation=None):
        psoil = self.find_physical_soil(simulation)
        psoil.LL15 = ll15

    def get_crop_ll(self, simulation=None):
        psoil = self.find_physical_soil(simulation)
        sc = psoil.FindChild[SoilCrop]()
        return np.array(sc.LL)

    def set_crop_ll(self, ll, simulation=None):
        psoil = self.find_physical_soil(simulation)
        sc = psoil.FindChild[SoilCrop]()
        sc.LL = ll

    def get_soil(self):
        sat = self.get_sat()
        dul = self.get_dul().copy()
        ll15 = self.get_ll15().copy()
        cll = self.get_crop_ll()
        return pd.DataFrame({"LL15" : ll15, "DUL" : dul, "Crop LL" : cll, "SAT" : sat})

class Simulation(object):

    def __init__(self, simulation):
        self.simulation = simulation
        self.zones = [Zone(z) for z in simulation.FindAllChildren[Models.Core.Zone]()]

    def find_physical_soil(self):
        soil = self.simulation.FindDescendant[Soil]()
        psoil = soil.FindDescendant[Physical]()
        return psoil

    # TODO should these be linked to zones instead?
    def get_dul(self):
        psoil = self.find_physical_soil()
        return np.array(psoil.DUL)

    def set_dul(self, dul):
        psoil = self.find_physical_soil()
        psoil.DUL = dul

class Zone(object):

    def __init__(self, zone):
        self.zone = zone
        self.name = zone.Name
        self.soil = self.zone.FindDescendant[Soil]()
        self.physical_soil = self.soil.FindDescendant[Physical]()

    # TODO should these be linked to zones instead?
    @property
    def dul(self):
        return np.array(self.physical_soil.DUL)

    @dul.setter
    def dul(self, dul):
        self.physical_soil.DUL = dul
















