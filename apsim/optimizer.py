import numpy as np
import pandas as pd
import nlopt

class OptimizerBase(object):

    def __init__(self, model, params, obs_yield, obs_lai, harvest_date, zone_names = None):

        obs_yield.rename(str.lower, axis=1, inplace=True)
        obs_lai.rename(str.lower, axis=1, inplace=True)

        obs_lai.columns = obs_lai.columns.str.replace(r".*lai.*", "obs_lai", regex=True)
        obs_yield.columns = obs_yield.columns.str.replace(r".*yield.*", "obs_yield", regex=True)

        if zone_names is not None:
            if type(zone_names) == str:
                zone_names = [zone_names]

            obs_yield = obs_yield[obs_yield.zone.apply(lambda x : x in zone_names)]
            obs_lai = obs_lai[obs_lai.zone.apply(lambda x : x in zone_names)]

        self.zone_names = zone_names
        self.model = model
        self.obs_lai = obs_lai
        self.obs_yield = obs_yield
        self.harvest_date = harvest_date

        self.optvars = sorted(params.keys())
        self.params = params.copy()
        self.N = len(self.optvars)

        self._iter = 0
        self.print_interval = 10
        self.log_level = 1

    def _print(self, clai, cyield, charvest, cost):
        if (self._iter % self.print_interval) == 0:
            print(f"\tIteration {self._iter}, cost: LAI {clai:.3f} Yield {cyield:.3f}, Harvest {charvest:.3f}, Total {cost:.3f}")

    def merge_sim_obs(self):
        df = self.model.results
        df.rename(str.lower, axis=1, inplace=True)
        df["sim_lai"] = df["lai"]
        sim_lai = df[["clocktoday", "zone", "sim_lai"]]
        sim_lai = self.obs_lai.merge(sim_lai, how="left")

        sim_yield = df[["zone", "yield", "lai"]].groupby("zone", as_index=False).agg(np.max)
        sim_yield.rename({"yield" : "sim_yield", "lai" : "sim_lai"}, axis=1, inplace=True)
        sim_yield =  sim_yield.merge(self.obs_yield)

        htime = self.model.harvest_date
        if htime is None:
            print("Out of range")
            harvest_time = df["clocktoday"].max()
        else:
            harvest_time = htime.iloc[0]["ClockToday"]
        return sim_yield, sim_lai, harvest_time

    def cost_function(self, p, grad = []):
        self.step(p, grad)
        self._iter += 1
        sim_yield, lai_df, harvest_time = self.merge_sim_obs()
        # Calculate errors
        e_lai = (lai_df.sim_lai - lai_df.obs_lai)
        e_yield = (sim_yield.sim_yield - sim_yield.obs_yield)
        e_harvest = (harvest_time - self.harvest_date).total_seconds() /(24*60*60)
        # Apply cost function
        return self.scaled_lsq(e_lai, e_yield, e_harvest)

    def scaled_lsq(self, e_lai, e_yield, e_harvest):
        cost_lai = np.mean((e_lai)**2)/np.max(self.obs_lai["obs_lai"])
        cost_yield = np.mean(np.abs(e_yield))/np.max(self.obs_yield["obs_yield"].max())
        cost_harvest = abs(e_harvest) / 7 # Error or 1 week in same scale with other variables
        total_cost =  cost_lai + cost_yield + cost_harvest
        if self.log_level > 0:
            self._print(cost_lai, cost_yield, cost_harvest, total_cost)
        return total_cost

    def rand_start(self, r):
        l, h = r
        vs = np.linspace(l,h, 10)
        idx = np.random.randint(0, len(vs))
        return vs[idx]

    @property
    def optimized_values(self):
        yld, lai, htime = self.merge_sim_obs()
        ovs = pd.DataFrame({self.optvars[i] :self.opt_values[i] for i in range(self.N)}, index=[0])
        yld = pd.concat([yld, ovs], axis=1)
        return yld, lai, htime

    def optimize(self, alg = nlopt.GN_DIRECT_L, maxeval = 5):
        if self.log_level > 0:
            print(f"Optimizing {self.N} parameters, max {maxeval} iterations")
        opt = nlopt.opt(alg, self.N)
        opt.set_min_objective(self.cost_function)

        opt.set_lower_bounds([self.params[v][0] for v in self.optvars])
        opt.set_upper_bounds([self.params[v][1] for v in self.optvars])

        opt.set_maxeval(maxeval)
        init = [self.rand_start(self.params[v]) for v in self.optvars]

        self.opt = opt
        self.opt_values = opt.optimize(init)

        self.step(self.opt_values)
        if self.log_level > 0:
            print(f"Done after {self._iter} iterations")
            print(str.join(", ", [f"{self.optvars[i]} = {self.opt_values[i]:.2f}" for i in range(self.N)]))


class SoilOptimizer(OptimizerBase):

        def __init__(self, *args, **kwargs):
            super(SoilOptimizer, self).__init__(*args, **kwargs)

            #Save starting values
            self.ll15 = self.model.get_ll15().copy()
            self.dul = self.model.get_dul().copy()
            self.sat = self.model.get_sat().copy()
            self.cll = self.model.get_crop_ll().copy()
            self.initial_no3 = self.model.get_initial_no3().copy()
            self.initial_nh4 = self.model.get_initial_nh4().copy()
            self.initial_urea = self.model.get_initial_urea().copy()

        def step(self, p, grad=[]):
            for zone in self.zone_names:
                for i in range(self.N):
                    v = self.optvars[i]
                    if v == "ll15":
                        self.model.set_ll15(self.ll15 + p[i], zone)
                    if v == "dul":
                        new_dul = self.dul + p[i]
                        self.model.set_dul(new_dul, zone)
                        self.model.set_sat(self.sat + p[i], zone)
                        #Make sure crop LL is below DUL
                        tmp_cll = self.cll.copy()
                        for j in range(len(tmp_cll)):
                            if tmp_cll[j] >= new_dul[j]:
                                tmp_cll[j] = new_dul[j] - 0.01
                            self.model.set_crop_ll(tmp_cll, zone)
                    if v.lower() == "no3":
                        self.model.set_initial_no3(self.initial_no3 + p[i])
                    if v.lower() == "nh4":
                        self.model.set_initial_nh4(self.initial_nh4 + p[i])
                    if v.lower() == "urea":
                        self.model.set_initial_urea(self.initial_urea + p[i])

            self.model.run(self.zone_names)

class ApsimOptimizer(object):

    def __init__(self, model, params, harvest_date, s2_lai, zones, zone_names = None):
        if zone_names is not None:
            if type(zone_names) == str:
                zone_names = [zone_names]
            zones = zones[zones.zone.apply(lambda x : x in zone_names)]

        self.zone_names = zone_names
        self.model = model
        self.cultivar_vars = sorted(params.keys())
        self.ll_vars = sorted(list(zones.zone))
        self.optvars = self.cultivar_vars + self.ll_vars

        self.N = len(self.optvars)

        self.Nc = len(self.cultivar_vars)
        self.Nll = len(self.ll_vars)

        self.cultivar_params = params
        self.ll_params = {}
        for i in range(self.Nll):
            #self.ll_params[self.ll_vars[i]] = (0, 50) # divided by 1000 when set to APSIM
            self.ll_params[self.ll_vars[i]] = (-60, 20) # divided by 1000 when set to APSIM

        self.params = params.copy()
        self.params.update(self.ll_params)

        # Average LL15 across zones
        self.ll15 = self.model.get_ll15().copy()
        self.dul = self.model.get_dul().copy()
        self.sat = self.model.get_sat().copy()
        self.cll = self.model.get_crop_ll().copy()

        # Fork to check the effect of DUL
        #self.model.set_ll15(self.ll15 + 0.02)

        #np.random.randint(0, 7)/100
        self.s2_lai = s2_lai[["Zone", "ClockToday", "LAI"]]
        self.zones = zones

        self.opt_values = None
        self.harvest_date = harvest_date
        self.print_count = 0
        self._iter = 0

    def range(self, x):
        return np.max(x) - np.min(x)

    def update_ll15(self, p):
        #print("LL: ", len(p), np.std(p))
        for i in range(len(p)):
            #self.model.set_ll15(self.ll15 + (p[i]/1000), self.ll_vars[i])
            self.model.set_dul(self.dul + (p[i]/1000), self.ll_vars[i])
            self.model.set_sat(self.sat + (p[i]/1000), self.ll_vars[i])
            tmp_cll = self.cll.copy()
            tmp_cll[-1] = (self.dul + (p[i]/1000)).min() - 0.01
            self.model.set_crop_ll(tmp_cll, self.ll_vars[i])


    def _print(self, clai, cyield, charvest, cost):
        self._iter += 1
        if self.print_count == 9:
            print(f"\tIteration {self._iter}, cost: LAI {clai:.3f} Yield {cyield:.3f}, Harvest {charvest:.3f}, Total {cost:.3f}")
            self.print_count = 0
        else:
            self.print_count += 1

    def merge_sim_obs(self):
        df = self.model.results
        df["sim_LAI"] = df["WheatLAI"]
        sim_lai = df[["ClockToday", "Zone", "sim_LAI"]]

        lai_df = self.s2_lai.merge(sim_lai, how="left")
        sim_yield = df[["Zone", "Yield", "WheatLAI"]].groupby("Zone", as_index=False).agg(np.max)
        sim_yield.rename({"Zone" : "zone", "Yield" : "sim_yield", "WheatLAI" : "sim_LAI"}, axis=1, inplace=True)
        sim_yield =  sim_yield.merge(self.zones)
        htime = df.loc[df.WheatPhenologyCurrentStageName  == 'HarvestRipe', "ClockToday"]
        if len(htime) > 0:
            harvest_time = htime.iloc[0]
        else:
            print("Out of range")
            harvest_time = pd.Timestamp("2022-10-31")
        return sim_yield, lai_df, harvest_time

    def normalized_cost_function(self, p, grad = []):
        c = {self.optvars[i] : p[i] for i in range(self.Nc)}
        if self.Nc > 0:
            self.model.update_cultivar(c)
        self.update_ll15(p[self.Nc : self.N])
        self.model.run(self.zone_names, clean=False)

        sim_yield, lai_df, harvest_time = self.merge_sim_obs()
        e_lai = (lai_df.sim_LAI - lai_df.LAI)
        e_yield = (sim_yield.sim_yield - (sim_yield["yield_kg"])) # We prefer over estimation of yield instead of underestimation
        e_harvest = (self.harvest_date - harvest_time).total_seconds() /(24*60*60)
        clai = np.mean((e_lai)**2)/np.max(lai_df.LAI)    #self.range(lai_df.LAI)
        cyield = np.mean(np.abs(e_yield))/np.max(sim_yield["yield_kg"])  #self.range(sim_yield["yield_kg"])
        charvest = abs(e_harvest) / 7 # Error or 1 week in same scale with other variables
        cost =  clai + cyield + charvest
        self._print(clai, cyield, charvest, cost)
        return cost

    def _optimized_model(self):
        c = {self.optvars[i] : self.opt_values[i] for i in range(self.Nc)}
        if self.Nc > 0:
            self.model.update_cultivar(c)
        self.update_ll15(self.opt_values[self.Nc : self.N])

    def get_lls(self):
        lls = []
        for i in range(self.Nc, self.N):
            ll = {"zone" : self.optvars[i], "ll15_delta" : self.opt_values[i]/1000}
            lls.append(ll)
        return pd.DataFrame(lls)

    def rand_start(self, r):
        l, h = r
        vs = np.linspace(l,h, 10)
        idx = np.random.randint(0, len(vs))
        return vs[idx]

    def optimize(self, alg = nlopt.GN_DIRECT_L, maxeval = 5):
        print(f"Optimizing {self.N} parameters, max {maxeval} iterations")

        opt = nlopt.opt(alg, self.N)
        opt.set_min_objective(self.normalized_cost_function)

        opt.set_lower_bounds([self.params[v][0] for v in self.optvars])
        opt.set_upper_bounds([self.params[v][1] for v in self.optvars])

        opt.set_maxeval(maxeval)
        init = [self.rand_start(self.params[v]) for v in self.optvars]
        #print(init)
        self.opt = opt
        self.opt_values = opt.optimize(init)
        self._optimized_model()
        self.model.run(self.zone_names)
        print("Done!")
