import numpy as np
import pandas as pd
import nlopt

class ApsimOptimizer(object):

    def __init__(self, model, params, harvest_date, s2_lai, zones, zone_names = None):
        if zone_names is not None:
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
            print(f"Iteration {self._iter}, cost: LAI {clai:.1f} Yield {cyield:.1f}, Harvest {charvest:.1f}, Total {cost:.1f}")
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
        self.model.run(self.zone_names)

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

    def optimize(self, alg = nlopt.GN_DIRECT_L, maxeval = 5, normalized=True):
        print(f"Optimizing {self.N} parameters")

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
