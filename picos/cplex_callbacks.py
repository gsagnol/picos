#-----------------------------------------------------------#
#       This code is a modification of the example mipex4.py
#       which comes with the cplex distrubution
#

from __future__ import print_function

import cplex
from cplex.callbacks import MIPInfoCallback, IncumbentCallback, BranchCallback, NodeCallback
import time

class nbIncCallback(IncumbentCallback):
        
        def __call__(self):
                if not self.aborted:
                        self.cursol+=1
                if self.cursol == self.nbsol:
                        self.aborted=1
                        self.abort()

class nbSolCallback(BranchCallback):
        
        def __call__(self):
                if not self.aborted:
                        if self.is_integer_feasible():
                                self.cursol+=1
                if self.cursol == self.nbsol:
                        self.aborted=1
                        self.abort()

class PicosInfoCallback(MIPInfoCallback):
        def __call__(self):
                v1 = self.get_incumbent_objective_value()
                v2 = self.get_best_objective_value()
                self.ub=max(v1,v2)
                self.lb=min(v1,v2)
                timeused = time.time() - self.starttime
                if not self.bounds is None:
                        self.bounds.append((timeused,self.lb,self.ub))
                if (self.lbound is not None) and (self.lb > self.lbound):
                        print("specified lower bound reached, quitting")
                        self.aborted=1
                        self.abort()
                if (self.ubound is not None) and (self.ub < self.ubound):
                        print("specified upper bound reached, quitting")
                        self.aborted=1
                        self.abort()
                if (self.timelimit is not None):
                        gap = 100.0 * self.get_MIP_relative_gap()
                        if timeused > self.timelimit and (
                                (self.acceptablegap is None) or (gap < self.acceptablegap)):
                                print("Good enough solution at", timeused, "sec., gap =",
                                      gap, "%, quitting.")
                                self.aborted = 1
                                self.abort()                  
                        

""" older version, several MIPInfoCallback was not possible?
class TimeLimitCallback(MIPInfoCallback):

    def __call__(self):
        if not self.aborted and self.has_incumbent():
            gap = 100.0 * self.get_MIP_relative_gap()
            timeused = time.time() - self.starttime
            if timeused > self.timelimit and (
             (self.acceptablegap is None) or (gap < self.acceptablegap)):
                print "Good enough solution at", timeused, "sec., gap =", \
                      gap, "%, quitting."
                self.aborted = 1
                self.abort()

class lboundCallback(MIPInfoCallback):
        
        def __call__(self):
                if not self.aborted:
                        v1 = self.get_incumbent_objective_value()
                        v2 = self.get_best_objective_value()
                        self.lb = min(v1,v2)
                if self.lb > self.bound:
                        self.aborted=1
                        self.abort()
              
class uboundCallback(MIPInfoCallback):
        
        def __call__(self):
                if not self.aborted:
                        v1 = self.get_incumbent_objective_value()
                        v2 = self.get_best_objective_value()
                        self.ub = max(v1,v2)
                if self.ub < self.bound:
                        self.aborted=1
                        self.abort()
                        
class boundMonitorCallback(MIPInfoCallback):
        def __call__(self):
                v1 = self.get_incumbent_objective_value()
                v2 = self.get_best_objective_value()
                timeused = time.time() - self.starttime
                self.bounds.append((timeused,min(v1,v2),max(v1,v2)))
"""