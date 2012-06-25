#-----------------------------------------------------------#
#       This code is a modification of the example mipex4.py
#       which comes with the cplex distrubution
#

import cplex
from cplex.callbacks import MIPInfoCallback, IncumbentCallback, BranchCallback
import time

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
                        