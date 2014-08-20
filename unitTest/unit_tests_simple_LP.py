# Test Picos
import unittest
import picos as pic
import cvxopt as cvx
import networkx as nx

class PicosSimpleLPTests(unittest.TestCase):
        
        def test_trivial_unfeasibility(self):
                import picos as pic
                P = pic.Problem()
                x = P.add_variable('x',1)
                P.add_constraint(x<1)
                #0==1
                P.add_constraint(pic.sum([x for i in []])==1)
                P.set_objective('max',x)
                for sol in pic.tools.available_solvers():
                        infeasible = False
                        try:
                                P.solve(solver = sol,verbose=0)
                                if 'infeas' in P.status:
                                        infeasible = True
                        except:
                                infeasible = True
                        self.assertEqual(infeasible,True, 'infeasibility not detected with '+sol)
                




def main():
    unittest.main()


if __name__ == '__main__':
    main()
    