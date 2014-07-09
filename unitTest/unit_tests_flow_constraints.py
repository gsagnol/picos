# Test Picos
import unittest
import picos as pic
import cvxopt as cvx
import networkx as nx

class PicosFlowConstraintsTests(unittest.TestCase):

	def testFlowConstraintSubProbConstraintNumber(self):
		G = nx.DiGraph()
		G.add_edge('S','A', capacity=1); G.add_edge('A','B', capacity=1); G.add_edge('B','T', capacity=1)
		pb = pic.Problem()
		f={}
		for e in G.edges():
			f[e]=pb.add_variable('f[{0}]'.format(e),1)
		F = pb.add_variable('F',1)
		flowCons = pic.flow_Constraint(G, f, source='S', sink='T', capacity='capacity', flow_value= F, graphName='G')
		self.assertEqual(len(flowCons.Ptmp.constraints), 10, 'Number of constraint in the subproblem is wrong.')

	# check the output string of a flow constraint
	def testFlowConstraintStringOutput(self):
		G = nx.DiGraph()
		G.add_edge('S','A', capacity=1); G.add_edge('A','B', capacity=1); G.add_edge('B','T', capacity=1)
		pb = pic.Problem()
		f={}
		for e in G.edges():
			f[e]=pb.add_variable('f[{0}]'.format(e),1)
		F = pb.add_variable('F',1)
		flowCons = pic.flow_Constraint(G, f, source='S', sink='T', capacity='capacity', flow_value= F, graphName='G')
		self.assertEqual(str(flowCons), '# flow constraint : Flow conservation in G from S to T with value F#', 'String output of flow constraint is wrong.')

	# Compute a trivial flow constraint problem
	def testFlowConstraintProblemSolution(self):
		G = nx.DiGraph()
		G.add_edge('S','A', capacity=1); G.add_edge('A','B', capacity=1); G.add_edge('B','T', capacity=1)
		pb = pic.Problem()
		f={}
		for e in G.edges():
			f[e]=pb.add_variable('f[{0}]'.format(e),1)
		F = pb.add_variable('F',1)
		flowCons = pic.flow_Constraint(G, f, source='S', sink='T', capacity='capacity', flow_value= F, graphName='G')
		pb.addConstraint(flowCons)
		pb.set_objective('max',F)
		sol = pb.solve(verbose=0)
		flow = pic.tools.eval_dict(f)
		self.assertEqual(str(flow), "{('S', 'A'): 1.0, ('A', 'B'): 1.0, ('B', 'T'): 1.0}", 'Cannot compute solve a simple flow problem.')

	def testFlowConstraintMultiSinkProblemSolution(self):
		G = nx.DiGraph()
		G.add_edge('S','A', capacity=5); G.add_edge('S','B', capacity=5); G.add_edge('A','T1', capacity=1); G.add_edge('B','T2', capacity=3)
		pb = pic.Problem()
		f={}
		for e in G.edges():
			f[e]=pb.add_variable('f[{0}]'.format(e),1)
		F1 = pb.add_variable('F1',1)
		F2=pb.add_variable('F2',1)
		flowCons = pic.flow_Constraint(G, f, source='S', sink=['T1', 'T2'], capacity='capacity', flow_value= [F1, F2], graphName='G')
		pb.addConstraint(flowCons)
		pb.set_objective('max',F1+F2)
		sol = pb.solve(verbose=0)
		flow = pic.tools.eval_dict(f)
		self.assertEqual(str(flow), "{('S', 'A'): 1.0, ('S', 'B'): 3.0, ('B', 'T2'): 3.0, ('A', 'T1'): 1.0}", 'Cannot compute solve a simple multisink problem.')

	def testFlowConstraintMultiSourceProblemSolution(self):
		G = nx.DiGraph()
		G.add_edge('S1','A', capacity=5); G.add_edge('S2','B', capacity=5); G.add_edge('A','T', capacity=4); G.add_edge('B','T', capacity=3)
		pb = pic.Problem()
		f={}
		for e in G.edges():
			f[e]=pb.add_variable('f[{0}]'.format(e),1)
		F1 = pb.add_variable('F1',1)
		F2=pb.add_variable('F2',1)
		flowCons = pic.flow_Constraint(G, f, source=['S1', 'S2'], sink='T', capacity='capacity', flow_value= [F1, F2], graphName='G')
		pb.addConstraint(flowCons)
		pb.set_objective('max',F1+F2)
		sol = pb.solve(verbose=0)
		flow = pic.tools.eval_dict(f)
		self.assertEqual(str(flow), "{('S1', 'A'): 4.0, ('S2', 'B'): 3.0, ('A', 'T'): 4.0, ('B', 'T'): 3.0}", 'Cannot compute solve a simple multisource problem.')

        
def main():
    unittest.main()


if __name__ == '__main__':
    main()
    
