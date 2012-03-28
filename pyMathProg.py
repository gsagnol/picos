# coding: utf-8
import cvxopt as cvx
import numpy as np
import sys
from progress_bar import ProgressBar

global MSK_INFINITY
MSK_INFINITY=1e16


#----------------------------------------------------
#        Grouping constraints, summing expressions
#----------------------------------------------------

def sum(lst,it=None,indices=None):
        import __builtin__
        affSum=__builtin__.sum(lst)
        if not it is None:
                sumstr='_'
                if  not indices is None:
                        sumstr+='{'
                if isinstance(it,tuple) and len(it)==2 and isinstance(it[1],int):
                        it=(it,)
                if isinstance(it,list):
                        it=tuple(it)                
                if not isinstance(it,tuple):
                        it=(it,)
                if isinstance(it[0],tuple):
                        sumstr+=str(it[0][0])
                else:
                        sumstr+=str(it[0])
                for k in [k for k in range(len(it)) if k>0]:
                        if isinstance(it[k],tuple):
                                sumstr+=','+str(it[k][0])
                        else:
                                sumstr+=','+str(it[k])
                if not indices is None:
                        sumstr+=' in '+indices+'}'
                try:
		  indstr=putIndices([l.affstring() for l in lst],it)
                except Exception:
		  indstr='['+str(len(lst))+' expressions (first: '+lst[0].string+')]'
                sumstr+=' '+indstr
                sigma='Σ' #'u'\u03A3'.encode('utf-8')
                affSum.string=sigma+sumstr
        return affSum

def lse(exp):
	"""log-sum-exp"""
	if isinstance(exp,AffinExpr):
		return LogSumExp(exp)
	else:
		term,termString=_retrieve_matrix(exp,None)
		Exp=AffinExpr(factors={},constant=term,size=term.size,string=termString,variables=self.variables)
		return lse(Exp)

def allIdent(lst):
        if len(lst)<=1:
                return(True)
        return (np.array([lst[i]==lst[i+1] for i in range(len(lst)-1)]).all() )

def findEndOfInd(string,curInd,curIndName=''):
        if curIndName=='':
                return findEndOfInd(string,curInd+1,string[curInd])
        if curIndName[0]=="'":
                if (string[curInd]=="'"):
                        return (curInd+1,curIndName+"'")
        if curIndName[0]=='"':
                if (string[curInd]=='"'):
                        return (curInd+1,curIndName+'"')
        if (curIndName[0].isdigit() or curIndName[0]=='.'):
                if (curInd>=len(string)) or (not (string[curInd].isdigit() or string[curInd]=='.')):
                        return (curInd,curIndName)
        return findEndOfInd(string,curInd+1,curIndName+string[curInd])
                

def isFirstCharIndex(st):
        return (st in ["'",'"','.'] or st.isdigit() )

def onlyFirstIndices(lst):
        return (np.array([isFirstCharIndex(lst[i]) for i in range(len(lst))]).all() )

def putIndices(lsStrings,it):
        #for multiple indices
        toMerge=[]
        for k in it:
                if isinstance(k,tuple):
                        itlist=list(it)
                        ik=itlist.index(k)
                        itlist.remove(k)
                        for i in range(k[1]):
                                itlist.insert(ik,k[0]+'__'+str(i))
                                ik+=1
                        toMerge.append((k[0],itlist[ik-k[1]:ik]))
                        it=tuple(itlist)
        #main function
        n=len(lsStrings)
        curInd=n*[0]
        frame=''
        namedIndex=[]
        tmpName=[]
        foundIndices=0
        for k in range(n):
                namedIndex.append(len(it)*[None])
                tmpName.append(None)
        while curInd[0]<len(lsStrings[0]):
                #import pdb;pdb.set_trace()
                currentFramePiece=''
                while allIdent([lsStrings[k][curInd[k]] for k in range(n)]):
                        currentFramePiece+=lsStrings[0][curInd[0]]
                        curInd=[c+1 for c in curInd]
                        if curInd[0]>=len(lsStrings[0]):
                                break
                #patch for sub-sum of length 1:
                if (curInd[0]<len(lsStrings[0]) ) and (
                         not onlyFirstIndices([lsStrings[k][curInd[k]] for k in range(n)]) ):
                        listIndices=[i for i in range(n) if isFirstCharIndex(lsStrings[i][curInd[i]]) ]
                        listNotIndices=[i for i in range(n) if not(
                                isFirstCharIndex(lsStrings[i][curInd[i]])) ]
                        endOfInd={}
                        nextChar=None
                        for k in listIndices:
                                endOfInd[k]=findEndOfInd(lsStrings[k],curInd[k])[0]
                                if nextChar is None:
                                        nextChar=lsStrings[k][endOfInd[k]]
                                else:
                                        if nextChar!=lsStrings[k][endOfInd[k]]:
                                                raise Exception('found a different char')
                        indString=None
                        for j in listNotIndices:
                                iendj=lsStrings[j].index(nextChar,curInd[j])
                                if indString is None:
                                        indString=lsStrings[j][curInd[j]:iendj]
                                else:
                                        if indString!=lsStrings[j][curInd[j]:iendj]:
                                                raise Exception('found a different index')
                        for k in listIndices:
                                lsStrings[k]=lsStrings[k][:curInd[k]]+indString+lsStrings[k][endOfInd[k]:]
                        frame+=currentFramePiece                        
                        continue
                if (curInd[0]<len(lsStrings[0]) ):
                        while ( np.array([lsStrings[i][curInd[i]-1].isdigit() for i in range(
                                        len(lsStrings))]).all()   ): #last char was a digit                
                                currentFramePiece=currentFramePiece[:-1]
                                curInd=[c-1 for c in curInd]
                frame+=currentFramePiece
                if curInd[0]<len(lsStrings[0]):
                        #import pdb;pdb.set_trace()
                        currentFramePiece=''
                        for k in range(n):
                                curInd[k],tmpName[k]=findEndOfInd(lsStrings[k],curInd[k])
                        for ind in range(foundIndices):
                                if ([namedIndex[k][ind] for k in range(n)]==tmpName):
                                        currentFramePiece=it[ind]
                        if currentFramePiece=='':
                                #previous index not found
                                for k in range(n):
                                        namedIndex[k][foundIndices]=tmpName[k]
                                currentFramePiece=it[foundIndices]
                                foundIndices+=1
                        frame+=currentFramePiece
        #merge multiple indices
        import re
        import string
        for x in toMerge:
                rexp='(\(( )*'+string.join(x[1],',( )*')+'( )*\)|('+string.join(x[1],',( )*')+'))'
                m=re.search(rexp,frame)
                while(m):
                        frame=frame[:m.start()]+x[0]+frame[m.end():]
                        m=re.search(rexp,frame)
        return frame

        
        
                
def eval_dict(dict_of_variables):
        """evaluates all the variables in the dictionary
        and returns the same dictionary, but evaluated"""
        for k in dict_of_variables:
                dict_of_variables[k] = dict_of_variables[k].eval()
                if dict_of_variables[k].size == (1,1):
                        dict_of_variables[k] = dict_of_variables[k][0]
        return dict_of_variables



#---------------------------------------------
#        Tools of the interface
#---------------------------------------------

def blocdiag(X,n,sub1=0,sub2='n'):
        """
        makes diagonal blocs of X, for indices in [sub1,sub2[
        n indicates the total number of blocks (horizontally)
        """
        if sub2=='n':
                sub2=n
        ''' OLD VERSION (inefficient)
        zz=cvx.spmatrix([],[],[],(X.size[0],X.size[1]))
        mat=[]
        for i in range(n):
                col=[]
                for k in range(n):
                        if (k>=sub1 and k<sub2):
                                if (i==k):
                                        col.append(X)
                                else:
                                        col.append(zz)
                mat.append(col)
        return cvx.sparse(mat)
        '''
        if not isinstance(X,cvx.base.spmatrix):
                X=cvx.sparse(X)
        I=[]
        J=[]
        V=[]
        i0=0
        for k in range(sub1,sub2):
                I.extend([xi+i0 for xi in X.I])
                J.extend([xj+X.size[1]*k for xj in X.J])
                V.extend(X.V)
                i0+=X.size[0]
        return cvx.spmatrix(V,I,J,(i0,X.size[1]*n))



def _retrieve_matrix(mat,exSize=None):
        """
        parses the variable *mat* and convert it to a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
        If the variable **exSize** is provided, the function tries
        to return a matrix that matches this expected size, or raise an
        error.
        
        .. WARNING:: If there is a conflit between the size of **mat** and
                     the expected size **exsize**, the function might still
                     return something without raising an error !
        
        .. todo:: Better Exception handling
        
        :param mat: The value to be converted into a cvx.spmatrix.
                    The function will try to parse this variable and
                    format it to a vector/matrix. *mat* can be of one
                    of the following types:
                             
                        * ``list`` [creates a vecor of dimension len(list)]
                        * :func:`cvxopt matrix <cvxopt:cvxopt.matrix>`
                        * :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`
                        * :func:`numpy array <numpy:numpy.array>`
                        * ``int`` or ``real`` [creates a vector/matrix of the size exSize *(or of size (1,1) if exSize is None)*,
                          whith all entries equal to **mat**.
                        * following strings: 
                                
                                * '``|a|``' for a matrix with all terms equal to a
                                * '``|a|(n,m)``' for a matrix forced to be of size n x m, with all terms equal to a
                                * '``e_i(n,m)``' matrix of size (n,m), with a 1 on the ith coordinate (and 0 elsewhere)
                                * '``e_i,j(n,m)``' matrix  of size (n,m), with a 1 on the (i,j)-entry (and 0 elsewhere)
                                * '``I``' for the identity matrix
                                * '``I(n)``' for the identity matrix, forced to be of size n x n.
                                * '``a%s``', where ``%s`` is one of the above string: the matrix that 
                                  should be returned when **mat** == ``%s``, multiplied by the scalar a.
        :returns: A tuple of the form (**M**, **s**), where **M** is the conversion of **mat** into a
                  :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`, and **s**
                  is a string representation of **mat**
        
        **Example:**
        
        >>> import pyMathProg as MP
        >>> MP._retrieve_matrix([1,2,3])
        (<3x1 sparse matrix, tc='d', nnz=3>, '[ 3 x 1 MAT ]')
        >>> MP._retrieve_matrix('e_5(7,1)')
        (<7x1 sparse matrix, tc='d', nnz=1>, 'e_5')
        >>> print MP._retrieve_matrix('e_11(7,2)')[0] #doctest: +NORMALIZE_WHITESPACE
        [   0        0       ]
        [   0        0       ]
        [   0        0       ]
        [   0        0       ]
        [   0        1.00e+00]
        [   0        0       ]
        [   0        0       ]
        >>> print MP._retrieve_matrix('5.3I',(2,2))
        (<2x2 sparse matrix, tc='d', nnz=2>, '5.3I')

        """
        retstr=None
        if isinstance(mat,np.ndarray):
                retmat=cvx.matrix(mat,tc='d')
        elif isinstance(mat,cvx.base.matrix):
                if mat.typecode=='d':
                        retmat=mat
                else:
                        retmat=cvx.matrix(mat,tc='d')
        elif isinstance(mat,cvx.base.spmatrix):
                retmat=mat
        elif isinstance(mat,list):
                retmat=cvx.matrix(np.array(mat),tc='d')
        elif (isinstance(mat,float) or isinstance(mat,int) or isinstance(mat,np.float64) ):
                if isinstance(mat,np.float64):
                        mat=float(mat)
                if mat==0:
                        if exSize is None:
                                #no exSize-> scalar
                                retmat=cvx.matrix(0,(1,1))
                        elif isinstance(exSize,int):
                                #exSize is an int -> 0 * identity matrix
                                retmat=cvx.spmatrix([],[],[], (exSize,exSize) )
                        elif isinstance(exSize,tuple):
                                #exSize is a tuple -> zeros of desired size
                                retmat=cvx.spmatrix([],[],[], exSize )
                        retstr=''
                else:
                        if exSize is None:
                                #no exSize-> scalar
                                retmat=cvx.matrix(mat,(1,1))
                        elif isinstance(exSize,int):
                                #exSize is an int -> alpha * identity matrix
                                retmat=mat*cvx.spdiag([1.]*exSize)
                        elif isinstance(exSize,tuple):
                                #exSize is a tuple -> zeros of desired size
                                retmat=mat*cvx.matrix(1., exSize )
                        retstr=str(mat)
        elif isinstance(mat,str):
                retstr=mat
                if mat[0]=='-':
                        alpha=-1.
                        mat=mat[1:]
                else:
                        alpha=1.
                ind=1
                try:
                        while True:
                                junk=float(mat[:ind])
                                ind+=1
                except Exception:
                        ind-=1
                        if ind>0:
                                alpha*=float(mat[:ind])
                        mat=mat[ind:]
                transpose=False
                if mat[-2:]=='.T':
                        transpose=True
                        mat=mat[:-2]
                #|alpha| for a matrix whith all alpha
                #|alpha|(n,m) for a matrix of size (n,m)
                if (mat.find('|')>=0):
                        i1=mat.find('|')
                        i2=mat.find('|',i1+1)
                        if i2<0:
                                raise Exception('There should be a 2d bar')
                        fact=float(mat[i1+1:i2])
                        i1=mat.find('(')
                        if i1>=0:
                                i2=mat.find(')')
                                ind=mat[i1+1:i2]
                                i1=ind.split(',')[0]
                                #checks
                                try:
                                        i2=ind.split(',')[1]
                                except IndexError:
                                        raise Exception('index of |1| should be i,j')
                                if not i1.isdigit():
                                        raise Exception('first index of |1| should be int')
                                if not i2.isdigit():
                                        raise Exception('second index of |1| should be int')
                                i1=int(i1)
                                i2=int(i2)
                        elif isinstance(exSize,tuple):
                                i1,i2=exSize
                        else:
                                raise Exception('size unspecified')
                        retmat=fact*cvx.matrix(1.,  (i1, i2)  )
                #unit vector
                elif (mat.find('e_')>=0):
                        mspl=mat.split('e_')
                        if len(mspl[0])>0:
                                raise NameError('unexpected case')
                        mind=mspl[1][:mspl[1].index('(')]
                        if (mind.find(',')>=0):
                                idx=mind.split(',')
                                idx=(int(idx[0]),int(idx[1]))
                        else:
                                idx=int(mind)
                        i1=mat.find('(')
                        if i1>=0:
                                i2=mat.find(')')
                                ind=mat[i1+1:i2]
                                i1=ind.split(',')[0]
                                #checks
                                try:
                                        i2=ind.split(',')[1]
                                except IndexError:
                                        raise Exception('index of e_ should be i,j')
                                if not i1.isdigit():
                                        raise Exception('first index of e_ should be int')
                                if not i2.isdigit():
                                        raise Exception('second index of e_ should be int')
                                i1=int(i1)
                                i2=int(i2)
                        elif isinstance(exSize,tuple):
                                i1,i2=exSize
                        else:
                                raise Exception('size unspecified')
                        retmat=cvx.spmatrix([],[],[],(i1,i2) )
                        retmat[idx]=1
                #identity
                elif (mat.startswith('I')):
                        if len(mat)>1 and mat[1]=='(':
                                if mat[-1]!=')':
                                        raise Exception('this string shlud have the format "I(n)"')
                                szstr=mat[2:-1]
                                if not(szstr.isdigit()):
                                        raise Exception('this string shlud have the format "I(n)"')
                                sz=int(szstr)
                                if (not exSize is None) and (
                                        (isinstance(exSize,int) and  exSize!=sz) or
                                        (isinstance(exSize,tuple) and ((exSize[0]!=sz) or (exSize[1]!=sz)))):
                                        raise Exception('exSize does not match the n in "I(n)"')
                                exSize=(sz,sz)
                                retstr='I'
                        if exSize is None:
                                raise Exception('size unspecified')
                        if isinstance(exSize,tuple):
                                if exSize[0]!=exSize[1]:
                                        raise Exception('matrix should be square')
                                retmat=cvx.spdiag([1.]*exSize[0])
                        else:#we have an integer
                                retmat=cvx.spdiag([1.]*exSize)
                else:
                        raise NameError('unexpected mat variable')
                if transpose:
                        retmat=retmat.T
                retmat*=alpha
        else:
                raise NameError('unexpected mat variable')
        #look for a more appropriate string...
        if retstr is None:
                retstr='[ {0} x {1} MAT ]'.format(retmat.size[0],retmat.size[1])
        if not retmat: #|0|
                retstr=''
        elif retmat.size==(1,1):
                retstr=str(retmat[0])
        elif max(retmat+0.)==min(retmat+0.): #|1| (+0. to avoid sparse evaluation)
                if retmat[0]==0:
                        retstr=''
                elif retmat[0]==1:
                        retstr='|1|'
                else:
                        retstr='|'+str(retmat[0])+'|'
        elif cvx.sparse(retmat).I.size[0]==1: #e_x
                spm=cvx.sparse(retmat)
                i=spm.I[0]
                j=spm.J[0]
                retstr=''
                if spm.V[0]!=1:
                        retstr=str(spm.V[0])+'*'
                if retmat.size[1]>1:
                        retstr+='e_'+str(i)+','+str(j)
                else:
                        retstr+='e_'+str(i)
        #(1,1) matrix but not appropriate size
        if retmat.size==(1,1) and (exSize not in [(1,1),1,None]):
                return _retrieve_matrix(retmat[0],exSize)
        return cvx.sparse(retmat),retstr

def svec(mat):
        """
        mat must be symmetric,
        return the svec representation of mat.
        """
        if not isinstance(mat,cvx.spmatrix):
                mat=cvx.sparse(mat)
        
        s0=mat.size[0]
        if s0!=mat.size[1]:
                raise ValueError('mat but be symmetric')
        
        I=[]
        J=[]
        V=[]
        for (i,j,v) in zip((mat.I),(mat.J),(mat.V)):
                if mat[j,i]!=v:
                        raise ValueError('mat but be symmetric')
                if i<=j:
                        isvec=j*(j+1)/2+i
                        J.append(0)
                        I.append(isvec)
                        if i==j:
                                V.append(v)
                        else:
                                V.append(np.sqrt(2)*v)

        return cvx.spmatrix(V,I,J,(s0*(s0+1)/2,1))

def svecm1(vec):
        if vec.size[1]>1:
                raise ValueError('should be a column vector')
        v=vec.size[0]
        n=int(np.sqrt(1+8*v)-1)/2
        if n*(n+1)/2 != v:
                raise ValueError('vec should be of dimension n(n+1)/2')
        if not isinstance(vec,cvx.spmatrix):
                vec=cvx.sparse(vec)
        I=[]
        J=[]
        V=[]
        for i,v in zip(vec.I,vec.V):
                c=int(np.sqrt(1+8*i)-1)/2
                r=i-c*(c+1)/2
                I.append(r)
                J.append(c)
                if r==c:
                        V.append(v)
                else:
                        I.append(c)
                        J.append(r)
                        V.extend([v/np.sqrt(2)]*2)
        return cvx.spmatrix(V,I,J,(n,n))
             
                
        
        
def _svecm1_identity(vtype,size):
        """
        row wise svec-1 transformation of the
        identity matrix of size size[0]*size[1]
        """
        if vtype=='symmetric':
                s0=size[0]
                if size[1]!=s0:
                        raise ValueError('should be square')
                I=range(s0*s0)
                J=[]
                V=[]
                for i in I:
                        rc= (i%s0,i/s0)
                        (r,c)=(min(rc),max(rc))
                        j=c*(c+1)/2+r
                        J.append(j)
                        if r==c:
                                V.append(1)
                        else:
                                V.append(1/np.sqrt(2))
                idmat=cvx.spmatrix(V,I,J,(s0*s0,s0*(s0+1)/2))
        
        else:
                sp=size[0]*size[1]
                idmat=cvx.spmatrix([1]*sp,range(sp),range(sp),(sp,sp))
                
        return idmat
                
"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
--------                                Problem class                                        ------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

class Problem:
	"""This class represents an optimization problem.
	The constructor creates an empty problem.
        Some options can be provided under the form
        **key** = **value**.
	See the list of available options
        in :func:`set_all_options_to_default`
	"""
	def __init__(self,**options):
		self.objective = ('find',None) #feasibility problem only
		self.constraints = {}
		self.variables = {}
		
		self.countVar=0
		self.countCons=0
		self.numberOfVars=0
		self.numberAffConstraints=0
		self.numberConeVars=0
		self.numberConeConstraints=0
		self.numberLSEConstraints=0
		self.numberQuadConstraints=0
		self.numberQuadNNZ=0
		self.numberSDPConstraints=0
		self.numberSDPVars=0

		self.cvxoptVars={'c':None,'A':None,'b':None,'Gl':None,
				'hl':None,'Gq':None,'hq':None,'Gs':None,'hs':None,
				'F':None,'g':None, 'quadcons': None}
		
		self.gurobi_Instance = None
		self.grbvar = {}
		
		self.cplex_Instance = None

		self.msk_env=None
		self.msk_task=None

		self.groupsOfConstraints = {}
		self.listOfVars = {}
		self.consNumbering=[]
		
		self.options = {}
		if options is None: options={}
		self.set_all_options_to_default()
		self.update_options(**options)

		self.number_solutions=0
		
		self.longestkey=0 #for a nice display of constraints
		self.varIndices=[]

	def __str__(self):
		probstr='---------------------\n'		
		probstr+='optimization problem:\n'
		probstr+='{0} variables, {1} affine constraints'.format(
				self.numberOfVars,self.numberAffConstraints)
		if self.numberConeVars>0:
			probstr+=', {0} vars in a SO cone'.format(
				self.numberConeVars)
		if self.numberLSEConstraints>0:
			probstr+=', {0} vars in a LOG-SUM-EXP'.format(
				self.numberLSEConstraints)
                if self.numberSDPConstraints>0:
                        probstr+=', {0} vars in a SD cone'.format(
                                self.numberSDPVars)
		probstr+='\n'

		printedlis=[]
		for vkey in self.variables.keys():
			if '[' in vkey and ']' in vkey:
				lisname=vkey[:vkey.index('[')]
				if not lisname in printedlis:
					printedlis.append(lisname)
					var=self.listOfVars[lisname]
					probstr+='\n'+lisname+' \t: '
					probstr+=var['type']+' of '+str(var['numvars'])+' variables, '
					if var['size']=='different':
						probstr+='different sizes'
					else:
						probstr+=str(var['size'])
					if var['vtype']=='different':
						probstr+=', different type'
					else:
						probstr+=', '+var['vtype']
			else:			
				var = self.variables[vkey]
				probstr+='\n'+vkey+' \t: '+str(var.size)+', '+var.vtype
		probstr+='\n'
		if self.objective[0]=='max':
			probstr+='\n\tmaximize '+self.objective[1].string+'\n'
		elif self.objective[0]=='min':
			probstr+='\n\tminimize '+self.objective[1].string+'\n'
		elif self.objective[0]=='find':
			probstr+='\n\tfind vars\n'
		probstr+='such that\n'		
		k=0
		while k<self.countCons:
			if k in self.groupsOfConstraints.keys():
				lcur=len(self.groupsOfConstraints[k][2])				
				if lcur>0:
					lcur+=2
					probstr+='('+self.groupsOfConstraints[k][2]+')'
				if self.longestkey==0:
					ntabs=0
				else:
					ntabs=int(np.ceil((self.longestkey+2)/8.0))
				missingtabs=int(  np.ceil(((ntabs*8)-lcur)/8.0)  )
				for i in range(missingtabs):
					probstr+='\t'
				if lcur>0:
					probstr+=': '
				else:
					probstr+='  '
				probstr+=self.groupsOfConstraints[k][1]
				k=self.groupsOfConstraints[k][0]+1
			else:
				probstr+=self.constraints[k].keyconstring(self.longestkey)+'\n'
				k+=1
		probstr+='---------------------'
		return probstr
	

	"""
	----------------------------------------------------------------
	--                       Utilities                            --
	----------------------------------------------------------------
	"""

	def remove_all_constraints(self):
		self.constraints = {}
		self.countCons=0
		self.numberAffConstraints=0
		self.numberConeVars=0
		self.numberConeConstraints=0
		self.numberQuadConstraints=0
		self.numberLSEConstraints=0
		self.groupsOfConstraints ={}
		self.consNumbering=[]
		self.numberSDPConstraints=0
		self.numberSDPVars=0
		
	
	def obj_value(self):
                """
                If the problem was already solved, returns the objective value.
                Otherwise, it raises an ``AttributeError``.
                """
                return self.objective[1].eval()[0]

	def get_varName(self,Id):
		return [k for k in self.variables.keys() if  self.variables[k].Id==Id][0]
	
	def set_objective(self,typ,expr):
                """
                Defines the objective function of the problem.
                
                :param typ: can be either 'max' (maximization problem),
                            'min' (minimization problem),
                            or 'find' (feasibility problem).
                :type typ: str.
                :param expr: an :class:`Expression`. The expression to be minimized
                             or maximized. This parameter will be ignored if typ=='find'.
                """
		if (isinstance(expr,AffinExpr) and expr.size<>(1,1)):
			raise Exception('objective should be scalar')
		if not (isinstance(expr,AffinExpr) or isinstance(expr,LogSumExp)
			or isinstance(expr,QuadExp) or isinstance(expr,GeneralFun)):
			raise Exception('unsupported objective')
		if isinstance(expr,LogSumExp):
			self.numberLSEConstraints+=expr.Exp.size[0]*expr.Exp.size[1]
		if isinstance(expr,QuadExp):
			self.numberQuadConstraints+=1
			self.numberQuadNNZ+=expr.nnz()
		self.objective=(typ,expr)
        
        def set_var_value(self,name,value,optimalvar=False):
                """
                sets a variable to the given value. This can be useful to check
                the value of a complicated :class:`Expression`, or to use
                a solver with a *hot start algorithm (not implemented yet)*.
                
                :param name: name of the variable to which the value will be given
                :type name: str.
                :param value: The value for the variable. The function will try to
                              parse this variable as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`
                              of the desired size by using
                              the function :func:`_retrieve_matrix`
                              
                **Example**
                
                >>> prob=MP.Problem()
                >>> x=prob.add_variable('x',2)
                >>> prob.set_var_value('x',[3,4])
                >>> abs(x)**2
                #quadratic expression: ||x||**2 #
                >>> (abs(x)**2).eval()
                25.0
                >>> 

                .. Todo::
                Virer cette doc, passer en private, et donner doc de set_value d'une exp
                """
                ind = None
                if isinstance(name,tuple): # alternative solution
                        ind=name[0]
                        name=name[1]

                if not name in self.variables.keys():
			raise Exception('unknown variable name')
		valuemat,valueString=_retrieve_matrix(value,self.variables[name].size)
		if valuemat.size<>self.variables[name].size:
			raise Exception('should be of size {0}'.format(self.variables[name].size))
		if self.variables[name].vtype=='symmetric':
                        valuemat=svec(valuemat)
		if ind is None:
                        self.variables[name].value=valuemat
                        if optimalvar:
                                self.number_solutions=max(self.number_solutions,1)
                else:
                        self.variables[name].value_alt[ind]=valuemat
                        if optimalvar:
                                self.number_solutions=max(self.number_solutions,ind+1)

        def new_param(self,name,value):
                """
                Declare a parameter for the problem, that will be stored
                as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
                It is possible to give a ``list`` or a ``dict`` of parameters.
                The function returns a constant :class:`AffinExpr` 
                (or a ``list`` or a ``dict`` of :class:`AffinExpr`) representing this parameter.
                
                .. note :: Declaring parameters is optional, since the expression can
                           as well be given by using normal variables. (see Example below).
                           However, if you use this function to declare your parameters,
                           the names of the parameters will be display when you **print**
                           an :class:`Expression` or a :class:`Constraint`
                
                :param name: The name given to this parameter.
                :type name: str.
                :param value: The value (resp ``list`` of values, ``dict`` of values) of the parameter.
                              The type of **value** (resp. the elements of the ``list`` **value**,
                              the values of the ``dict`` **value**) should be understandable by
                              the function :func:`_retrieve_matrix`.
                :returns: A constant affine expression (:class:`AffinExpr`)
                          (resp. a ``list`` of :class:`AffinExpr` of the same length as **value**,
                          a ``dict`` of :class:`AffinExpr` indexed by the keys of **value**)
                          
                **Example:**
                
                >>> import cvxopt as cvx
                >>> prob=MP.Problem()
                >>> x=prob.add_variable('x',3)
                >>> B={'foo':17.4,'matrix':cvx.matrix([[1,2],[3,4],[5,6]]),'ones':'|1|(4,1)'}
                >>> B['matrix']*x+B['foo']
                # (2 x 1)-affine expression: [ 2 x 3 MAT ]*x + |17.4| #
                >>> #(in the string above, |17.4| represents the 2-dim vector [17.4,17.4])
                >>> B=prob.new_param('B',B)
                >>> B['matrix']*x+B['foo']
                # (2 x 1)-affine expression: B[matrix]*x + |B[foo]| #
                """
		if isinstance(value,list):
			L=[]			
			for i,l in enumerate(value):
				L.append( self.new_param(name+'['+str(i)+']',l) )
			return L
		elif isinstance(value,dict):
			D={}
			for k in value.keys():
				D[k]=self.new_param(name+'['+str(k)+']',value[k])
			return D
		else:
			term,termString=_retrieve_matrix(value,None)
			return AffinExpr({},constant=term[:],size=term.size,string=name,variables=self.variables)


	def _makeGandh(self,affExpr):
		"""if affExpr is an affine expression,
		this method creates a bloc matrix G to be multiplied by the large
		vectorized vector of all variables,
		and returns the vector h corresponding to the constant term.
		"""
		n1=affExpr.size[0]*affExpr.size[1]
		#matrix G		
		"""
		Gmats=[]
		import pdb; pdb.set_trace()
		for i in self.varIndices:
			nam=self.get_varName(i)
			if nam in affExpr.factors.keys():
				Gmats.append([affExpr.factors[nam]])
			else:
				zz=cvx.spmatrix([],[],[],(n1,
					self.variables[nam].size[0]*self.variables[nam].size[1]) )
				Gmats.append([zz])
		G=cvx.sparse(Gmats,tc='d')		
		"""		
		
		I=[]
		J=[]
		V=[]
		for nam in affExpr.factors:
			si = self.variables[nam].startIndex
			if type(affExpr.factors[nam])<>cvx.base.spmatrix:
				affExpr.factors[nam]=cvx.sparse(affExpr.factors[nam])
			I.extend(affExpr.factors[nam].I)
			J.extend([si+j for j in affExpr.factors[nam].J])
			V.extend(affExpr.factors[nam].V)
		G=cvx.spmatrix(V,I,J,(n1,self.numberOfVars))
		
		#is it really sparse ?
		#if cvx.nnz(G)/float(G.size[0]*G.size[1])>0.5:
		#	G=cvx.matrix(G,tc='d')
		#vector h
		if affExpr.constant is None:
			h=cvx.matrix(0,(n1,1),tc='d')
		else:
			h=affExpr.constant
                if not isinstance(h,cvx.matrix):
                        h=cvx.matrix(h,tc='d')
		if h.typecode<>'d':
			h=cvx.matrix(h,tc='d')
		return G,h

                
        def set_all_options_to_default(self):
                """set all the options to their default.
                The following options are available:
                
                * tol=1e-7 : optimality tolerance for the solver
                * feastol=1e-7 : feasibility tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                * abstol=1e-7 : absolute tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                * reltol=1e-6 : relative tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                * maxit=50 : maximum number of iterations
                * verbose=1 : verbosity level
                * solver='cvxopt' : currently the other available solvers are 'cplex','mosek','smcp','zibopt'
                * step_sqp=1 : 'first step length for the sequential quadratic programming procedure'
                * harmonic_steps=True : step at the ith step of the sqp procedure is step_sqp/i]
                * onlyChangeObjective'=False : useful when we want to recompute the solution of a
                  problem but with a different objective function. If set to *True* and a new
                  objective function has been passed to the solver, the constraints
                  of the problem will not be parsed next time :func:`solve` is called (this can lead
                  to a huge gain of time).
                * noduals=False : if True, do not retrieve the dual variables
                * nbsol=None (solver default) : maximum number of computed
                  solutions in the solution pool.
                * timelimit =None (infinity) : time limit for the solver
                * acceptableGap =None (0%) : If the time limit is reached, consider the solution as
                  acceptable if the gap is less than this value.
                * treemomory = None (solver default) : size of the buffer for the branch and bound tree,
                  in Megabytes.
                * gaplim = None (0%) : returns a solution as soon as this gap is reached.
                * pool_gap = None (0%) : keeps only the solution
                  within this gap in the pool
                                 
                                 
                .. Warning:: Not all options are handled by all solvers yet.
                
                .. Todo:: Organize above options
                """
                #Additional, hidden option (requires a patch of smcp, to use conlp to
                #interface the feasible starting point solver):
                #
                #* 'smcp_feas'=False [if True, use the feasible start solver with SMCP]
                default_options={'tol'            :1e-7,
                                 'feastol'        :1e-7,
                                 'abstol'         :1e-7,
                                 'reltol'         :1e-6,
                                 'maxit'          :50,
                                 'verbose'        :1,
                                 'solver'         :'CVXOPT',
                                 'step_sqp'       :1,
                                 'harmonic_steps' :1,
                                 'onlyChangeObjective':False,
                                 'noduals'        :False,
                                 'smcp_feas'      :False,
                                 'nbsol'          :None,
                                 'timelimit'      :None,
                                 'acceptablegap'  :None,
                                 'treememory'     :None,
                                 'gaplim'         :None,
                                 'pool_gap'       :None
                                 }
                                 
                                 
                self.options=default_options

        def set_option(self,key,val):
                """
                Sets the option **key** to the value **val**.
                
                :param key: The key of an option (see the list of keys in :func:`set_all_options_to_default`).
                :type key: str.
                :param val: New value for the option **key**.
                """
                if key not in self.options:
                        raise AttributeError('unkown option key :'+str(key))
                self.options[key]=val
                if key=='tol':
                        self.options['feastol']=val
                        self.options['abstol']=val
                        self.options['reltol']=val*10

        
        def update_options(self, **options):
                """
                update the option ``dict``, for the pairs
                key = value. For a list of available options,
                see :func:`set_all_options_to_default`.
                """
                
                for k in options.keys():
                        self.set_option(k,options[k])
                
        def eliminate_useless_variables(self):
                """
                Removes from the problem the variables that do not
                appear in any constraint or in the objective function.
                """
                for var in self.variables.keys():
                        found=False
                        for cons in self.constraints.keys():
                                if isinstance(self.constraints[cons].Exp1,AffinExpr):
                                        if var in self.constraints[cons].Exp1.factors.keys():
                                                found=True
                                        if var in self.constraints[cons].Exp2.factors.keys():
                                                found=True
                                        if not self.constraints[cons].Exp3 is None:
                                                if var in self.constraints[cons].Exp3.factors.keys():
                                                        found=True
                                elif isinstance(self.constraints[cons].Exp1,QuadExp):
                                        if var in self.constraints[cons].Exp1.aff.factors.keys():
                                                found=True
                                        for ij in self.constraints[cons].Exp1.quad:
                                                if var in ij:
                                                        found=True
                                #TODO manque case LSE ?
                        if not self.objective[1] is None:
                                if isinstance(self.objective[1],AffinExpr):
                                        if var in self.objective[1].factors.keys():
                                                found=True
                                elif isinstance(self.objective[1],QuadExp):
                                        if var in self.objective[1].aff.factors.keys():
                                                found=True
                                        for ij in self.objective[1].quad:
                                                if var in ij:
                                                        found=True
                                elif isinstance(self.objective[1],LogSumExp):
                                        if var in self.objective[1].Exp.factors.keys():
                                                found=True
                        if not found:
                                self.remove_variable(var)
                                print('variable '+var+' was useless and has been removed')

        """
        ----------------------------------------------------------------
        --                TOOLS TO CREATE AN INSTANCE                 --
        ----------------------------------------------------------------
        """

        def add_variable(self,name,size=1, vtype = 'continuous' ):
                """
                adds a variable in the problem,
                and returns an :class:`AffinExpr` representing this variable.

                For example,
                
                >>> prob=MP.Problem()
                >>> x=prob.add_variable('x',3)
                >>> x
                # (3 x 1)-affine expression: x #
                
                :param name: The name of the variable.
                :type name: str.
                :param size: The size of the variable.
                             
                             Can be either
                             
                                * an ``int`` *n* , in which case the variable is a **vector of dimension n**
                                * or a ``tuple`` *(n,m)*, and the variable is a **n x m-matrix**.
                
                :type size: int or tuple.
                :param vtype: variable type. Can be : 'continuous', 'binary', 'integer', 'semicont', or 'semiint'
                :type vtype: str.
                :returns:  an instance of the class :class:`AffinExpr` -- Affine expression representing the created variable.
                
                """

		if name in self.variables:
			raise Exception('this variable already exists')
		if isinstance(size,int):
			size=(size,1)
		if len(size)==1:
			size=(size[0],1)

		if '[' in name and ']' in name:#list or dict of variables
			lisname=name[:name.index('[')]
			ind=name[name.index('[')+1:name.index(']')]
			if lisname in self.listOfVars:
				oldn=self.listOfVars[lisname]['numvars']
				self.listOfVars[lisname]['numvars']+=1
				if size<>self.listOfVars[lisname]['size']:
					self.listOfVars[lisname]['size']='different'
				if vtype<>self.listOfVars[lisname]['vtype']:
					self.listOfVars[lisname]['vtype']='different'
				if self.listOfVars[lisname]['type']=='list' and ind<>str(oldn):
					self.listOfVars[lisname]['type']='dict'
			else:
				self.listOfVars[lisname]={'numvars':1,'size':size,'vtype':vtype}
				if ind=='0':
					self.listOfVars[lisname]['type']='list'
				else:
					self.listOfVars[lisname]['type']='dict'
		self.variables[name]=Variable(name,size,self.countVar,self.numberOfVars, vtype)
		if vtype=='symmetric':
                        if size[0]!=size[1]:
                                raise ValueError('symmetric variables must be square')
                        s0=size[0]
                        self.numberOfVars+=s0*(s0+1)/2
                else:
                        self.numberOfVars+=size[0]*size[1]
		self.varIndices.append(self.countVar)
		self.countVar+=1
		
		#svec operation
		idmat=_svecm1_identity(vtype,size)
		
		return AffinExpr({name:idmat},
                                 size=size,
                                 string=name,
                                 variables=self.variables)
	
	def remove_variable(self,name):
                """
                removes the variable **name** from the problem
                """
                if '[' in name and ']' in name:#list or dict of variables
                        lisname=name[:name.index('[')]
                        if lisname in self.listOfVars:
                                del self.listOfVars[lisname] #not a complete list of vars anymore
                if name not in self.variables.keys():
                        raise Exception('variable does not exist')
                Id=self.variables[name].Id
                self.countVar-=1
                sz=self.variables[name].size
                self.numberOfVars-=sz[0]*sz[1]
                self.varIndices.remove(Id)
                del self.variables[name]
                self._recomputeStartEndIndices()
        
        def _recomputeStartEndIndices(self):
                ind=0
                for i in self.varIndices:
                        nam=self.get_varName(i)
                        self.variables[nam].startIndex=ind
                        ind+=self.variables[nam].size[0]*self.variables[nam].size[1]
                        self.variables[nam].endIndex=ind

        def add_constraint(self,cons, key=None):
                """Adds a constraint in the problem.
                
                :param cons: The constraint to be added.
                :type cons: :class:`Constraint`
                :param key: Optional parameter to describe the constraint with a key string.
                :type key: str.
                """
                cons.key=key
                if not key is None:
                        self.longestkey=max(self.longestkey,len(key))
                self.constraints[self.countCons]=cons
                self.consNumbering.append(self.countCons)
                self.countCons+=1
                if cons.typeOfConstraint[:3]=='lin':
                        self.numberAffConstraints+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                elif cons.typeOfConstraint[2:]=='cone':
                        self.numberConeVars+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                        self.numberConeConstraints+=1
                elif cons.typeOfConstraint=='lse':
                        self.numberLSEConstraints+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                elif cons.typeOfConstraint=='quad':
                        self.numberQuadConstraints+=1
                        self.numberQuadNNZ+=cons.Exp1.nnz()
                elif cons.typeOfConstraint[:3]=='sdp':
                        self.numberSDPConstraints+=1
                        self.numberSDPVars+=(cons.Exp1.size[0]*(cons.Exp1.size[0]+1))/2
                

        def add_list_of_constraints(self,lst,it=None,indices=None,key=None):
                u"""adds a list of constraints in the problem.
                This fonction can be used with lists created on the fly by python (see the example below).
                
                :param lst: ``list`` of :class:`Constraint`.
                :param it: Description of the letters which should
                           be used to replace the dummy indices.
                           The function tries to find a template
                           for the string representations of the
                           constraints in the list. If several indices change in the
                           list, their letters should be given as a
                           tuple of string, in their order of appearance in the
                           resulting string. For example, if three indices
                           change in the constraints, and you want them to be named
                           'i','j' and 'k', set **it** = ('i','j','k').
                           You can also group two indices which always appear together,
                           e.g. if 'i' always appear next to 'j' you
                           could set **it** = (('ij',2),'k'). Here, the number 2
                           indicates that 'ij' replaces 2 indices.
                           If **it** = None, or if the function is not able to find a
                           template, the string of the first constraint will be used to
                           represent this list of constraints.
                :type it: None or str or tuple.
                :param indices: a string to denote the set where the indices belong to.
                :type indices: str.
                :param key: Optional parameter to describe the list of constraints with a key string.
                :type key: str.
                                
                **Example:**

                >>> import pyMathProg as MP
                >>> import cvxopt as cvx
                >>> prob=MP.Problem()
                >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(5)]
                >>> x #doctest: +NORMALIZE_WHITESPACE
                [# (2 x 1)-affine expression: x[0] #,
                 # (2 x 1)-affine expression: x[1] #,
                 # (2 x 1)-affine expression: x[2] #,
                 # (2 x 1)-affine expression: x[3] #,
                 # (2 x 1)-affine expression: x[4] #]
                >>> y=prob.add_variable('y',5)
                >>> IJ=[(1,2),(2,0),(4,2)]
                >>> w={}
                >>> for ij in IJ:
                ...         w[ij]=prob.add_variable('w[{0}]'.format(ij),3)
                ... 
                >>> u=prob.new_param('u',cvx.matrix([2,5]))
                >>> prob.add_list_of_constraints(
                ... [u.T()*x[i]<y[i] for i in range(5)],
                ... 'i',
                ... '[5]')
                >>> 
                >>> prob.add_list_of_constraints(
                ... [abs(w[i,j])<y[j] for (i,j) in IJ],
                ... (('ij',2),),
                ... 'IJ')
                >>> 
                >>> # (The function will not succeed to find a string template in the next example)
                >>> prob.add_list_of_constraints(
                ... [y[t] > y[t+1] for t in range(4)],
                ... 't',
                ... '[4]')
                >>> 
                >>> print prob #doctest: +NORMALIZE_WHITESPACE
                ---------------------
                optimization problem:
                24 variables, 9 affine constraints, 9 vars in a SO cone
                <BLANKLINE>
                x   : list of 5 variables, (2, 1), continuous
                w   : dict of 3 variables, (3, 1), continuous
                y   : (5, 1), continuous
                <BLANKLINE>
                    find vars
                such that
                  u.T*x[i] < y[i] for all i in [5]
                  ||w[ij]|| < y[ij__1] for all ij in IJ
                  [4 constraints (first: y[0] > y[1])]
                ---------------------

                """
                firstCons=self.countCons
                for ks in lst:
                        self.add_constraint(ks)
                        self.consNumbering.pop()
                self.consNumbering.append(range(firstCons,self.countCons))
                lastCons=self.countCons-1
                if key is None:
                        key=''
                else:
                        self.longestkey=max(self.longestkey,len(key))
                if it is None:
                        strlis='['+str(len(lst))+' constraints (first: '+lst[0].constring()+')]\n'
                else:
                        strlis=' for all '
                        if len(it)>1:
                                strlis+='('                        
                        for x in it:
                                if isinstance(x,tuple):
                                        strlis+=x[0]
                                else:
                                        strlis+=x
                                strlis+=','
                        strlis=strlis[:-1] #remvove the last comma
                        if len(it)>1:
                                strlis+=')'
                        if not indices is None:
                                strlis+=' in '+indices
                        if isinstance(it,tuple) and len(it)==2 and isinstance(it[1],int):
                                it=(it,)
                        if isinstance(it,list):
                                it=tuple(it)
                        if not isinstance(it,tuple):
                                it=(it,)
                        lstr=[l.constring() for l in lst]
                        try:
                                indstr=putIndices(lstr,it)
                                strlis=indstr+strlis+'\n'
                        except Exception as ex:
                                strlis='['+str(len(lst))+' constraints (first: '+lst[0].constring()+')]\n'
                self.groupsOfConstraints[firstCons]=[lastCons,strlis,key]
                        
        def get_valued_variable(self,name):
                """
                Returns the optimal value of the variable (as an :class:`AffinExpr`)
                with the given **name**.
                If **name** is a list (resp. dict) of variables,
                named with the template 'name[index]' (resp. 'name[key]'),
                then the function returns the list (resp. dict)
                of these variables.
                
                .. Warning:: If the problem has not been solved,
                             this function will raise an Exception.
                """
                exp=self.get_variable(name)
                if isinstance(exp,list):
                        for i in xrange(len(exp)):
                                exp[i]=exp[i].eval()
                elif isinstance(exp,dict):
                        for i in exp:
                                exp[i]=exp[i].eval()
                else:
                        exp=exp.eval()
                return exp
                

        def get_variable(self,name):
                """
                Returns the variable (as an :class:`AffinExpr`) with the given **name**.
                If **name** is a list (resp. dict) of variables,
                named with the template 'name[index]' (resp. 'name[key]'),
                then the function returns the list (resp. dict)
                of these variables.
                """
                var=name
                if var in self.listOfVars.keys():
                        if self.listOfVars[var]['type']=='dict':
                                rvar={}
                        else:
                                rvar=[0]*self.listOfVars[var]['numvars']
                        seenKeys=[]
                        for ind in [vname[len(var)+1:-1] for vname in self.variables.keys() if \
                                 (vname[:len(var)] ==var and vname[len(var)]=='[')]:
                                if ind.isdigit():
                                        key=int(ind)
                                        if key not in seenKeys:
                                                seenKeys.append(key)
                                        else:
                                                key=ind
                                elif ',' in ind:
                                        isplit=ind.split(',')
                                        if isplit[0].startswith('('):
                                                isplit[0]=isplit[0][1:]
                                        if isplit[-1].endswith(')'):
                                                isplit[-1]=isplit[-1][:-1]
                                        if all([i.isdigit() for i in isplit]):
                                                key=tuple([int(i) for i in isplit])
                                                if key not in seenKeys:
                                                        seenKeys.append(key)
                                                else:
                                                        key=ind
                                        else:
                                                key=ind
                                else:
                                        try:
                                                key=float(ind)
                                        except ValueError:
                                                key=ind
                                sz=self.variables[var+'['+ind+']'].size
                                rvar[key]=AffinExpr({var+'['+ind+']':cvx.spdiag([1.]*sz[0]*sz[1])},
                                        constant=0,
                                        size=self.variables[var+'['+ind+']'].size,string=var+'['+ind+']',
                                        variables=self.variables)
                        return rvar
                else:
                        sz=self.variables[var].size
                        return AffinExpr({var:cvx.spdiag([1.]*sz[0]*sz[1])},constant=0,
                                size=sz,string=var,
                                variables=self.variables)


        def get_constraint(self,ind):
                u"""
                returns a constraint of the problem.
                
                :param ind: There are two ways to index a constraint.
                
                               * if ind is an ``int`` *n*, then the nth constraint (starting from 0)
                                 will be returned, where all the constraints are counted
                                 in the order where they were passed to the problem.
                                 
                               * if ind is a ``tuple`` *(k,i)*, then the ith constraint
                                 from the kth group of constraints is returned
                                 (starting from 0). By            
                                 *group of constraints*, it is meant a single constraint
                                 or a list of constraints added together with the
                                 function :func:`add_list_of_constraints`.
                
                :type ind: int or tuple.
                
                **Example:**
                
                >>> import pyMathProg as MP
                >>> import cvxopt as cvx
                >>> prob=MP.Problem()
                >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(5)]
                >>> y=prob.add_variable('y',5)
                >>> prob.add_list_of_constraints(
                ... [(1|x[i])<y[i] for i in range(5)],
                ... 'i',
                ... '[5]')
                >>> prob.add_constraint(y>0)
                >>> print prob #doctest: +NORMALIZE_WHITESPACE
                ---------------------
                optimization problem:
                15 variables, 10 affine constraints
                <BLANKLINE>
                x   : list of 5 variables, (2, 1), continuous
                y   : (5, 1), continuous
                <BLANKLINE>
                    find vars
                such that
                  〈 |1| | x[i] 〉 < y[i] for all i in [5]
                  y > 0
                ---------------------
                >>> prob.get_constraint(1)
                # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #
                >>> prob.get_constraint((0,3))
                # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #
                >>> prob.get_constraint((1,))
                # (5x1)-affine constraint: y > 0 #
                >>> prob.get_constraint(5)
                # (5x1)-affine constraint: y > 0 #
                
                """
                indtuple=ind
                if isinstance(indtuple,int):
                        return self.constraints[indtuple]
                lsind=self.consNumbering                
                for k in indtuple:
                        if not isinstance(lsind,list):
                                raise Exception('too many indices')
                        if k>=len(lsind):
                                raise Exception('index is too large')
                        lsind=lsind[k]
                if isinstance(lsind,list):
                                raise Exception('too few indices')
                return self.constraints[lsind]
                
        def _eval_all(self):
                """
                Returns the big vector with all variable values,
                in the order induced by sorted(self.variables.keys()).
                """
                xx=cvx.matrix([],(0,1))
                for v in sorted(self.variables.keys()):
                        xx=cvx.matrix([xx,self.variables[v].value[:]])
                return xx

        """
        ----------------------------------------------------------------
        --                BUILD THE VARIABLES FOR A SOLVER            --
        ----------------------------------------------------------------
        """        

        '''GUROBi does not work with our version of Python ? 
        def makeGUROBI_Instance(self):
                """
                defines the variables gurobi_Instance and grbvar
                """
                self._set_as_current_problem()                
                self.options['solver'] = 'GUROBI'
                m = Model()
                
                grb_type = {        'continuous' : GRB.CONTINUOUS, 
                                'binary' : GRB.BINARY, 
                                'integer' : GRB.INTEGER, 
                                'semicont' : GRB.SEMICONT, 
                                'semiint' : GRB.SEMIINT }
                
                # variables
                for kvar in self.variables.keys():
                        variable = self.variables[kvar]
                        # a vector
                        objective = self.objective[1].factors
                        if variable.size[1] == 1:
                                # objective vector
                                if kvar in objective.keys():
                                        vectorObjective = objective[kvar][0]
                                else:
                                        vectorObjective = []
                                        for i in range(variable.size[0]):
                                                vectorObjective.append[0]
                                
                                for i in range(variable.size[0]):
                                        #lb (optional): Lower bound for new variable.
                                        #ub (optional): Upper bound for new variable.
                                        #obj (optional): Objective coefficient for new variable.
                                        #vtype (optional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT).
                                        #name (optional): Name for new variable.
                                        #column (optional): Column object that indicates the set of constraints in which the new variable participates, and the associated coefficients. 
                                        newvar = m.addVar(obj = vectorObjective[i], vtype = grb_type[variable.vtype])
                                        self.grbvar[kvar].append(n)
                        # not a vector
                        else:
                                raise ValueError("the variable is not a vector, not implemented yet")
                m.update()
                
                for constr in self.constraints:
                        if constr.typeOfConstraint == 'lin<':
                                sense = GRB.LESS_EQUAL
                        elif constr.typeOfConstraint == 'lin>':
                                sense = GRB.GREATER_EQUAL
                        elif constr.typeOfConstraint == 'lin=':
                                sense = GRB.EQUAL
                        else:
                                raise ValueError('Impossible linear constraint')
                        
                        # decompose vector in i relations
                        for i in range(constr.Exp1.size[0]):
                                lhsvar = []
                                lhsparam = []
                                rhsvar = []
                                rhsparam = []
                                # left
                                for kvar in constr.Exp1.factors.keys():
                                        lhsvar.append(self.grbvar[kvar])
                                        lhsparam.append(constr.Exp1.factors[kvar][i,:])
                                # right
                                for kvar in constr.Exp2.factors.keys():
                                        rhsvar.append(self.grbvar[kvar])
                                        rhsparam.append(constr.Exp1.factors[kvar][i,:])
                                #adding the constraint
                                lhs = LinExpr(lhsparam, lhsvar)
                                rhs = LinExpr(rhsparam, rhsvar)
                                m.addConstr(lhs, sense, rhs)
        '''
                
        def is_continuous(self):
                """ Returns True if there are only continuous variables"""
                for kvar in self.variables.keys():
                        if self.variables[kvar].vtype != 'continuous':
                                return False
                return True
                
        def _make_cplex_instance(self):
                """
                Defines the variables cplex_Instance and cplexvar,
                used by the cplex solver.
                
                TODO: feasibility problems
                """
                try:
                        import cplex
                except:
                        Exception('cplex library not found')
                
                c = cplex.Cplex()
                import itertools
                
                if not self.options['timelimit'] is None:
                        import timelimitcallback
                        import time
                        timelim_cb = c.register_callback(timelimitcallback.TimeLimitCallback)
                        timelim_cb.starttime = time.time()
                        timelim_cb.timelimit = self.options['timelimit']
                        if not self.options['acceptablegap'] is None:
                                timelim_cb.acceptablegap =self.options['acceptableGap']
                        else:
                                timelim_cb.acceptablegap = 100
                        timelim_cb.aborted = 0
                        #c.parameters.tuning.timelimit.set(self.options['timelimit']) #DOES NOT WORK LIKE THIS ?
                if not self.options['treememory'] is None:
                        c.parameters.mip.limits.treememory.set(self.options['treememory'])
                if not self.options['gaplim'] is None:
                        c.parameters.mip.tolerances.mipgap.set(self.options['gaplim'])
                if not self.options['nbsol'] is None:
                        c.parameters.mip.limits.solutions.set(self.options['nbsol'])
                if not self.options['pool_gap'] is None:
                        c.parameters.mip.pool.relgap.set(self.options['pool_gap'])
                
                
                sense_opt = self.objective[0]
                if sense_opt == 'max':
                        c.objective.set_sense(c.objective.sense.maximize)
                elif sense_opt == 'min':
                        c.objective.set_sense(c.objective.sense.minimize)
                #else:
                #        raise ValueError('feasibility problems not implemented ?')
                
                self.options['solver'] = 'cplex'
                
                cplex_type = {  'continuous' : c.variables.type.continuous, 
                                'binary' : c.variables.type.binary, 
                                'integer' : c.variables.type.integer, 
                                'semicont' : c.variables.type.semi_continuous, 
                                'semiint' : c.variables.type.semi_integer }                
                
                
                if self.options['verbose']>0:
                        limitbar=self.numberOfVars
                        prog = ProgressBar(0,limitbar, 77, mode='fixed')
                        oldprog = str(prog)
                        print('Creating variables...')
                        print
                
                #variables
                
                colnames=['']*self.numberOfVars
                obj=[0]*self.numberOfVars
                types=['C']*self.numberOfVars
                
                #specify bounds later, in constraints
                ub=[cplex.infinity]*self.numberOfVars
                lb=[-cplex.infinity]*self.numberOfVars
                
                if self.objective[1] is None:
                        objective = {}
                else:
                        objective = self.objective[1].factors
                
                for kvar,variable in self.variables.iteritems():
                        sj=self.variables[kvar].startIndex
                        if objective.has_key(kvar):
                                vectorObjective = objective[kvar]
                        else:
                                vectorObjective = [0]*(variable.size[0]*variable.size[1])
                        for k in range(variable.size[0]*variable.size[1]):
                                colnames[sj+k]=kvar+'_'+str(k)
                                obj[sj+k]=vectorObjective[k]
                                types[sj+k]=cplex_type[variable.vtype]
                                
                                if self.options['verbose']>0:
                                        #<--display progress
                                        prog.increment_amount()
                                        if oldprog != str(prog):
                                                print prog, "\r",
                                                sys.stdout.flush()
                                                oldprog=str(prog)
                                        #-->
                
                if self.options['verbose']>0:
                        prog.update_amount(limitbar)
                        print prog, "\r",
                        print
                
                #constraints
                
                #progress bar
                if self.options['verbose']>0:
                        print
                        print('adding constraints...')
                        print 
                        limitbar=self.numberAffConstraints
                        prog = ProgressBar(0,limitbar, 77, mode='fixed')
                        oldprog = str(prog)
                
                rows=[]
                cols=[]
                vals=[]
                rhs=[]
                senses= ''
                
                irow=0
                for constrKey,constr in self.constraints.iteritems():
                        nnz=0
                        for kvar,lin_expr_fact in constr.Exp1.factors.iteritems():
                                # add each var one by one if val =! 0
                                sj=self.variables[kvar].startIndex
                                for i,j,v in itertools.izip(lin_expr_fact.I,lin_expr_fact.J,lin_expr_fact.V):
                                        rows.append(irow+i)
                                        cols.append(sj+j)
                                        vals.append(v)
                                        nnz+=1
                                
                        for kvar,lin_expr_fact in constr.Exp2.factors.iteritems():
                                # add each var one by one if val =! 0
                                sj=self.variables[kvar].startIndex
                                for i,j,v in itertools.izip(lin_expr_fact.I,lin_expr_fact.J,lin_expr_fact.V):
                                        rows.append(irow+i)
                                        cols.append(sj+j)
                                        vals.append(-v)
                                        nnz+=1
                        
                        szcons = constr.Exp1.size[0]*constr.Exp1.size[1]
                        rhstmp = cvx.matrix(0.,(szcons,1))
                        constant1 = constr.Exp1.constant #None or a 1*1 matrix
                        constant2 = constr.Exp2.constant
                        if not constant1 is None:
                                rhstmp = rhstmp-constant1
                        if not constant2 is None:
                                rhstmp = rhstmp+constant2
                        
                        rhs.extend(rhstmp)
                        
                        if nnz == 1:
                                #BOUND
                                i=rows.pop()
                                j=cols.pop()
                                v=vals.pop()
                                r=rhs.pop()
                                b=r/float(v)
                                if len(rhstmp)>1:
                                        raise Exception('bound constraint with an RHS of dimension >1')
                                if constr.typeOfConstraint in ['lin<','lin=']:
                                        if b<ub[j]:
                                                ub[j]=b
                                if constr.typeOfConstraint in ['lin>','lin=']:
                                        if b>lb[j]:
                                                lb[j]=b
                        else:
                                if constr.typeOfConstraint == 'lin<':
                                        senses += "L"*szcons # lower
                                elif constr.typeOfConstraint == 'lin>':
                                        senses += "G"*szcons # greater
                                elif constr.typeOfConstraint == 'lin=':
                                        senses += "E"*szcons # equal
                                irow+=szcons
                        
                        if self.options['verbose']>0:
                                #<--display progress
                                prog.increment_amount()
                                if oldprog != str(prog):
                                        print prog, "\r",
                                        sys.stdout.flush()
                                        oldprog=str(prog)
                                #-->
                
                if self.options['verbose']>0:
                        prog.update_amount(limitbar)
                        print prog, "\r",
                        print
                
                print
                print('Passing to cplex...')
                c.variables.add(obj = obj, ub = ub, lb=lb, names = colnames,types=types)
                c.linear_constraints.add(rhs = rhs, senses = senses)
                c.linear_constraints.set_coefficients(zip(rows, cols, vals))
                
                # define problem type
                if self.is_continuous():
                        c.set_problem_type(c.problem_type.LP)
                
                self.cplex_Instance = c
                print('CPLEX INSTANCE created')
                return c, self

# -------------------- Tool for cplex -----------------
        def cvxInList(self, cvxArray):
                if cvxArray.size[0] != 1:
                        raise ValueError('cannot be converted as a list')
                listOut = []
                for i in range(cvxArray.size[1]):
                        listOut.append(cvxArray[0,i])
                return listOut

                
        def _make_cvxopt_instance(self):
                """
                defines the variables in self.cvxoptVars, used by the cvxopt solver
                """
                ss=self.numberOfVars
                #initial values                
                self.cvxoptVars['A']=cvx.spmatrix([],[],[],(0,ss),tc='d')
                self.cvxoptVars['b']=cvx.matrix([],(0,1),tc='d')
                self.cvxoptVars['Gl']=cvx.spmatrix([],[],[],(0,ss),tc='d')
                self.cvxoptVars['hl']=cvx.matrix([],(0,1),tc='d')
                self.cvxoptVars['Gq']=[]
                self.cvxoptVars['hq']=[]
                self.cvxoptVars['Gs']=[]
                self.cvxoptVars['hs']=[]
                self.cvxoptVars['quadcons']=[]
                #objective
                if isinstance(self.objective[1],QuadExp):
                        self.cvxoptVars['quadcons'].append(('_obj',-1))
                        objexp=self.objective[1].aff
                elif isinstance(self.objective[1],LogSumExp):
                        objexp=self.objective[1].Exp
                else:
                        objexp=self.objective[1]
                if self.numberLSEConstraints==0:
                        if self.objective[0]=='find':
                                self.cvxoptVars['c']=cvx.matrix(0,(ss,1),tc='d')
                        elif self.objective[0]=='min':
                                (c,constantInObjective)=self._makeGandh(objexp)
                                self.cvxoptVars['c']=cvx.matrix(c,tc='d').T
                        elif self.objective[0]=='max':
                                (c,constantInObjective)=self._makeGandh(objexp)
                                self.cvxoptVars['c']=-cvx.matrix(c,tc='d').T
                else:
                        if self.objective[0]=='find':
                                self.cvxoptVars['F']=cvx.matrix(0,(1,ss),tc='d')
                                self.cvxoptVars['K']=[0]
                        else:
                                (F,g)=self._makeGandh(objexp)
                                self.cvxoptVars['K']=[F.size[0]]
                                if self.objective[0]=='min':
                                        self.cvxoptVars['F']=cvx.matrix(F,tc='d')
                                        self.cvxoptVars['g']=cvx.matrix(g,tc='d')
                                elif self.objective[0]=='max':
                                        self.cvxoptVars['F']=-cvx.matrix(F,tc='d')
                                        self.cvxoptVars['g']=-cvx.matrix(g,tc='d')
                
                if self.options['verbose']>0:
                        limitbar=self.numberAffConstraints + self.numberConeConstraints + self.numberQuadConstraints + self.numberLSEConstraints + self.numberSDPConstraints
                        prog = ProgressBar(0,limitbar, 77, mode='fixed')
                        oldprog = str(prog)
                
                #constraints                
                for k in self.constraints.keys():
                        #linear constraints                        
                        if self.constraints[k].typeOfConstraint[:3]=='lin':
                                sense=self.constraints[k].typeOfConstraint[3]
                                (G_lhs,h_lhs)=self._makeGandh(self.constraints[k].Exp1)
                                (G_rhs,h_rhs)=self._makeGandh(self.constraints[k].Exp2)
                                if sense=='=':
                                        self.cvxoptVars['A']=cvx.sparse([self.cvxoptVars['A'],G_lhs-G_rhs])
                                        self.cvxoptVars['b']=cvx.matrix([self.cvxoptVars['b'],h_rhs-h_lhs])
                                elif sense=='<':
                                        self.cvxoptVars['Gl']=cvx.sparse([self.cvxoptVars['Gl'],G_lhs-G_rhs])
                                        self.cvxoptVars['hl']=cvx.matrix([self.cvxoptVars['hl'],h_rhs-h_lhs])
                                elif sense=='>':
                                        self.cvxoptVars['Gl']=cvx.sparse([self.cvxoptVars['Gl'],G_rhs-G_lhs])
                                        self.cvxoptVars['hl']=cvx.matrix([self.cvxoptVars['hl'],h_lhs-h_rhs])
                                else:
                                        raise NameError('unexpected case')
                        elif self.constraints[k].typeOfConstraint=='SOcone':
                                (A,b)=self._makeGandh(self.constraints[k].Exp1)
                                (c,d)=self._makeGandh(self.constraints[k].Exp2)
                                self.cvxoptVars['Gq'].append(cvx.sparse([-c,-A]))
                                self.cvxoptVars['hq'].append(cvx.matrix([d,b]))
                        elif self.constraints[k].typeOfConstraint=='RScone':
                                (A,b)=self._makeGandh(self.constraints[k].Exp1)
                                (c1,d1)=self._makeGandh(self.constraints[k].Exp2)
                                (c2,d2)=self._makeGandh(self.constraints[k].Exp3)
                                self.cvxoptVars['Gq'].append(cvx.sparse([-c1-c2,-2*A,c2-c1]))
                                self.cvxoptVars['hq'].append(cvx.matrix([d1+d2,2*b,d1-d2]))
                        elif self.constraints[k].typeOfConstraint=='lse':
                                (F,g)=self._makeGandh(self.constraints[k].Exp1)
                                self.cvxoptVars['F']=cvx.sparse([self.cvxoptVars['F'],F])
                                self.cvxoptVars['g']=cvx.matrix([self.cvxoptVars['g'],g])
                                self.cvxoptVars['K'].append(F.size[0])
                        elif self.constraints[k].typeOfConstraint=='quad':
                                self.cvxoptVars['quadcons'].append((k,self.cvxoptVars['Gl'].size[0]))
                                #quadratic part handled later
                                (G_lhs,h_lhs)=self._makeGandh(self.constraints[k].Exp1.aff)
                                self.cvxoptVars['Gl']=cvx.sparse([self.cvxoptVars['Gl'],G_lhs])
                                self.cvxoptVars['hl']=cvx.matrix([self.cvxoptVars['hl'],-h_lhs])
                        elif self.constraints[k].typeOfConstraint[:3]=='sdp':
                                sense=self.constraints[k].typeOfConstraint[3]
                                (G_lhs,h_lhs)=self._makeGandh(self.constraints[k].Exp1)
                                (G_rhs,h_rhs)=self._makeGandh(self.constraints[k].Exp2)
                                if sense=='<':
                                        self.cvxoptVars['Gs'].append(G_lhs-G_rhs)
                                        self.cvxoptVars['hs'].append(h_rhs-h_lhs)
                                elif sense=='>':
                                        self.cvxoptVars['Gs'].append(G_rhs-G_lhs)
                                        self.cvxoptVars['hs'].append(h_lhs-h_rhs)
                                else:
                                        raise NameError('unexpected case')
                        else:
                                raise NameError('unexpected case')
                        if self.options['verbose']>0:
                                #<--display progress
                                prog.increment_amount()
                                if oldprog != str(prog):
                                        print prog, "\r",
                                        sys.stdout.flush()
                                        oldprog=str(prog)
                                #-->
                        
                #reshape hs matrices as square matrices
                #for m in self.cvxoptVars['hs']:
                #        n=int(np.sqrt(len(m)))
                #        m.size=(n,n)
                   
                if self.options['verbose']>0:
                        prog.update_amount(limitbar)
                        print prog, "\r",
                        sys.stdout.flush()
                        print

        #-----------
        #mosek tool
        #-----------
        
        # Define a stream printer to grab output from MOSEK
        def streamprinter(self,text):
                sys.stdout.write(text)
                sys.stdout.flush()

        def _make_mosek_instance(self):
                """
                defines the variables msk_env and msk_task used by the solver mosek.
                """
                if self.options['verbose']>0:
                        print('build mosek instance')
                
                try:
                        import mosek
                except ImportError:
                        print('mosek library not found')

                #only change the objective coefficients
                if self.options['onlyChangeObjective']:
                        if self.msk_task is None:
                                raise Exception('option is only available when msk_task has been defined before')
                        newobj=self.options['onlyChangeObjective']
                        (cobj,constantInObjective)=self._makeGandh(newobj)
                        self.cvxoptVars['c']=cvx.matrix(cobj,tc='d').T
                        
                        for j in range(len(self.cvxoptVars['c'])):
                        # Set the linear term c_j in the objective.
                                self.msk_task.putcj(j,self.cvxoptVars['c'][j])
                        return
                                
                # Make a MOSEK environment
                env = mosek.Env ()
                # Attach a printer to the environment
                if self.options['verbose']>=1:
                        env.set_Stream (mosek.streamtype.log, self.streamprinter)
                # Create a task
                task = env.Task(0,0)
                # Attach a printer to the task
                if self.options['verbose']>=1:
                        task.set_Stream (mosek.streamtype.log, self.streamprinter)                                
                                
                                
                                
                #patch for quadratic problems with a single var
                if self.numberOfVars==1 and self.numberQuadConstraints>0:
                        ptch=self.add_variable('_ptch_',1)
                        self.add_constraint( ptch>0 )                                
                                
                                
                # Give MOSEK an estimate of the size of the input data.
                # This is done to increase the speed of inputting data.                                
                                
                self._make_cvxopt_instance()
                NUMVAR = self.numberOfVars+int(sum([Gk.size[0] for Gk in self.cvxoptVars['Gq']]))
                NUMCON = self.numberAffConstraints+int(sum([Gk.size[0] for Gk in self.cvxoptVars['Gq']]))
                NUMCONE = self.numberConeConstraints
                NUMANZ= len(self.cvxoptVars['A'].I)+len(self.cvxoptVars['Gl'].I)
                NUMQNZ= self.numberQuadNNZ

                if bool(self.cvxoptVars['Gs']) or bool(self.cvxoptVars['F']):
                        raise Exception('SDP or GP constraints are not implemented in mosek')

                # Append 'NUMCON' empty constraints.
                # The constraints will initially have no bounds.
                task.append(mosek.accmode.con,NUMCON)
                #Append 'NUMVAR' variables.
                # The variables will initially be fixed at zero (x=0).
                task.append(mosek.accmode.var,NUMVAR)

                #specifies the integer variables
                for k in self.variables:
                        if self.variables[k].vtype=='binary':
                                raise Exception('not implemented yet')
                        elif self.variables[k].vtype=='integer':
                                for i in xrange(self.variables[k].startIndex,self.variables[k].endIndex):
                                        task.putvartype(i,mosek.variabletype.type_int)

                for j in range(NUMVAR):
                        # Set the linear term c_j in the objective.
                        if j< self.numberOfVars:
                                if self.objective[0]=='max':         #max is handled directly by MOSEK,
                                                                #revert to initial value        
                                        task.putcj(j,-self.cvxoptVars['c'][j])
                                else:
                                        task.putcj(j,self.cvxoptVars['c'][j])
                        #make the variable free
                        task.putbound(mosek.accmode.var,j,mosek.boundkey.fr,0.,0.)

                #equality constraints:
                Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                ijvs=sorted(zip(Ai,Aj,Av))
                del Ai,Aj,Av
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                iaff=0
                for i,jv in itojv.iteritems():
                        J=[jvk[0] for jvk in jv]
                        V=[jvk[1] for jvk in jv]
                        if len(J)==1:
                                #fixed variable
                                b=self.cvxoptVars['b'][i]/V[0]
                                task.putbound(mosek.accmode.var,J[0],mosek.boundkey.fx,b,b)
                        else:
                        
                                #affine inequality
                                b=self.cvxoptVars['b'][i]
                                task.putaijlist([iaff]*len(J),J,V)
                                task.putbound(mosek.accmode.con,iaff,mosek.boundkey.fx,
                                                b,b)
                                iaff+=1

                #inequality constraints:
                Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                ijvs=sorted(zip(Gli,Glj,Glv))
                del Gli,Glj,Glv
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                        
                for i,jv in itojv.iteritems():
                        J=[jvk[0] for jvk in jv]
                        V=[jvk[1] for jvk in jv]
                        if len(J)==1 and (not (i in [t[1] for t in self.cvxoptVars['quadcons']])):
                                #bounded variable
                                bk,bl,bu=task.getbound(mosek.accmode.var,J[0])
                                b=self.cvxoptVars['hl'][i]/V[0]
                                if V[0]>0:
                                        #less than
                                        bu=min(b,bu)
                                if V[0]<0:
                                        #greater than
                                        bl=max(b,bl)
                                if bu==bl:
                                        task.putbound(mosek.accmode.var,J[0],mosek.boundkey.fx,bl,bu)
                                elif bl>bu:
                                        raise Exception('unfeasible bound for var '+str(J[0]))
                                else:
                                        if bl<-MSK_INFINITY:
                                                if bu>MSK_INFINITY:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.fr,bl,bu)
                                                else:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.up,bl,bu)
                                        else:
                                                if bu>MSK_INFINITY:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.lo,bl,bu)
                                                else:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.ra,bl,bu)
                        else:
                                #affine inequality
                                b=self.cvxoptVars['hl'][i]
                                task.putaijlist([iaff]*len(J),J,V)
                                task.putbound(mosek.accmode.con,iaff,mosek.boundkey.up,-MSK_INFINITY,b)
                                if i in [t[1] for t in self.cvxoptVars['quadcons']]:
                                        #affine part of a quadratic constraint
                                        qcons= [qc for (qc,l) in self.cvxoptVars['quadcons'] if l==i][0]
                                        qconsindex=self.cvxoptVars['quadcons'].index((qcons,i))
                                        self.cvxoptVars['quadcons'][qconsindex]=(qcons,iaff)
                                        #we replace the line number in Gl by the index of the MOSEK constraint
                                iaff+=1
                
                #conic constraints:
                icone=self.numberOfVars
                for k in range(NUMCONE):
                        #Gk x + sk = hk
                        istart=icone
                        for i in range(self.cvxoptVars['Gq'][k].size[0]):
                                J=list(self.cvxoptVars['Gq'][k][i,:].J)
                                V=list(self.cvxoptVars['Gq'][k][i,:].V)
                                h=self.cvxoptVars['hq'][k][i]
                                J.append(icone)
                                V.append(1)
                                task.putaijlist([iaff]*(1+len(J)),J,V)
                                task.putbound(mosek.accmode.con,iaff,mosek.boundkey.fx,h,h)
                                iaff+=1
                                icone+=1
                        iend=icone
                        #sk in quadratic cone
                        task.appendcone(mosek.conetype.quad, 0.0, range(istart,iend))

                #quadratic constraints:
                task.putmaxnumqnz(NUMQNZ)
                for (k,iaff) in self.cvxoptVars['quadcons']:
                        subI=[]
                        subJ=[]
                        subV=[]
                        if k=='_obj':
                                qexpr=self.objective[1]
                        else:
                                qexpr=self.constraints[k].Exp1

                        for i,j in qexpr.quad:
                                si,ei=self.variables[i].startIndex,self.variables[i].endIndex
                                sj,ej=self.variables[j].startIndex,self.variables[j].endIndex
                                Qij=qexpr.quad[i,j]
                                if not isinstance(Qij,cvx.spmatrix):
                                        Qij=cvx.sparse(Qij)
                                if si==sj:#put A+A' on the diag
                                        sI=list((Qij+Qij.T).I+si)
                                        sJ=list((Qij+Qij.T).J+sj)
                                        sV=list((Qij+Qij.T).V)
                                        for u in range(len(sI)-1,-1,-1):
                                                #remove when j>i
                                                if sJ[u]>sI[u]:
                                                        del sI[u]
                                                        del sJ[u]
                                                        del sV[u]
                                elif si>=ej: #add A in the lower triang
                                        sI=list(Qij.I+si)
                                        sJ=list(Qij.J+sj)
                                        sV=list(Qij.V)
                                else: #add B' in the lower triang
                                        sI=list((Qij.T).I+sj)
                                        sJ=list((Qij.T).J+si)
                                        sV=list((Qij.T).V)
                                subI.extend(sI)
                                subJ.extend(sJ)
                                subV.extend(sV)
                        
                        if k=='_obj':
                                task.putqobj(subI,subJ,subV)
                        else:
                                task.putqconk(iaff,subI,subJ,subV)
                #objective sense
                if self.objective[0]=='max':
                        task.putobjsense(mosek.objsense.maximize)
                else:
                        task.putobjsense(mosek.objsense.minimize)
                
                self.msk_env=env
                self.msk_task=task

                if self.options['verbose']>0:
                        print('mosek instance built')


        def _make_zibopt(self):
                """
                Defines the variables scip_solver and scip_vars, used by the zibopt solver.
                
                .. TODO:: handle the quadratic problems
                """
                try:
                        from zibopt import scip
                except:
                        raise Exception('scip library not found')
                
                scip_solver = scip.solver(quiet=not(self.options['verbose']))
                
                if bool(self.cvxoptVars['Gs']) or bool(self.cvxoptVars['F']) or bool(self.cvxoptVars['Gq']):
                        raise Exception('SDP, SOCP, or GP constraints are not implemented in mosek')
                if bool(self.cvxoptVars['quadcons']):
                        raise Exception('not implemented yet')
                
                self._make_cvxopt_instance()
                #TODO integer vars
                #max handled directly by scip
                if self.objective[0]=='max':
                        self.cvxoptVars['c']=-self.cvxoptVars['c']
                
                zib_types={ 'continuous':scip.CONTINUOUS,
                            'integer'   :scip.INTEGER,
                            'binary'    :scip.BINARY,
                           }
                types=[0]*self.cvxoptVars['A'].size[1]
                for var in self.variables.keys():
                                si=self.variables[var].startIndex
                                ei=self.variables[var].endIndex
                                vtype=self.variables[var].vtype
                                try:
                                        types[si:ei]=[zib_types[vtype]]*(ei-si)
                                except:
                                        raise Exception('this vtype is not handled by scip: '+str(vtype))
                
                x=[]
                for i in range(self.cvxoptVars['A'].size[1]):
                    x.append(scip_solver.variable(types[i],
                                lower=-MSK_INFINITY,
                                coefficient=self.cvxoptVars['c'][i])
                            )
                
                Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                ijvs=sorted(zip(Ai,Aj,Av))
                del Ai,Aj,Av
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                        
                for i,jv in itojv.iteritems():
                        exp=0
                        for term in jv:
                                exp+= term[1]*x[term[0]]
                        scip_solver += exp == self.cvxoptVars['b'][i]
                        
                #inequalities
                Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                ijvs=sorted(zip(Gli,Glj,Glv))
                del Gli,Glj,Glv
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                        
                for i,jv in itojv.iteritems():
                        exp=0
                        for term in jv:
                                exp+= term[1]*x[term[0]]
                        scip_solver += exp <= self.cvxoptVars['hl'][i]
                                        
                
                self.scip_solver=scip_solver
                self.scip_vars=x
                
                
                

                
        """
        -----------------------------------------------
        --                CALL THE SOLVER            --
        -----------------------------------------------
        """        

        def solve(self, **options):
                """
                Solves the problem.
                
                Once the problem has been solved, the optimal variables
                can be obtained thanks to the :func:``eval``. The
                duals variables can be accessed by the method :func:``dual``
                of the class :class:``Constraint``.
                
                :keyword options: A list of options to update before
                                  the call to the solver. In particular, 
                                  the solver can
                                  be specified here,
                                  under the form key = value.
                                  See the list of available options
                                  in :func:`set_all_options_to_default`
                :returns: A dictionary **sol** which contains the information
                            returned by the solver.
                
                .. TODO:: probleme ik ou il pas defini absence con= ou con< ?
                """
                if options is None: options={}
                self.update_options(**options)

                #self.eliminate_useless_variables()

                if isinstance(self.objective[1],GeneralFun):
                        return self._sqpsolve(options)
                
                #WARNING: Bug with cvxopt-mosek ?
                if (self.options['solver']=='CVXOPT' #obolete name, use lower case
                    or self.options['solver']=='cvxopt-mosek'
                    or self.options['solver']=='smcp'
                    or self.options['solver']=='cvxopt'):

                        primals,duals,obj,sol=self._cvxopt_solve()
                        
                # For cplex (only LP and MIP implemented)
                elif (self.options['solver']=='cplex'):
                        
                        primals,duals,obj,sol=self._cplex_solve()

                # for mosek
                elif (self.options['solver']=='MSK' #obsolete value, use lower case
                        or self.options['solver']=='mosek'):
                        
                        primals,duals,obj,sol=self._mosek_solve()

                
                elif (self.options['solver']=='zibopt'):
                        
                        primals,duals,obj,sol=self._zibopt_solve()

                else:
                        raise Exception('unknown solver')                       
                        #TODO:Other solvers (GUROBI, ...)
                
                for k in primals.keys():
                        if not primals[k] is None:
                                self.set_var_value(k,primals[k],optimalvar=True)
                                
                if 'noduals' in self.options and self.options['noduals']:
                        pass
                else:
                        for i,d in enumerate(duals):
                                self.constraints[i].set_dualVar(d)
                if obj=='toEval' and not(self.objective[1] is None):
                        obj=self.objective[1].eval()
                sol['obj']=obj
                return sol

                
        def _cvxopt_solve(self):
                """
                Solves a problem with the cvxopt solver.
                
                .. Todo:: * handle quadratic problems
                
                .. Warning:: CVXOPT will raise an error if an equality of
                             the form 0==0 is implied by matrices A and b.
                """
                
                #--------------------#
                # makes the instance #
                #--------------------#
                
                if self.options['onlyChangeObjective']:
                        if self.cvxoptVars['c'] is None:
                                raise Exception('option is only available when cvxoptVars has been defined before')
                        newobj=self.objective[1]
                        (cobj,constantInObjective)=self._makeGandh(newobj)
                        self.cvxoptVars['c']=cvx.matrix(cobj,tc='d').T
                else:
                        self._make_cvxopt_instance()
                #--------------------#        
                #  sets the options  #
                #--------------------#
                import cvxopt.solvers
                cvx.solvers.options['maxiters']=self.options['maxit']
                cvx.solvers.options['abstol']=self.options['abstol']
                cvx.solvers.options['feastol']=self.options['feastol']
                cvx.solvers.options['reltol']=self.options['reltol']
                cvx.solvers.options['show_progress']=bool(self.options['verbose']>0)
                if self.options['solver'].upper()=='CVXOPT':
                        currentsolver=None
                elif self.options['solver']=='cvxopt-mosek':
                        currentsolver='mosek'
                elif self.options['solver']=='smcp':
                        currentsolver='smcp'
                #-------------------------------#
                #  runs the appropriate solver  #
                #-------------------------------#
                if  self.numberQuadConstraints>0:#QCQP
                        probtype='QCQP'
                        raise Exception('CVXOPT with Quadratic constraints is not handled')
                elif self.numberLSEConstraints>0:#GP
                        if len(self.cvxoptVars['Gq'])+len(self.cvxoptVars['Gs'])>0:
                                raise Exception('cone constraints + LSE not implemented')
                        probtype='GP'
                        if self.options['verbose']>0:
                                print '-----------------------------------'
                                print '         cvxopt GP solver'
                                print '-----------------------------------'
                        sol=cvx.solvers.gp(self.cvxoptVars['K'],
                                                self.cvxoptVars['F'],self.cvxoptVars['g'],
                                                self.cvxoptVars['Gl'],self.cvxoptVars['hl'],
                                                self.cvxoptVars['A'],self.cvxoptVars['b'])
                #changes to adapt the problem for the conelp interface:
                elif currentsolver=='mosek':
                        if len(self.cvxoptVars['Gs'])>0:
                                raise Exception('CVXOPT does not handle SDP with MOSEK')                            
                        if len(self.cvxoptVars['Gq'])+len(self.cvxoptVars['Gs']):
                                if self.options['verbose']>0:
                                        print '------------------------------------------'
                                        print '  mosek LP solver interfaced by cvxopt'
                                        print '------------------------------------------'
                                sol=cvx.solvers.lp(self.cvxoptVars['c'],
                                                self.cvxoptVars['Gl'],self.cvxoptVars['hl'],
                                                self.cvxoptVars['A'],self.cvxoptVars['b'],
                                                solver=currentsolver)
                                probtype='LP'
                        else:
                                if self.options['verbose']>0:
                                        print '-------------------------------------------'
                                        print '  mosek SOCP solver interfaced by cvxopt'
                                        print '-------------------------------------------'
                                sol=cvx.solvers.socp(self.cvxoptVars['c'],
                                                        self.cvxoptVars['Gl'],self.cvxoptVars['hl'],
                                                        self.cvxoptVars['Gq'],self.cvxoptVars['hq'],
                                                        self.cvxoptVars['A'],self.cvxoptVars['b'],
                                                        solver=currentsolver)
                                probtype='SOCP'
                else:
                        dims={}
                        dims['s']=[int(np.sqrt(Gsi.size[0])) for Gsi in self.cvxoptVars['Gs']]
                        dims['l']=self.cvxoptVars['Gl'].size[0]
                        dims['q']=[Gqi.size[0] for Gqi in self.cvxoptVars['Gq']]
                        G=self.cvxoptVars['Gl']
                        h=self.cvxoptVars['hl']
                        # handle the equalities as 2 ineq for smcp
                        if currentsolver=='smcp':
                                if self.cvxoptVars['A'].size[0]>0:
                                       G=cvx.sparse([G,self.cvxoptVars['A']]) 
                                       G=cvx.sparse([G,-self.cvxoptVars['A']])
                                       h=cvx.matrix([h,self.cvxoptVars['b']])
                                       h=cvx.matrix([h,-self.cvxoptVars['b']])
                                       dims['l']+=(2*self.cvxoptVars['A'].size[0])

                        for i in range(len(dims['q'])):
                                G=cvx.sparse([G,self.cvxoptVars['Gq'][i]])
                                h=cvx.matrix([h,self.cvxoptVars['hq'][i]])

                                         
                        for i in range(len(dims['s'])):
                                G=cvx.sparse([G,self.cvxoptVars['Gs'][i]])
                                h=cvx.matrix([h,self.cvxoptVars['hs'][i]])
                        if currentsolver=='smcp':
                                try:
                                        import smcp
                                except:
                                        raise Exception('library smcp not found')
                                if self.options['smcp_feas']:
                                        sol=smcp.solvers.conelp(self.cvxoptVars['c'],
                                                        G,h,dims,feas=self.options['smcp_feas'])
                                else:
                                        sol=smcp.solvers.conelp(self.cvxoptVars['c'],
                                                        G,h,dims)
                        else:

                                if self.options['verbose']>0:
                                        print '--------------------------'
                                        print '  cvxopt CONELP solver'
                                        print '--------------------------'
                                sol=cvx.solvers.conelp(self.cvxoptVars['c'],
                                                        G,h,dims,
                                                        self.cvxoptVars['A'],
                                                        self.cvxoptVars['b'])
                        probtype='ConeLP'
                        
                                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                 
                primals={}
                if not (sol['x'] is None):
                        for var in self.variables.keys():
                                si=self.variables[var].startIndex
                                ei=self.variables[var].endIndex
                                varvect=sol['x'][si:ei]
                                if self.variables[var].vtype=='symmetric':
                                        varvect=svecm1(varvect) #varvect was the svec
                                                                #representation of X
                                
                                primals[var]=cvx.matrix(varvect, self.variables[var].size)
                else:
                        print('##################################')
                        print('WARNING: Primal Solution Not Found')
                        print('##################################')
                        primals=None

                #--------------------#
                # retrieve the duals #
                #--------------------#
                duals=[]
                if 'noduals' in self.options and self.options['noduals']:
                        pass
                else:
                        
                        printnodual=False
                        (indy,indzl,indzq,indznl,indzs)=(0,0,0,0,0)
                        if probtype=='LP' or probtype=='ConeLP':
                                zkey='z'
                        else:
                                zkey='zl'
                        zqkey='zq'
                        zskey='zs'
                        if probtype=='ConeLP':
                                indzq=dims['l']
                                zqkey='z'
                                zskey='z'
                                indzs=dims['l']+sum(dims['q'])
                        ykey='y'
                        if currentsolver=='smcp':
                                nbeq=self.cvxoptVars['A'].size[0]
                                indy=dims['l']-2*nbeq
                                ykey='z'

                        for k in self.constraints.keys():
                                #Equality
                                if self.constraints[k].typeOfConstraint=='lin=':
                                        if not (sol[ykey] is None):
                                                consSz=np.product(self.constraints[k].Exp1.size)
                                                duals.append(sol[ykey][indy:indy+consSz])
                                                indy+=consSz
                                                if currentsolver=='smcp':
                                                        dualm=sol[ykey][indy-consSz+nbeq:indy+nbeq]
                                                        duals[-1]-=dualm
                                        else:
                                                printnodual=True
                                                duals.append(None)
                                #Inequality
                                elif self.constraints[k].typeOfConstraint[:3]=='lin':
                                        if not (sol[zkey] is None):
                                                consSz=np.product(self.constraints[k].Exp1.size)
                                                if self.constraints[k].typeOfConstraint[3]=='<':
                                                        duals.append(sol[zkey][indzl:indzl+consSz])
                                                else:
                                                        duals.append(sol[zkey][indzl:indzl+consSz])
                                                indzl+=consSz
                                        else:
                                                printnodual=True
                                                duals.append(None)
                                #SOCP constraint [Rotated or not]
                                elif self.constraints[k].typeOfConstraint[2:]=='cone':
                                        if not (sol[zqkey] is None):
                                                if probtype=='ConeLP':
                                                        consSz=np.product(self.constraints[k].Exp1.size)+1
                                                        duals.append(sol[zqkey][indzq:indzq+consSz])
                                                        indzq+=consSz
                                                else:
                                                        duals.append(sol[zqkey][indzq])
                                                        indzq+=1
                                        else:
                                                printnodual=True
                                                duals.append(None)
                                #SDP constraint
                                elif self.constraints[k].typeOfConstraint[:3]=='sdp':
                                        if not (sol[zskey] is None):
                                                if probtype=='ConeLP':
                                                        consSz=np.product(self.constraints[k].Exp1.size)
                                                        duals.append(sol[zskey][indzs:indzs+consSz])
                                                        indzs+=consSz
                                                else:
                                                        duals.append(sol[zskey][indzs])
                                                        indzs+=1
                                        else:
                                                printnodual=True
                                                duals.append(None)
                                #GP constraint
                                elif self.constraints[k].typeOfConstraint=='lse':
                                        if not (sol['znl'] is None):
                                                consSz=np.product(self.constraints[k].Exp1.size)
                                                duals.append(sol['znl'][indznl:indznl+consSz])
                                                indznl+=consSz
                                        else:
                                                printnodual=True
                                                duals.append(None)
                                else:
                                        raise Exception('constraint cannot be handled')
                                
                        if printnodual:
                                print('################################')
                                print('WARNING: Dual Solution Not Found')
                                print('################################')                
                
                #-----------------#
                # objective value #
                #-----------------#
                if self.numberLSEConstraints>0:#GP
                        obj='toEval'
                else:#LP or SOCP
                        if sol['primal objective'] is None:
                                if sol['dual objective'] is None:
                                        obj=None
                                else:
                                        obj=sol['dual objective']
                        else:
                                if sol['dual objective'] is None:
                                        obj=sol['primal objective']
                                else:
                                        obj=0.5*(sol['primal objective']+sol['dual objective'])
                        
                        if self.objective[0]=='max' and not obj is None:
                                obj = -obj
                
                return (primals,duals,obj,sol)
                

        def  _cplex_solve(self):
                """
                Solves a problem with the cvxopt solver.
                
                .. Todo::  * set solver settings
                           * handle SOCP ?
                """
                #----------------------------#
                #  create the cplex instance #
                #----------------------------#
                self._make_cplex_instance()
                c = self.cplex_Instance
                
                if c is None:
                        raise ValueError('a cplex instance should have been created before')
                
                
                #TODO : settings parameters
                
                #--------------------#
                #  call the solver   #
                #--------------------#                
              
                if not self.options['nbsol'] is None:
                        try:
                                c.populate_solution_pool()
                        except CplexSolverError:
                                print "Exception raised during populate"
                else:
                        try:
                                c.solve()
                        except CplexSolverError:
                                print "Exception raised during solve"
                                
        
                self.cplex_Instance = c
                
                # solution.get_status() returns an integer code
                print("Solution status = " , c.solution.get_status(), ":",)
                # the following line prints the corresponding string
                print(c.solution.status[c.solution.get_status()])
                
                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                
                #primals
                numsol = c.solution.pool.get_num()
                if numsol>1:
                        objvals=[]
                        for i in range(numsol):
                                objvals.append((c.solution.pool.get_objective_value(i),i))
                        indsols=[]
                        rev=(self.objective[0]=='max')
                        for ob,ind in sorted(objvals,reverse=rev)[:self.options['nbsol']]:
                                indsols.append(ind)
                
                primals = {}
                for kvar in self.variables:
                        value = []
                        sz_var = self.variables[kvar].size[0]*self.variables[kvar].size[1]
                        for i in range(sz_var):
                                name = kvar + '_' + str(i)
                                value.append(c.solution.get_values(name))
                                             
                        primals[kvar] = cvx.matrix(value,self.variables[kvar].size)
                
                if numsol>1:
                        for ii,ind in enumerate(indsols):
                                for kvar in self.variables:
                                        value = []
                                        sz_var = self.variables[kvar].size[0]*self.variables[kvar].size[1]
                                        for i in range(sz_var):
                                                name = kvar + '_' + str(i)
                                                value.append(c.solution.pool.get_values(ind,name))
                                             
                                        primals[(ii,kvar)] = cvx.matrix(value,self.variables[kvar].size)
                        
                #--------------------#
                # retrieve the duals #
                #--------------------#
                duals = [] 
                if 'noduals' in self.options and self.options['noduals']:
                        pass
                else:
                        # not available for a MIP (only for LP)
                        if self.is_continuous():
                                pos_cplex = 0 # the next scalar constraint line to study (num of cplex)
                                # pos_interface the next vect constraint in our interface
                                # for each constraint
                                for pos_interface in self.constraints.keys():
                                        dim = self.constraints[pos_interface].Exp1.size[0]
                                        # take all the composants of the constraint
                                        dual_lines = range(pos_cplex, pos_cplex + dim)
                                        dual_values = c.solution.get_dual_values(dual_lines)
                                        duals.append(cvx.matrix(dual_values))
                                        pos_cplex += dim
                #-----------------#
                # objective value #
                #-----------------#             
                obj = c.solution.get_objective_value()
                sol = {'cplex_solution':c.solution}
                return (primals,duals,obj,sol)
                

        def _mosek_solve(self):
                """
                Solves the problem with mosek
                
                .. Todo:: * Solver settings
                          * Dual vars for QP
                          * Direct handling of rotated cones
                """
                #----------------------------#
                #  create the mosek instance #
                #----------------------------#
                import mosek
                self._make_mosek_instance()
                task=self.msk_task

                #TODO : settings parameters
                
                #--------------------#
                #  call the solver   #
                #--------------------# 
                #optimize
                task.optimize()
                
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)
                prosta = []
                solsta = []

                if self.is_continuous():
                        soltype=mosek.soltype.itr
                else:
                        soltype=mosek.soltype.itg

                [prosta,solsta] = task.getsolutionstatus(soltype)
                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                # Output a solution
                xx = np.zeros(self.numberOfVars, float)
                task.getsolutionslice(
                        soltype, mosek.solitem.xx, 0,self.numberOfVars, xx)
                #PRIMAL VARIABLES        
                primals={}
                if (solsta == mosek.solsta.optimal or
                        solsta == mosek.solsta.near_optimal or
                        solsta == mosek.solsta.unknown or
                        solsta == mosek.solsta.integer_optimal):
                        for var in self.variables.keys():
                                si=self.variables[var].startIndex
                                ei=self.variables[var].endIndex
                                varvect=xx[si:ei]
                                primals[var]=cvx.matrix(varvect, self.variables[var].size)
                        if solsta == mosek.solsta.near_optimal or solsta == mosek.solsta.unknown:
                                print('warning, solution status is ' +repr(solsta))
                else:
                        raise Exception('unknown status (solsta)')

                #--------------------#
                # retrieve the duals #
                #--------------------#
                duals=[]
                if 'noduals' in self.options and self.options['noduals']:
                        pass
                else:
                        idvarcone=self.numberOfVars #index of variables in cone
                        ideq=0 #index of equality constraint in cvxoptVars['A']
                        idconeq=0 #index of equality constraint in mosekcons (without fixed vars)
                        idin=0 #index of inequality constraint in cvxoptVars['Gl']
                        idconin=len([1 for ida in range(self.cvxoptVars['A'].size[0])
                                        if len(self.cvxoptVars['A'][ida,:].J)>1])
                                #index of inequality constraint in mosekcons (without fixed vars)
                        idcone=0 #number of seen cones
                        Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                        
                        #indices for ineq constraints
                        ijvs=sorted(zip(Gli,Glj,Glv),reverse=True)
                        if len(ijvs)>0:
                                (ik,jk,vk)=ijvs.pop()
                                curik=-1
                                delNext=False
                        del Gli,Glj,Glv
                        
                        #indices for eq constraints
                        Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                        ijls=sorted(zip(Ai,Aj,Av),reverse=True)
                        if len(ijls)>0:
                                (il,jl,vl)=ijls.pop()
                                curil=-1
                                delNext=False
                        del Ai,Aj,Av
                        
                        #now we parse the constraints
                        for k in self.constraints.keys():
                                #conic constraint
                                if self.constraints[k].typeOfConstraint[2:]=='cone':
                                        szcone=self.cvxoptVars['Gq'][idcone].size[0]
                                        v=np.zeros(szcone,float)
                                        task.getsolutionslice(soltype,mosek.solitem.snx,
                                                        idvarcone,idvarcone+szcone,v)
                                        duals.append(cvx.matrix(v))
                                        idvarcone+=szcone
                                        idcone+=1
                                elif self.constraints[k].typeOfConstraint=='lin=':
                                        szcons=int(np.product(self.constraints[k].Exp1.size))
                                        fxd=[]
                                        while il<ideq+szcons:
                                                if il!=curil:
                                                        fxd.append((il-ideq,jl,vl))
                                                        curil=il
                                                        delNext=True
                                                elif delNext:
                                                        del fxd[-1]
                                                        delNext=False
                                                try:
                                                        (il,jl,vl)=ijls.pop()
                                                except IndexError:
                                                        break
                                        
                                        v=np.zeros(szcons-len(fxd),float)
                                        if len(v)>0:
                                                task.getsolutionslice(soltype,mosek.solitem.y,
                                                        idconeq,idconeq+szcons-len(fxd),v)
                                        v=v.tolist()
                                        for (l,var,coef) in fxd: #dual of fixed var constraints
                                                duu=np.zeros(1,float)
                                                dul=np.zeros(1,float)
                                                task.getsolutionslice(soltype,mosek.solitem.sux,
                                                                        var,var+1,duu)
                                                task.getsolutionslice(soltype,mosek.solitem.slx,
                                                                        var,var+1,dul)
                                                v.insert(l,(dul[0]-duu[0])/coef)
                                        duals.append(cvx.matrix(v))
                                        ideq+=szcons
                                        idconeq+=(szcons-len(fxd))
                                elif self.constraints[k].typeOfConstraint[:3]=='lin':#inequality
                                        szcons=int(np.product(self.constraints[k].Exp1.size))
                                        fxd=[]
                                        while ik<idin+szcons:
                                                if ik!=curik:
                                                        fxd.append((ik-idin,jk,vk))
                                                        curik=ik
                                                        delNext=True
                                                elif delNext:
                                                        del fxd[-1]
                                                        delNext=False
                                                try:
                                                        (ik,jk,vk)=ijvs.pop()
                                                except IndexError:
                                                        break
                                                                                
                                        #for k in range(szcons):
                                                #if len(self.cvxoptVars['Gl'][idin+k,:].J)==1:
                                                        #fxd.append((k,
                                                                #self.cvxoptVars['Gl'][idin+k,:].J[0],
                                                                #self.cvxoptVars['Gl'][idin+k,:].V[0]))
                                        v=np.zeros(szcons-len(fxd),float)
                                        if len(v)>0:
                                                task.getsolutionslice(soltype,mosek.solitem.y,
                                                        idconin,idconin+szcons-len(fxd),v)
                                        v=v.tolist()
                                        for (l,var,coef) in fxd: #dual of simple bound constraints
                                                du=np.zeros(1,float)
                                                bound=self.cvxoptVars['hl'][idin+l]/coef
                                                bk,bl,bu=task.getbound(mosek.accmode.var,var)
                                                if coef>0: #upper bound
                                                        if bound==bu:
                                                                task.getsolutionslice(soltype,mosek.solitem.sux,
                                                                        var,var+1,du)
                                                                v.insert(l,-du[0]/coef)
                                                        else:
                                                                v.insert(l,0.) #inactive bound
                                                else:   #lower bound
                                                        if bound==bl:
                                                                task.getsolutionslice(soltype,mosek.solitem.slx,
                                                                        var,var+1,du)
                                                                v.insert(l,du[0]/coef)
                                                        else:
                                                                v.insert(l,0.) #inactive bound
                                        duals.append(cvx.matrix(v))
                                        idin+=szcons
                                        idconin+=(szcons-len(fxd))
                                else:
                                                print('dual for this constraint is not handled yet')
                                                duals.append(None)
                #-----------------#
                # objective value #
                #-----------------#  
                #OBJECTIVE
                sol = {'mosek_task':task}
                obj = 'toEval'

                #delete the patch variable for quad prog with 1 var
                if '_ptch_' in self.variables:
                        self.remove_variable('_ptch_')
                        del primals['_ptch_']
                
                return (primals,duals,obj,sol)                
                
        def _zibopt_solve(self):
                """
                Solves the problem with the zib optimization suite
                
                .. Todo:: * dual variables ?
                          * solver parameters
                """
                #-----------------------------#
                #  create the zibopt instance #
                #-----------------------------#
                self._make_zibopt()
                
                timelimit=10000000.
                gaplim=0.
                nbsol=-1
                if not self.options['timelimit'] is None:
                        timelimit=self.options['timelimit']
                if not self.options['gaplim'] is None:        
                        gaplim=self.options['gaplim']
                
                #if fact, nbsol is a limit on the number of feasible nodes visited
                #if not self.options['nbsol'] is None:
                #        nbsol=self.options['nbsol']
                
                #--------------------#
                #  call the solver   #
                #--------------------#                 
                if self.objective[0]=='max':
                        sol=self.scip_solver.maximize(time=timelimit,
                                                        gap=gaplim,
                                                        nsol=nbsol)
                else:
                        sol=self.scip_solver.minimize(time=timelimit,
                                                        gap=gaplim,
                                                        nsol=nbsol)
                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                val=sol.values()
                primals={}
                for var in self.variables.keys():
                        si=self.variables[var].startIndex
                        ei=self.variables[var].endIndex
                        varvect=self.scip_vars[si:ei]
                        primals[var]=cvx.matrix([val[v] for v in varvect],
                                self.variables[var].size)


                duals = []
                #-----------------#
                # objective value #
                #-----------------#  
                obj=sol.objective
                solt={}
                solt['zibopt_sol']=sol
                return (primals,duals,obj,solt)
                
                
        def _sqpsolve(self,options):
                """
                Solves the problem by sequential Quadratic Programming.
                """
                import copy
                for v in self.variables:
                        if self.variables[v].value is None:
                                self.set_var_value(v,cvx.uniform(self.variables[v].size))
                #lower the display level for mosek                
                self.options['verbose']-=1
                oldvar=self._eval_all()
                subprob=copy.deepcopy(self)
                print('solve by SQP method with proximal convexity enforcement')
                print('it:     crit\t\tproxF\tstep')
                print('---------------------------------------')
                converged=False
                k=1
                while not converged:
                        obj,grad,hess=self.objective[1].fun(self.objective[1].Exp.eval())
                        diffExp=self.objective[1].Exp-self.objective[1].Exp.eval()
                        quadobj0=obj+grad.T*diffExp+0.5*diffExp.T()*hess*diffExp
                        proxF=self.options['step_sqp']
                        #VARIANT IN CONSTRAINTS: DO NOT FORCE CONVEXITY...
                        #for v in subprob.variables.keys():
                        #        x=subprob.get_varExp(v)
                        #        x0=self.get_variable(v).eval()
                        #        subprob.add_constraint((x-x0).T()*(x-x0)<0.5)
                        solFound=False
                        while (not solFound):
                                if self.objective[0]=='max':
                                        quadobj=quadobj0-proxF*abs(diffExp)**2 #(proximal + force convexity)
                                else:
                                        quadobj=quadobj0+proxF*abs(diffExp)**2 #(proximal + force convexity)
                                subprob=copy.deepcopy(self)                                
                                subprob.set_objective(self.objective[0],quadobj)
                                if self.options['harmonic_steps'] and k>1:
                                        for v in subprob.variables.keys():
                                                x=subprob.get_varExp(v)
                                                x0=self.get_variable(v).eval()
                                                subprob.add_constraint((x-x0).T()*(x-x0)<(10./float(k-1)))
                                try:
                                        sol=subprob.solve()
                                        solFound=True
                                except Exception as ex:
                                        if str(ex)[:6]=='(1296)': #function not convex
                                                proxF*=(1+cvx.uniform(1))
                                        else:
                                                #reinit the initial verbosity
                                                self.options['verbose']+=1
                                                raise
                        if proxF>=100*self.options['step_sqp']:
                                #reinit the initial verbosity
                                self.options['verbose']+=1
                                raise Exception('function not convex before proxF reached 100 times the initial value')

                        for v in subprob.variables:
                                self.set_var_value(v,subprob.get_valued_variable(v))
                        newvar=self._eval_all()
                        step=np.linalg.norm(newvar-oldvar)
                        if isinstance(step,cvx.matrix):
                                step=step[0]
                        oldvar=newvar
                        if k==1:
                                print('  {0}:         --- \t{1:6.3f} {2:10.4e}'.format(k,proxF,step))
                        else:
                                print('  {0}:   {1:16.9e} {2:6.3f} {3:10.4e}'.format(k,obj,proxF,step))
                        k+=1
                        #have we converged ?
                        if step<self.options['tol']:
                                converged=True
                        if k>self.options['maxit']:
                                converged=True
                                print('Warning: no convergence after {0} iterations'.format(k))

                #reinit the initial verbosity
                self.options['verbose']+=1
                sol['lastStep']=step
                return sol
                
                
#----------------------------------------
#                 Obsolete functions
#----------------------------------------

        def set_varValue(self,name,value):
                self.set_var_value(name,value)
                
        def defaultOptions(self,**opt):
                self.set_all_options_to_default(opt)
                
        def set_options(self, **options):
                self.update_options( **options)

        def addConstraint(self,cons):
                self.add_constraint(cons)
       
        def isContinuous(self):
                return self.is_continuous()
                
        def makeCplex_Instance(self):
                self._make_cplex_instance()
                
        def makeCVXOPT_Instance(self):
                self._make_cvxopt_instance()
#----------------------------------------
#                 Variable class
#----------------------------------------

class Variable:
        def __init__(self,name,size,Id,startIndex, vtype = 'continuous',value=None):
                self.name=name
                self.size=size
                self.Id=Id
                self.vtype=vtype
                self.startIndex=startIndex #starting position in the global vector of all variables
                if vtype=='symmetric':
                        self.endIndex=startIndex+(size[0]*(size[0]+1))/2 #end position +1
                else:
                        self.endIndex=startIndex+size[0]*size[1] #end position +1
                self.value=value
                self.value_alt = {} #alternative values for solution pools

        def __str__(self):
                return '<variable {0}:({1} x {2}),{3}>'.format(
                        self.name,self.size[0],self.size[1],self.vtype)

#----------------------------------
#                Expression
#----------------------------------
                
class Expression:
        """the parent class of AffinExpr, Norm, LogSumExp, QuadExp, GeneralFun"""
	def __init__(self,string,variables):
                self.string=string
                if variables is None:
                        raise Exception('unexpected case')
                self.variables=variables
		
#----------------------------------
#                 AffinExpr class
#----------------------------------

class AffinExpr(Expression):
	"""a class for defining vectorial (or matrix) affine expressions
	*The dictionary 'factors' stores, for each variable, a
	tuple (fact,string), where 'string' is a representation of the
	linear combination, and 'fact' is the factor by which the variable
	it is multiplied, (if the variable is a matrix, then the factor
	is with respect to the column-vectorization of the variable).
	The factor is stored as a cvx matrix or cvx spmatrix.
	For example (if x is a vector variable and X is a matrix variable): 
		_the product A*x is stored as a pair 'x':A
		_the product A*X is stored as a pair 'X':blkdiag(A,...,A)
			where the bloc diagonal matrix has a block
			corresponding to each column of X
		_the scalar product <A,X> is stored as 'X':A[:].T
			where A[:] is the colum-vectorization of A
	*Similarly, the 'constant' attribute stores a tuple
		( <vectorized constant>,string )
		If the constant is 0, then 'constant' can be (None,'0')
	"""
	
	def __init__(self,factors=None,constant=None,
                        size=(1,1),
                        string='0',
                        variables=None
                        ):
		if factors is None:
                        factors={}
		Expression.__init__(self,string,variables)
		self.factors=factors
		self.constant=constant
		self.size=size
		#self.string=string
		#self.variables=variables
		
	def __str__(self):
		affstr='# ({0} x {1})-affine expression: '.format(self.size[0],
								self.size[1])
		affstr+=self.affstring()
		affstr+=' #'
		return affstr

	def __repr__(self):
		if self.isconstant():
			return str(self.eval())
		else:
			return self.__str__()
	def affstring(self):
		return self.string

	def eval(self,ind=None):
		if self.constant is None:
			val=cvx.spmatrix([],[],[],(self.size[0]*self.size[1],1))
		else:
			val=self.constant
		#for k in self.factors.keys():
			#if MATH_PROG_PROBLEMS['current'] is None:
				#raise Exception(k+' is not valued')			
			#if not MATH_PROG_PROBLEMS['current'].variables[k].value is None:
				#val=val+self.factors[k]*MATH_PROG_PROBLEMS['current'].variables[k].value[:]
			#else:
				#raise Exception(k+' is not valued')
                for k in self.factors.keys():
                        if k not in self.variables:
                                raise Exception(k+' is an unknown variable')
                        if ind is None:
                                if not self.variables[k].value is None:
                                        val=val+self.factors[k]*self.variables[k].value[:]
                                else:
                                        raise Exception(k+' is not valued')
                        else:
                                if ind in self.variables[k].value_alt:
                                        val=val+self.factors[k]*self.variables[k].value_alt[ind][:]
                                else:
                                        raise Exception(k+' does not have a value for the index '+str(ind))
		return cvx.matrix(val,self.size)
		
        def set_value(self,value):
                #TODO: Improve for set_value on items ?
                if len(self.factors)>1:
                        raise Exception('set_value can only be called on a simple Expression representing a variable')
                mm=self.factors.values()[0] #should be identity matrix
                name=self.factors.keys()[0]
                if name not in self.variables:
                        raise Exception('unexpected variable name')
                vtype=self.variables[name].vtype
                size=self.variables[name].size
                idmat=_svecm1_identity(vtype,size)
                
                #test if mm==idmat
                mlist=sorted(zip(mm.I,mm.J,mm.V))
                idlist=sorted(zip(idmat.I,idmat.J,idmat.V))
                if (mlist!=idlist):
                        raise Exception('set_value can only be called on a simple Expression representing a variable.')
                
                valuemat,valueString=_retrieve_matrix(value,self.variables[name].size)
                if valuemat.size<>self.variables[name].size:
                        raise Exception('should be of size {0}'.format(self.variables[name].size))
                if vtype=='symmetric':
                        valuemat=svec(valuemat)
                self.variables[name].value=valuemat
                
	def is0(self):
		return ( not(bool(self.constant)) and self.factors=={})

	def is1(self):
		if (self.constant is None):
			return False
		return (self.size==(1,1) and self.constant[0]==1 and self.factors=={})

	def isconstant(self):
		return self.factors=={}

	def T(self):
		"""Transposition"""
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
					self.string,variables=self.variables)
		for k in selfcopy.factors:
			bsize=selfcopy.size[0]
			bsize2=selfcopy.size[1]
			I0=[(i/bsize)+(i%bsize)*bsize2 for i in selfcopy.factors[k].I]
			J=selfcopy.factors[k].J
			V=selfcopy.factors[k].V
			selfcopy.factors[k]=cvx.spmatrix(V,I0,J,selfcopy.factors[k].size)
			'''old version -- not very efficient
			newfac=cvx.spmatrix([],[],[],(0,selfcopy.factors[k].size[1]))
			n=selfcopy.size[1]
			m=selfcopy.size[0]
			for j in range(m):
				for i in range(n):
					newfac=cvx.sparse([newfac,selfcopy.factors[k][i*m+j,:]])
			selfcopy.factors[k]=newfac
			'''
			
		if not (selfcopy.constant is None):
			selfcopy.constant=cvx.matrix(selfcopy.constant,
					selfcopy.size).T[:]
		selfcopy.size=(selfcopy.size[1],selfcopy.size[0])
		if ( ('*' in selfcopy.affstring()) or ('/' in selfcopy.affstring())
			or ('+' in selfcopy.affstring()) or ('-' in selfcopy.affstring()) ):
			selfcopy.string='( '+selfcopy.string+' ).T'
		else:
			selfcopy.string+='.T'
		return selfcopy

	def __rmul__(self,fact):
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
					self.string,variables=self.variables)
		if isinstance(fact,AffinExpr):
			if fact.isconstant():
				fac,facString=fact.eval(),fact.string
			else:
				raise Exception('not implemented')
		else:
			fac,facString=_retrieve_matrix(fact,self.size[0])		
		if fac.size==(1,1) and selfcopy.size[0]<>1:
			fac=fac[0]*cvx.spdiag([1.]*selfcopy.size[0])
		if self.size==(1,1) and fac.size[1]<>1:
			oldstring=selfcopy.string
			selfcopy=selfcopy.diag(fac.size[1])
			selfcopy.string=oldstring
		if selfcopy.size[0]<>fac.size[1]:
			raise Exception('incompatible dimensions')
		bfac=blocdiag(fac,selfcopy.size[1])
		for k in selfcopy.factors:
			newfac=bfac*selfcopy.factors[k]
			selfcopy.factors[k]=newfac
		if selfcopy.constant is None:
			newfac=None
		else:
			newfac=bfac*selfcopy.constant
		selfcopy.constant=newfac
		selfcopy.size=(fac.size[0],selfcopy.size[1])
		#the following removes 'I' from the string when a matrix is multiplied
                #by the identity. We leave the 'I' when the factor of identity is a scalar
		if len(facString)>0:		
			if facString[-1]=='I' and (len(facString)==1
				 or facString[-2].isdigit() or facString[-2]=='.') and (
				 self.size != (1,1)):
				facString=facString[:-1]
		if len(facString)>0:
			if ('+' in selfcopy.affstring()) or ('-' in selfcopy.affstring()):
				selfcopy.string=facString+'*( '+selfcopy.string+' )'
			else:
				selfcopy.string=facString+'*'+selfcopy.string
		return selfcopy


	
	def __mul__(self,fact):
		#product of 2 affine expressions
		if isinstance(fact,AffinExpr):
			if fact.isconstant():
				fac,facString=fact.eval(),fact.string		
			elif self.isconstant():
				return fact.__rmul__(self)
			elif self.size[0]==1 and fact.size[1]==1 and self.size[1]==fact.size[0]:
				#quadratic expression
				linpart=AffinExpr({},constant=None,size=(1,1),variables=self.variables)
				if not self.constant is None:
					linpart=linpart+self.constant.T*fact
				if not fact.constant is None:
					linpart=linpart+self*fact.constant
				if not ((fact.constant is None) or (self.constant is None)):
					linpart=linpart-self.constant.T*fact.constant
				"""if not ( self.constant is None or fact.constant is None):
					linpart.constant=self.constant.T*fact.constant
				if not fact.constant is None:				
					for k in self.factors.keys():
						linpart.factors[k]=fact.constant.T*self.factors[k]
				if not self.constant is None:
					for k in fact.factors.keys():
						if k in linpart.factors.keys():
							linpart.factors[k]+=self.constant.T*fact.factors[k]
						else:
							linpart.factors[k]=self.constant.T*fact.factors[k]
				"""
				quadpart={}
				for i in self.factors.keys():
					for j in fact.factors.keys():
						quadpart[i,j]=self.factors[i].T*fact.factors[j]
				stleft=self.affstring()
				stright=fact.affstring()
				if ('+' in stleft) or ('-' in stleft):
					if len(stleft)>3 and not(stleft[0]=='(' and stleft[-3:]==').T'):
						stleft='( '+stleft+' )'
				if ('+' in stright) or ('-' in stright):
					stright='( '+stright+' )'				
				if self.size[1]==1:
					return QuadExp(quadpart,linpart,stleft+'*'+stright,LR=(self,fact),variables=self.variables)
				else:
					return QuadExp(quadpart,linpart,stleft+'*'+stright,variables=self.variables)
			else:
				raise Exception('not implemented')
		elif isinstance(fact,QuadExp):
			return QuadExp*fact
		#product with a constant
		else:
			if self.size==(1,1): #scalar mult. of the constant
                                fac,facString=_retrieve_matrix(fact,None)
                        else: #normal matrix multiplication, we expect a size        
                                fac,facString=_retrieve_matrix(fact,self.size[1])
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
				self.string,variables=self.variables)
		if fac.size==(1,1) and selfcopy.size[1]<>1:
			fac=fac[0]*cvx.spdiag([1.]*selfcopy.size[1])
		if self.size==(1,1) and fac.size[0]<>1:
			oldstring=selfcopy.string
			selfcopy=selfcopy.diag(fac.size[0])
			selfcopy.string=oldstring
		prod=(self.T().__rmul__(fac.T)).T()
		prod.size=(selfcopy.size[0],fac.size[1])
		#the following removes 'I' from the string when a matrix is multiplied
		#by the identity. We leave the 'I' when the factor of identity is a scalar
		if len(facString)>0:
			if facString[-1]=='I' and (len(facString)==1
				 or facString[-2].isdigit() or facString[-2]=='.') and(
				 self.size != (1,1)):
				facString=facString[:-1]
		if len(facString)>0:
			if ('+' in selfcopy.affstring()) or ('-' in selfcopy.affstring()):
				prod.string='( '+selfcopy.string+' )*'+facString
			else:
				prod.string=selfcopy.string+'*'+facString
		else:
			prod.string=selfcopy.string
		return prod
	
	def __or__(self,fact):#scalar product
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
			self.string,variables=self.variables)
		if isinstance(fact,AffinExpr):
			if fact.isconstant():
				fac,facString=fact.eval(),fact.string
			elif self.isconstant():
				return fact.__ror__(self)	
			else:
				raise Exception('not implemented')
		else:		
			fac,facString=_retrieve_matrix(fact,self.size)
		if selfcopy.size<>fac.size:
			raise Exception('incompatible dimensions')
		cfac=fac[:].T
		for k in selfcopy.factors:
			newfac=cfac*selfcopy.factors[k]
			selfcopy.factors[k]=newfac
		if selfcopy.constant is None:
			newfac=None
		else:
			newfac=cfac*selfcopy.constant
		selfcopy.constant=newfac
		selfcopy.size=(1,1)
		if facString[-1]=='I' and (len(facString)==1
				 or facString[-2].isdigit() or facString[-2]=='.'):
			selfcopy.string=facString[:-1]+'trace( '+selfcopy.string+' )'
		else:
			#selfcopy.string= u'\u2329 '+selfcopy.string+' | '+facString+u' \u232a'
			selfcopy.string='〈 '+selfcopy.string+' | '+facString+' 〉'
			#'\xe2\x8c\xa9 '+selfcopy.string+' | '+facString+' \xe2\x8c\xaa'
		return selfcopy

	def __ror__(self,fact):
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
			self.string,variables=self.variables)
		if isinstance(fact,AffinExpr):
			if fact.isconstant():
				fac,facString=fact.eval(),fact.string
			else:
				raise Exception('not implemented')
		else:		
			fac,facString=_retrieve_matrix(fact,self.size)
		if selfcopy.size<>fac.size:
			raise Exception('incompatible dimensions')
		cfac=fac[:].T
		for k in selfcopy.factors:
			newfac=cfac*selfcopy.factors[k]
			selfcopy.factors[k]=newfac
		if selfcopy.constant is None:
			newfac=None
		else:
			newfac=cfac*selfcopy.constant
		selfcopy.constant=newfac
		selfcopy.size=(1,1)
		if facString[-1]=='I' and (len(facString)==1
				 or facString[-2].isdigit() or facString[-2]=='.'):
			selfcopy.string=facString[:-1]+'trace( '+selfcopy.string+' )'
		else:
			#selfcopy.string=u'\u2329 '+facString+' | '+selfcopy.string+u' \u232a'
			selfcopy.string='\xe2\x8c\xa9 '+facString+' | '+selfcopy.string+' \xe2\x8c\xaa'
		return selfcopy

	
	def __add__(self,term):
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
					self.string,variables=self.variables)
		if isinstance(term,AffinExpr):
			if term.size==(1,1) and self.size<>(1,1):
				oldstring=term.string
				term=cvx.matrix(1.,self.size)*term.diag(self.size[1])
				term.string='|'+oldstring+'|'
			if self.size==(1,1) and term.size<>(1,1):
				oldstring=self.string
				selfone=cvx.matrix(1.,term.size)*self.diag(term.size[1])
				selfone.string='|'+oldstring+'|'
				return (selfone+term)
			if term.size<>selfcopy.size:
				raise Exception('incompatible dimension in the sum')
			for k in term.factors.keys():
				if k in selfcopy.factors.keys():
					newfac=selfcopy.factors[k]+term.factors[k]
					selfcopy.factors[k]=newfac
				else:
					selfcopy.factors[k]=term.factors[k]
			if selfcopy.constant is None and term.constant is None:
				pass
			else:
				newCons=cvx.spmatrix([],[],[],selfcopy.size)[:]
				if not selfcopy.constant is None:
					newCons=newCons+selfcopy.constant
				if not term.constant is None:
					newCons=newCons+term.constant
				selfcopy.constant=newCons
			if term.affstring() not in ['0','']:
				if term.string[0]=='-':
					import re					
					if ('+' not in term.string[1:]) and (
						'-' not in term.string[1:]):
						selfcopy.string=selfcopy.string+' '+term.affstring()
					elif (term.string[1]=='(') and (
			  			 re.search('.*\)((\[.*\])|(.T))*$',term.string) ): 								#a group in a (...)
						selfcopy.string=selfcopy.string+' '+term.affstring()
					else:
						selfcopy.string=selfcopy.string+' + ('+ \
								term.affstring()+')'
				else:
					selfcopy.string+=' + '+term.affstring()
			return selfcopy
		elif isinstance(term,QuadExp):
			if self.size<>(1,1):
				raise Exception('LHS must be scalar')
			expQE=QuadExp({},self,self.affstring(),variables=self.variables)
			return expQE+term
		else: #constant term
			term,termString=_retrieve_matrix(term,selfcopy.size)
			return self+AffinExpr({},constant=term[:],size=term.size,string=termString,variables=self.variables)

	def __radd__(self,term):
		return self.__add__(term)

	def __neg__(self):
		selfneg=(-1)*self		
		if self.string<>'':
			if self.string[0]=='-':
				import re
				if ('+' not in self.string[1:]) and ('-' not in self.string[1:]):
					selfneg.string=self.string[1:]
				elif (self.string[1]=='(') and (
				   re.search('.*\)((\[.*\])|(.T))*$',self.string) ): #a group in a (...)
					if self.string[-1]==')':
						selfneg.string=self.string[2:-1] #we remove the parenthesis
					else:
						selfneg.string=self.string[1:] #we keep the parenthesis
				else:
					selfneg.string='-('+self.string+')'
			else:
				if ('+' in self.string) or ('-' in self.string):
					selfneg.string='-('+self.string+')'
				else:
					selfneg.string='-'+self.string
		return selfneg
		
	def __sub__(self,term):
		if isinstance(term,AffinExpr) or isinstance(term,QuadExp):
			return self+(-term)
		else: #constant term
			term,termString=_retrieve_matrix(term,self.size)
			return self-AffinExpr({},constant=term[:],size=term.size,string=termString,variables=self.variables)

	def __rsub__(self,term):
		return term+(-self)

	def __div__(self,divisor): #division (by a scalar)
		if isinstance(divisor,AffinExpr):
			if divisor.isconstant():
				divi,diviString=divisor.eval(),divisor.string
			else:
				raise Exception('not implemented')
			if divi.size<>(1,1):
				raise Exception('not implemented')
			if divi[0]==0:
				raise Exception('Division By Zero')
			division=self * (1/divi)
			if ('+' in self.string) or ('-' in self.string):
				division.string = '('+ self.string + ') /' + diviString
			else:
				division.string =  self.string+ ' / ' + diviString
			return division
		else : #constant term
			divi,diviString=_retrieve_matrix(divisor,(1,1))
			return self/AffinExpr({},constant=divi[:],size=(1,1),string=diviString,variables=self.variables)

	def __rdiv__(self,divider):
		divi,diviString=_retrieve_matrix(divider,None)
		return AffinExpr({},constant=divi[:],size=divi.size,string=diviString,variables=self.variables)/self
						

	def __getitem__(self,index):
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
				self.string,variables=self.variables)
		def slicestr(sli):
			if not (sli.start is None or sli.stop is None):
				if (sli.stop==sli.start+1):
					return str(sli.start)
			ss=''
			if not sli.start is None:
				ss+=str(sli.start)
			ss+=':'
			if not sli.stop is None:
				ss+=str(sli.stop)
			if not sli.step is None:
				ss+=':'
				ss+=str(sli.step)
			return ss
		if isinstance(index,int):
			index=slice(index,index+1,None)
		if isinstance(index,slice):
			idx=index.indices(self.size[0]*self.size[1])
			rangeT=range(idx[0],idx[1],idx[2])
			for k in selfcopy.factors.keys():
				selfcopy.factors[k]=selfcopy.factors[k][rangeT,:]
			if not selfcopy.constant is None:
				selfcopy.constant=selfcopy.constant[rangeT]
			selfcopy.size=(len(rangeT),1)
			indstr=slicestr(index)
		elif isinstance(index,tuple):
			if isinstance(index[0],int):
				index=(slice(index[0],index[0]+1,None),index[1])
			if isinstance(index[1],int):
				index=(index[0],slice(index[1],index[1]+1,None))
			idx0=index[0].indices(self.size[0])
			idx1=index[1].indices(self.size[1])
			rangei=range(idx0[0],idx0[1],idx0[2])
			rangej=range(idx1[0],idx1[1],idx1[2])
			rangeT=[]
			for j in rangej:
				rangei_translated=[]
				for vi in rangei:
					rangei_translated.append(
						vi+(j*self.size[0]))
				rangeT.extend(rangei_translated)
			for k in selfcopy.factors.keys():
				selfcopy.factors[k]=selfcopy.factors[k][rangeT,:]
			if not selfcopy.constant is None:	
				selfcopy.constant=selfcopy.constant[rangeT]
			selfcopy.size=(len(rangei),len(rangej))
			indstr=slicestr(index[0])+','+slicestr(index[1])
		if ('*' in selfcopy.affstring()) or ('+' in selfcopy.affstring()) or (
			'-' in selfcopy.affstring()) or ('/' in selfcopy.affstring()):
			selfcopy.string='( '+selfcopy.string+' )['+indstr+']'
		else:
			selfcopy.string=selfcopy.string+'['+indstr+']'
		return selfcopy
		
			
	def __lt__(self,exp):
		if isinstance(exp,AffinExpr):
			if exp.size==(1,1) and self.size<>(1,1):
				oldstring=exp.string
				exp=cvx.matrix(1.,self.size)*exp.diag(self.size[1])
				exp.string='|'+oldstring+'|'
			if self.size==(1,1) and exp.size<>(1,1):
				oldstring=self.string
				selfone=cvx.matrix(1.,exp.size)*self.diag(exp.size[1])
				selfone.string='|'+oldstring+'|'
				return (selfone<exp)
			return Constraint('lin<',None,self,exp)
		elif isinstance(exp,QuadExp):
			if (self.isconstant() and self.size==(1,1)
				and (not exp.LR is None) and (not exp.LR[1] is None)
			):
				cst=AffinExpr( factors={},constant=cvx.matrix(np.sqrt(self.eval()),(1,1)),
					size=(1,1),string=self.string,variables=self.variables)
				return (Norm(cst)**2)<exp
			elif self.size==(1,1):
				return (-exp)<(-self)
			else:
				raise Exception('not implemented')
		else:			
			term,termString=_retrieve_matrix(exp,self.size)
			exp2=AffinExpr(factors={},constant=term[:],size=self.size,string=termString,variables=self.variables)
			return Constraint('lin<',None,self,exp2)

	def __gt__(self,exp):
		if isinstance(exp,AffinExpr):
			if exp.size==(1,1) and self.size<>(1,1):
				oldstring=exp.string
				exp=cvx.matrix(1.,self.size)*exp.diag(self.size[1])
				exp.string='|'+oldstring+'|'
			if self.size==(1,1) and exp.size<>(1,1):
				oldstring=self.string
				selfone=cvx.matrix(1.,exp.size)*self.diag(exp.size[1])
				selfone.string='|'+oldstring+'|'
				return (selfone>exp)	
			return Constraint('lin>',None,self,exp)
		elif isinstance(exp,QuadExp):
			return exp<self
		else:			
			term,termString=_retrieve_matrix(exp,self.size)
			exp2=AffinExpr(factors={},constant=term[:],size=self.size,string=termString,variables=self.variables)
			return Constraint('lin>',None,self,exp2)

	def __eq__(self,exp):
		if isinstance(exp,AffinExpr):
			if exp.size==(1,1) and self.size<>(1,1):
				oldstring=exp.string
				exp=cvx.matrix(1.,self.size)*exp.diag(self.size[1])
				exp.string='|'+oldstring+'|'
			if self.size==(1,1) and exp.size<>(1,1):
				oldstring=self.string
				selfone=cvx.matrix(1.,exp.size)*self.diag(exp.size[1])
				selfone.string='|'+oldstring+'|'
				return (selfone==exp)
			return Constraint('lin=',None,self,exp)
		else:			
			term,termString=_retrieve_matrix(exp,self.size)
			exp2=AffinExpr(factors={},constant=term[:],size=self.size,string=termString,variables=self.variables)
			return Constraint('lin=',None,self,exp2)

	def __abs__(self):
		return Norm(self)

	def __pow__(self,exponent):
		if (self.size==(1,1) and self.isconstant()):
			return AffinExpr(factors={},constant=self.eval()[0]**exponent,
				size=(1,1),string='('+self.string+')**2',variables=self.variables)
		if (exponent<>2 or self.size<>(1,1)):
			raise Exception('not implemented')
		return Norm(self)**2

	def diag(self,dim):
		if self.size<>(1,1):
			raise Exception('not implemented')
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,
				self.string,variables=self.variables)
		idx=cvx.spdiag([1.]*dim)[:].I
		for k in self.factors.keys():
			selfcopy.factors[k]=cvx.spmatrix([],[],[],(dim**2,self.factors[k].size[1]))
			for i in idx:
				selfcopy.factors[k][i,:]=self.factors[k]
		selfcopy.constant=cvx.matrix(0.,(dim**2,1))
		if not self.constant is None:		
			for i in idx:
				selfcopy.constant[i]=self.constant[0]
		selfcopy.size=(dim,dim)
		selfcopy.string='diag('+selfcopy.string+')'
		return selfcopy

	def __and__(self,exp):
		"""horizontal concatenation"""
		selfcopy=AffinExpr(self.factors.copy(),self.constant,self.size,self.string,variables=self.variables)
		if isinstance(exp,AffinExpr):
			if exp.size[0]<>selfcopy.size[0]:
				raise Exception('incompatible size for concatenation')
			for k in list(set(exp.factors.keys()).union(set(selfcopy.factors.keys()))):
				if (k in selfcopy.factors.keys()) and (k in exp.factors.keys()):
					newfac=cvx.sparse([[selfcopy.factors[k],exp.factors[k]]])
					selfcopy.factors[k]=newfac
				elif k in exp.factors.keys():
					s1=selfcopy.size[0]*selfcopy.size[1]
					s2=exp.factors[k].size[1]
					newfac=cvx.sparse([[cvx.spmatrix([],[],[],(s1,s2)),
							exp.factors[k]]])
					selfcopy.factors[k]=newfac
				else:
					s1=exp.size[0]*exp.size[1]
					s2=selfcopy.factors[k].size[1]
					newfac=cvx.sparse([[selfcopy.factors[k],
						cvx.spmatrix([],[],[],(s1,s2))]])
					selfcopy.factors[k]=newfac
			if selfcopy.constant is None and exp.constant is None:
				pass
			else:
				s1=selfcopy.size[0]*selfcopy.size[1]
				s2=exp.size[0]*exp.size[1]
				if not selfcopy.constant is None:
					newCons=selfcopy.constant
				else:
					newCons=cvx.spmatrix([],[],[],(s1,1))
				if not exp.constant is None:
					newCons=cvx.sparse([[newCons,exp.constant]])
				else:
					newCons=cvx.sparse([[newCons,cvx.spmatrix([],[],[],(s2,1))]])
				selfcopy.constant=newCons
			selfcopy.size=(exp.size[0],exp.size[1]+selfcopy.size[1])
			sstring=selfcopy.string
			estring=exp.string
			if sstring[0]=='[' and sstring[-1]==']':
				sstring=sstring[1:-1]
			if estring[0]=='[' and estring[-1]==']':
				estring=estring[1:-1]
			selfcopy.string='['+sstring+','+estring+']'
			return selfcopy
		else:
			Exp,ExpString=_retrieve_matrix(exp,self.size[0])
			exp2=AffinExpr(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
			return (self & exp2)

	def __rand__(self,exp):
		Exp,ExpString=_retrieve_matrix(exp,self.size[0])
		exp2=AffinExpr(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
		return (exp2 & self)
			
	def __floordiv__(self,exp):
		"""vertical concatenation"""
		if isinstance(exp,AffinExpr):
			concat=(self.T() & exp.T()).T()
			concat.size=(exp.size[0]+self.size[0],exp.size[1])
			sstring=self.string
			estring=exp.string
			if sstring[0]=='[' and sstring[-1]==']':
				sstring=sstring[1:-1]
			if estring[0]=='[' and estring[-1]==']':
				estring=estring[1:-1]
			concat.string='['+sstring+';'+estring+']'
			return concat
		else:
			Exp,ExpString=_retrieve_matrix(exp,self.size[1])
			exp2=AffinExpr(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
			return (self // exp2)

	def __rfloordiv__(self,exp):
		Exp,ExpString=_retrieve_matrix(exp,self.size[1])
		exp2=AffinExpr(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
		return (exp2 // self)

        def apply_function(self,fun):
                return GeneralFun(fun,self,fun())
        
        def __lshift__(self,exp):
                if self.size[0]<>self.size[1]:
                        raise Exception('both sides of << must be square')
                if isinstance(exp,AffinExpr):
                        return Constraint('sdp<',None,self,exp)
                else:
                       n=self.size[0]
                       Exp,ExpString=_retrieve_matrix(exp,(n,n))
                       exp2=AffinExpr(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                       return (self << exp2)
                       
        def __rshift__(self,exp):
                if self.size[0]<>self.size[1]:
                        raise Exception('both sides of << must be square')
                if isinstance(exp,AffinExpr):
                        return Constraint('sdp>',None,self,exp)
                else:
                       n=self.size[0]
                       Exp,ExpString=_retrieve_matrix(exp,(n,n))
                       exp2=AffinExpr(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                       return (self >> exp2)

#---------------------------------------------
#        Class Norm and ProductOfAffinExpr  
#---------------------------------------------

class Norm(Expression):
	def __init__(self,exp):
                Expression.__init__(self,'||'+exp.string+'||',exp.variables)
		self.exp=exp
	def __str__(self):
		normstr='# norm of a ({0} x {1})- expression: ||'.format(self.exp.size[0],
								self.exp.size[1])
		normstr+=self.exp.affstring()
		normstr+='||'
		normstr+=' #'
		return normstr		
		
	def eval(self, ind=None):
		vec=self.exp.eval(ind)
		return np.linalg.norm(vec)

	def __pow__(self,exponent):
		if (exponent<>2):
			raise Exception('not implemented')
		if self.exp.isconstant():
			Qnorm=QuadExp({},
				AffinExpr(factors={},constant=self.exp.eval(),size=(1,1),string='  ',variables=self.variables),
				string='  ',
				variables=self.variables)
		else:
			Qnorm=QuadExp(None,None,None,None,variables=self.variables)
			#Qnorm=(self.exp.T())*(self.exp)
		Qnorm.LR=(self.exp,None)
		#if self.exp.size<>(1,1):
		Qnorm.string='||'+self.exp.affstring()+'||**2'
		#else:
		#	Qnorm.string='('+self.exp.affstring()+')**2'
		return Qnorm

	def __lt__(self,exp):
		if isinstance(exp,AffinExpr):
			if self.exp.size<>(1,1):
				return Constraint('SOcone',None,self.exp,exp)
			else:
				cons = (self.exp // -self.exp) < (exp // exp)
				if exp.is1():
					cons.myconstring= '||'+self.exp.string+'|| < 1'
				else:
					cons.myconstring= '||'+self.exp.string+'|| < '+exp.string
				cons.myfullconstring='# (1x1)-SOC constraint '+cons.myconstring+' #'
				return cons
		else:#constant		
			term,termString=_retrieve_matrix(exp,(1,1))
			exp1=AffinExpr(factors={},constant=term,size=(1,1),string=termString,variables=self.variables)
			return self<exp1

class LogSumExp(Expression):
	def __init__(self,exp):
                Expression.__init__(self,'LSE['+exp.string+']',exp.variables)
		self.Exp=exp
	def __str__(self):
		lsestr='# log-sum-exp of an affine expression: '
		lsestr+=self.Exp.affstring()
		lsestr+=' #'
		return lsestr

	def affstring(self):
		return 'LSE['+self.Exp.affstring()+']'

	def eval(self, ind=None):
		return np.log(np.sum(np.exp(self.Exp.eval(ind))))

	def __lt__(self,exp):
		if exp<>0:
			raise Exception('lhs must be 0')
		else:
			return Constraint('lse',None,self.Exp,0)

class QuadExp(Expression):
	"""
	quad are the quadratic factors,
	aff is the affine part of the expression,
	string is a string,
	and LR stores a factorization of the expression for norms (||x|| -> LR=(x,None))
								and product of scalar expressions)
								
	.. Todo:: quad expression of dimension>1							
	"""
	def __init__(self,quad,aff,string,LR=None,variables=None):
		Expression.__init__(self,string,variables)
		self.quad=quad
		self.aff=aff
		#self.string=string
		self.LR=LR

        def __str__(self):
                return '#quadratic expression: '+self.string+' #'

        def __repr__(self):
                return '#quadratic expression: '+self.string+' #'

        def eval(self, ind=None):
                if not self.aff is None:
                        val=self.aff.eval(ind)
                else:
                        val=cvx.matrix(0.,(1,1))
                if not self.LR is None:
                        ex1=self.LR[0].eval(ind)
                        if self.LR[1] is None:
                                val+=(ex1.T*ex1)
                        else:
                                if self.LR[0].size!=(1,1) or self.LR[1].size!=(1,1):
                                        raise Exception('QuadExp of size (1,1) only are implemented')
                                else:
                                        ex2=self.LR[1].eval(ind)
                                        val+=(ex1*ex2)
                                
                                
                elif not self.quad is None:
                        for i,j in self.quad:
                                if not i in self.variables:
                                        raise Exception(i+' is unknown')
                                if not j in self.variables:
                                        raise Exception(j+' is unknown')
                                if ind is None:
                                        if self.variables[i].value is None:
                                                raise Exception(i+' is not valued')
                                        if self.variables[j].value is None:
                                                raise Exception(j+' is not valued')
                                        xi=self.variables[i].value[:]
                                        xj=self.variables[j].value[:]
                                else:
                                        if ind not in self.variables[i].value_alt:
                                                raise Exception(i+' does not have a value for the index '+str(ind))
                                        if ind not in self.variables[j].value_alt:
                                                raise Exception(j+' does not have a value for the index '+str(ind))
                                        xi=self.variables[i].value_alt[ind][:]
                                        xj=self.variables[j].value_alt[ind][:]
                                val=val+xi.T*self.quad[i,j]*xj

                
		return val[0]
	
	def nnz(self):
		nz=0
		for ij in self.quad:
			nz+=len(self.quad[ij].I)
		return nz

	#OVERLOADS:
	#division par un scalaire

	def __mul__(self,fact):
		if isinstance(fact,AffinExpr):
			if fact.isconstant() and fact.size==(1,1):
				import copy
				selfcopy=QuadExp(self.quad.copy(),copy.deepcopy(self.aff),self.string,variables=self.variables)
				for ij in self.quad:
					selfcopy.quad[ij]=fact.eval()[0]*selfcopy.quad[ij]
				selfcopy.aff=fact*selfcopy.aff
				selfcopy.string=fact.affstring()+'*('+self.string+')'
				if not self.LR is None:
					if self.LR[1] is None and (fact.eval()[0]>=0): #Norm squared
						selfcopy.LR=(np.sqrt(fact.eval())*self.LR[0],None)
					elif self.LR[1] is None and (fact.eval()[0]<0):
						selfcopy.LR=None
					else:
						selfcopy.LR=(fact*self.LR[0],self.LR[1])
				return selfcopy
			else:
				raise Exception('not implemented')
		else: #constant term
			fact,factString=_retrieve_matrix(fact,(1,1))
			return self*AffinExpr({},constant=fact[:],size=fact.size,string=factString,variables=self.variables)

	def __add__(self,term):
		if isinstance(term,QuadExp):
			import copy
			selfcopy=QuadExp(self.quad.copy(),copy.deepcopy(self.aff),self.string,variables=self.variables)
			for ij in self.quad:
				if ij in term.quad.keys():
					selfcopy.quad[ij]=selfcopy.quad[ij]+term.quad[ij]
			for ij in term.quad:
				if not (ij in self.quad):
					selfcopy.quad[ij]=term.quad[ij]
			selfcopy.aff=selfcopy.aff+term.aff
			selfcopy.LR=None
			if term.string not in ['0','']:
				if term.string[0]=='-':
					import re					
					if ('+' not in term.string[1:]) and (
						'-' not in term.string[1:]):
						selfcopy.string=selfcopy.string+' '+term.string
					elif (term.string[1]=='(') and (
			  			 re.search('.*\)((\[.*\])|(.T))*$',term.string) ): 								#a group in a (...)
						selfcopy.string=selfcopy.string+' '+term.string
					else:
						selfcopy.string=selfcopy.string+' + ('+ \
								term.string+')'
				else:
					selfcopy.string+=' + '+term.string
			return selfcopy
		elif isinstance(term,AffinExpr):
			if term.size<>(1,1):
				raise Exception('RHS must be scalar')
			expQE=QuadExp({},term,term.affstring(),variables=self.variables)
			return self+expQE
		else:
			term,termString=_retrieve_matrix(term,(1,1))
			expAE=AffinExpr(factors={},constant=term,size=term.size,string=termString,variables=self.variables)
			return self+expAE

	def __rmul__(self,fact):
		return self*fact

	def __neg__(self):
		selfneg = (-1)*self
		if self.string[0]=='-':
			import re
			if ('+' not in self.string[1:]) and ('-' not in self.string[1:]):
				selfneg.string=self.string[1:]
			elif (self.string[1]=='(') and (
			   re.search('.*\)((\[.*\])|(.T))*$',self.string) ): #a group in a (...)
				if self.string[-1]==')':
					selfneg.string=self.string[2:-1] #we remove the parenthesis
				else:
					selfneg.string=self.string[1:] #we keep the parenthesis
			else:
				selfneg.string='-('+self.string+')'
		else:
			if ('+' in self.string) or ('-' in self.string):
				selfneg.string='-('+self.string+')'
			else:
				selfneg.string='-'+self.string
		return selfneg

	def __sub__(self,term):
		return self+(-term)

	def __rsub__(self,term):
		return term+(-self)

	def __radd__(self,term):
		return self+term

	def __lt__(self,exp):
		if isinstance(exp,QuadExp):		
			if ((not self.LR is None) and (self.LR[1] is None)
				and (not exp.LR is None) and (not exp.LR[1] is None)
			): #SOCP constraint
				return Constraint('RScone',None,self.LR[0],exp.LR[0],exp.LR[1])
			else:
				return Constraint('quad',None,self-exp,0)
		if isinstance(exp,AffinExpr):
			if exp.size<>(1,1):
				raise Exception('RHS must be scalar')
			exp2=AffinExpr(factors={},constant=cvx.matrix(1.,(1,1)),size=(1,1),string='1',variables=self.variables)
			expQE=QuadExp({},exp,exp.affstring(),LR=(exp,exp2),variables=self.variables)
			return self<expQE
		else:
			term,termString=_retrieve_matrix(exp,(1,1))
			expAE=AffinExpr(factors={},constant=term,size=(1,1),string=termString,variables=self.variables)
			return self<expAE

	def __gt__(self,exp):
		if isinstance(exp,QuadExp):
			if (not exp.LR is None) and (exp.LR[1] is None): # a squared norm
				return exp<self
			return (-self)<(-exp)
		if isinstance(exp,AffinExpr):
			if exp.size<>(1,1):
				raise Exception('RHS must be scalar')
			if exp.isconstant():
				cst=AffinExpr( factors={},constant=cvx.matrix(np.sqrt(exp.eval()),(1,1)),
					size=(1,1),string=exp.string,variables=self.variables)
				return (Norm(cst)**2)<self
			else:
				return (-self)<(-exp)
		else:
			term,termString=_retrieve_matrix(exp,(1,1))
			expAE=AffinExpr(factors={},constant=term,size=(1,1),string=termString,variables=self.variables)
			return self>expAE
				

class GeneralFun(Expression):
	"""a class storing a general scalar function,
		applied to an affine expression"""
	def __init__(self,fun,Exp,funstring):
                Expression.__init__(self,self.funstring+'( '+Exp.string+')',
                                        Exp.variables)
		self.fun=fun
		self.Exp=Exp
		self.funstring=funstring
		self.string=self.funstring+'( '+self.Exp.affstring()+' )'

	def __str__(self):
		return '# general function '+self.string+' #'
		

	def eval(self,ind=None):
		val=self.Exp.eval(ind)
		o,g,h=self.fun(val)
		return o

#----------------------------------
#                 Constraint class
#----------------------------------

class Constraint:
        """a class for describing a constraint (see the method add_constraint)
        """

	def __init__(self,typeOfConstraint,Id,Exp1,Exp2,Exp3=None,dualVariable=None,key=None):
		self.typeOfConstraint=typeOfConstraint
		self.Exp1=Exp1
		self.Exp2=Exp2
		self.Exp3=Exp3
		self.Id=Id
		self.dualVariable=dualVariable
		self.key=None
		self.myconstring = None
		self.myfullconstring = None
		if typeOfConstraint=='RScone' and Exp3 is None:
			raise NameError('I need a 3d expression')
		if typeOfConstraint[:3]=='lin':
			if Exp1.size<>Exp2.size:
				raise NameError('incoherent lhs and rhs')
		if typeOfConstraint[2:]=='cone':			
			if Exp2.size<>(1,1):
				raise NameError('expression on the rhs should be scalar')
			if not Exp3 is None:
				if Exp3.size<>(1,1):
					raise NameError(
					'expression on the rhs should be scalar')
		if typeOfConstraint=='lse':
			if not (Exp2==0 or Exp2.is0()):
				raise NameError('lhs must be 0')
			self.Exp2=AffinExpr(factors={},constant=cvx.matrix(0,(1,1)),string='0',size=(1,1),variables=self.variables)
		if typeOfConstraint=='quad':
			if not (Exp2==0 or Exp2.is0()):
				raise NameError('lhs must be 0')
			self.Exp2=AffinExpr(factors={},constant=cvx.matrix(0,(1,1)),string='0',size=(1,1),variables=self.variables)
                if typeOfConstraint[:3]=='sdp':
                        if Exp1.size<>Exp2.size:
                                raise NameError('incoherent lhs and rhs')
                        if Exp1.size[0]<>Exp1.size[1]:
                                raise NameError('lhs and rhs should be square')

	def __str__(self):
		if not(self.myfullconstring is None):
			return self.myfullconstring
		if self.typeOfConstraint[:3]=='lin':
			constr='# ({0}x{1})-affine constraint '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='SOcone':
			constr='# ({0}x{1})-SOC constraint '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='RScone':
			constr='# ({0}x{1})-Rotated SOC constraint '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='lse':
			constr='# ({0}x{1})-Log-Sum-Exp constraint '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='quad':
			constr='#Quadratic constraint '
                if self.typeOfConstraint[:3]=='sdp':
                        constr='# ({0}x{1})-SDP constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
		if not self.key is None:
			constr+='('+self.key+')'
		constr+=': '
		return constr+self.constring()+' #'
	
	def __repr__(self):
		if self.typeOfConstraint[:3]=='lin':
			constr='# ({0}x{1})-affine constraint: '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='SOcone':
			constr='# ({0}x{1})-SOC constraint: '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='RScone':
			constr='# ({0}x{1})-Rotated SOC constraint: '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='lse':
			constr='# ({0}x{1})-Log-Sum-Exp constraint '.format(
						self.Exp1.size[0],self.Exp1.size[1])
		if self.typeOfConstraint=='quad':
			constr='#Quadratic constraint '
                if self.typeOfConstraint[:3]=='sdp':
                        constr='# ({0}x{1})-SDP constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
		return constr+self.constring()+' #'

	def constring(self):
		if not(self.myconstring is None):
			return self.myconstring
		if self.typeOfConstraint[:3]=='lin':
			sense=' '+self.typeOfConstraint[-1]+' '
			if self.Exp2.is0():
				return self.Exp1.affstring()+sense+'0'
			else:
				return self.Exp1.affstring()+sense+self.Exp2.affstring()
		if self.typeOfConstraint=='SOcone':
			if self.Exp2.is1():
				return '||'+ self.Exp1.affstring()+'|| < 1'
			else:
				return '||'+ self.Exp1.affstring()+ \
					'|| < '+self.Exp2.affstring()
		if self.typeOfConstraint=='RScone':
			#if self.Exp1.size==(1,1):
			#	if self.Exp1.isconstant():
			#		retstr=self.Exp1.affstring() # warning: no square to simplfy
			#	else:
			#		retstr='('+self.Exp1.affstring()+')**2'
			if self.Exp1.size==(1,1) and self.Exp1.isconstant():
				retstr=self.Exp1.affstring() # warning: no square to simplfy
			else:
				retstr= '||'+ self.Exp1.affstring()+'||^2'
			if (self.Exp2.is1() and self.Exp3.is1()):
				return retstr+' < 1'
			elif self.Exp2.is1():
				return retstr+' < '+self.Exp3.affstring()
			elif self.Exp3.is1():
				return retstr+' < '+self.Exp2.affstring()
			else:
				return retstr+' < ( '+ \
				self.Exp2.affstring()+')( '+self.Exp3.affstring()+')'
		if self.typeOfConstraint=='lse':
			return 'LSE[ '+self.Exp1.affstring()+' ] < 0'
		if self.typeOfConstraint=='quad':
			return self.Exp1.string+' < 0'
		if self.typeOfConstraint[:3]=='sdp':
                        #sense=' '+self.typeOfConstraint[-1]+' '
                        if self.typeOfConstraint[-1]=='<':
                                sense = '≼' #≺,≼,⊰
                        else:
                                sense = '≽' #≻,≽,⊱
                        if self.Exp2.is0():
                                return self.Exp1.affstring()+sense+'0'
                        elif self.Exp1.is0():
                                return '0'+sense+self.Exp2.affstring()
                        else:
                                return self.Exp1.affstring()+sense+self.Exp2.affstring()

	def keyconstring(self,lgstkey=None):
		constr=''		
		if not self.key is None:
			constr+='('+self.key+')'
		if lgstkey is None:		
			constr+=':\t'			
		else:
			if self.key is None:			
				lcur=0
			else:
				lcur=len(self.key)+2
			if lgstkey==0:
				ntabs=0
			else:
				ntabs=int(np.ceil((2+lgstkey)/8.0))
			missingtabs=int(  np.ceil(((ntabs*8)-lcur)/8.0)  )
			for i in range(missingtabs):
				constr+='\t'
			if lcur>0:
				constr+=': '
			else:
				constr+='  '
			constr+=self.constring()
		return constr

	def set_dualVar(self,value):
		self.dualVariable=value
	
	def dual(self):
		return self.dualVariable

	def slack(self):
		if self.typeOfConstraint=='lin<':
			return self.Exp2.eval()-self.Exp1.eval()
		elif self.typeOfConstraint=='lin>':
			return self.Exp1.eval()-self.Exp2.eval()
		elif self.typeOfConstraint=='lin=':
			return self.Exp1.eval()-self.Exp2.eval()
		elif self.typeOfConstraint=='SOcone':
			return self.Exp2.eval()-(abs(self.Exp1)).eval()
		elif self.typeOfConstraint=='RScone':
			return self.Exp2.eval()[0]*self.Exp3.eval()[0]-(abs(self.Exp1)**2).eval()
		elif self.typeOfConstraint=='lse':
			return -lse(self.Exp1).eval()
		elif self.typeOfConstraint=='quad':
			return -(Exp1.eval())
