# coding: utf-8

#-------------------------------------------------------------------
#Picos 0.1.4 : A pyton Interface To Conic Optimization Solvers
#Copyright (C) 2012  Guillaume Sagnol
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#For any information, please contact:
#Guillaume Sagnol
#sagnol@zib.de
#Konrad Zuse Zentrum für Informationstechnik Berlin (ZIB)
#Takustrasse 7
#D-14195 Berlin-Dahlem
#Germany 
#-------------------------------------------------------------------

import cvxopt as cvx
import numpy as np
import sys, os

__all__=['_retrieve_matrix',
        '_svecm1_identity',
        'eval_dict',
        'putIndices',
        '_blocdiag',
        'svec',
        'svecm1',
        'ltrim1',
        'sum',
        '_bsum',
        'diag',
        'new_param',
        'available_solvers',
        'offset_in_lil',
        'diag_vect',
        '_quad2norm',
        '_copy_exp_to_new_vars',
        'ProgressBar',
        'QuadAsSocpError',
        'NotAppropriateSolverError',
        'NonConvexError',
        'DualizationError',
        'geomean',
        '_flatten',
        '_remove_in_lil',
        'norm',
        '_read_sdpa'
]


#----------------------------------------------------
#        Grouping constraints, summing expressions
#----------------------------------------------------

def sum(lst,it=None,indices=None):
        u"""sum of a list of affine expressions.
        This fonction can be used with python list comprehensions
        (see the example below).
        
        :param lst: list of :class:`AffinExp <picos.AffinExp>`.
        :param it: Description of the letters which should
                        be used to replace the dummy indices.
                        The function tries to find a template
                        for the string representations of the
                        affine expressions in the list.
                        If several indices change in the
                        list, their letters should be given as a
                        list of strings, in their order of appearance in
                        the resulting string. For example, if three indices
                        change in the summands, and you want them to be
                        named ``'i'``, ``'j'`` and ``'k'``, set ``it = ['i','j','k']``.
                        You can also group two indices which always appear together,
                        e.g. if ``'i'`` always appear next to ``'j'`` you
                        could set ``it = [('ij',2),'k']``. Here, the number 2
                        indicates that ``'ij'`` replaces 2 indices.
                        If ``it`` is set to ``None``, or if the function is not
                        able to find a template, the string of
                        the first summand will be used for
                        the string representation of the sum.
        :type it: None or str or list.
        :param indices: a string to denote the set where the indices belong to.
        :type indices: str.
                                
        **Example:**

        >>> import picos as pic
        >>> prob=pic.Problem()
        >>> x={}
        >>> names=['foo','bar','baz']
        >>> for n in names:
        ...   x[n]=prob.add_variable( 'x[{0}]'.format(n),(3,5) )
        >>> x #doctest: +NORMALIZE_WHITESPACE
        {'baz': # variable x[baz]:(3 x 5),continuous #,
         'foo': # variable x[foo]:(3 x 5),continuous #,
         'bar': # variable x[bar]:(3 x 5),continuous #}
        >>> pic.sum([x[n] for n in names],'n','names')
        # (3 x 5)-affine expression: Σ_{n in names} x[n] #
        >>> pic.sum([(i+1) * x[n] for i,n in enumerate(names)],['i','n'],'[3] x names') #two indices
        # (3 x 5)-affine expression: Σ_{i,n in [3] x names} i*x[n] #
        >>> IJ = [(1,2),(2,4),(0,1),(1,3)]
        >>> pic.sum([x['foo'][ij] for ij in IJ],[('ij',2)],'IJ') #double index
        # (1 x 1)-affine expression: Σ_{ij in IJ} x[foo][ij] #
        
        """
        from .expression import Expression
        from .expression import AffinExp
        if len(lst)==0:
                return AffinExp({},constant=0,size=(1,1),string='0')
        if not(all([isinstance(exi,Expression) for exi in lst])):
                import __builtin__
                return __builtin__.sum(lst)
        affSum=new_param('',cvx.matrix(0.,lst[0].size))
        for lsti in lst:
                affSum+=lsti
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

def _bsum(lst):
        """builtin sum operator"""
        import __builtin__
        return __builtin__.sum(lst)
        
def geomean(exp):
        """returns a :class:`GeoMeanExp <picos.GeoMeanExp>` object representing the geometric mean of the entries of ``exp[:]``.
        This can be used to enter inequalities of the form ``t <= geomean(x)``.
        Note that geometric mean inequalities are internally reformulated as a
        set of SOC inequalities.
        
        ** Example: **
        
        >>> import picos as pic
        >>> prob = pic.Problem()
        >>> x = prob.add_variable('x',1)
        >>> y = prob.add_variable('y',3)
        >>> # the following line adds the constraint x <= (y0*y1*y2)**(1./3) in the problem:
        >>> prob.add_constraint(x<pic.geomean(y))
       
        """
        from .expression import AffinExp
        from .expression import GeoMeanExp
        if not isinstance(exp,AffinExp):
                mat,name=_retrieve_matrix(exp)
                exp = AffinExp({},constant=mat[:],size=mat.size,string=name)
        return GeoMeanExp(exp)
        
def norm(exp,num=2,denom=1):
        """returns a :class:`NormP_Exp <picos.NormP_Exp>` object representing the (generalized-) p-norm of the entries
        of ``exp[:]``.
        This can be used to enter constraints of the form :math:`\Vert x \Vert_p \leq t` with :math:`p\geq1`.
        Generalized norms are also defined for :math:`p<1`, by using the usual formula
        :math:`\operatorname{norm}(x,p) := \Big(\sum_i x_i^p\Big)^{1/p}`. Note that this function
        is concave (for :math:`p<1`) over the set of vectors with nonnegative coordinates.
        When a constraint of the form :math:`\operatorname{norm}(x,p) > t` with :math:`p\leq1` is entered,
        PICOS implicitely assumes that :math:`x` is a nonnegative vector.
        
        The exponent :math:`p` of the norm must be specified either by
        a couple numerator (2d argument) / denominator (3d arguments),
        or directly by a float ``p`` given as second argument. In the latter case a rational
        approximation of ``p`` will be used.
        
        **Examples: **
        
        >>> import picos as pic
        >>> prob = pic.Problem()
        >>> x = prob.add_variable('x',1)
        >>> y = prob.add_variable('y',3)
        >>> pic.norm(y,7,3) < x
        # p-norm ineq : norm_7/3( y)<x#
        >>> pic.norm(y,-0.4) > x
        # generalized p-norm ineq : norm_-2/5( y)>x#
        
        """
        from .expression import AffinExp
        from .expression import NormP_Exp
        if not isinstance(exp,AffinExp):
                mat,name=_retrieve_matrix(exp)
                exp = AffinExp({},constant=mat[:],size=mat.size,string=name)
        if num == 2 and denom == 1:
                return abs(exp)
        p = float(num)/float(denom)
        if p==0:
                raise Exception('undefined for p=0')
        from fractions import Fraction
        frac = Fraction(p).limit_denominator(1000)
        return NormP_Exp(exp,frac.numerator,frac.denominator)
        
        
        
        
        
def allIdent(lst):
        if len(lst)<=1:
                return(True)
        return (np.array([lst[i]==lst[i+1] for i in range(len(lst)-1)]).all() )
               
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
        fr = cut_in_frames(lsStrings)
        frame = put_indices_on_frames(fr,it)
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

def is_index_char(char):
        return char.isalnum() or char=='_' or char=='.'
        
def findEndOfInd(string,curInd):
        indx=''
        while curInd<len(string) and is_index_char(string[curInd]):
                indx+=string[curInd]
                curInd+=1
        if indx=='':
                raise Exception('empty index')
        return curInd,indx

def cut_in_frames(lsStrings):
        n=len(lsStrings)
        curInd=n*[0]
        frame=[]
        while curInd[0]<len(lsStrings[0]):
                tmpName=[None]*n
                currentFramePiece=''
                piece_of_frame_found = False
                #go on while we have the same char
                while allIdent([lsStrings[k][curInd[k]] for k in range(n)]):
                        currentFramePiece+=lsStrings[0][curInd[0]]
                        piece_of_frame_found = True
                        curInd=[c+1 for c in curInd]
                        if curInd[0]>=len(lsStrings[0]):
                                break
                if not piece_of_frame_found:
                        #there was no template frame between successive indices
                        if curInd[0]==0: #we are still at the beginning
                                pass
                        else:
                                raise Exception('unexpected template')
                #go back until we get a non index char
                #import pdb;pdb.set_trace()
                if curInd[0]<len(lsStrings[0]):
                        while curInd[0]>0 and is_index_char(lsStrings[0][curInd[0]-1]):
                                currentFramePiece=currentFramePiece[:-1]
                                curInd=[c-1 for c in curInd]
                frame.append(currentFramePiece)
                if curInd[0]<len(lsStrings[0]):
                        for k in range(n):
                                curInd[k],tmpName[k]=findEndOfInd(lsStrings[k],curInd[k])
                        frame.append(tmpName)
        return frame
        
def put_indices_on_frames(frames,indices):
        frames_index=[]
        #find indices of frames
        for i,f in enumerate(frames):
                if isinstance(f,list):
                        frames_index.append(i)
        replacement_index=[]
        index_types=[]
        non_replaced_index=[]
        
        #find index types
        for num_index,fi in enumerate(frames_index):
                alpha=[]
                num=0
                for t in frames[fi]:
                        tsp=t.split('.')
                        if (len(tsp)<=2 and
                                all([s.isdigit() for s in tsp if len(s)>0])
                           ):
                                num+=1
                        else:
                                alpha.append(t)
                #we have a mix of numeric and alpha types in a frame,
                #with always the same alpha: 
                # -> patch for sub sums with only one term,
                # that was not replaced by its index
                if len(alpha)>0 and num>0:
                        if allIdent(alpha):
                                #patch
                                replacement_index.append(alpha[0])
                                index_types.append('resolved')
                                non_replaced_index.append(num_index)
                        else:
                                raise Exception('mix of numeric and alphabetic indices'+
                                                'for the index number {0}'.format(num_index))
                elif len(alpha)>0 and num==0:
                        replacement_index.append(None)
                        index_types.append('alpha')
                elif len(alpha)==0 and num>0:
                        replacement_index.append(None)
                        index_types.append('num')
                else:
                        raise Exception('unexpected index type'+
                                        'for the index number {0}'.format(num_index))

        #set a replacement index
        previous_numeric_index=[]
        previous_alphabetic_index=[]
        ind_next_index=0
        for num_index,fi in enumerate(frames_index):
                if replacement_index[num_index] is None:
                        if index_types[num_index]=='num':
                                #check if we have a constant offset with a previous index
                                for j in previous_numeric_index:
                                        prev_frame = frames[frames_index[j]]
                                        diff=[float(i)-float(p) for (p,i) in zip(prev_frame,frames[fi])]
                                        if allIdent(diff):
                                                ind=replacement_index[j]
                                                if diff[0]>0:
                                                        ind+='+'
                                                if diff[0]!=0:
                                                        offset= diff[0]
                                                        if offset == int(offset):
                                                                offset=int(offset)
                                                        ind+=str(offset)
                                                replacement_index[num_index]=ind
                                                break
                                if not(replacement_index[num_index] is None):
                                        continue
                        elif index_types[num_index]=='alpha':
                                #check if we have the same index
                                for j in previous_alphabetic_index:
                                        prev_frame = frames[frames_index[j]]
                                        same = [st==st2 for st,st2 in zip(prev_frame,frames[fi])]
                                        if all(same):
                                               replacement_index[num_index]=replacement_index[j]
                                               break
                                if not(replacement_index[num_index] is None):
                                        continue
                                               
                        if ind_next_index >= len(indices):
                                raise Exception('too few indices')
                        replacement_index[num_index]=indices[ind_next_index]
                        ind_next_index += 1
                        if index_types[num_index]=='num':
                                previous_numeric_index.append(num_index)
                        if index_types[num_index]=='alpha':        
                                previous_alphabetic_index.append(num_index)
                        
        if len(indices)!=ind_next_index:
                raise Exception('too many indices')
        
        #return the complete frame
        ret_string=''
        for num_index,fi in enumerate(frames_index):
                frames[fi]=replacement_index[num_index]
        for st in frames:
                ret_string+=st
        return ret_string
                

def eval_dict(dict_of_variables):
        """
        if ``dict_of_variables`` is a dictionary
        mapping variable names (strings) to :class:`variables <picos.Variable>`,
        this function returns the dictionary ``names -> variable values``. 
        """
        valued_dict={}
        for k in dict_of_variables:
                valued_dict[k] = dict_of_variables[k].eval()
                if valued_dict[k].size == (1,1):
                        valued_dict[k] = valued_dict[k][0]
        return valued_dict



#---------------------------------------------
#        Tools of the interface
#---------------------------------------------

def _blocdiag(X,n,sub1=0,sub2='n'):
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

def lse(exp):
        """
        shorter name for the constructor of the class :class:`LogSumExp <picos.LogSumExp>`
        
        **Example**
        
        >>> import picos as pic
        >>> import cvxopt as cvx
        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',3)
        >>> A=pic.new_param('A',cvx.matrix([[1,2],[3,4],[5,6]]))
        >>> pic.lse(A*x)<0
        # (2x1)-Geometric Programming constraint LSE[ A*x ] < 0 #
        
        """
        from .expression import LogSumExp
        return LogSumExp(exp)

def diag(exp,dim=1):
        r"""
        if ``exp`` is an affine expression of size (n,m),
        ``diag(exp,dim)`` returns a diagonal matrix of size ``dim*n*m`` :math:`\times` ``dim*n*m``,
        with ``dim`` copies of the vectorized expression ``exp[:]`` on the diagonal.
        
        In particular:
        
          * when ``exp`` is scalar, ``diag(exp,n)`` returns a diagonal
            matrix of size :math:`n \times n`, with all diagonal elements equal to ``exp``.
          
          * when ``exp`` is a vector of size :math:`n`, ``diag(exp)`` returns the diagonal
            matrix of size :math:`n \times n` with the vector ``exp`` on the diagonal
        
        
        **Example**
        
        >>> import picos as pic
        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',1)
        >>> y=prob.add_variable('y',1)
        >>> pic.tools.diag(x-y,4)
        # (4 x 4)-affine expression: Diag(x -y) #
        >>> pic.tools.diag(x//y)
        # (2 x 2)-affine expression: Diag([x;y]) #
        
        """
        from .expression import AffinExp
        if not isinstance(exp,AffinExp):
                mat,name=_retrieve_matrix(exp)
                exp = AffinExp({},constant=mat[:],size=mat.size,string=name)
        (n,m)=exp.size
        expcopy=AffinExp(exp.factors.copy(),exp.constant,exp.size,
                        exp.string)
        idx=cvx.spdiag([1.]*dim*n*m)[:].I
        for k in exp.factors.keys():
                #ensure it's sparse
                mat=cvx.sparse(expcopy.factors[k])
                I,J,V=list(mat.I),list(mat.J),list(mat.V)
                newI=[]
                for d in range(dim):
                        for i in I:
                                newI.append(idx[i+n*m*d])
                expcopy.factors[k]=cvx.spmatrix(V*dim,newI,J*dim,((dim*n*m)**2,exp.factors[k].size[1]))
        expcopy.constant=cvx.matrix(0.,((dim*n*m)**2,1))
        if not exp.constant is None:           
                for k,i in enumerate(idx):
                        expcopy.constant[i]=exp.constant[k%(n*m)]
        expcopy._size=(dim*n*m,dim*n*m)
        expcopy.string='Diag('+exp.string+')'
        return expcopy

def diag_vect(exp):
        """
        Returns the vector with the diagonal elements of the matrix expression ``exp``
        
        **Example**
        
        >>> import picos as pic
        >>> prob=pic.Problem()
        >>> X=prob.add_variable('X',(3,3))
        >>> pic.tools.diag_vect(X)
        # (3 x 1)-affine expression: diag(X) #
        
        """
        from .expression import AffinExp
        (n,m)=exp.size
        n=min(n,m)
        idx=cvx.spdiag([1.]*n)[:].I
        expcopy=AffinExp(exp.factors.copy(),exp.constant,exp.size,
                        exp.string)
        proj=cvx.spmatrix([1.]*n,range(n),idx,(n,exp.size[0]*exp.size[1]))
        for k in exp.factors.keys():
                expcopy.factors[k] = proj * expcopy.factors[k]
        if not exp.constant is None:
                expcopy.constant = proj * expcopy.constant
        expcopy._size=(n,1)
        expcopy.string='diag('+exp.string+')'
        return expcopy
        
def _retrieve_matrix(mat,exSize=None):
        """
        parses the variable *mat* and convert it to a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
        If the variable **exSize** is provided, the function tries
        to return a matrix that matches this expected size, or raise an
        error.
        
        .. WARNING:: If there is a conflit between the size of **mat** and
                     the expected size **exsize**, the function might still
                     return something without raising an error !
        
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
        
        >>> import picos as pic
        >>> pic.tools._retrieve_matrix([1,2,3])
        (<3x1 sparse matrix, tc='d', nnz=3>, '[ 3 x 1 MAT ]')
        >>> pic.tools._retrieve_matrix('e_5(7,1)')
        (<7x1 sparse matrix, tc='d', nnz=1>, 'e_5')
        >>> print pic.tools._retrieve_matrix('e_11(7,2)')[0] #doctest: +NORMALIZE_WHITESPACE
        [   0        0       ]
        [   0        0       ]
        [   0        0       ]
        [   0        0       ]
        [   0        1.00e+00]
        [   0        0       ]
        [   0        0       ]
        >>> print pic.tools._retrieve_matrix('5.3I',(2,2))
        (<2x2 sparse matrix, tc='d', nnz=2>, '5.3I')

        """
        retstr=None
        from .expression import Expression
        if isinstance(mat,Expression) and mat.is_valued():
                if isinstance(mat.value,cvx.base.spmatrix) or isinstance(mat.value,cvx.base.matrix):
                        retmat=mat.value
                else:
                        retmat=cvx.matrix(mat.value)
        elif isinstance(mat,np.ndarray):
                retmat=cvx.matrix(mat,tc='d')
        elif isinstance(mat,cvx.base.matrix):
                if mat.typecode=='d':
                        retmat=mat
                else:
                        retmat=cvx.matrix(mat,tc='d')
        elif isinstance(mat,cvx.base.spmatrix):
                retmat=mat
        elif isinstance(mat,list):
                if (isinstance(exSize,tuple)
                    and len(exSize)==2 
                    and not(exSize[0] is None)
                    and not(exSize[1] is None)
                    and len(mat)==exSize[0]*exSize[1]):
                        retmat=cvx.matrix(np.array(mat),exSize,tc='d')
                else:
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
        
        #make sure it's sparse
        retmat=cvx.sparse(retmat)
        
        #look for a more appropriate string...
        if retstr is None:
                retstr='[ {0} x {1} MAT ]'.format(retmat.size[0],retmat.size[1])
        if not retmat: #|0|
                if retmat.size==(1,1):
                        retstr='0'
                else:
                        retstr='|0|'
        elif retmat.size==(1,1):
                retstr=str(retmat[0])
        elif (len(retmat.V) == retmat.size[0]*retmat.size[1]) and (
              max(retmat.V)==min(retmat.V)): #|alpha|
                if retmat[0]==0:
                        retstr='|0|'
                elif retmat[0]==1:
                        retstr='|1|'
                else:
                        retstr='|'+str(retmat[0])+'|'
        elif retmat.I.size[0]==1: #e_x
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
        return retmat,retstr

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

def svecm1(vec,triu=False):
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
                        if triu:
                                V.append(v/np.sqrt(2))
                        else:
                                I.append(c)
                                J.append(r)
                                V.extend([v/np.sqrt(2)]*2)
        return cvx.spmatrix(V,I,J,(n,n))
             
                
def ltrim1(vec):
        if vec.size[1]>1:
                raise ValueError('should be a column vector')
        v=vec.size[0]
        n=int(np.sqrt(1+8*v)-1)/2
        if n*(n+1)/2 != v:
                raise ValueError('vec should be of dimension n(n+1)/2')
        if not isinstance(vec,cvx.matrix):
                vec=cvx.matrix(vec)
        M = cvx.matrix(0.,(n,n))
        r = 0
        c = 0
        for v in vec:
                if r==n:
                        c+=1
                        r=c
                M[r,c]=v
                if r>c:
                        M[c,r]=v
                r+=1

        return M
        
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
        
def new_param(name,value):
        """
        Declare a parameter for the problem, that will be stored
        as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
        It is possible to give a list or a dictionary of parameters.
        The function returns a constant :class:`AffinExp <picos.AffinExp>` 
        (or a ``list`` or a ``dict`` of :class:`AffinExp <picos.AffinExp>`) representing this parameter.
        
        .. note :: Declaring parameters is optional, since the expression can
                        as well be given by using normal variables. (see Example below).
                        However, if you use this function to declare your parameters,
                        the names of the parameters will be displayed when you **print**
                        an :class:`Expression <picos.Expression>` or a :class:`Constraint <picos.Constraint>`
        
        :param name: The name given to this parameter.
        :type name: str.
        :param value: The value (resp ``list`` of values, ``dict`` of values) of the parameter.
                        The type of **value** (resp. the elements of the ``list`` **value**,
                        the values of the ``dict`` **value**) should be understandable by
                        the function :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`.
        :returns: A constant affine expression (:class:`AffinExp <picos.AffinExp>`)
                        (resp. a ``list`` of :class:`AffinExp <picos.AffinExp>` of the same length as **value**,
                        a ``dict`` of :class:`AffinExp <picos.AffinExp>` indexed by the keys of **value**)
                        
        **Example:**
        
        >>> import cvxopt as cvx
        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',3)
        >>> B={'foo':17.4,'matrix':cvx.matrix([[1,2],[3,4],[5,6]]),'ones':'|1|(4,1)'}
        >>> B['matrix']*x+B['foo']
        # (2 x 1)-affine expression: [ 2 x 3 MAT ]*x + |17.4| #
        >>> #(in the string above, |17.4| represents the 2-dim vector [17.4,17.4])
        >>> B=pic.new_param('B',B)
        >>> #now that B is a param, we have a nicer display:
        >>> B['matrix']*x+B['foo']
        # (2 x 1)-affine expression: B[matrix]*x + |B[foo]| #
        """
        from .expression import AffinExp
        if isinstance(value,list):
                if all([isinstance(x,int) or isinstance(x,float) for x in value]):
                        #list with numeric data
                        term,termString=_retrieve_matrix(value,None)
                        return AffinExp({},constant=term[:],size=term.size,string=name)
                elif (all([isinstance(x,list) for x in value]) and
                    all([len(x)==len(value[0]) for x in value]) and
                    all([isinstance(xi,int) or isinstance(xi,float) for x in value for xi in x])
                    ):
                        #list of numeric lists of the same length
                        sz=len(value),len(value[0])
                        term,termString=_retrieve_matrix(value,sz)
                        return AffinExp({},constant=term[:],size=term.size,string=name)
                else:
                        L=[]
                        for i,l in enumerate(value):
                                L.append( new_param(name+'['+str(i)+']',l) )
                        return L
        elif isinstance(value,tuple):
                #handle as lists, but ignores numeric list and tables (vectors or matrices)
                L=[]
                for i,l in enumerate(value):
                        L.append( new_param(name+'['+str(i)+']',l) )
                return L
        elif isinstance(value,dict):
                D={}
                for k in value.keys():
                        D[k]=new_param(name+'['+str(k)+']',value[k])
                return D
        else:
                term,termString=_retrieve_matrix(value,None)
                return AffinExp({},constant=term[:],size=term.size,string=name)

def available_solvers():
        """Lists all available solvers"""
        lst=[]
        try:
                import cvxopt as co
                lst.append('cvxopt')
                del co
        except ImportError:
                pass
        try:
                import smcp as sm
                lst.append('smcp')
                del sm
        except ImportError:
                pass
        try:
		import mosek7 as mo7
		lst.append('mosek7')
		del mo7
		try:
                        import mosek as mo
                        version7 = not(hasattr(mo,'cputype'))
                        if not version7:
                                lst.append('mosek6')
                        del mo
                except ImportError:
                        pass
	except ImportError:#only one default mosek available
		try:
                        import mosek as mo
                        version7 = not(hasattr(mo,'cputype')) #True if this is the beta version 7 of MOSEK
                        del mo
                        if version7:
                                lst.append('mosek7')
                        else:
                                lst.append('mosek6')
                except ImportError:
                        pass
        try:
                import cplex as cp
                lst.append('cplex')
                del cp
        except ImportError:
                pass
        try:
                import zibopt as zo
                lst.append('zibopt')
                del zo
        except ImportError:
                pass
        try:
                import gurobipy as grb
                lst.append('gurobi')
                del grb
        except ImportError:
                pass
        #TRICK to force mosek6 during tests
        #if 'mosek7' in lst:
        #        lst.remove('mosek7')
        return lst
        
def offset_in_lil(lil,offset,lower):
        """
        substract the ``offset`` from all elements of the
        (recursive) list of lists ``lil``
        which are larger than ``lower``.
        """
        for i,l in enumerate(lil):
                if isinstance(l,int):
                      if l>lower:
                              lil[i]-=offset
                elif isinstance(l,list):
                        lil[i]=offset_in_lil(l,offset,lower)
                else:
                        raise Exception('elements of lil must be int or list')
        return lil

def _flatten(l):
        """ flatten a (recursive) list of list """
        for el in l:
                if hasattr(el, "__iter__") and not isinstance(el, basestring):
                        for sub in _flatten(el):
                                yield sub
                else:
                        yield el
                                
def _remove_in_lil(lil,elem):
        """ remove the element ``elem`` from a (recursive) list of list ``lil``.
            empty lists are removed if any"""
        if elem in lil:
                lil.remove(elem)
        for el in lil:
                if isinstance(el,list):
                        _remove_in_lil(el,elem)                
                        _remove_in_lil(el,[])
        if [] in lil: lil.remove([])
        
def _quad2norm(qd):
        """
        transform the list of bilinear terms qd
        in an equivalent squared norm
        (x.T Q x) -> ||Q**0.5 x||**2
        """
        #find all variables
        qdvars=[]
        for xy in qd:
                p1=(xy[0].startIndex,xy[0])
                p2=(xy[1].startIndex,xy[1])
                if p1 not in qdvars:
                        qdvars.append(p1)
                if p2 not in qdvars:
                        qdvars.append(p2)
        #sort by start indices
        qdvars=sorted(qdvars)
        qdvars=[v for (i,v) in qdvars]
        offsets={}
        ofs=0
        for v in qdvars:
                offsets[v]=ofs
                ofs+=v.size[0]*v.size[1]
               
        #construct quadratic matrix
        Q = cvx.spmatrix([],[],[],(ofs,ofs))
        I,J,V=[],[],[]
        for (xi,xj),Qij in qd.iteritems():
                oi=offsets[xi]
                oj=offsets[xj]
                Qtmp=cvx.spmatrix(Qij.V,Qij.I+oi,Qij.J+oj,(ofs,ofs))
                Q+=0.5*(Qtmp+Qtmp.T)
        #cholesky factorization V.T*V=Q
        #remove zero rows and cols
        nz=set(Q.I)
        P=cvx.spmatrix(1.,range(len(nz)),list(nz),(len(nz),ofs))
        Qp=P*Q*P.T
        try:
                import cvxopt.cholmod
                F=cvxopt.cholmod.symbolic(Qp)
                cvxopt.cholmod.numeric(Qp,F)
                Z=cvxopt.cholmod.spsolve(F,Qp,7)
                V=cvxopt.cholmod.spsolve(F,Z,4)
                V=V*P
        except ArithmeticError:#Singular or Non-convex, we must work on the dense matrix
                import cvxopt.lapack
                sig=cvx.matrix(0.,(len(nz),1),tc='z')
                U=cvx.matrix(0.,(len(nz),len(nz)))
                cvxopt.lapack.gees(cvx.matrix(Qp),sig,U)
                sig=sig.real()
                if min(sig)<-1e-7:
                        raise NonConvexError('I cannot convert non-convex quads to socp')
                for i in range(len(sig)): sig[i]=max(sig[i],0)
                V=cvx.spdiag(sig**0.5)*U.T
                V=cvx.sparse(V)*P
        allvars=qdvars[0]
        for v in qdvars[1:]:
                if v.size[1]==1:
                        allvars=allvars//v
                else:
                        allvars=allvars//v[:]
        return abs(V*allvars)**2
        
def _copy_dictexp_to_new_vars(dct,cvars):
        D = {}
        import copy
        for var,value in dct.iteritems():
                if isinstance(var,tuple):#quad
                        D[cvars[var[0].name],cvars[var[1].name]] = copy.copy(value)
                else:
                        D[cvars[var.name]] = copy.copy(value)
        return D

def _copy_exp_to_new_vars(exp,cvars):
        from .expression import Variable, AffinExp, Norm, LogSumExp, QuadExp, GeneralFun, GeoMeanExp,NormP_Exp
        import copy
        if isinstance(exp,Variable):
                return cvars[exp.name]
        elif isinstance(exp,AffinExp):
                newfacs = _copy_dictexp_to_new_vars(exp.factors,cvars)
                return AffinExp(newfacs,copy.copy(exp.constant),exp.size,exp.string)
        elif isinstance(exp,Norm):
                newexp =  _copy_exp_to_new_vars(exp.exp,cvars)
                return Norm(newexp)
        elif isinstance(exp,LogSumExp):
                newexp =  _copy_exp_to_new_vars(exp.Exp,cvars)
                return LogSumExp(newexp)
        elif isinstance(exp,QuadExp): 
                newaff =  _copy_exp_to_new_vars(exp.aff,cvars)
                newqds = _copy_dictexp_to_new_vars(exp.quad,cvars)
                if exp.LR is None:
                        return QuadExp(newqds,newaff,exp.string,None)
                else:
                        LR0 = _copy_exp_to_new_vars(exp.LR[0],cvars)
                        LR1 = _copy_exp_to_new_vars(exp.LR[1],cvars)
                        return QuadExp(newqds,newaff,exp.string,(LR0,LR1))
        elif isinstance(exp,GeneralFun): 
                newexp =  _copy_exp_to_new_vars(exp.Exp,cvars)
                return LogSumExp(exp.fun,newexp,exp.funstring)
        elif isinstance(exp,GeoMeanExp):
                newexp =  _copy_exp_to_new_vars(exp.exp,cvars)
                return GeoMeanExp(newexp)
        elif isinstance(exp,NormP_Exp):
                newexp =  _copy_exp_to_new_vars(exp.exp,cvars)
                return NormP_Exp(newexp)
        elif exp is None:
                return None
        else:
                raise Exception('unknown type of expression')          
        
def _copy_exp_to_new_vars_old(exp,cvars):#TOREM
        from .expression import Variable, AffinExp, Norm, LogSumExp, QuadExp, GeneralFun
        import copy
        ex2=copy.deepcopy(exp)
        if isinstance(exp,Variable):
                return cvars[exp.name]
        if isinstance(exp,AffinExp):
                for f in ex2.factors.keys():
                        mat=ex2.factors[f]
                        ex2.factors[cvars[f.name]]=mat
                        del ex2.factors[f]
                return ex2
        elif isinstance(exp,Norm):
                ex2.exp=_copy_exp_to_new_vars(ex2.exp,cvars)
                return ex2
        elif isinstance(exp,LogSumExp) or isinstance (exp,GeneralFun):
                ex2.Exp=_copy_exp_to_new_vars(ex2.Exp,cvars)
                return ex2
        elif isinstance(exp,QuadExp): 
                ex2.aff=_copy_exp_to_new_vars(ex2.aff,cvars)
                for (f,g) in ex2.quad.keys():
                        mat=ex2.quad[(f,g)]
                        ex2.quad[(cvars[f.name],cvars[g.name])]=mat
                        del ex2.quad[(f,g)]
                return ex2
        elif exp is None:
                return None
        else:
                raise Exception('unknown type of expression')
                
                
def _read_sdpa(filename):
                """TODO, remove dependence; currently relies on smcp sdpa_read
                cone constraints ||x||<t are recognized if they have the arrow form [t,x';x,t*I]>>0
                """
                f=open(filename,'r')
                import smcp
                from .problem import Problem
                A, bb, blockstruct = smcp.misc.sdpa_read(f)
                
                #make list of tuples (size,star,end) for each block
                blocks = []
                ix = 0
                for b in blockstruct:
                        blocks.append((b,ix,ix+b))
                        ix+=b
                
                P = Problem()
                nn = A.size[1]-1
                x = P.add_variable('x',nn)
                
                #linear (in)equalities
                Aineq = []
                bineq = []
                Aeq = []
                beq = []
                mm = int((A.size[0])**0.5)
                oldv = None
                for (_,i,_) in  [(sz,start,end) for (sz,start,end) in blocks if sz==1]:
                        ix = i*(mm+1)
                        v = A[ix,:]
                        if (oldv is not None) and np.linalg.norm((v+oldv).V)<1e-7:
                                Aineq.pop()
                                bineq.pop()
                                Aeq.append(v[1:].T)
                                beq.append(v[0])
                        else:
                                Aineq.append(v[1:].T)
                                bineq.append(v[0])
                        oldv = v
                Aeq = cvx.sparse(Aeq)
                Aineq = cvx.sparse(Aineq)
                if Aeq:
                        P.add_constraint(Aeq*x == beq)
                if Aineq:
                        P.add_constraint(Aineq*x > bineq)
                
                #sdp and soc constraints
                for (sz,start,end) in  [(sz,start,end) for (sz,start,end) in blocks if sz>1]:
                        MM = [cvx.sparse(cvx.matrix(A[:,i],(mm,mm)))[start:end,start:end] for i in range(nn+1)]
                        
                        #test whether it is an arrow pattern
                        isarrow = True
                        issoc = True
                        isrsoc = True
                        for M in MM:
                                for i,j in zip(M.I,M.J):
                                        if i>j and j>0:
                                                isarrow = False
                                                break
                                isrsoc = isrsoc and isarrow and all([M[i,i]==M[1,1] for i in xrange(2,sz)])
                                issoc = issoc and isrsoc and (M[1,1]==M[0,0])
                                if not(isrsoc): break
                                
                        if issoc or isrsoc:
                                Acone = cvx.sparse([M[:,0].T for M in MM[1:]]).T
                                bcone = MM[0][:,0]
                                if issoc:
                                        P.add_constraint(abs(Acone[1:,:]*x-bcone[1:]) < Acone[0,:]*x-bcone[0])
                                else:
                                        arcone = cvx.sparse([M[1,1] for M in MM[1:]]).T
                                        brcone = MM[0][1,1]
                                        P.add_constraint(abs(Acone[1:,:]*x-bcone[1:])**2 < (Acone[0,:]*x-bcone[0])*(arcone*x-brcone))
                                
                        else:
                                CCj = MM[0]+MM[0].T-cvx.spdiag([MM[0][i,i] for i in range(sz)])
                                MMj = [M+M.T-cvx.spdiag([M[i,i] for i in range(sz)]) for M in MM[1:]]
                                P.add_constraint(sum([x[i]*MMj[i] for i in range(nn) if MMj[i]],'i')>>CCj)
                 
                #objective
                P.set_objective('min',bb.T*x)
                return P      



# A Python Library to create a Progress Bar.
# Copyright (C) 2008  BJ Dierkes <wdierkes@5dollarwhitebox.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# This class is an improvement from the original found at:
#
#   http://code.activestate.com/recipes/168639/
#
 
class ProgressBar:
        def __init__(self, min_value = 0, max_value = 100, width=None,**kwargs):
                self.char = kwargs.get('char', '#')
                self.mode = kwargs.get('mode', 'dynamic') # fixed or dynamic
                if not self.mode in ['fixed', 'dynamic']:
                        self.mode = 'fixed'
        
                self.bar = ''
                self.min = min_value
                self.max = max_value
                self.span = max_value - min_value
                if width is None:
                        width=self.getTerminalSize()[1]-10
                self.width = width
                self.amount = 0       # When amount == max, we are 100% done 
                self.update_amount(0) 
        
        
        def increment_amount(self, add_amount = 1):
                """
                Increment self.amount by 'add_ammount' or default to incrementing
                by 1, and then rebuild the bar string. 
                """
                new_amount = self.amount + add_amount
                if new_amount < self.min: new_amount = self.min
                if new_amount > self.max: new_amount = self.max
                self.amount = new_amount
                self.build_bar()
        
        
        def update_amount(self, new_amount = None):
                """
                Update self.amount with 'new_amount', and then rebuild the bar 
                string.
                """
                if not new_amount: new_amount = self.amount
                if new_amount < self.min: new_amount = self.min
                if new_amount > self.max: new_amount = self.max
                self.amount = new_amount
                self.build_bar()
        
        def get_amount(self):
                return self.amount

        def build_bar(self):
                """
                Figure new percent complete, and rebuild the bar string base on 
                self.amount.
                """
                diff = float(self.amount - self.min)
                percent_done = int(round((diff / float(self.span)) * 100.0))
        
                # figure the proper number of 'character' make up the bar 
                all_full = self.width - 2
                num_hashes = int(round((percent_done * all_full) / 100))
        
                if self.mode == 'dynamic':
                        # build a progress bar with self.char (to create a dynamic bar
                        # where the percent string moves along with the bar progress.
                        self.bar = self.char * num_hashes
                else:
                        # build a progress bar with self.char and spaces (to create a 
                        # fixe bar (the percent string doesn't move)
                        self.bar = self.char * num_hashes + ' ' * (all_full-num_hashes)
        
                percent_str = str(percent_done) + "%"
                self.bar = '[ ' + self.bar + ' ] ' + percent_str
        
        
        def __str__(self):
                return str(self.bar)
                
                
        def getTerminalSize(self):
                """
                returns (lines:int, cols:int)
                """
                import os, struct
                def ioctl_GWINSZ(fd):
                        import fcntl, termios
                        return struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))
                # try stdin, stdout, stderr
                for fd in (0, 1, 2):
                        try:
                                return ioctl_GWINSZ(fd)
                        except:
                                pass
                # try os.ctermid()
                try:
                        fd = os.open(os.ctermid(), os.O_RDONLY)
                        try:
                                return ioctl_GWINSZ(fd)
                        finally:
                                os.close(fd)
                except:
                        pass
                # try `stty size`
                try:
                        return tuple(int(x) for x in os.popen("stty size", "r").read().split())
                except:
                        pass
                # try environment variables
                try:
                        return tuple(int(os.getenv(var)) for var in ("LINES", "COLUMNS"))
                except:
                        pass
                # i give up. return default.
                return (25, 80)
                
class QuadAsSocpError(Exception):
        """
        Exception raised when the problem can not be solved
        in the current form, because quad constraints are not handled.
        User should try to convert the quads as socp.
        """
        def __init__(self,msg):
                self.msg=msg
                
        def __str__(self): return self.msg
        def __repr__(self): return "QuadAsSocpError('"+self.msg+"')"

class NotAppropriateSolverError(Exception):
        """
        Exception raised when trying to solve a problem with
        a solver which cannot handle it
        """
        def __init__(self,msg):
                self.msg=msg
                
        def __str__(self): return self.msg
        def __repr__(self): return "NotAppropriateSolverError('"+self.msg+"')"

class NonConvexError(Exception):
        """
        Exception raised when non-convex quadratic constraints
        are passed to a solver which cannot handle them.
        """
        def __init__(self,msg):
                self.msg=msg 
                
        def __str__(self): return self.msg
        def __repr__(self): return "NonConvexError('"+self.msg+"')"
        
class DualizationError(Exception):
        """
        Exception raised when a non-standard conic problem is being dualized.
        """
        def __init__(self,msg):
                self.msg=msg 
                
        def __str__(self): return self.msg
        def __repr__(self): return "DualizationError('"+self.msg+"')"