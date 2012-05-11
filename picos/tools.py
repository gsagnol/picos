# coding: utf-8
import cvxopt as cvx
import numpy as np
import sys
from progress_bar import ProgressBar


__all__=['_retrieve_matrix',
        '_svecm1_identity',
        'eval_dict',
        'putIndices',
        '_blocdiag',
        'svec',
        'svecm1',
        'sum',
        'diag',
        'new_param',
        'available_solvers'
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
        """TODOC:
        evaluates all the variables in the dictionary
        and returns the same dictionary, but evaluated"""
        for k in dict_of_variables:
                dict_of_variables[k] = dict_of_variables[k].eval()
                if dict_of_variables[k].size == (1,1):
                        dict_of_variables[k] = dict_of_variables[k][0]
        return dict_of_variables



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

def diag(exp,dim):
        r"""
        if ``exp`` is a scalar affine expression,
        ``diag(exp)`` returns a diagonal matrix of size ``dim`` :math:`\times` ``dim``,
        wiith the diagonal elements equal to ``exp``.
        
        **Example**
        
        >>> import picos as pic
        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',1)
        >>> y=prob.add_variable('y',1)
        >>> pic.tools.diag(x-y,4)
        # (4 x 4)-affine expression: diag(x -y) #
        
        """
        from .expression import AffinExp
        if exp.size<>(1,1):
                raise Exception('not implemented')
        expcopy=AffinExp(exp.factors.copy(),exp.constant,exp.size,
                        exp.string)
        idx=cvx.spdiag([1.]*dim)[:].I
        for k in exp.factors.keys():
                expcopy.factors[k]=cvx.spmatrix([],[],[],(dim**2,exp.factors[k].size[1]))
                for i in idx:
                        expcopy.factors[k][i,:]=exp.factors[k]
        expcopy.constant=cvx.matrix(0.,(dim**2,1))
        if not exp.constant is None:           
                for i in idx:
                        expcopy.constant[i]=exp.constant[0]
        expcopy.size=(dim,dim)
        expcopy.string='diag('+expcopy.string+')'
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
        elif max(retmat+0.)==min(retmat+0.): #|1| (+0. to avoid sparse evaluation)
                if retmat[0]==0:
                        retstr='|0|'
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
        
def new_param(name,value):
        """TODOC again
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
                import mosek as mo
                lst.append('mosek')
                del mo
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
        return lst