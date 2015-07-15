`PICOS <http://picos.zib.de/>`_ is a user friendly interface
to several conic and integer programming solvers,
very much like `YALMIP <http://users.isy.liu.se/johanl/yalmip/>`_
or `CVX <http://cvxr.com/cvx/>`_  under `MATLAB <http://www.mathworks.com/>`_.

The main motivation for PICOS is to have the possibility to
enter an optimization problem as a *high level model*,
and to be able to solve it with several *different solvers*.
Multidimensional and matrix variables are handled in a natural fashion,
which makes it painless to formulate a SDP or a SOCP.
This is very useful for educational purposes,
and to quickly implement some models and
test their validity on simple examples.

Furthermore, with PICOS you can take advantage of the
python programming language to read and write data,
construct a list of constraints by using python list comprehensions,
take slices of multidimensional variables, etc. 


Author
======

Picos initial author and current primary developer is:
                
                  `Guillaume Sagnol <http://www.zib.de/sagnol>`_, <sagnol( a t )zib.de>

Contributors
============
                  
People who actively contributed to the code of Picos (in no particular order)

        * `Sergio Callegari <http://www.unibo.it/faculty/sergio.callegari>`_ 

        * `Petter Wittek <http://peterwittek.com/>`_

        * Paul Fournel
        
        * Arno Ulbricht

        * Bertrand Omont

Thanks also to
==============

People who contributed to the improvement of Picos by sending
their comments, ideas, questions, ... (in no particular order):

        * `Dan Stahlke <http://www.stahlke.org/>`_
        
        * `Marco Dalai <http://www.ing.unibs.it/~marco.dalai/>`_

        * `Matteo Seminaroti <http://www.cwi.nl/people/2683/>`_
        
        * `Warren Schudy <http://cs.brown.edu/~ws/>`_
        
        * `Elmar Swarat <http://www.zib.de/swarat>`_
