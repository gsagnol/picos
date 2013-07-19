PICOS is a user friendly interface
to several conic and integer programming solvers,
very much like `YALMIP <http://users.isy.liu.se/johanl/yalmip/>`_ under
`MATLAB <http://www.mathworks.com/>`_.

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


Contributors
============

`Guillaume Sagnol <http://www.zib.de/sagnol>`_

Thanks also to
==============

Bertrand Omont

`Elmar Swarat <http://www.zib.de/swarat>`_
