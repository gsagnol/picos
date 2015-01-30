from distutils.core import setup

setup(
    name='PICOS',
    version='1.0.2',
    author='G. Sagnol',
    author_email='sagnol@zib.de',
    packages=['picos'],
    url='http://pypi.python.org/pypi/PICOS/',
    license='LICENSE.txt',
    description='A Python Interface to Conic Optimization Solvers.',
    long_description=open('README.txt').read(),
    install_requires=[
        "CVXOPT >= 1.1.4",
        "numpy  >= 1.6.2",
    ],
) 
