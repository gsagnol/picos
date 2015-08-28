from distutils.core import setup

setup(
    name='PICOS',
    version='1.1.1',
    author='G. Sagnol',
    author_email='sagnol@zib.de',
    packages=['picos'],
    license='LICENSE.txt',
    description='A Python Interface to Conic Optimization Solvers.',
    long_description=open('README.rst').read(),
    install_requires=[
        "CVXOPT >= 1.1.4",
        "numpy  >= 1.6.2",
        "six >= 1.8.0"
    ],
    url='http://picos.zib.de',
    download_url='http://picos.zib.de/dist/PICOS-1.1.1dev.tar.gz'
) 
