Neutron Star Merger Simulations
===============================

This repository contains slides and homework assignments for the lecture on
"Neutron star merger simulations" to be delivered at the "Neutron star mergers
for non-experts: GW170817 in the multi-messenger astronomy and FRIB eras"
summer school at Michigan State University.

I will post the solutions to the homework problems at the end of the summer
school.

To download this repository to your laptop use the command

``
    $ git clone https://github.com/dradice/JINA_MSU_School_2018.git
``

To run the example scripts you will need to have Jupyter with Python-3.6, 

We will use a custom python-3 library for the solution of balance laws in 1D,
which is implemented using [numpy](http://www.numpy.org/). This notebook also
uses [matplotlib](https://matplotlib.org/) and
[ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) for the
visualization and the root finding routines from
[scipy](https://www.scipy.org/). All of these packages are easily available
through the [anaconda python
distribution](https://anaconda.org/anaconda/python)

On macOS, in alternative to Anaconda, you can also install the required
packages using [MacPorts](https://www.macports.org/). Assuming that MacPorts is
installed, you should be able to download all required packages with the
command

``
    $ sudo -s 
    $ yes | port install py36-matplotlib py36-notebook py36-ipywidgets py36-scipy
``

If everything is installed successfully you should be able to run the Jupyter
notebook with the command (in the homework folder)

``
    $ jupyter-notebook-3.6
``
