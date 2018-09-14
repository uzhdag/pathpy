<img src="https://github.com/IngoScholtes/pathpy/blob/master/pathpy_logo.png" width="300" alt="pathpy logo" />

# Introduction

`pathpy` is an OpenSource python package for the analysis of time series data on networks using higher- and multi-order network models.

`pathpy` is specifically tailored to analyse temporal networks as well as time series and sequence data that capture multiple short, independent paths observed in an underlying graph or network. Examples for data that can be analysed with pathpy include time-stamped social networks, user click streams in information networks, biological  pathways, citation networks, or information cascades in social networks.

Unifying the modelling and analysis of path statistics and temporal networks, `pathpy` provides efficient methods to extract causal or time-respecting paths from time-stamped network data. The current package distributed via the PyPI name `pathpy2` supersedes the packages [`pyTempnets`](https://github.com/IngoScholtes/pyTempNets) as well as [version 1.0 of `pathpy`](https://github.com/IngoScholtes/pathpy).

`pathpy` facilitates the analysis of temporal correlations in time series data on networks.
It uses a principled model selection technique to infer higher-order graphical representations that capture both topological and temporal characteristics. 
It specifically allows to answer the question when a network abstraction of time series data is justified and when higher-order network representations are needed.

`pathpy` facilitates the analysis of temporal correlations in time series data on networks. It uses model selection and statistical learning to generate optimal higher- and multi-order models that capture both topological and temporal characteristics. It can help to answer the important question when a network abstraction of complex systems is justified and when higher-order representations are needed instead.

The theoretical foundation of this package, **higher- and multi-order network models**, was developed in the following published works:

1. I Scholtes: [When is a network a network? Multi-Order Graphical Model Selection in Pathways and Temporal Networks](http://dl.acm.org/citation.cfm?id=3098145), In KDD'17 - Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Halifax, Nova Scotia, Canada, August 13-17, 2017
2. I Scholtes, N Wider, A Garas: [Higher-Order Aggregate Networks in the Analysis of Temporal Networks: Path structures and centralities](http://dx.doi.org/10.1140/epjb/e2016-60663-0), The European Physical Journal B, 89:61, March 2016
3. I Scholtes, N Wider, R Pfitzner, A Garas, CJ Tessone, F Schweitzer: [Causality-driven slow-down and speed-up of diffusion in non-Markovian temporal networks](http://www.nature.com/ncomms/2014/140924/ncomms6024/full/ncomms6024.html), Nature Communications, 5, September 2014
4. R Pfitzner, I Scholtes, A Garas, CJ Tessone, F Schweitzer: [Betweenness preference: Quantifying correlations in the topological dynamics of temporal networks](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.198701), Phys Rev Lett, 110(19), 198701, May 2013

`pathpy` extends higher-order modelling approaches towards multi-order models for paths that capture temporal correlations at multiple length scales simultaneously. All mathematical details of the framework can be found in this [openly available preprint](https://arxiv.org/abs/1702.05499).

<img src="https://github.com/uzhdag/pathpy/blob/master/multiorder.png" width="500" alt="Illustration of Multi-Order Model" />

A broader view on optimal higher-order models in the analyis of complex systems can be found [here](https://arxiv.org/abs/1806.05977).

`pathpy` is fully integrated with `jupyter`, providing rich and interactive in-line visualisations of networks, temporal networks, higher-, and multi-order models. Visualisations can be exported to HTML5 files that can be shared and published onthe Web.

# Download and installation

`pathpy` is pure python code. It has no platform-specific dependencies and should thus work on all platforms. pathpy requires python 3.x. It builds on numpy and scipy. The latest release version 2.0 of pathpy can be installed by typing:

`> pip install pathpy2`

Please make sure that you use the pyPI name `pathpy2` as the package name `pathpy` is currently blocked.

If you want to install the latest development version, you can install it direvctly from git by typing: 

`> pip install git+git://github.com/uzhdag/pathpy.git`

# Tutorial

A comprehensive 3 hour hands-on tutorial that shows how you can use `pathpy` to analyse data on pathways and temporal networks is available [online](https://ingoscholtes.github.io/csh2018-tutorial/pathpy).

An explanatory video that introduces the science behind `pathpy` is available here:

[![Watch promotional video](https://img.youtube.com/vi/CxJkVrD2ZlM/0.jpg)](https://www.youtube.com/watch?v=CxJkVrD2ZlM)

A promotional video showcasing some of `pathpy`'s features is available here:

[![Watch promotional video](https://img.youtube.com/vi/QIPqFaR2Z5c/0.jpg)](https://www.youtube.com/watch?v=QIPqFaR2Z5c)

# Documentation

The code is fully documented via docstrings which are accessible through python's built-in help system. Just type help(SYMBOL_NAME) to see the documentation of a class or method. A reference manual is available here https://uzhdag.github.io/pathpy/hierarchy.html

# Releases and Versioning

The first public beta release of `pathpy` (released February 17 2017) is v1.0-beta. Following versions are named MAJOR.MINOR.PATCH according to semantic versioning. The latest release version is 2.0.0.

# Known issues

- Depending on whether or not `scipy` has been compiled with `MKL` or `openblas`, considerable numerical differences can occur, e.g. for eigenvalue centralities, PageRank, spectral clustering, and other measures that depend on the eigenvectors and eigenvalues of matrices. Please refer to `scipy.show_config()` to show compilation flags. We are investigating this issue.
- Interactive visualisations in `jupyter` are currently only supported for `juypter` notebooks, stand-alone HTML files, and the jupyter display integrated in IDEs like Visual Studio Code (which we highly recommend to work with `pathpy`). Due to its new widget mechanism, interactive `d3js` visualizations are currently not available for `jupyterLab`. Due to the complex document object model generated by `jupyter` notebooks, visualization performance is best in stand-alone HTML files and in Visual Studio Code.
- The visualisation module currently does not support the drawing of edge arrows for temporal networks with directed edges. However, a powerful templating mechanism is available to support custom interactive and dynamic visualisations both for static and temporal networks.
- The visualisation of paths in terms of alluvial diagrams within `jupyter` is currently unstable. This is due to the asynchronous loading of external scripts and possible network latencies e.g. in wireless networks. We will replace this in a future version.

# Acknowledgements

The research and development behind `pathpy` is generously funded by the Swiss National Science Foundation via [grant 176938](http://p3.snf.ch/Project-176938).

The research behind this data analytics package was previously funded by the Swiss State Secretariate for Education, Research and Innovation via grant C14.0036.  The development of the predecessor package `pyTempNets` was supported by the MTEC Foundation in the context of the project "The Influence of Interaction Patterns on Success in Socio-Technical Systems: From Theory to Practice".

# Contributors

[Ingo Scholtes](http://www.ifi.uzh.ch/dag) (project lead, development)  
[Luca Verginer](https://www.verginer.eu/) (development, test suite integration)  

# Past Contributors

Roman Cattaneo (development)  
Nicolas Wider (testing)  

# Copyright

`pathpy` is licensed under the [GNU Affero General Public License](https://choosealicense.com/licenses/agpl-3.0/).

(c) Copyright ETH Zürich & University of Zurich, 2015-2018
