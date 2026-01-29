# EmbeddedBenchmark.jl
Comparative benchmark study for several embedded / immersed / unfitted Finite Element methods, namely: CutFEM [1], AgFEM [2], SBM [3] and WSBM [4]. The goal of this study is to compare the aforementioned methods within a single numerical framework. A comparison is made on accuracy and condition numbers for varying number of elements and for first and second polynomial order in both 2D and 3D, we consider as well implicit (level-set) and explicit (STL) geometrical representation of the embedded boundary. For the 2D case with implicit embedded boundary we also conduct a performance analysis, comparing the minimal compute time and minimal number of allocations. The minimal is selected from a large number of re-executions of the same code, such that the bias introduced by system noise is minimized. 

[1]: Burman, Erik, et al. "CutFEM: discretizing geometry and partial differential equations." International Journal for Numerical Methods in Engineering 104.7 (2015): 472-501.

[2]: Badia, Santiago, Francesc Verdugo, and Alberto F. Martín. "The aggregated unfitted finite element method for elliptic problems." Computer Methods in Applied Mechanics and Engineering 336 (2018): 533-553.

[3]: Main, Alex, and Guglielmo Scovazzi. "The shifted boundary method for embedded domain computations. Part I: Poisson and Stokes problems." Journal of Computational Physics 372 (2018): 972-995.

[4]: Colomés, Oriol, et al. "A weighted shifted boundary method for free surface flow problems." Journal of Computational Physics 424 (2021): 109837.
