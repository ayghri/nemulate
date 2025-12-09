"""
Adapted from https://github.com/chadagreene/CDT/blob/master/cdt/windstress.m
\tau = \rho C_D  W^2
\tau_x = \tau . sin(\theta)
\tau_y = \tau . cos(\theta)
where \rho: air density
      C_d : drag coefficient
      W   : wind speed, W^2 = u^2 + v^2
      \theta: angle of the wind from true north

sin(\theta) = u/W
cos(\theta) = v/W

Read more: https://www.nrsc.gov.in/sites/default/files/pdf/oceanproducts/DIVA_Windstresscurl.pdf
Examples: https://www.chadagreene.com/CDT/cdtcurl_documentation.html
"""

# % Drag coefficient, a global average from Kara et al., 2007 (http://dx.doi.org/10.1175/2007JCLI1825.1)
Cd = 1.25e-3
# % density of air in kg/m^3
rho = 1.225

import numpy as np


def windstress(u10, v10):
    W2 = u10**2 + v10**2
    U = np.sqrt(W2)
    tau = rho * Cd * W2
    tau_x = tau * u10 / U
    tau_y = tau * v10 / U
    return tau_x, tau_y
