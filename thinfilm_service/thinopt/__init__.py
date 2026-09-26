"""thinopt: characteristic-matrix thin-film optics, as an HTTP service.

Module map (one responsibility each):

* :mod:`thinopt.optics`      Snell's law, cos(theta), s/p effective admittances,
                             phase thickness.
* :mod:`thinopt.matrices`    complex 2x2 characteristic matrices and their product.
* :mod:`thinopt.solver`      single-point reflectance / transmittance / absorptance.
* :mod:`thinopt.spectrum`    wavelength scans built from the same solver.
* :mod:`thinopt.validation`  request parsing and physical-input validation.
* :mod:`thinopt.presets`     built-in quarter-wave AR demo coating.
* :mod:`thinopt.api`         Flask HTTP layer (JSON only).
"""

from .api import create_app

__all__ = ["create_app"]
__version__ = "1.0.0"
