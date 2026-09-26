"""Built-in demo coating: single quarter-wave anti-reflection layer on glass.

The canonical lab check — a MgF2-like layer (n = 1.38) on a glass substrate
(n = 1.52) in air, with physical thickness chosen so the optical thickness
is one quarter of the design wavelength:

    d = lambda0 / (4 * n_layer)  =  550 nm / (4 * 1.38)  ~  99.64 nm

At the design wavelength and normal incidence this drops the reflectance
from ~4.26 % (bare air/glass interface) to ~1.26 %, which is exactly the
relationship the demo endpoint and the test-suite reproduce.
"""

from __future__ import annotations

from .matrices import Layer, Stack

#: Design wavelength of the demo coating (nm).
DESIGN_WAVELENGTH = 550.0
#: Refractive indices of the demo system.
INCIDENT_INDEX = 1.0
LAYER_INDEX = 1.38
SUBSTRATE_INDEX = 1.52
#: Physical thickness giving quarter-wave optical thickness at the design wavelength.
QUARTER_WAVE_THICKNESS = DESIGN_WAVELENGTH / (4.0 * LAYER_INDEX)


def demo_stack() -> Stack:
    """Air | quarter-wave layer | glass."""
    return Stack(
        incident_index=INCIDENT_INDEX,
        substrate_index=SUBSTRATE_INDEX,
        layers=(Layer(index=LAYER_INDEX, thickness=QUARTER_WAVE_THICKNESS),),
    )


def bare_stack() -> Stack:
    """The same system without the coating: bare air/glass interface."""
    return Stack(
        incident_index=INCIDENT_INDEX,
        substrate_index=SUBSTRATE_INDEX,
        layers=(),
    )
