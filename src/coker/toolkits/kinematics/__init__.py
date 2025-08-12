from coker.toolkits.kinematics.rigid_body import *

try:
    from coker.toolkits.kinematics.visualiser import *
except ImportError:
    import warnings

    warnings.warn(
        "Visualiser not available; install matplotlib for visualisation"
    )
