import warnings
from gisutils import projection


def __getattr__(name):
    if name in {'get_proj_str', 'project'}:
        warnings.warn("The 'project' module was renamed to 'projection' "
                      "to avoid confusion with the project() function.",
                      DeprecationWarning)
        return projection.__dict__[name]
    raise AttributeError('No function named ' + name)
