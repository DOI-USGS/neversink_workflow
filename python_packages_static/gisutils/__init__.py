import warnings
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from gisutils.projection import get_proj_str, get_authority_crs
from gisutils.projection import project as project_fn
from gisutils.raster import get_values_at_points, get_raster_crs, write_raster
from gisutils.shapefile import df2shp, shp2df, get_shapefile_crs


def __getattr__(name):
    if name == 'project':
        warnings.warn("The 'project' module was renamed to 'projection' "
                      "to avoid confusion with the project() function.",
                      DeprecationWarning)
        return project_fn
    raise AttributeError('No module named ' + name)
