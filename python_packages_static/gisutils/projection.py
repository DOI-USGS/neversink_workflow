"""
Functions for working with coordinate reference systems and projections.
"""
import warnings
from functools import partial
import numpy as np
from shapely.ops import transform
from shapely.geometry.base import BaseMultipartGeometry
try:
    import rasterio
except:
    rasterio = False
import pyproj
from osgeo import osr
from gisutils.utils import is_sequence


def __getattr__(name):
    if name == 'project':
        warnings.warn("The 'project' module was renamed to 'projection' "
                      "to avoid confusion with the project() function.",
                      DeprecationWarning)
        return project
    raise AttributeError('No module named ' + name)


def get_proj_str(prj):
    """Get the PROJ string from the well-known text in an ESRI projection file.

    Parameters
    ----------
    prj : string (filepath)
        ESRI Shapefile or projection file

    Returns
    -------
    proj_str : string (http://trac.osgeo.org/proj/)

    """
    prjfile = prj[:-4] + '.prj' # allows shp or prj to be argued
    try:
        with open(prjfile) as src:
            prjtext = src.read()
        srs = osr.SpatialReference()
        srs.ImportFromESRI([prjtext])
        proj_str = srs.ExportToProj4()
        return proj_str
    except:
        pass


def project(geom, projection1, projection2):
    """Reproject shapely geometry object(s) or scalar
    coodrinates to new coordinate system

    Parameters
    ----------
    geom: shapely geometry object, list of shapely geometry objects,
          list of (x, y) tuples, or (x, y) tuple.
    projection1: string
        Proj4 string specifying source projection
    projection2: string
        Proj4 string specifying destination projection
    """
    # pyproj 2 style
    # https://pyproj4.github.io/pyproj/dev/gotchas.html
    transformer = pyproj.Transformer.from_crs(projection1, projection2, always_xy=True)

    # check for x, y values instead of shapely objects
    if isinstance(geom, tuple):
        # tuple of scalar values
        if np.isscalar(geom[0]):
            return transformer.transform(*geom)
        elif is_sequence(geom[0]):
            return transformer.transform(*geom)

    # sequence of tuples or shapely objects
    if isinstance(geom, BaseMultipartGeometry):
        geom0 = geom
    elif is_sequence(geom):
        geom = list(geom) # in case it's a generator
        geom0 = geom[0]
    else:
        geom0 = geom

    # sequence of tuples
    if isinstance(geom0, tuple):
        a = np.array(geom)
        x = a[:, 0]
        y = a[:, 1]
        return transformer.transform(x, y)

    project = partial(transformer.transform)

    # do the transformation!
    if is_sequence(geom) and not isinstance(geom, BaseMultipartGeometry):
        return [transform(project, g) for g in geom]
    return transform(project, geom)


def get_authority_crs(crs):
    """Try to get the authority representation
    for a CRS, for more robust comparison with other CRS
    objects.

    Parameters
    ----------
    crs : obj
        A Python int, dict, str, or pyproj.crs.CRS instance
        passed to the pyproj.crs.from_user_input
        See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
        Can be any of:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class

    Returns
    -------
    authority_crs : pyproj.crs.CRS instance
        CRS instance initiallized with the name
        and authority code (e.g. epsg: 5070) produced by
        pyproj.crs.CRS.to_authority()

    Notes
    -----
    pyproj.crs.CRS.to_authority() will return None if a matching
    authority name and code can't be found. In this case,
    the input crs instance will be returned.

    References
    ----------
    http://pyproj4.github.io/pyproj/stable/api/crs/crs.html

    """
    if crs is not None:
        crs = pyproj.crs.CRS.from_user_input(crs)
        authority = crs.to_authority()
        if authority is not None:
            return pyproj.CRS.from_user_input(authority)
        return crs


def project_raster(source_raster, dest_raster, dest_crs,
                   resampling=1, resolution=None, num_threads=2,
                   driver='GTiff'):
    """Reproject a raster from one coordinate system to another using Rasterio
    code from: https://github.com/mapbox/rasterio/blob/master/docs/reproject.rst

    Parameters
    ----------
    source_raster : str
        Filename of source raster.
    dest_raster : str
        Filename of reprojected (destination) raster. Extension of filename should
        match the driver (e.g., '.tif' for GeoTIFF)
    dest_crs : obj
        A Python int, dict, str, or pyproj.crs.CRS instance
        passed to the pyproj.crs.from_user_input
        See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
        Can be any of:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class
    resampling : int
        Type of resampling to use when reprojecting the raster
        (see rasterio source code: https://github.com/mapbox/rasterio/blob/master/rasterio/enums.py)
        nearest = 0
        bilinear = 1
        cubic = 2
        cubic_spline = 3
        lanczos = 4
        average = 5
        mode = 6
        gauss = 7
        max = 8
        min = 9
        med = 10
        q1 = 11
        q3 = 12
    resolution : tuple of floats (len 2)
        cell size of the output raster
        (x resolution, y resolution)
    driver : str
        GDAL driver/format to use for writing dst_raster. Default is GeoTIFF.
    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(source_raster) as src:
        print('reprojecting {}...\nfrom:\n{}, res: {:.2e}, {:.2e}\n'.format(
                source_raster,
                src.crs.to_string(),
                src.res[0], src.res[1],
                dest_crs), end='')
        affine, width, height = calculate_default_transform(
            src.crs, dest_crs, src.width, src.height, *src.bounds, resolution=resolution)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dest_crs,
            'transform': affine,
            'affine': affine,
            'width': width,
            'height': height,
            'driver': driver
        })
        with rasterio.open(dest_raster, 'w', **kwargs) as dst:
            print('to:\n{}, res: {:.2e}, {:.2e}...'.format(dst.crs.to_string(), *dst.res))
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=affine,
                    dst_crs=dest_crs,
                    resampling=resampling,
                    num_threads=num_threads)
    print('wrote {}.'.format(dest_raster))


def get_rasterio_crs(crs):
    """Returns a rasterio.crs.CRS representation
    of the input coordinate reference system.

    Parameters
    ----------
    crs : obj
        A Python int, dict, str, or pyproj.crs.CRS instance
        passed to the pyproj.crs.from_user_input
        See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
        Can be any of:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class

    Returns
    -------
    rasterio_crs : rasterio.crs.CRS instance
    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    crs = get_authority_crs(crs)
    rasterio_crs = rasterio.crs.CRS.from_user_input(crs.srs)
    return rasterio_crs