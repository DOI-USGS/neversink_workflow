"""
Functions for working with rasters.
"""
from pathlib import Path
import os
import warnings
import time
import fiona
from shapely.geometry import box, mapping, Polygon
from shapely import wkt
try:
    import rasterio
    from rasterio import Affine
    from rasterio.mask import mask
except:
    rasterio = False

import numpy as np
from scipy import interpolate
try:
    from osgeo import gdal
except:
    gdal = False

from gisutils.projection import project, get_authority_crs, project_raster
from gisutils.shapefile import get_shapefile_crs, shp2df


def get_transform(xul, yul, dx, dy=None, rotation=0.):
    """Returns an affine.Affine instance that can be
    used to locate raster grids in space. See
    https://www.perrygeo.com/python-affine-transforms.html
    https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html

    Parameters
    ----------
    xul : float
        x-coorindate of upper left corner of raster grid
    yul : float
        y-coorindate of upper left corner of raster grid
    dx : float
        cell spacing in the x-direction
    dy : float
        cell spacing in the y-direction
    rotation :
        rotation of the raster grid in degrees, clockwise
    Returns
    -------
    affine.Affine instance
    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    if dy is None:
        dy = -dx
    return Affine(dx, 0., xul,
                  0., dy, yul) * \
           Affine.rotation(rotation)


def get_raster_crs(raster):
    """Get the coordinate reference system for a shapefile.

    Parameters
    ----------
    raster : str (filepath)
        Path to a raster

    Returns
    -------
    crs : pyproj.CRS instance

    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    with rasterio.open(raster) as src:
        if src.crs is not None:
            crs = get_authority_crs(src.crs)
            return crs


def get_values_at_points(rasterfile, x=None, y=None, band=1,
                         points=None, points_crs=None,
                         out_of_bounds_errors='coerce',
                         method='nearest', size_thresh=1e9):
    """Get raster values single point or list of points. Points in
    a different coordinate reference system (CRS) specified with a points_crs will be
    reprojected to the raster CRS prior to sampling.

    Parameters
    ----------
    rasterfile : str
        Filename of raster.
    x : 1D array
        X coordinate locations
    y : 1D array
        Y coordinate locations
    points : list of tuples or 2D numpy array (npoints, (row, col))
        Points at which to sample raster.
    points_crs : obj, optional
        Coordinate reference system for points or x, y. Only needed if
        different than the CRS for the raster, in which case the points will be
        reprojected to the raster CRS prior to getting the values.
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
    out_of_bounds_errors : {‘raise’, ‘coerce’}, default 'raise'
        * If 'raise', then x, y locations outside of the raster will raise an exception.
        * If 'coerce', then x, y locations outside of the raster will be set to NaN.
    method : str 'nearest' or 'linear'
        If 'nearest', the rasterio.DatasetReader.index() method is used to
        return the raster values at the nearest cell centers. If 'linear',
        scipy.interpolate.interpn is used for bilinear interpolation of values
        between raster cell centers.
    size_thresh : float
        Prior to reading any data, the raster size (height * width) is evaluated. If
        the size is larger than size_thresh, point values are read using
        :meth:`rasterio.io.DatasetReader.sample` (regardless of the specified method),
        which gets nearest pixel values without reading the whole dataset into memory.
        A 32-bit raster of size=1e9 would require approximately 4 GB of memory
        (at 4 bytes per pixel).
        By default, 1e9.


    Returns
    -------
    list of floats

    Notes
    -----
    requires rasterio
    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')

    # read in sample points
    array_shape = None
    if x is not None and isinstance(x[0], tuple):
        x, y = np.array(x).transpose()
        warnings.warn(
            "new argument input for get_values_at_points is x, y, or points"
        )
    elif x is not None:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(x.shape) > 1:
            array_shape = x.shape
            x = x.ravel()
        if len(y.shape) > 1:
            array_shape = y.shape
            y = y.ravel()
    elif points is not None:
        if not isinstance(points, np.ndarray):
            x, y = np.array(points)
        else:
            x, y = points[:, 0], points[:, 1]
    else:
        print('Must supply x, y or list/array of points.')

    assert os.path.exists(rasterfile), "raster {} not found".format(rasterfile)
    t0 = time.time()

    print("reading data from {}...".format(rasterfile))
    data = None
    with rasterio.open(rasterfile) as src:
        meta = src.meta
        nodata = meta['nodata']
        size = src.shape[0] * src.shape[1]
        if size < size_thresh:
            data = src.read(band)

        # reproject coordinates if needed
        if points_crs is not None:
            points_crs = get_authority_crs(points_crs)
            raster_crs = get_authority_crs(src.crs)
            if raster_crs is None:
                warnings.warn(f'Input raster {rasterfile} does not have a projection (CRS) assigned!')
            else:
                if points_crs is not None and points_crs != raster_crs:
                    x, y = project((x, y), points_crs, raster_crs)

        if data is None:
            results = src.sample(list(zip(x, y)))
            results = np.squeeze(list(results))

    if data is None:
        pass
    elif method == 'nearest':
        i, j = src.index(x, y)
        i = np.array(i, dtype=int)
        j = np.array(j, dtype=int)
        nrow, ncol = data.shape

        # mask row, col locations outside the raster
        within = (i >= 0) & (i < nrow) & (j >= 0) & (j < ncol)

        # get values at valid point locations
        results = np.ones(len(i), dtype=float) * np.nan
        results[within] = data[i[within], j[within]]
        if out_of_bounds_errors == 'raise' and np.any(np.isnan(results)):
            n_invalid = np.sum(np.isnan(results))
            raise ValueError("{} points outside of {} extent.".format(n_invalid, rasterfile))
    else:
        # map the points to interpolate to onto the raster coordinate system
        # (in case the raster is rotated)
        x_rx, y_ry = ~src.transform * (x, y)
        # coordinates of raster pixel centers in raster coordinate system
        # (e.g. i,j = 0, 0 = 0.5, 0.5)
        pad = 0.5  # extra padding, in pixels, so that points within the outer pixels are still counted
        padding = np.arange(0.5 - pad, 0.5)
        rx = padding.tolist() + list(np.arange(src.width) + 0.5) + list(src.width - padding)
        ry = padding.tolist() + list(np.arange(src.height) + 0.5) + list(src.height - padding)
        # pad the coordinates and the data
        pad_width = int(np.ceil(pad))
        padded = np.pad(data.astype(float), pad_width=pad_width, mode='edge')

        # exclude nodata points prior to interpolating
        padded[padded == nodata] = np.nan
        bounds_error = False
        if out_of_bounds_errors == 'raise':
            bounds_error = True
        results = interpolate.interpn((ry, rx), padded,
                                      (y_ry, x_rx), method=method,
                                       bounds_error=bounds_error, fill_value=nodata)
    # convert nodata values to np.nans
    results[results == nodata] = np.nan

    # reshape to input shape
    if array_shape is not None:
        results = np.reshape(results, array_shape)
    print("finished in {:.2f}s".format(time.time() - t0))
    return results


def points_to_raster(points_shapefiles, nodata_value=-99,
                     data_col='values',
                     output_resolution=250,
                     outfile='surface.tif', dest_crs=None):
    """Interpolate point data to a regular grid using scipy.interpolate.griddata;
    write results to a GeoTiff.

    Parameters
    ----------
    points_shapefiles : shapefile or list of shapefiles
        Point shapefiles with estimated data, assumed to be on a regular grid.
    nodata_value : numeric
        Value in `points_shapefiles` indicating no data
    data_col : str
        Field in `points_shapefiles` with estimated data.
    output_resolution : numeric
        Cell spacing of the output raster
    outfile : stf
        Output GeoTiff
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

    Notes
    -----

    """

    df = shp2df(points_shapefiles, dest_crs=dest_crs)

    if dest_crs is None:
        dest_crs = get_shapefile_crs(points_shapefiles)

    # reshape the values column to a nrow x ncol array; convert invalid values to nans
    data = df[data_col].values
    data[data == nodata_value] = np.nan

    # coordinates for the orignal grid (aligned with NHG cell corners)
    x_points = np.array([g.x for g in df.geometry])
    y_points = np.array([g.y for g in df.geometry])

    # specifications for a new output_resolution grid aligned with NHG
    # xul, yul is the cell center of the first cell (in the upper left corner)
    xul = x_points.min()
    yul = y_points.max()
    dxy = output_resolution

    # 1D arrays of x and y coordinates for each column, row
    x = np.arange(np.min(x_points), np.max(x_points) + dxy, dxy)
    y = np.arange(np.min(y_points), np.max(y_points) + dxy, dxy)

    # 2D arrays of x and y coordinates for each point
    X, Y = np.meshgrid(x, y)

    # interpolate the values onto the new grid
    # using bilinear interpolation
    # `bounds_error=False` means extrapolated points will be filled with nans
    results = interpolate.griddata((x_points, y_points), data, (X, Y),
                                   method='linear')
    results = np.flipud(results)

    results = np.ma.masked_array(results, mask=np.isnan(results))
    write_raster(outfile, results, xul=xul, yul=yul,
                 dx=dxy, dy=dxy, rotation=0., crs=dest_crs,
                 nodata=-9999)


def write_raster(filename, array, xll=0., yll=0., xul=None, yul=None,
                 dx=1., dy=None, rotation=0., proj_str=None, crs=None,
                 nodata=-9999, verbose=False,
                 **kwargs):
    """
    Write a numpy array to Arc Ascii grid or shapefile with the model
    reference.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of output file. Export format is determined by
        file extention.
        '.asc'  Arc Ascii grid
        '.tif'  GeoTIFF (requries rasterio package)
    array : 2D numpy.ndarray
        Array to export
    xll : float
        x-coordinate of lower left corner of raster grid.
        Either xul, yul or xll, yll must be specified.
        Default = 0.
    yll : float
        y-coordinate of lower left corner of raster grid
        Default = 0.
    xul : float
        x-coordinate of upper left corner of raster grid.
        Either xul, yul or xll, yll must be specified.
    yul : float
        y-coordinate of upper left corner of raster grid
    dx : float
        cell spacing in the x-direction
    dy : float
        cell spacing in the y-direction
        (optional, assumed equal to dx by default)
    rotation :
        rotation of the raster grid in degrees, clockwise
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
    nodata : scalar
        Value to assign to np.nan entries (default -9999)
    kwargs:
        keyword arguments to np.savetxt (ascii)
        rasterio.open (GeoTIFF)
        or flopy.export.shapefile_utils.write_grid_shapefile2

    Notes
    -----
    Rotated grids will be either be unrotated prior to export,
    using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
    included in their transform property (GeoTiff format). In either case
    the pixels will be displayed in the (unrotated) projected geographic
    coordinate system, so the pixels will no longer align exactly with the
    model grid (as displayed from a shapefile, for example). A key difference
    between Arc Ascii and GeoTiff (besides disk usage) is that the
    unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
    will have the same number of rows and pixels as the original.

    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    t0 = time.time()
    a = array
    # third dimension is the number of bands
    if len(a.shape) == 2:
        a = np.reshape(a, (1, a.shape[0], a.shape[1]))
    count, height, width = a.shape

    if proj_str is not None:
        warnings.warn('gisutils.write_raster: the proj_str argument is deprecated; use crs instead',
                      DeprecationWarning)
        crs = proj_str

    if xul is not None and yul is not None:
        # default to decreasing y coordinates if upper left is specified
        if dy is None:  
            dy = -dx
        xll = _xul_to_xll(xul, height * dy, rotation)
        yll = _yul_to_yll(yul, height * dy, rotation)
    elif xll is not None and yll is not None:
        # default to increasing y coordinates if lower left is specified
        if dy is None:
            dy = dx
        xul = _xll_to_xul(xll, height * dy, rotation)
        yul = _yll_to_yul(yll, height * dy, rotation)
    if str(filename).lower().endswith(".tif"):
        trans = get_transform(xul=xul, yul=yul,
                              dx=dx, dy=-np.abs(dy), rotation=rotation)

        # third dimension is the number of bands
        if len(a.shape) == 2:
            a = np.reshape(a, (1, a.shape[0], a.shape[1]))

        if a.dtype == np.int64:
            a = a.astype(np.int32)
        meta = {'count': count,
                'width': width,
                'height': height,
                'nodata': nodata,
                'dtype': a.dtype,
                'driver': 'GTiff',
                'crs': crs,
                'transform': trans,
                'compress': 'lzw'
                }
        meta.update(kwargs)
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(a)
            if isinstance(a, np.ma.masked_array):
                dst.write_mask(~a.mask.transpose(1, 2, 0))
        print('wrote {}'.format(filename))

    elif str(filename).lower().endswith(".asc"):
        path, fname = os.path.split(filename)
        fname, ext = os.path.splitext(fname)
        for band in range(count):
            if count == 1:
                filename = os.path.join(path, fname + '.asc')
            else:
                filename =  os.path.join(path, fname + '_{}.asc'.format(band))
            write_arc_ascii(a[band], filename, xll=xll, yll=yll,
                            cellsize=dx,
                            nodata=nodata, **kwargs)
    if verbose:
        print("raster creation took {:.2f}s".format(time.time() - t0))


def zonal_stats(feature, raster, out_shape=None,
                stats=['mean'], **kwargs):
    try:
        from rasterstats import zonal_stats
    except:
        raise ImportError("This function requires rasterstats.")
    if not isinstance(feature, str):
        feature_name = 'feature'
    else:
        feature_name = feature
    t0 = time.time()
    print('computing {} {} for zones in {}...'.format(raster,
                                                      ', '.join(stats),
                                                      feature_name
                                                      ))
    print(stats)
    results = zonal_stats(feature, raster, stats=stats, **kwargs)
    print(out_shape)
    if out_shape is None:
        out_shape = (len(results),)
    #print(results)
    #means = [r['mean'] for r in results]
    #means = np.asarray(means)
    #means = np.reshape(means, out_shape).astype(float)
    #results = means

    #results = np.reshape(results, out_shape)
    #results = np.reshape(results, out_shape).astype(float)
    results_dict = {}
    for stat in stats:
        res = [r[stat] for r in results]
        res = np.asarray(res)
        res = np.reshape(res, out_shape).astype(float)
        results_dict[stat] = res
    print("finished in {:.2f}s".format(time.time() - t0))
    return results_dict


def read_arc_ascii(filename, shape=None):
    with open(filename) as src:
        meta = {}
        for i in range(6):
            k, v = next(src).strip().split()
            v = float(v) if '.' in v else int(v)
            meta[k.lower()] = v

        # make a gdal-style geotransform
        dx = meta['cellsize']
        dy = meta['cellsize']
        xul = meta['xllcorner']
        yul = meta['yllcorner'] + dy * meta['nrows']
        rx, ry = 0, 0
        meta['geotransform'] = dx, rx, xul, ry, -dy, yul

        if shape is not None:
            assert (meta['nrows'], meta['ncols']) == shape, \
                "Data in {} are {}x{}, expected {}x{}".format(filename,
                                                              meta['nrows'],
                                                              meta['ncols'],
                                                              *shape)
        arr = np.loadtxt(src)
    return arr, meta


def write_arc_ascii(array, filename, xll=0, yll=0, cellsize=1.,
                    nodata=-9999, **kwargs):
    """Write numpy array to Arc Ascii grid.

    Parameters
    ----------
    array : 2D numpy.ndarray
    filename : str (file path)
        Name of output arc ascii file
    xll : scalar
        X-coordinate of lower left corner of grid
    yll : scalar
        Y-coordinate of lower left corner of grid
    cellsize : scalar
        Grid spacing
    nodata : scalar
        Value indicating cells with no data.
    kwargs: keyword arguments to numpy.savetxt
    """
    array = array.copy()
    array[np.isnan(array)] = nodata

    filename = '.'.join(filename.split('.')[:-1]) + '.asc'  # enforce .asc ending
    nrow, ncol = array.shape
    txt = 'ncols  {:d}\n'.format(ncol)
    txt += 'nrows  {:d}\n'.format(nrow)
    txt += 'xllcorner  {:f}\n'.format(xll)
    txt += 'yllcorner  {:f}\n'.format(yll)
    txt += 'cellsize  {}\n'.format(cellsize)
    txt += 'NODATA_value  {:.0f}\n'.format(nodata)
    with open(filename, 'w') as output:
        output.write(txt)
    with open(filename, 'ab') as output:
        np.savetxt(output, array, **kwargs)
    print('wrote {}'.format(filename))


def _xul_to_xll(xul, height, rotation=0.):
    theta = rotation * np.pi / 180
    return xul - (np.sin(theta) * height)


def _xll_to_xul(xll, height, rotation=0.):
    theta = rotation * np.pi / 180
    return xll + (np.sin(theta) * height)


def _yul_to_yll(yul, height, rotation=0.):
    theta = rotation * np.pi / 180
    return yul - (np.cos(theta) * height)


def _yll_to_yul(yul, height, rotation=0.):
    theta = rotation * np.pi / 180
    return yul + (np.cos(theta) * height)


def clip_raster(inraster, clip_features, outraster,
                clip_features_crs=None,
                clip_kwargs=None,
                project_kwargs=None, **kwargs):
    """Clip raster to feature extent(s), write the output
    to a new raster file. If the feature extent(s) are in
    a different coordinate reference system, the raster will first
    be reprojected to that CRS and then clipped. The output raster
    will be in the CRS of the clip features.

    Parameters
    ----------
    inraster : str
        Path to a raster file readable by rasterio.open
    clip_features : str or list-like
        Shapefile or sequence of features. Features can be in
        any format accepted by gisutils.raster.get_feature_geojson()
    outraster : str
        Filename for output raster.
    clip_features_crs : obj
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
    clip_kwargs: dict
        Keyword arguments to rasterio.mask
    project_kwargs : dict
        Key word arguments to gisutils.projection.project_raster()
        These are only used if the clip features are
        in a different coordinate system, in which case
        the raster will be reprojected into that coordinate
        system.
    kwargs : keyword arguments
        Keyword arguments to rasterio.open for writing the output raster.
    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    if clip_kwargs is None:
        clip_kwargs = {}
    if project_kwargs is None:
        project_kwargs = {}

    with rasterio.open(inraster) as src:
        raster_crs = get_authority_crs(src.crs)

    # start with assumption of same coordinates
    if clip_features_crs is None:
        clip_features_crs = raster_crs
    # get the clip feature crs from shapefile
    if isinstance(clip_features, str) or isinstance(clip_features, Path):
        if Path(clip_features).exists():
            clip_features_crs = get_shapefile_crs(clip_features)
    # otherwise if clip feature crs was specified
    else:
        clip_features_crs = get_authority_crs(clip_features_crs)

    # convert the clip_features to geojson
    geoms = get_feature_geojson(clip_features)
    print('input raster crs:\n{}\n\n'.format(raster_crs),
          'clip feature crs:\n{}\n'.format(clip_features_crs))
    # if the coordinate systems are not the same
    # reproject the raster first before clipping
    # this could be greatly sped up by first clipping the input raster prior to reprojecting
    if raster_crs != clip_features_crs or len(project_kwargs) > 0:
        tmpraster = 'tmp.tif'
        tmpraster2 = 'tmp2.tif'
        print('Input raster and clip feature(s) are in different coordinate systems.\n'
              'Reprojecting input raster from\n{}\nto\n{}\n'.format(raster_crs, clip_features_crs))
        # make prelim clip of raster to speed up reprojection
        xmin, xmax, ymin, ymax = get_geojson_collection_bounds(geoms)
        longest_side = np.max([xmax - xmin, ymax - ymin])
        bounds = box(xmin, ymin, xmax, ymax).buffer(longest_side * 0.1)
        bounds = project(bounds, clip_features_crs, raster_crs)
        _clip_raster(inraster, [bounds], tmpraster, clip_kwargs=clip_kwargs)
        project_raster(tmpraster, tmpraster2, clip_features_crs, **project_kwargs, **kwargs)
        inraster = tmpraster2

    _clip_raster(inraster, geoms, outraster, clip_kwargs=clip_kwargs, **kwargs)

    if raster_crs != clip_features_crs:
        for tmp in [tmpraster, tmpraster2]:
            if os.path.exists(tmp):
                print('removing {}...'.format(tmp))
                os.remove(tmp)
    print('Done.')


def _clip_raster(inraster, features, outraster, clip_kwargs, **kwargs):
    """Clips a raster to clip_features in the same coordinate
    reference system, write the output to a new raster file.


    Parameters
    ----------
    inraster : str
        Path to a raster file readable by rasterio.open
    features : str or list-like
        Shapefile or sequence of clip_features. Features can be in
        any format accepted by gisutils.raster.get_feature_geojson()
    outraster : str
        Filename for output raster.
    clip_kwargs : dict
        Keyword arguments to rasterio.mask for clipping the raster
    kwargs : keyword arguments
        Keyword arguments to rasterio.open for writing the output raster.
    """
    if not rasterio:
        raise ModuleNotFoundError('This function requires rasterio. Please conda install rasterio.')
    # convert the clip_features to geojson
    geoms = get_feature_geojson(features)
    with rasterio.open(inraster) as src:
        print('clipping {}...'.format(inraster))

        defaults = {'crop': True,
                    'nodata': src.nodata,
                    #'pad': True,
                    'all_touched': True
                    }
        defaults.update(clip_kwargs)

        out_image, out_transform = mask(src, geoms, **defaults)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "compress": "lzw",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        out_meta.update(kwargs)

        with rasterio.open(outraster, "w", **out_meta) as dest:
            dest.write(out_image)
            print('wrote {}'.format(outraster))


def get_feature_geojson(features):
    """convert input clip_features to list of clip_features in the
    geojson format.

    Parameters
    ----------
    features : str (shapefile path) or list of clip_features
        If clip_features is a list, the clip_features can be in shapely polygon
        or wkt format (e.g. input to shapely.geometry.shape())

    Returns
    -------
    geoms : list of geojson clip_features
    """

    # clip_features is a single shapely geometry
    if isinstance(features, Polygon):
        geoms = [mapping(features)]
    # clip_features is a single geojson geometry
    elif isinstance(features, dict):
        geoms = [features]
    elif isinstance(features, str) or isinstance(features, Path):
        # clip_features is a single wkt string
        try:
            geoms = [mapping(wkt.loads(features))]
        # assume clip_features are in a shapefile
        except:
            if os.path.exists(features):
                with fiona.open(features, "r") as shp:
                    geoms = [feature["geometry"] for feature in shp]
            else:
                raise TypeError('Unrecognized feature type: {}'.format(features))
    elif isinstance(features, list):
        # clip_features are geo-json
        if isinstance(features[0], dict):
            geoms = features
        # clip_features are wkt strings
        elif isinstance(features[0], str):
            try:
                geoms = [mapping(wkt.loads(f)) for f in features]
            except:
                raise TypeError('Unrecognized feature type: {}'.format(features))
        # clip_features are shapely geometries
        else:
            try:
                mapping(features[0])
                geoms = [mapping(f) for f in features]
            except:
                raise TypeError('Unrecognized feature type: {}'.format(features))
    else:
        raise TypeError('Unrecognized feature type: {}'.format(features))
    return geoms


def get_geojson_collection_bounds(geojsoncollection):
    """Get the bounds for a collection of geojson clip_features.

    Parameters
    ----------
    geojsoncollection : sequence
        Sequence of geojson clip_features

    Returns
    -------
    xmin, xmax, ymin, ymax

    """
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for feature in geojsoncollection:
        for crds in feature['coordinates']:
            a = np.array(crds)
            x, y = a[:, 0], a[:, 1]
            xmin, xmax = np.min(x, xmin), np.max(x, xmax)
            ymin, ymax = np.min(y, ymin), np.max(y, ymax)
    return xmin, xmax, ymin, ymax