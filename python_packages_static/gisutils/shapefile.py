"""
Functions for working with shapefiles.
"""
from distutils.version import LooseVersion
import warnings
from pathlib import Path
import os
import collections
import shutil
import fiona
from shapely.geometry import shape, mapping
import numpy as np
import pandas as pd
import pyproj
from pyproj.enums import WktVersion
from gisutils.projection import get_authority_crs, project
from gisutils.utils import is_sequence


def df2shp(dataframe, shpname, geo_column='geometry', index=False,
           retain_order=False,
           prj=None, epsg=None, proj_str=None, crs=None):
    """Write a DataFrame with a column of shapely geometries to a shapefile.

    Parameters
    ----------
    dataframe : pandas.DataFrame
    shpname : str, filepath
        Output shapefile
    geo_column : str
        Name of column in dataframe with feature geometries (default 'geometry')
    index : bool
        If True, include the DataFrame index in the written shapefile
    retain_order : bool
        Retain column order in dataframe, using an OrderedDict. Shapefile will
        take about twice as long to write, since OrderedDict output is not
        supported by the pandas DataFrame object.
    prj : str
        Path to ESRI projection file describing the coordinate reference system of the feature geometries
        in the 'geometry' column. (specify one of prj, epsg, proj_str)
    epsg : int
        EPSG code describing the coordinate reference system of the feature geometries
        in the 'geometry' column.
    proj_str : str
        PROJ string describing the coordinate reference system of the feature geometries
        in the 'geometry' column.
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
    writes a shapefile to shpname
    """

    # first check if output path exists
    output_folder = os.path.abspath(os.path.split(shpname)[0])
    if os.path.split(shpname)[0] != '' and not os.path.isdir(output_folder):
        raise IOError("Output folder doesn't exist:\n{}".format(output_folder))

    # check for empty dataframe
    if len(dataframe) == 0:
        raise IndexError("DataFrame is empty!")

    df = dataframe.copy()  # make a copy so the supplied dataframe isn't edited

    # reassign geometry column if geo_column is special (e.g. something other than "geometry")
    if geo_column != 'geometry':
        df['geometry'] = df[geo_column]
        df.drop(geo_column, axis=1, inplace=True)

    # assign none for geometry, to write a dbf file from dataframe
    Type = None
    if 'geometry' not in df.columns:
        df['geometry'] = None
        Type = 'None'
        mapped = [None] * len(df)

    # reset the index to integer index to enforce ordering
    # retain index as attribute field if index=True
    df.reset_index(inplace=True, drop=not index)

    # enforce 10 character limit
    df.columns = rename_fields_to_10_characters(df.columns)

    properties = shp_properties(df)
    del properties['geometry']

    # set projection (or use a prj file, which must be copied after shp is written)
    # alternatively, provide a crs in dictionary form as read using fiona
    # from a shapefile like fiona.open(inshpfile).crs
    crs_wkt = None
    if epsg is not None:
        warnings.warn('gisutils.df2shp: the epsg argument is deprecated; use crs instead',
                      DeprecationWarning)
        from fiona.crs import from_epsg
        crs = from_epsg(int(epsg))
    elif proj_str is not None:
        warnings.warn('gisutils.df2shp: the proj_str argument is deprecated; use crs instead',
                      DeprecationWarning)
        from fiona.crs import from_string
        crs = from_string(proj_str)
    elif crs is not None:
        proj_crs = get_authority_crs(crs)
        # https://pyproj4.github.io/pyproj/stable/crs_compatibility.html#converting-from-pyproj-crs-crs-for-fiona
        if LooseVersion(fiona.__gdal_version__) < LooseVersion("3.0.0"):
            crs_wkt = proj_crs.to_wkt(WktVersion.WKT1_GDAL)
        else:
            # GDAL 3+ can use WKT2
            crs_wkt = proj_crs.to_wkt()
        crs = None
    else:
        pass

    if Type != 'None':
        for g in df.geometry:
            try:
                Type = g.type
            except:
                continue
        mapped = [mapping(g) for g in df.geometry]

    schema = {'geometry': Type, 'properties': properties}
    length = len(df)

    if not retain_order:
        props = df.drop('geometry', axis=1).astype(object).to_dict(orient='records')
    else:
        props = [collections.OrderedDict(r) for i, r in df.drop('geometry', axis=1).astype(object).iterrows()]
    print('writing {}...'.format(shpname), end='')
    #with fiona.collection(shpname, "w", driver="ESRI Shapefile", crs=crs, crs_wkt=crs_wkt, schema=schema) as output:
    with fiona.open(shpname, "w", driver="ESRI Shapefile", crs=crs, crs_wkt=crs_wkt, schema=schema) as output:
        for i in range(length):
            output.write({'properties': props[i],
                          'geometry': mapped[i]})
    if prj is not None:
        try:
            print('copying {} --> {}...'.format(prj, "{}.prj".format(shpname[:-4])))
            shutil.copyfile(prj, "{}.prj".format(shpname[:-4]))
        except IOError:
            print('Warning: could not find specified prj file. shp will not be projected.')
    print(' Done')


def rename_fields_to_10_characters(columns, limit=10):
    fields = list(map(str, columns))  # convert columns to strings in case some are ints
    newfields = []
    for s in (fields):
        if s[:limit] not in newfields:
            newfields.append(s[:limit])
        else:
            for i in range(100):
                if i < 10:
                    if '{}{}'.format(s[:limit-1], str(i)) not in newfields:
                        newfields.append(s[:limit-1] + str(i))
                        break
                elif i < 100:
                    if '{}{}'.format(s[:limit-2], str(i)) not in newfields:
                        newfields.append(s[:limit-2] + str(i))
                        break
    return newfields


def shp2df(shplist, index=None, index_dtype=None, clipto=[], filter=None,
           true_values=None, false_values=None, layer=None, dest_crs=None,
           skip_empty_geom=True):
    """Read shapefile/DBF, list of shapefiles/DBFs, or File geodatabase (GDB)
     into pandas DataFrame.

    Parameters
    ----------
    shplist : string or list
        of shapefile/DBF name(s) or FileGDB
    index : string
        Column to use as index for dataframe
    index_dtype : dtype
        Enforces a datatype for the index column (for example, if the index field is supposed to be integer
        but pandas reads it as strings, converts to integer)
    clipto : list
        limit what is brought in to items in index of clipto (requires index)
    filter : tuple (xmin, ymin, xmax, ymax)
        bounding box to filter which records are read from the shapefile.
    true_values : list
        same as argument for pandas read_csv
    false_values : list
        same as argument for pandas read_csv
    layer : str
        Layer name to read (if opening FileGDB)
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
    skip_empty_geom : True/False, default True
        Drops shapefile entries with null geometries.
        DBF files (which specify null geometries in their schema) will still be read.

    Returns
    -------
    df : DataFrame
        with attribute fields as columns; feature geometries are stored as
    shapely geometry objects in the 'geometry' column.
    """
    if isinstance(shplist, str) or isinstance(shplist, Path):
        shplist = [shplist]
    if not isinstance(true_values, list) and true_values is not None:
        true_values = [true_values]
    if not isinstance(false_values, list) and false_values is not None:
        false_values = [false_values]
    if len(clipto) > 0 and index:
        clip = True
    else:
        clip = False

    # destination crs for geometries read from shapefile(s)
    if dest_crs is not None:
        dest_crs = get_authority_crs(dest_crs)

    df = pd.DataFrame()
    for shp in shplist:
        print("\nreading {}...".format(shp))
        if not os.path.exists(shp):
            raise IOError("{} doesn't exist".format(shp))

        # crs of current shapefile
        shp_crs = get_shapefile_crs(shp)
        # set the destination CRS if none was specified
        # so that heterogenious shapefiles will be output to
        # the same CRS
        if dest_crs is None and shp_crs is not None:
            dest_crs = shp_crs

        with fiona.open(shp, 'r', layer=layer) as shp_obj:

            if index is not None:
                # handle capitolization issues with index field name
                fields = list(shp_obj.schema['properties'].keys())
                index = [f for f in fields if index.lower() == f.lower()][0]

            attributes = []
            # for reading in shapefiles
            meta = shp_obj.meta
            if meta['schema']['geometry'] != 'None':
                if filter is not None:
                    print('filtering on bounding box {}, {}, {}, {}...'.format(*filter))
                if clip:  # limit what is brought in to items in index of clipto
                    for line in shp_obj.filter(bbox=filter):
                        props = line['properties']
                        if not props[index] in clipto:
                            continue
                        props['geometry'] = line.get('geometry', None)
                        attributes.append(props)
                else:
                    for line in shp_obj.filter(bbox=filter):
                        props = line['properties']
                        props['geometry'] = line.get('geometry', None)
                        attributes.append(props)
                print('--> building dataframe... (may take a while for large shapefiles)')
                shp_df = pd.DataFrame(attributes)
                # reorder fields in the DataFrame to match the input shapefile
                if len(attributes) > 0:
                    shp_df = shp_df[list(attributes[0].keys())]

                # handle null geometries
                if len(shp_df) == 0:
                    print('Empty dataframe! No clip_features were read.')
                    if filter is not None:
                        print('Check filter {} for consistency \
    with shapefile coordinate system'.format(filter))
                # shp_df will only have a geometry column if it isn't empty
                else:
                    geoms = shp_df.geometry.tolist()
                    if geoms.count(None) == 0:
                        shp_df['geometry'] = [shape(g) for g in geoms]
                    elif skip_empty_geom:
                        null_geoms = [i for i, g in enumerate(geoms) if g is None]
                        shp_df.drop(null_geoms, axis=0, inplace=True)
                        shp_df['geometry'] = [shape(g) for g in shp_df.geometry.tolist()]
                    else:
                        shp_df['geometry'] = [shape(g) if g is not None else None
                                              for g in geoms]

            # for reading in DBF files (just like shps, but without geometry)
            else:
                if clip:  # limit what is brought in to items in index of clipto
                    for line in shp_obj:
                        props = line['properties']
                        if not props[index] in clipto:
                            continue
                        attributes.append(props)
                else:
                    for line in shp_obj:
                        attributes.append(line['properties'])
                print('--> building dataframe... (may take a while for large shapefiles)')
                shp_df = pd.DataFrame(attributes)
                # reorder fields in the DataFrame to match the input shapefile
                if len(attributes) > 0:
                    shp_df = shp_df[list(attributes[0].keys())]

        if len(shp_df) == 0:
            continue
        # set the dataframe index from the index column
        if index is not None:
            if index_dtype is not None:
                shp_df[index] = shp_df[index].astype(index_dtype)
            shp_df.index = shp_df[index].values

        # reproject geometries to dest_crs if needed
        if shp_crs is not None and dest_crs is not None and shp_crs != dest_crs:
            shp_df['geometry'] = project(shp_df['geometry'], shp_crs, dest_crs)

        df = df.append(shp_df)

        # convert any t/f columns to numpy boolean data
        if true_values is not None or false_values is not None:
            replace_boolean = {}
            for t in true_values:
                replace_boolean[t] = True
            for f in false_values:
                replace_boolean[f] = False

            # only remap columns that have values to be replaced
            cols = [c for c in df.columns if c != 'geometry']
            for c in cols:
                if len(set(replace_boolean.keys()).intersection(set(df[c]))) > 0:
                    df[c] = df[c].map(replace_boolean)

    return df


def shp_properties(df):

    newdtypes = {'bool': 'str',
                 'object': 'str',
                 'datetime64[ns]': 'str'}

    # fiona/OGR doesn't like numpy ints
    # shapefile doesn't support 64 bit ints,
    # but apparently leaving the ints alone is more reliable
    # than intentionally downcasting them to 32 bit
    # pandas is smart enough to figure it out on .to_dict()?
    for c in df.columns:
        if c != 'geometry':
            df[c] = df[c].astype(newdtypes.get(df.dtypes[c].name,
                                               df.dtypes[c].name))
        if 'int' in df.dtypes[c].name:
            if np.max(np.abs(df[c])) > 2**31 -1:
                df[c] = df[c].astype(str)

    # strip dtypes to just 'float', 'int' or 'str'
    def stripandreplace(s):
        return ''.join([i for i in s
                        if not i.isdigit()]).replace('object', 'str')
    dtypes = [stripandreplace(df[c].dtype.name)
              if c != 'geometry'
              else df[c].dtype.name for c in df.columns]
    properties = collections.OrderedDict(list(zip(df.columns, dtypes)))
    return properties


def get_shapefile_crs(shapefile):
    """Get the coordinate reference system for a shapefile.

    Parameters
    ----------
    shapefile : str
        Path to a shapefile

    Returns
    -------
    crs : pyproj.CRS instance

    """
    if not isinstance(shapefile, str) and \
            is_sequence(shapefile):
        shapefile = shapefile[0]
    shapefile = Path(shapefile)

    prjfile = shapefile.with_suffix('.prj')
    if prjfile.exists():
        with open(prjfile) as src:
            wkt = src.read()
            crs = pyproj.crs.CRS.from_wkt(wkt)
            return get_authority_crs(crs)


