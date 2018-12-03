
import os
import glob

"""Utility functions to help with deep learning"""

import numpy as np
import pandas as pd


def read_data_file(datafile):
    """
    Read the csv file that describes each image/patch
    Parameters
    ----------
    datafile : str
        Location of the datafile

    Returns
    -------
    df : DataFrame
        Pandas dataframe of the csv file
    """

    """Name for the columns of the DataFrame"""
    names = ['id', 'fid', 'file', 'mmfile', 'plate', 'row', 'column',
             'field', 'yc', 'xc']

    df = pd.read_csv(datafile, sep=",", names=names)
    mmfile = df['mmfile']
    maxrow = df['row'].max()
    well = maxrow*(df['column'] - 1) + df['row'] - 1
    df['well'] = well
    """Get rid of the home directory in the mmfile path"""
    mmfile = mmfile.str.replace("/Users/cjw/", '')
    df['mmfile'] = mmfile
    return df

def clean_mmfilename(df):
    mmcol = df['mmfile']

def getXY(mmdict, df, rowid, size):
    """
    Get the X and Y (and some other things) coordinates for row rowid of the dataframe
    
    Parameters
    ----------
    mmdict : dictionary of mmfiles
    df : the dataframe
    rowid : row interest in the dataframe
    size : size of the image patch

    Returns
    -------
    tuple : (mfile, fid, x, y, well)
        mfile: memmap file
        fid : file id form dataframe
        x : x coordinate
        y : y coordinate
        well: what well the image is in
        
    """
    rowd = df[df['id'] == rowid]
    if len(rowd) == 0:
        print(len(rowd), rowid)
        
    row = rowd.iloc[0]
    fid = int(row['fid'])
    xc = row['xc']
    yc = row['yc']
    well = row['well']
    mfile = mmdict[row['mmfile'].strip()]
    x = int(xc) - size//2
    y = int(yc) - size//2

    shape = mfile.shape
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > (shape[2] - size):
        x = shape[2] - size
    if y > (shape[1] - size):
        y = shape[1] - size

    return mfile, fid, x, y, well


def getbatch(mmdict, df, start, batchsize, size, nchannels, channels=None):
    """
    Get a batch of images for training or inference
    
    Parameters
    ----------
    mmdict : dict
        memmap dictionary
    df : dataframe
    start : int
        start position in the list of patches
    batchsize : int
        number of patched to return
    size : int
        size (width) of each image patch
    nchannels : int
        number of channels to return
    channels : np.array
        list of channels to return

    Returns
    -------
    tuple (batch, well, rownums)
        batch : array of the batch of images
        wells : list of what well each image is in
        rownums : row from the dataframe for each image
    """


    if channels is None:
        channels = np.arange(nchannels)
    dfsize = len(df)
    dx = dfsize//batchsize
    rownums = np.linspace(start, start + dx*(batchsize - 1), batchsize,
                          dtype=np.int32)
    #print(len(rownums))
    #print(rownums)
    batch = np.zeros((batchsize, size, size, nchannels))
    wells = []
    for i, v in enumerate(rownums):
        mfile, fid, x, y, well = getXY(mmdict, df, v,  size)
        z = mfile[fid][y:y + size, x:x + size, channels]
        wells.append(well)
        batch[i] = z
    wells = np.asarray(wells)
    return batch, wells, rownums


def get_df(mmdict, df, size, nchannels, channels=None):
    if channels == None:
        channels = np.arange(nchannels)

    dfsize = len(df)
    images = np.zeros((dfsize, size, size, nchannels))
    for i, (index, irow) in enumerate(df.iterrows()):
        rid = irow['id'] ## id in the whole dataframe
        fid = irow['fid'] ## id in the mmfile the image is in
        xc = irow['xc']
        yc = irow['yc']
        x = int(xc) - size//2
        y = int(yc) - size//2
        mfile = mmdict[irow['mmfile'].strip()]
        
        shape = mfile.shape
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > (shape[2] - size):
            x = shape[2] - size
        if y > (shape[1] - size):
            y = shape[1] - size

        z = mfile[fid][y: y + size, x: x + size, channels]
        images[i] = z
    return images

def get_sample(mmdict, df, n, size, nchannels, channels=None):
    if channels == None:
        channels = np.arange(0, nchannels, 1)

    sampledf = df.sample(n)
    images = get_df(mmdict, sampledf, size, nchannels, channels)
    return images
    
    
def getWell(mmdict, df, size, row, column, nchannels, channels=None):
    """
    Get every image patch in a well
    
    Parameters
    ----------
    mmdict : dict
        dictionary of memmap file
    df : DataFrame
    size : int
        size of image patches
    row : int
        row in the plate
    column: int
        column in the plate
    nchannels : int
        number of channels
    channels : np.array
        list of channels for the batch
        

    Returns
    -------
        images : numpy array of the images
    """

    if channels is None:
        channels = np.arange(nchannels)

    welldf = df[df['row'] == row]
    welldf = welldf[welldf['column'] == column]
    dfsize = len(welldf)

    images = np.zeros((dfsize, size, size, nchannels))
    for i, (index, irow) in enumerate(welldf.iterrows()):
        rid = irow['id']
        fid = irow['fid']
        xc = irow['xc']
        yc = irow['yc']
        x = int(xc) - size // 2
        y = int(yc) - size // 2
        mfile = mmdict[irow['mmfile'].strip()]

        shape = mfile.shape
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > (shape[2] - size):
            x = shape[2] - size
        if y > (shape[1] - size):
            y = shape[1] - size

        z = mfile[fid][y:y + size, x:x + size, channels]
        images[i] = z
    return images

def create_mm_dataframe(mmfiles_dict):
    pass

def list_mmfiles(dir):
    if not dir.endswith("/"):
        dir += "/"
        
    mmfiles = glob.glob(dir + "*" + ".mm")
    return mmfiles
