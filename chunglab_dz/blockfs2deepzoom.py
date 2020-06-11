import typing
from blockfs import Directory
from deepzoom import ImageCreator
import json
from nuggt.utils.warp import Warper
import numpy as np
import pandas
from scipy import ndimage
import tempfile
import tifffile

AXES_XY = "xy"
AXES_XZ = "xz"
AXES_YZ = "yz"

def warp_coords(alignment_file:str, x:np.ndarray, y:np.ndarray, z:np.ndarray, dest_shape:typing.Tuple[int,  int, int])\
        -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the warping in an alignment file (from reference to moving) to a set of reference coordinates

    :param alignment_file:
    :param x:
    :param y:
    :param z:
    :param dest_shape:
    :return:
    """
    with open(alignment_file) as fd:
        alignment = json.load(fd)
    zss, yss, xss = [np.linspace(0, _, 50) for _ in (dest_shape[0], dest_shape[1], dest_shape[2])]
    warper = Warper(alignment["reference"], alignment["moving"]).approximate(zss, yss, xss)
    result = warper(np.column_stack((z, y, x)))
    return result[:, 2], result[:, 1], result[:, 0]


def read_image_values(src:str, xs:np.ndarray, ys:np.ndarray, zs:np.ndarray) -> np.ndarray:
    global block
    """
    Read image values from a blockfs file.

    :param src: the precomputed.blockfs file
    :param xs: the X coordinates of the image values to read
    :param ys: the Y coordinates of the image values to read
    :param zs: the Z coordinates of the image values to read
    :return: a 1-d array of the values at each x, y and z
    """
    directory = Directory.open(src)
    xb = np.floor(xs / directory.x_block_size).astype(np.int16)
    yb = np.floor(ys / directory.y_block_size).astype(np.int16)
    zb = np.floor(zs / directory.z_block_size).astype(np.int16)
    df = pandas.DataFrame(dict(x=xs.astype(np.int32), y=ys.astype(np.int32), z=zs.astype(np.int32),
                               xb=xb, yb=yb, zb=zb))
    dg = df.groupby(["xb", "yb", "zb"])
    result = np.zeros(len(xs), directory.dtype)
    for (xi, yi, zi), idxs in dg.groups.items():
        x_off = xi * directory.x_block_size
        y_off = yi * directory.y_block_size
        z_off = zi * directory.z_block_size
        if xi < 0 or yi < 0 or zi < 0 or\
              x_off >= directory.x_extent or \
              y_off >= directory.y_extent or \
              z_off >= directory.z_extent:
            continue
        block = directory.read_block(x_off, y_off, z_off)
        sub_df = dg.get_group((xi, yi, zi))
        values = ndimage.map_coordinates(block, [sub_df.z.values - z_off,
                                                 sub_df.y.values - y_off,
                                                 sub_df.x.values - x_off])
        result[idxs] = values
    return result


def blockfs2deepzoom(
        src:str,
        dest:str,
        alignment:str,
        reference_shape:typing.Tuple[int, int, int],
        axes:str,
        plane:int,
        magnification:float=10.0,
        tile_size:int=254,
        tile_overlap:int=1,
        tile_format:str="jpg",
        clip:int=None,
        save_file:str=None):
    """
    Read a plane from precomputed tif and convert to the deep zoom format

    :param src: source data URL
    :param dest: path to destination for the deep zoom
    :param alignment: the alignment file to use to translate.
    :param reference_shape: the shape of the reference image (z, y, x)
    :param axes: either "xy", "xz" or "yz" for the axes of the plane to be created
    :param plane: the coordinate of the plane in the axis perpendicular to the plane
    :param magnification: magnify the reference volume by this amount
    :param tile_size: the size of a tile image
    :param tile_overlap: amount of overlap between tiles
    :param tile_format: file format, e.g. "png" or "jpg"
    :param clip: high intensity clipping value
    :param save_file: if present, save the warped image here instead of in a temporary file
    :return:
    """
    n_x = int(reference_shape[2] * magnification)
    n_y = int(reference_shape[1] * magnification)
    axx = np.linspace(0, reference_shape[2] - 1, n_x).reshape(1, 1, -1)
    axy = np.linspace(0, reference_shape[1] - 1, n_y).reshape(1, -1, 1)
    n_z = int(reference_shape[0] * magnification)
    axz = np.linspace(0, reference_shape[0] - 1, n_z).reshape(-1, 1, 1)
    ax_plane = np.array([plane]).reshape(1, 1, 1)
    if axes == AXES_XY:
        n_z = 1
        axz = ax_plane
        dest_volume_shape = (1, n_y, n_x)
        dest_plane_shape = (n_y, n_x)
    elif axes == AXES_XZ:
        n_y = 1
        axy = ax_plane
        dest_volume_shape = (n_z, 1, n_x)
        dest_plane_shape = (n_z, n_x)
    elif axes == AXES_YZ:
        n_x = 1
        axx = ax_plane
        dest_volume_shape = (n_z, n_y, n_x)
        dest_plane_shape = (n_z, n_y)
    xr = (np.ones(dest_volume_shape, np.float32) * axx).flatten()
    yr = (np.ones(dest_volume_shape, np.float32) * axy).flatten()
    zr = (np.ones(dest_volume_shape, np.float32) * axz).flatten()
    xs, ys, zs = warp_coords(alignment, xr, yr, zr, reference_shape)
    img = read_image_values(src, xs, ys, zs).reshape(dest_plane_shape[0], dest_plane_shape[1])
    img = np.rot90(img, 1)
    if clip is not None:
        img = np.clip(img, 0, clip)
    img = (img.astype(np.uint32) * 255 / img.max()).astype(np.uint8)
    if save_file is None:
        with tempfile.NamedTemporaryFile("wb", suffix=".tif") as fd:
            tifffile.imsave(fd, img)
            creator = ImageCreator(tile_size=tile_size,
                                   tile_overlap=tile_overlap,
                                   tile_format=tile_format)
            creator.create(fd.name, dest)
    else:
        tifffile.imsave(save_file, img)
        creator = ImageCreator(tile_size=tile_size,
                               tile_overlap=tile_overlap,
                               tile_format=tile_format)
        creator.create(save_file, dest)







