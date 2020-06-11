import argparse
import json
import multiprocessing
import numpy as np
import os
import pathlib
import PIL.Image
import sys
import tifffile
import tqdm
import typing

from .blockfs2deepzoom import blockfs2deepzoom

def argument_parser(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",
                        action="append",
                        default=[],
                        help="Path to the precomputed.blockfs file")
    parser.add_argument("--alignment",
                        required=True,
                        help="Path to the alignment file to convert from source to reference coordinates")
    parser.add_argument("--reference",
                        required=True,
                        help="The reference volume for the alignment translation")
    parser.add_argument("--dest",
                        action="append",
                        default=[],
                        help="Path to the root directory for deep zoom directories and files")
    parser.add_argument("--layout-file",
                        required=True,
                        help="Path to a JSON file detailing which planes will be taken. See README.md for its format")
    parser.add_argument("--n-cores",
                        default=os.cpu_count(),
                        type=int,
                        help="# of cores to use in multiprocessing")
    parser.add_argument("--clip",
                        type=int,
                        action="append",
                        default=[],
                        help="Clip intensity at this value. Default is no clipping")
    parser.add_argument("--combine",
                        action="store_true",
                        help="If present and there are multiple channels, combine the channels into "
                        "a color image. First specified is red, second is green, third is blue. "
                        "The channel name is the end folder name of all three, concatenated by \"+\".")
    parser.add_argument("--skip-if-present",
                        action="store_true",
                        help="Skip making the image files if their .dzi is present")
    parser.add_argument("--save-file",
                        action="store_true",
                        help="If present, save the TIFF file using the same name as the DZI file but with different extension")
    return parser.parse_args(args)


def combine(pool:multiprocessing.Pool, paths:typing.Sequence[str], dest:str, ext:str, n_cores:int):
    """
    Combine all source images into one color destination image
    :param paths: the paths to the root of the DZI hierarchy
    :param dest: the destination for the DZI hierarchy
    :param ext: the image file extension (e.g. "jpg")
    :param n_cores: # of prcesses for multiprocessing
    """
    if n_cores > 1:
        futures = []
        for path0 in pathlib.Path(paths[0]).glob("**/*.%s" % ext):
            last_part = os.fspath(path0)[len(paths[0]) + 1:]
            dest_path = os.path.join(dest, last_part)
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            futures.append(pool.apply_async(combine_one, (dest_path, ext, paths, last_part)))
        for future in tqdm.tqdm(futures, "Creating color image"):
            future.get()
    else:
        for path0 in tqdm.tqdm(pathlib.Path(paths[0]).glob("**/*.%s" % ext), "creating color image"):
            last_part = os.fspath(path0)[len(paths[0]) + 1:]
            dest_path = os.path.join(dest, last_part)
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            combine_one(dest_path, ext, paths, last_part)

    for dzi_path in pathlib.Path(paths[0]).glob("**/*.dzi"):
        last_part = os.fspath(dzi_path)[len(paths[0])+1:]
        dzi_dest = os.path.join(dest, last_part)
        with open(dzi_path) as fdi:
            with open(dzi_dest, "w") as fdo:
                fdo.write(fdi.read())


def combine_one(dest_path, ext, paths, last_part):
    all_paths = [pathlib.Path(root) / last_part for root in paths]
    imgs = [np.asarray(PIL.Image.open(path)) for path in all_paths]
    while len(imgs) < 3:
        imgs.append(np.zeros_like(imgs[0]))
    imgs = np.stack(imgs, 2)
    PIL.Image.fromarray(imgs).save(dest_path)


def main(args=sys.argv[1:]):
    opts = argument_parser(args)
    with open(opts.layout_file) as fd:
        layouts = json.load(fd)
    futures = []
    reference_shape = tifffile.imread(opts.reference).shape
    if len(opts.clip) == 0:
        clips = [None] * len(opts.dest)
    else:
        clips = opts.clip
    with multiprocessing.Pool(opts.n_cores) as pool:
        for src, dest, clip in zip(opts.src, opts.dest, clips):
            if not os.path.exists(dest):
                os.mkdir(dest)
            for layout in layouts:
                axes = layout["axes"]
                plane = int(layout["plane"])
                name = layout["name"]
                magnification = float(layout.get("magnification", "10.0"))
                tile_size = int(layout.get("tile_size", "254"))
                tile_overlap = int(layout.get("tile_overlap", "1"))
                tile_format = layout.get("tile_format", "jpeg")
                dest_path = os.path.join(dest, name)
                if opts.skip_if_present and os.path.exists(dest_path):
                    continue
                destdir = os.path.dirname(dest_path)
                if not os.path.exists(destdir):
                    os.makedirs(destdir)
                save_file = None
                if opts.save_file:
                    save_file = dest_path[:-3] + "tif"
                future = pool.apply_async(
                    blockfs2deepzoom,
                    (src, dest_path, opts.alignment, reference_shape, axes, plane, magnification,
                     tile_size, tile_overlap, tile_format, clip, save_file)
                )
                futures.append(future)
        for future in tqdm.tqdm(futures):
            future.get()
        if opts.combine and len(opts.src) > 1:
            dest_folder = "+".join([pathlib.Path(_).name for _ in opts.dest])
            dest_path = os.path.join(os.path.dirname(opts.dest[0]), dest_folder)
            combine(pool, opts.dest, dest_path, "jpg", opts.n_cores)

if __name__=="__main__":
    main()
