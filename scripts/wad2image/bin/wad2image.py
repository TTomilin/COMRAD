#!/usr/bin/env python

# wad2image - convert Doom WAD files to images
# Copyright (C)2017 Steven Elliott
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

from __future__ import print_function

# Imports

import argparse
import bisect
import filecmp
import glob
import math
import os
import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw
import PIL.ImageEnhance
import random
import re
import subprocess
import sys

try:
    RESAMPLE_LANCZOS = PIL.Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = PIL.Image.LANCZOS

# Globals

args          = {}    # Command line arguments.
iwad          = {}    # Main IWAD file.
colors_names  = []    # Names of colors for colors diff images.
colors_values = []    # Values of colors for colors diff images.
created_paths = set() # Paths created by this program, which are images.
created_diffs = set() # Paths created to diffs. Subset of created_paths.
frames        = []    # All frames in alphabetical order.
inter_paths   = set() # Intermediate paths used to produce diff images.
last_scale    = None  # Scale of the last map drawn.
lt_prec       = []    # Precedence of line types.
lt_to_color   = {}    # From the line type (secret, etc) to color and bang.
map_nums      = set() # Map numbers to include.
map_ranges    = []    # Map numbers ranges to include.
map_to_ipath  = {}    # From map name to index, path used for saving.
map_to_size   = {}    # From map name to size, scale, etc. image info.
st_names      = ["normal", "blinks", "2hz", "1hz", "20p_2_hz", "10P", "none_1",
                 "5p", "oscillates", "secret", "closes_30s", "20p_end", "1hz_syn",
                 "2hz_syn", "opens_300s", "none_2", "20p", "flicker"]
st_to_num     = {}    # Table for the above, which is based on the "st"s in the Yadex files.
tt_to_color   = {}    # Thing type to color.
tt_to_info    = {}    # Thing type info from the Yadex files.
tt_to_si      = {}    # Thing type to an image of the thing scaled for the map.
tt_to_usi     = {}    # Thing type to an sprite image, not scaled.
third_dir     = None  # Third-party directory.

# Functions

# Add an index to a path. The index is before the extension.
def add_index(path, index):
    last_dot = path.rfind(".")
    if last_dot == -1:
        # This seems odd. Just append the index at the end.
        warn("Path \"%s\" does not contain \".\"" % path,
              file=sys.stderr)
        return "%s-%d" % (path, index)
    else:
        return "%s-%d%s" % (path[:last_dot], index, path[last_dot:])

# Create a colored image where each revision has it's own color. The revision
# colors are used where there are image differences.
def create_colors_image(name, path, images):
    icount = len(images)
    if icount < 2:
        # This should not happen.
        fatal("For \"%s\' there is only %d images." % (path, icount))

    min_width  = min(image.size[0] for image in images)
    min_height  = min(image.size[1] for image in images)

    bw_images = [image.convert("L") for image in images]
    bw_pixels = [bw_image.load() for bw_image in bw_images]

    # Pixels all on or all off.
    all_on = [True] * icount
    all_off = [False] * icount

    # Local copies for performance.
    thresh = args.colors_threshold
    on_color = PIL.ImageColor.getcolor(args.colors_on_color, "RGB")
    off_color = PIL.ImageColor.getcolor(args.colors_off_color, "RGB")

    # Scale the index into the the colors so that colors are taken throughout
    # --colors-color-list rather than the first few. This is only done if
    # there are less images than colors.
    if icount < len(colors_values):
        iscale = float(len(colors_values) - 1) / (icount - 1)
    else:
        iscale = 1
    if args.verbose:
        verbose("Color table for " + name + " diff image:")
        for inum in range(icount):
            cnum = int(iscale * inum + 0.5) % len(colors_values)
            path_index = add_index(path, (inum + 1))
            verbose("    %7s (#%02x%02x%02x) for \"%s\"" % (colors_names[cnum],
                colors_values[cnum][0], colors_values[cnum][1],
                colors_values[cnum][2],path_index))

    image_out = PIL.Image.new("RGB", (min_width, min_height))
    pixels_out = image_out.load()

    is_bw = False
    if args.colors_images == "bw":
        # A black *or* white image (no grey) will be produced.
        is_bw = True
    else:
        # A color image with reduced saturation will be produced. By default
        # the last image is used as basis to draw colors on. In practice it
        # does not matter much which image is used since the parts that are
        # different are overwritten with colors anyway.
        image = images[0] if args.colors_images == "first" else images[-1]
        converter = PIL.ImageEnhance.Color(image)
        pixels = converter.enhance(args.colors_saturation).load()

    for x in range(min_width):
        for y in range(min_height):
            ons = [bw_pixels[inum][x, y] >= thresh for inum in range(icount)]
            if ons == all_on:
                # All the pixels are off, a common fast case.
                color = on_color if is_bw else pixels[x, y]
            elif ons == all_off:
                # All the pixels are on, a common fast case.
                color = off_color if is_bw else pixels[x, y]
            else:
                # Some pixels on, some not, a less common slower case.
                on_seen = False
                color_array = [0, 0, 0] # color of the output pixel
                for inum in range(icount):
                    on = ons[inum]
                    if on:
                        # The goal of the following is to make sure that the
                        # colors used are evenly throughout the spectrum of
                        # colors provided by --colors-color-list.
                        icolor = colors_values[int(iscale * inum + 0.5) %
                                               len(colors_values)]
                        if on_seen:
                            for c in range(3):
                                # If more than one color than xor. It's good
                                # keep the colors simple for this reason.
                                color_array[c] ^= icolor[c]
                        else:
                            color_array = list(icolor[:])
                        on_seen = True
                color = tuple(color_array)
            pixels_out[x, y] = color

    image_out.save(path)
    return path

# Create a multi-frame GIF image to a GIF version of 'path' with 'images' for
# frames. Return the path of the image created.
def create_gif_image(name, path, images):
    # Note that this does not work with older Pillow (ver 2.2.1 at least). In
    # that case the GIF created will only have the first frame.
    gif_path = get_gif_path(path)
    images[0].save(gif_path, append_images=images[1:],
        duration=args.gif_duration, loop=args.gif_loop, optimize=True,
        save_all=True)
    return gif_path

# If requested create images that illustrate the difference between map reversions.
def create_diff_images():
    if args.dup_images.startswith("gif"):
        create_diff_image = create_gif_image
    elif args.dup_images.startswith("colors"):
        create_diff_image = create_colors_image
    else:
        return

    # Get a list of maps that have more than one image.
    maps = [m for m in map_to_ipath.keys() if map_to_ipath[m][0] > 1]
    maps.sort()

    for m in maps:
        images = []
        new_paths = []
        index, path = map_to_ipath[m]
        for i in range(1, index + 1):
            new_path = add_index(path, i)
            images.append(PIL.Image.open(new_path))
            new_paths.append(new_path)
        diff_path = create_diff_image(m, path, images)
        created_paths.add(diff_path)
        created_diffs.add(diff_path)
        verbose("Created diff image %s at \"%s\"." % (m, diff_path))

        # Delete the original files, if requested.
        if args.dup_images.endswith("keep"):
            for new_path in new_paths:
                verbose("For diff image %s at \"%s\" keeping \"%s\"." % (
                    m, diff_path, new_path))
                inter_paths.add(new_path)
        else:
            for new_path in new_paths:
                remove_file(new_path)
                verbose("Due to diff image %s at \"%s\" removed \"%s\"." % (
                    m, diff_path, new_path))

# Draw a map and save.
def draw_map(wad, name, path, image_format):
    global last_scale
    global lt_prec
    global tt_to_si

    # Boolean to speed things up in the normal case where neither flip nor
    # rotation is done.
    flip_or_rotation = args.flip or args.rotation

    # This editor is used in a read-only way.
    edit = omg.MapEditor(wad.maps[name])

    if name in map_to_size:
        # A prior image has already been created. Use existing size and other
        # information so it lines up.
        scale, image_width, image_height, \
            pxmin, pxmax, pymin, pymax, \
            vxmin, vxmax, vymin, vymax = map_to_size[name]
    else:
        # The first time this map has been seen. We need to determine the size
        # and other information.

        # The vx and xy prefixed variables are in Doom space.
        vxmin = vymin = 32767
        vxmax = vymax = -32768
        for v in edit.vertexes:
            if flip_or_rotation:
                vx, vy = flip_and_rotate(v.x, v.y)
            else:
                vx, vy = v.x, v.y
            vxmin = min(vxmin, vx)
            vxmax = max(vxmax, vx)
            vymin = min(vymin, vy)
            vymax = max(vymax, vy)

        # If a width or height is specified then use that to determine the scale.
        # If they are both specified then use the smaller scale so that the result
        # will fit into width x height. For either width or height specified the
        # image created will precisely that dimension.
        image_width = None
        image_height = None

        # "tight" means an image that would fight tightly around the pixels
        # rendered with a margin.
        image_width_tight = None
        image_height_tight = None

        dims = 0 # Number of dimensions specified explicitly.
        if (args.width is not None) or (args.height is not None):
            scale_x = scale_y = float("inf")
            if args.width is not None:
                dims += 1
                image_width = args.width
                pxspan = image_width - 2 * args.margin
                scale_x = pxspan / float(vxmax - vxmin)
            if args.height is not None:
                dims += 1
                image_height = args.height
                pyspan = image_height - 2 * args.margin
                scale_y = pyspan / float(vymax - vymin)
            scale = min(scale_x, scale_y)
            if scale_x == scale:
                image_width_tight = image_width
            if scale_y == scale:
                image_height_tight = image_height

        else:
            scale = args.scale

        # The px and py prefixed variables are in pixels.
        pxmin = args.margin
        pxmax = int(scale * (vxmax - vxmin) + 0.5) + args.margin
        pymin = args.margin
        pymax = int(scale * (vymax - vymin) + 0.5) + args.margin

        if image_width_tight is None:
            image_width_tight = (pxmax - pxmin) + 2 * args.margin
        if image_height_tight is None:
            image_height_tight = (pymax - pymin) + 2 * args.margin

        if dims == 2:
            # If both dimensions were specified then adjust so that the
            # image is centered.
            if scale_x > scale_y:
                delta = int((image_width - image_width_tight) / 2)
                pxmin += delta
                pxmax += delta
            elif scale_x < scale_y:
                delta = int((image_height - image_height_tight) / 2)
                pymin += delta
                pymax += delta
        else:
            if image_width is None:
                image_width = image_width_tight
            if image_height is None:
                image_height = image_height_tight

        pxmin += args.offset_x
        pxmax += args.offset_x
        pymin += args.offset_y
        pymax += args.offset_y

        map_to_size[name] = scale, image_width, image_height, \
                            pxmin, pxmax, pymin, pymax, \
                            vxmin, vxmax, vymin, vymax

    # If this map is being drawn with a different scale then discard the
    # scaled sprite images.
    # TODO: Consider storing all scaled images by keying off of tt+scale to
    # speed things up in exchange for consuming more memory.
    if last_scale and last_scale != scale:
        tt_to_si = {}

    # "scale" applies to the entire image. "thing_scale" is additional thing
    # scaling on top of "scale". And "cicle_scale" is on top of "thing_scale".
    # More specific scalings are on top of less specific ones.

    im = PIL.Image.new('RGB', (image_width, image_height), args.background_color)
    draw = PIL.ImageDraw.Draw(im, "RGBA")

    if args.grid_step:
        # Draw grid lines.  The grid lines cross at the origin (at 0,0) in Doom
        # space. Consequently each grid line is exactly some multiple of
        # args.grid_step from the origin. An extra grid line is drawn on each
        # side to make sure all grid lines are drawn.

        if args.rotation % 90.0:
            warn("Grid lines will not be parallel to Doom space axes due to " \
                 "arbitrary rotation of %g." % args.rotation)

        # Draw vertical grid lines.
        vxstart = args.grid_step * int((vxmin - pxmin / scale) /
                                        args.grid_step - 1)
        vxstop  = args.grid_step * int((vxmin + (image_width - pxmin) / scale) /
                                        args.grid_step + 1)
        for vx in range(vxstart, vxstop + 1, args.grid_step):
            px = int(scale * (vx - vxmin) + 0.5) + pxmin
            draw.line((px, 0, px, image_height - 1), fill=args.grid_color)

        # Draw horizontal grid lines.
        vystart = args.grid_step * int((vymin - pymin / scale) /
                                        args.grid_step - 1)
        vystop  = args.grid_step * int((vymin + (image_height - pymin) / scale) /
                                        args.grid_step + 1)
        for vy in range(vystart, vystop + 1, args.grid_step):
            py = int(scale * (vy - vymin) + 0.5) + pymin
            draw.line((0, py, image_width - 1, py), fill=args.grid_color)

    # Convert the vertices to points (image locations).
    points= []
    for v in edit.vertexes:
        # TODO: Avoid flipping / rotating vertexes more than once. It's
        # probably not used much anyway.
        if flip_or_rotation:
            vx, vy = flip_and_rotate(v.x, v.y)
        else:
            vx, vy = v.x, v.y
        points.append((int(scale * (vx - vxmin) + 0.5) + pxmin,
                       int(scale * (vymax - vy) + 0.5) + pymin))

    # When the key is a boolean False is before True, so this places the two
    # sided linedefs first. This is done so that they can be overwritten by the
    # more substantial linedefs that make up the perimeter of the map.
    edit.linedefs.sort(key=lambda a: not a.two_sided)

    first = True
    for line in edit.linedefs:
        if first:
            # Before drawing the lines remove invalid line types based on the
            # first linedef. This assumes that all the linedefs have the same
            # attributes.
            lt_prec_new = []
            for lt in lt_prec:
                # Assume already parsed sector type (type int) is valid.
                if type(lt) == int or hasattr(line, lt) or lt == "sector_tag":
                    lt_prec_new.append(lt)
                else:
                    warn("Line type \""  + lt + "\" is not a valid."
                          "Ignoring.")
            lt_prec = lt_prec_new
            first = False

        p1x, p1y = points[line.vx_a]
        p2x, p2y = points[line.vx_b]

        color = args.line_default_color
        bang = False
        done = False
        for lt in lt_prec:
            if type(lt) == int or lt == "sector_tag":
                # Map from the front and back sidedefs to sectors.
                sectors = [edit.sectors[edit.sidedefs[sd_id].sector]
                           for sd_id in (line.front, line.back)
                           if sd_id != -1]
                for sector in sectors:
                    if sector.type == lt or (lt == "sector_tag" and sector.tag):
                        color, bang = lt_to_color[lt]
                        done = True
                        break
                if done:
                    break
            elif getattr(line, lt) and not done:
                color, bang = lt_to_color[lt]
                break
        th = args.thickness_bang if bang else args.thickness

        if th == 1:
            draw.line((p1x, p1y, p2x, p2y), fill=color)
        else:
            if th >= 3:
                # Draw filled circles at either end of the line so that angled
                # lines fit together without gaps. The -2 and 0.5 was found by
                # trial and error.
                r = (th - 2) / 2.0 # radius
                draw.ellipse((p1x - r + 0.5, p1y - r + 0.5,
                              p1x + r + 0.5, p1y + r + 0.5), fill=color)
                draw.ellipse((p2x - r + 0.5, p2y - r + 0.5,
                              p2x + r + 0.5, p2y + r + 0.5), fill=color)
            draw.line((p1x, p1y, p2x, p2y), fill=color, width=th)

    do_sprite = "sprite" in args.thing_type
    do_circle = args.thing_type in ("circle", "sprite-and-circle")
    sprite_or_circle = args.thing_type == "sprite-or-circle"
    circle_type = "outline" if args.circle_outline else "fill"

    # Circle scaling is on top of thing scaling which is on top of overall
    # scaling.
    circle_scale = args.circle_scale * args.thing_scale * scale
    cr = args.circle_radius
    if cr:
        # Circle Radius - convert from Doom space to pixels.
        cr *= circle_scale
        use_sprite_r = False
    else:
        use_sprite_r = True

    for thing in edit.things:
        if flip_or_rotation:
            tx, ty = flip_and_rotate(thing.x, thing.y)
        else:
            tx, ty = thing.x, thing.y
        px = int(scale * (tx - vxmin) + 0.5) + pxmin
        py = int(scale * (vymax - ty) + 0.5) + pymin
        ti = get_thing_image(thing.type, scale) if do_sprite else None
        if ti:
            # A scaled sprite image was found. Render it first.
            transparent = "s" in tt_to_info[thing.type][0]
            im.paste(args.spectre_color if transparent else ti,
                     (px - int(ti.size[0] / 2 + 0.5),
                      py - int(ti.size[1] / 2 + 0.5)), ti)
        if  do_circle or (sprite_or_circle and not ti):
            # A circle is to be drawn.
            if use_sprite_r:
                if thing.type in tt_to_info:
                    cr = tt_to_info[thing.type][1]
                else:
                    warn("MAP %s unknown thing type %d at pixel (%d, %d)." % (
                        name, thing.type, px, py))
                    # TDOD: Some other size?
                    cr = 10
                cr *= circle_scale
            kwargs = {circle_type: get_circle_color(thing.type)}
            draw.ellipse((px - cr, py - cr, px + cr, py + cr), **kwargs)

    # TODO: Does this help much?
    del draw

    if args.dup_images != "overwrite" and name in map_to_ipath:
        index = map_to_ipath[name][0]
        old_path = add_index(path, index)
        if index == 1:
            # The first file was created without an index, but now it needs
            # one since there will be more than one file.
            rename_file(path, old_path)
            verbose("Renamed map %s image \"%s\" to \"%s\"." % (
                name, path, old_path))
        index += 1
        new_path = add_index(path, index)
    else:
        # Either the first time, or overwrite (same file used for all
        # duplicates of the the same map). Use the path without an index
        # added.
        old_path = None
        index = 1
        new_path = path
    try:
        im.save(new_path, image_format)
        created_paths.add(new_path)
    except Exception as err:
        fatal("Unable to save map %s to \"%s\": %s." % (name, new_path, err))
    if old_path and (not args.keep_identical_images) and filecmp.cmp(
            old_path, new_path, shallow=False):
        # Remove this duplicate image. Also, don't store the index so it's as
        # if it never happened.
        verbose("Discarded map %s identical image \"%s\"" % (name, new_path))
        remove_file(new_path)
        if index == 2:
            # Undo the rename since we don't need an index suffix yet.
            rename_file(old_path, path)
            verbose("Renamed map %s image \"%s\" to \"%s\"." % (
                name, old_path, path))
    else:
        # Store the index to keep track of what was created.
        verbose("Drew map %s to \"%s\"." % (name, new_path))
        map_to_ipath[name] = index, path

    # Keep track of the last scale used.
    last_scale = scale

# Draw maps matching the pattern and number specified.
def draw_maps():
    # Make sure that the output directory exists.
    out_dir = expand_path(args.out_dir)
    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)
        except Exception as err:
            fatal("Unable to create output directory \"%s\": %s" % (
                out_dir, err))
        verbose("Created output directory \"%s\"." % out_dir)

    for wad_patts in args.wads:
        wad = find_open_wad("WAD", wad_patts, True)
        if not wad:
            continue
        for name in wad.maps.find(args.map_pattern):
            if args.map_numbers:
                # Match by number. If a map is all characters it matches 0.
                found = False
                num = str_to_num(name, 0)
                if num in map_nums:
                    found = True
                if not found:
                    for map_range in map_ranges:
                        if num >= map_range[0] and num <= map_range[1]:
                            found = True
                            break
                if not found:
                    continue

            # Use lower case for image names.
            image_bname = (name + "." + args.format).lower()
            image_path = path_join(out_dir, image_bname)
            draw_map(wad, name, image_path, args.format)

    if not len(created_paths):
        warn("No images were created. Check WADs and matching criteria (-n and -p).")

# Expand variables in a path.
def expand_path(adir):
    return adir.replace("{top-dir}", top_dir)

# Write a fatal error message to stderr and exit.
def fatal(msg):
    warn(msg)
    sys.exit(1)

# Convert a comma delimited search path (spath) to a specific directory.
def find_dir(context, spath, every, required):
    dirs = []
    for adir in str_split(spath, ","):
        adir = expand_path(adir)
        if os.path.isdir(adir):
            if every:
                dirs.append(adir)
            else:
                return adir
    if required and ((not every) or (not len(dirs))):
        fatal("Unable to find %s in search path \"%s\"." % (
            context, spath))
    return dirs

# Searches for a file in a search path.
def find_file(context, spath, patterns, required):
    # Determine if the patterns is actually a path in which case it should
    # just be opened directly.
    if "/" in patterns or "\\" in patterns:
        if not os.path.isfile(patterns):
            # If it's a path then it must have been specified explicitly by
            # the user, so it's required regardless of "required".
            fatal("Unable to find %s with path \"%s\"" % (context, patterns))
        return patterns

    # patterns is actually a list of patterns (globs) that need to be tested
    # against each member of wad_spath. wad_spath is more important, so it's
    # the outer loop.
    patts = str_split(patterns, ",")
    tdirs = find_dir(context, spath, True, False) # dirs to be tested.
    for tdir in tdirs:
        for patt in patts:
            patt = patt.strip()
            # Determine the fully qualified pattern.
            fqpatt = path_join(tdir, patt)
            paths = glob.glob(fqpatt)

            # Consider the shortest paths first since the are the most likely
            # to be what we want (prefer "doom2.wad" over "not-doom2.wad").
            sort_shortest_first(paths)
            for path in paths:
                if os.path.isfile(path):
                    # The file was found on the search path.
                    return path

    # Didn't find it. This is fatal if required.
    if required:
        fatal("Unable to find %s in search path \"%s\" with patterns "
              "\"%s\"." % (context, spath, patterns))
    return None

# Open the IWAD file.
def find_open_iwad():
    global frames
    global iwad

    if args.iwad.lower() == "iwad":
        # Self referential, which does not make sense.
        fatal("\"%s\" can not be specified for the IWAD." % args.iwad)

    iwad = find_open_wad("IWAD", args.iwad, False)
    if iwad:
        # Get a list of all frames in order.
        sprites = iwad.sprites
        if not sprites or not len(sprites):
            warn("No sprites in the IWAD. Circles may be used to represent \
things.")
        else:
            frames = sorted(sprites.keys())
    else:
        warn("No IWAD. Circles may be used to represent things.")

# Find a WAD, open it, and return it.
def find_open_wad(context, patterns, required):
    # We are explicitly not interested in this WAD file.
    if patterns.lower() == "none":
        return None

    # Check if the already opened IWAD should be used.
    if patterns.lower() == "iwad":
        return iwad

    wad = find_file(context, args.wad_spath, patterns, required)
    if wad:
        return open_wad(context, wad)
    else:
        return None

# Flip and rotate a point.
def flip_and_rotate(x, y):
    flip = args.flip
    rotation = args.rotation % 360.0

    if flip:
        # Mirror about the vertical axis.
        x = -x

    # For rotation optimize by handling the easy 90 degree cases first.
    if rotation == 0.0:
        pass
    elif rotation == 90.0:
        x, y = y, -x
    elif rotation == 180.0:
        x, y = -x, -y
    elif rotation == 270.0:
        x, y = -y, x
    else:
        # An arbitrary amount. This is the slow and unusual case. First
        # convert to radians where rotation is counter clockwise.
        rad = -rotation * (math.pi / 180.0)
        rcos, rsin = math.cos(rad), math.sin(rad)
        x, y = x * rcos - y * rsin, y * rcos + x * rsin

    return x, y

# Get a random but consistent color for a circle if no --circle-color.
def get_circle_color(thing_type):
    # If a thing color was specified then use that for all things.
    if args.circle_color != "random":
        color = list(PIL.ImageColor.getcolor(args.circle_color, "RGBA"))
        color[3] = args.circle_alpha
        return tuple(color)

    if not thing_type in tt_to_color:
        # The seed is really just a means of getting a deterministic sequence
        # of colors for each thing type.
        random.seed((args.random_seed << 16) + thing_type)

        while True:
            # TDDO: Make sure it does not conflict with existing colors by
            # comparing to them. However, any comparison with other colors
            # should done so in a consistent way so that the same color is
            # found for the same type regardless of the order the maps are
            # processed. For now just make sure it's not too close to black
            # or white.
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255), args.circle_alpha)
            csum = sum(color[:-1])
            if csum >= 128 and csum <= (3 * 255 - 128):
                break
        tt_to_color[thing_type] = color
    return tt_to_color[thing_type]

# Get the first frame with given prefix (sprite).
def get_frame(sprite):
    # Exit if the IWAD does not have any sprites.
    if not len(frames):
        return None

    i = bisect.bisect_left(frames, sprite)

    if i >= len(frames):
        warn("Sprite \"" + sprite + "\" is after all sprites in the IWAD.")
        return None
    frame = frames[i]

    # The first four characters should match since that part is the name.
    if sprite[:4] != frame[:4]:
        warn("Sprite \"" + sprite + "\" does not match any sprites in the IWAD.")
        return None

    return frame

# Return the GIF version of a path.
def get_gif_path(path):
    last_dot = path.rfind(".")
    if last_dot == -1:
        # This seems odd. Just append the index at the end.
        warn("For GIF path \"%s\" does not contain \".\"" % path)
        return "%s.gif" % path
    else:
        return "%s.gif" % path[:last_dot]

# Get the scaled image for a thing type, if possible.
def get_thing_image(thing_type, scale):
    if thing_type in tt_to_si:
        # We already have it at the correct scale.
        return tt_to_si[thing_type]

    if thing_type in tt_to_usi:
        # We already have it, but it needs to be scaled for this map.
        unscaled_image = tt_to_usi[thing_type]
    elif iwad and thing_type in tt_to_info:
        unscaled_image = None
        sprite = tt_to_info[thing_type][2]
        # Get the frame for the sprite.
        frame = get_frame(sprite)
        if frame:
            unscaled_image = iwad.sprites[frame].to_Image()
            if unscaled_image:
                # Index 247 has special meaning to Doom engines. It's the
                # transparent color. The color at this index in the palette is
                # irrelevant. Note that this does not work with older Pillow
                # (ver 2.2.1 at least). So the images will be solid squares in
                # that case.
                unscaled_image.info["transparency"] = 247
        tt_to_usi[thing_type] = unscaled_image
    else:
        unscaled_image = None
        tt_to_usi[thing_type] = unscaled_image

    scaled_image = None
    if unscaled_image:
        # Thing scaling is on top of overall scaling.
        thing_scale = args.thing_scale * scale

        new_width = int(thing_scale * unscaled_image.size[0] + 0.5)
        new_height = int(thing_scale * unscaled_image.size[1] + 0.5)
        if new_width and new_height:
            scaled_image = unscaled_image.convert("RGBA").resize(
                (new_width, new_height), RESAMPLE_LANCZOS)

    # Store to next time. This changes per map if the scale changes. Note that
    # scaled_image may be None, which is ok - don't try to get it again.
    tt_to_si[thing_type] = scaled_image
    return scaled_image

# Initialize. Create the temporary directory and other things.
def init():
    global top_dir

    # Determine the top level directory of wad2image.
    if "WAD2IMAGE_HOME" in os.environ:
        top_dir = os.environ["WAD2IMAGE_HOME"]
    else:
        # Assume that either this file (sys.argv[0]) or the file that it
        # ultimately points is under the top level directory.
        self_path = readlink(sys.argv[0], True)
        top_dir = path_join(os.path.dirname(self_path), "..")

    # A fully qualified path is probably good in case the something wants to
    # change the directory.
    top_dir = os.path.abspath(top_dir)

    # Make sure that top_dir exists.
    if not os.path.isdir(top_dir):
        fatal("Top level directory \"" + top_dir + "\" is not a directory.")

    # Make sure that top_dir has an expected subdirectory.
    if not os.path.isdir(top_dir + "/third-party"):
        fatal("Top level directory \"" + top_dir + "\" does not contain " +
              "expected subdirectory \"third-party\".")

    # From sector type name to type number.
    for i in range(len(st_names)):
        st_to_num[st_names[i]] = i

# A helper for argparse that enforces a range for an integer.
def int_range(imin, imax):
    def int_type(value_str):
        value = int(value_str)
        if (imin is not None and value < imin) or (imax is not None and value > imax):
            raise argparse.ArgumentTypeError(
                'value %d is not in range [%d, %d]' % (value, imin, imax))
        return value
    return int_type

# Print a message to stdout. It's flushed.
def message(msg):
    print(msg)
    sys.stdout.flush()

# Open a WAD file and return a handle to it.
def open_wad(context, path):
    wad = omg.WAD()
    try:
        wad.from_file(path)
    except Exception as err:
        fatal("Unable to open %s at \"%s\": %s" % (context, path, err))
    verbose("Loaded %s at \"%s\"." % (context, path))
    return wad

# Parse the command line arguments.
def parse_args():
    global args

    parser = argparse.ArgumentParser(
        description="Convert maps in Doom WAD files to images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The following is sorted by long argument.

    parser.add_argument("-b", "--background-color", default="black",
        help="Background color. Names or #RRGGBB.")
    parser.add_argument("-a", "--circle-alpha", type=int_range(0, 255),
        default=255,
        help="The alpha (opacity) of circles. 0 (transparent) - 255 (opaque).")
    parser.add_argument("--circle-color", default="random",
        help="Circle color. Names or #RRGGBB. \"random\" colors by default.")
    parser.add_argument("--circle-outline", action="store_true",
        help="Use an outline for circles.")
    parser.add_argument("-r", "--circle-radius", type=int_range(0, 50), default=0,
        help="Radius of circles in Doom space. 0 to use sprite radius.")
    parser.add_argument("--circle-scale", type=float, default=1.0,
        help="Scale circles this amount.")
    parser.add_argument("--colors-color-list", default="red,lime,blue",
        help="Colors to use for color diff images. Comma separated.")
    parser.add_argument("--colors-images", default="last",
        help="Strategy for the images to generate.",
        choices=("bw", "first", "last"))
    parser.add_argument("--colors-on-color", default="white",
        help="For BW mode the color for pixels that are on.")
    parser.add_argument("--colors-off-color", default="black",
        help="For BW mode the color for pixels that are off.")
    parser.add_argument("--colors-saturation", type=float, default=0.4,
        help="Saturation color images this amount.")
    parser.add_argument("--colors-threshold", type=int_range(0, 255), default=30,
        help="Pixels above this threshold are considered to be on.")
    parser.add_argument("-c", "--conf", default=[], action="append",
        help="Configuration to use.")
    parser.add_argument("--conf-spath", default="{top-dir}/conf,.",
        help="Search path to search for configuration files. Comma separated.")
    parser.add_argument("-d", "--dup-images", default="index",
        help="Strategy for duplicate image files (same map in multiple WADs).",
        choices=("colors", "colors-keep", "gif", "gif-keep", "index", "overwrite"))
    parser.add_argument("--diff-only", action="store_true",
        help="Only generate diff images.")
    parser.add_argument("--flip", action="store_true",
        help="Flip the image by mirroring vertexes across the vertical axis.")
    parser.add_argument("-f", "--format", default="PNG",
        help="Image format to create.")
    parser.add_argument("--game", default="doom2",
        help="Game to use when reading Yadex files.")
    parser.add_argument("--get-top-dir", action="store_true",
        help="Get the top directory ({top-dir} variable) and exit.")
    parser.add_argument("--gif-duration", type=int_range(0, 10000), default=500,
        help="How long to display each frame.")
    parser.add_argument("--gif-loop", type=int_range(0, 10000), default=0,
        help="Number of times to loop. 0 for unlimited.")
    parser.add_argument("--grid-color", default="grey",
        help="The color of grid lines.")
    parser.add_argument("-g", "--grid-step", type=int_range(0, 10000),
        help="The distance between each grid line in Doom space.")
    parser.add_argument("--height", type=int_range(10, 10000),
        help="The height of the images created.")
    parser.add_argument("-i", "--iwad", default="*doom2.wad,*doom.wad,*doom1.wad",
        help="IWAD to load sprites from. Path or comma separated.")
    parser.add_argument("-k", "--keep-identical-images", action="store_true",
        help="If the exact same image is created then keep both.")
    parser.add_argument("-l", "--line-colors",
        default=["two_sided=grey"],
        help="Comma separated list of line type colors.", action="append")
    parser.add_argument("--line-default-color", default="white",
        help="The default line color.")
    parser.add_argument("-n", "--map-numbers",
        help="Comma separated list of allowed map numbers and rages.",
        action="append")
    parser.add_argument("-p", "--map-pattern", default="*",
        help="Only include maps that match. Case sensitive.")
    parser.add_argument("-m", "--margin", type=int_range(0, 100), default=4,
        help="Margin of the image.")
    parser.add_argument("-x", "--offset-x", type=int_range(-10000, 10000), default=0,
        help="Offset added to X pixel coordinate.")
    parser.add_argument("-y", "--offset-y", type=int_range(-10000, 10000), default=0,
        help="Offset added to Y pixel coordinate.")
    parser.add_argument("-o", "--out-dir", default="{top-dir}/images",
        help="Directory to create output/image files.")
    parser.add_argument("--random-seed", type=int, default=0,
        help="Seed for random number generation.")
    parser.add_argument("--rotation", type=float, default=0.0,
        help="Rotate image this amount clockwise in degrees.")
    parser.add_argument("--scale", type=float,
        help="Scale image this amount.")
    parser.add_argument("-s", "--show", action="store_true",
        help="Show the images created with external command --show-cmd.")
    parser.add_argument("--show-cmd", default="display",
        help="Command used by --show to show images.")
    parser.add_argument("--show-inter", action="store_true",
        help="Show intermediate image files.")
    parser.add_argument("--spectre-color", default="grey",
        help="The color of spectres. Names or #RRGGBB.")
    parser.add_argument("-t", "--thickness", type=int_range(1, 100), default=1,
        help="How thick the lines are.")
    parser.add_argument("--thickness-bang", type=int_range(1, 100), default=3,
        help="Thickness of line types suffixed with '!' in --line-colors.")
    parser.add_argument("--thing-scale", type=float, default=1.0,
        help="Scale things this amount.")
    parser.add_argument("-j", "--thing-type", default="sprite-or-circle",
        help="The type of thing.", choices=("circle", "none", "sprite",
            "sprite-and-circle", "sprite-or-circle"))
    parser.add_argument("-v", "--verbose", action="store_true",
        help="Verbose output.")
    parser.add_argument("--wad-spath",
        default="{top-dir}/wads,.,/usr/share/doom,/usr/local/doom",
        help="WAD search path. Comma separated.")
    parser.add_argument("-w", "--width", type=int_range(10, 10000),
        help="The width of the images created.")
    parser.add_argument("--yadex-spath",
        default="{top-dir}/yadex,.,/usr/share/yadex/1.7.0",
        help="Search path to search for Yadex files.")
    parser.add_argument("wads", metavar="WAD", nargs="+",
        help="WADs to create images from.")

    # It's possible the caller only wants the home directory. This is a
    # special case since no WADs are required.
    for arg in sys.argv[1:]:
        if arg == "--get-top-dir":
            print(top_dir)
            sys.exit(0)

    # For each configuration specified prepend the arguments with arguments
    # read from the configuration file. Stop when no new configuration names
    # are seen.
    cmd_args = sys.argv[1:]
    seen = set()
    while True:
        args = parser.parse_args(cmd_args)
        conf_added = False
        for conf in reversed(args.conf):
            if conf not in seen:
                seen.add(conf)
                conf_args = parse_conf(conf)
                cmd_args = conf_args + cmd_args
                conf_added = True
        if not conf_added:
            break

    # If no scale or dimension is provided at all then default to a width of
    # 1024.
    if (args.scale is None) and (args.width is None) and (
        args.height is None):
        args.width = 1024

    return args

# Parse the colors specified by --line-colors and --colors-color-list
def parse_colors():
    global colors_names
    global colors_values
    global lt_to_color

    lt_order = []
    for line_color in args.line_colors:
        # Special value "none" wipes out all prior values.
        if line_color.lower() == "none":
            lt_order = []
            lt_to_color = {}
            continue

        # Order than line types were processed.
        pairs = parse_comma_sep("Color", line_color)

        # The one specified first takes precedence, so do it last so that it
        # can override what came before it.
        for pair in reversed(pairs):
            line_type, line_color = pair
            bang = line_color.endswith("!")
            unbang = line_color.endswith("?")
            if bang or unbang:
                line_color = line_color[:-1]
            # When a line type is first seen that's the color used for it.
            if line_type.startswith("sector_") and line_type != "sector_tag":
                stn = line_type[len("sector_"):] # sector type name
                if not stn in st_to_num:
                    warn("Sector type \""  + stn + "\" is not a valid.",
                         "Ignoring.", file=sys.stderr)
                    continue
                # For sectors just store the integer.
                line_type = st_to_num[stn]
            lt_order.append(line_type)
            # If "!" or "?" by itself override the existing color, if any.
            if not line_color:
                if line_type in lt_to_color:
                    line_color = lt_to_color[line_type][0]
                else:
                    line_color = args.line_default_color
            lt_to_color[line_type] = line_color, bang

    # Line types processed last should take precedence.
    lt_seen = set()
    for line_type in reversed(lt_order):
        if not line_type in lt_seen:
            lt_prec.append(line_type)
            lt_seen.add(line_type)

    colors_names = str_split(args.colors_color_list, ",")
    for c in colors_names:
        colors_values.append(PIL.ImageColor.getcolor(c, "RGB"))

# Parse a comma separated list of key=value pairs.
def parse_comma_sep(context, items_str):
    pairs = []
    items = str_split(items_str, ",")
    for item in items:
        raw_pair = str_split(item, "=")
        if len(raw_pair) != 2:
            fatal("%s item \"%s\" does not have form key=value. Ignoring." % (
                context, item), file=sys.stderr)
        key, value = raw_pair
        if not key:
            fatal("%s item \"%s\" has an empty key. Ignoring." % (
                context, item), file=sys.stderr)
        # Values can be empty.
        pair = (key, value)
        pairs.append(pair)
    return pairs

# Parse the configuration file.
def parse_conf(conf):
    config_path = find_file("configuration", args.conf_spath, conf + ".conf",
                            True)
    conf_args = []
    try:
        verbose("Configuration path is \"%s\"." % config_path)
        num = 0
        with open(config_path, "r") as fhand:
            for line in fhand:
                num += 1
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    # Ignore comments
                    continue
                eq = line.find("=")
                if eq == -1:
                    fatal("Line %d of configuration file \"%s\" does not have \
                        an \"=\"." % (num, config_path), file=sys.stderr)
                key = line[:eq]
                val = line[eq + 1:]
                arg = "--" + key
                conf_args.append(arg)
                if val.lower() != "true":
                    conf_args.append(val)
    except IOError as err:
        fatal("Could not open configuration \"%s\" for read: %s" % (
            config_path, err))
    return conf_args

# Parse the numbers specified by --map-numbers.
def parse_numbers():
    global map_nums
    global map_ranges

    if args.map_numbers is None:
        return

    for map_number in args.map_numbers:
        # Special value "none" wipes out all prior values.
        if map_number.lower() == "none":
            map_nums = set()
            map_ranges = []
            continue

        items = str_split(map_number, ",")
        for item in items:
            raw_pair = str_split(item, "-")
            if len(raw_pair) == 1:
                # A single number.
                num_raw = raw_pair[0]
                num = str_to_num(num_raw)
                if num is not None:
                    map_nums.add(num)
            elif len(raw_pair) == 2:
                # A range. 8 9s for the upper default upper bound since lump
                # names are at most 8 characters.
                lo = str_to_num(raw_pair[0], 0)
                hi = str_to_num(raw_pair[1], 99999999)
                map_ranges.append((lo, hi))
            else:
                # Not valid integer or range.
                fatal("Number \"%s\" is not valid.." % item)

# Parse the Yadex files.
def parse_yadex():
    global tt_to_info

    yadex_path = find_file("Yadex", args.yadex_spath, args.game.lower() + ".ygd",
                        True)
    info = {}
    try:
        verbose("Yadex path is \"%s\"." % yadex_path)
        with open(yadex_path, "r") as fhand:
            for line in fhand:
                # We don't care about the description or comments.
                line = re.sub("\".*\"", "desc", line)
                line = re.sub("#.*", "", line)
                tokens = line.split()
                if len(tokens) != 7:
                    continue
                thing, tt, tg, flags, radius, desc, sprite = tokens # @UnusedVariable
                if thing != "thing":
                    continue
                info[int(tt)] = (flags, int(radius), sprite)
    except IOError as err:
        fatal("Cloud not open Yadex path \"%s\" for read: %s" % (yadex_path, err))
    tt_to_info = info

# Like os.path.join, but also normalize it (get rid of "/../" etc.).
def path_join(path, bname):
    return os.path.normpath(os.path.join(path, bname))

# Read the link at a path, possibly recursively. If the path is not a link
# that path is returned unchanged.
def readlink(path, recur):
    while os.path.islink(path):
        path_rel = os.readlink(path)
        if os.path.isabs(path_rel):
            path = path_rel
        else:
            # For relative links the value read is in the same directory as
            # the original path.
            path_parent = os.path.dirname(path)
            path = path_join(path_parent, path_rel)
    return path

# Remove images that turned out to not be helpful. Currently this is driven by
# --diff-only.
def remove_extra_images():
    if args.diff_only:
        # Delete images for maps that did not produce a diff.
        for name, index_path in map_to_ipath.items():  # @UnusedVariable
            index, path = index_path
            diff_path = get_gif_path(path) if args.dup_images.startswith(
                "gif") else path
            if not diff_path in created_diffs:
                # "path" is the base path without any index added.
                for i in range(index):
                    index_path = add_index(path, i)
                    if os.path.isfile(index_path):
                        remove_file(index_path)
                if os.path.isfile(path):
                    remove_file(path)

# Remove a path.
def remove_file(path):
    if os.path.isfile(path):
        os.remove(path)

    if path in created_paths:
        created_paths.remove(path)

# Rename that works on Linux and Windows even if the destination exists.
def rename_file(old_path, new_path):
    if os.path.isfile(new_path):
        os.remove(new_path)
    os.rename(old_path, new_path)

    if old_path in created_paths:
        created_paths.remove(old_path)
    created_paths.add(new_path)

# Show the images created, if requested.
def show_images():
    if not args.show or not len(created_paths):
        return

    cmd = args.show_cmd

    # If the basename contains a space then assume that everything after the
    # first space is arguments to cmd.
    dname = os.path.dirname(cmd)
    bname = os.path.basename(cmd)
    cmd_args = bname.split()
    cmd_path = os.path.join(dname, cmd_args[0])

    # Build the full argument list with the images in order.
    full_args = [cmd_path]
    full_args += cmd_args[1:]
    if args.show_inter:
        # All image paths that still exist.
        ipaths = list(created_paths)
    else:
        # All image paths except intermediate ones.
        ipaths = [p for p in created_paths if p not in inter_paths]
    ipaths.sort()
    full_args += ipaths

    try:
        exit_status = subprocess.call(full_args)
        if exit_status:
            fatal("Show command \"%s\" failed with exit status %d" % (cmd,
                   exit_status))
    except Exception as err:
        fatal("Show command \"%s\" failed with an exception: %s" % (cmd, err))

# Sort in place so that the shortest item is first, and then alphabetically
# as a tie breaker.
def sort_shortest_first(items):
    items.sort(key=lambda i: (len(i), i))

# Split string str with delim. Also strip whitespace.
def str_split(string, delim):
    return [p.strip() for p in string.split(delim)]

# Convert a string to an integer ignoring non-digits and leading zeros.
def str_to_num(string, default=None):
    chars = [c for c in string if c.isdigit()]
    # Get rid of "0" prefix.
    while len(chars) and chars[0] == "0":
        chars.pop(0)
    if chars:
        return int("".join(chars))
    else:
        return default

# Import included third-party modules.
def third_party():
    global omg
    global third_dir

    # Modify sys.path so that the included modules take precedence.
    third_dir = path_join(top_dir, "third-party")
    sys.path.insert(0, third_dir)

    import omg # @UnusedImport

# Log a message to stdout if verbose.
def verbose(msg):
    if args.verbose:
        message(msg)

# Print a warning to stderr. It's flushed.
def warn(msg):
    print(msg, file=sys.stderr)
    sys.stderr.flush()

# Main

init()
third_party()
parse_args()
parse_colors()
parse_numbers()
parse_yadex()
find_open_iwad()
draw_maps()
create_diff_images()
remove_extra_images()
show_images()
