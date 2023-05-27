import math
import re
import subprocess
from collections import UserList
from math import ceil, cos, floor, pi, sin, sqrt
from random import randint, random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import cairo
import click
import numpy as np
from foronoi import Polygon, Voronoi

Point = tuple[float, float]
Points = list[Point]
RGBColor = tuple[float, float, float]

## Making voronoi diagrams


class Grid(UserList):
    """
    A 2 dimensional grid represented as a list for storing valid sample points.
    Represents the background grid outlined in the Fast Poisson Disk Sampling algorithm
    """

    def __init__(self, src_width: int, src_height: int, radius: int) -> None:
        super().__init__()

        self.cellsize = radius / sqrt(2)
        self.radius = radius
        self.width = ceil(src_width / self.cellsize)
        self.height = ceil(src_height / self.cellsize)

        for _ in range(self.width * self.height):
            self.append(None)

    def coords(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Returns the grid coordinates for a given sample point"""
        return floor(point[0] / self.cellsize), floor(point[1] / self.cellsize)

    @staticmethod
    def distance(point_1: Tuple[float, float], point_2: Tuple[float, float]) -> float:
        """Returns the euclidean distance between two points"""
        dx = point_1[0] - point_2[0]
        dy = point_1[1] - point_2[1]
        return sqrt(dx * dx + dy * dy)

    def fits(self, point: Tuple[float, float]) -> bool:
        """Scan the grid to check if a given point will fit. Only test nearby samples"""
        grid_x, grid_y = self.coords(point)
        yrange = range(max(grid_y - 2, 0), min(grid_y + 3, self.height))
        for point_x in range(max(grid_x - 2, 0), min(grid_x + 3, self.width)):
            for point_y in yrange:
                self.grid_point = self[point_x + point_y * self.width]
                if self.grid_point is None:
                    continue
                if self.distance(point, self.grid_point) <= self.radius:
                    return False
        return True


class PoissonDisk:
    """
    Implements a fast poisson disk sampling pattern in 2 dimensions
    as referenced here https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    """

    def __init__(
        self, width: int, height: int, radius: int, k_points: int = 300
    ) -> None:
        self.width = width
        self.height = height
        self.radius = radius
        self.k_points = k_points

        self.grid = Grid(width, height, radius)
        point = width * random(), height * random()
        self.queue = [point]
        grid_x, grid_y = self.grid.coords(point)
        self.grid[grid_x + grid_y * self.grid.width] = point

    @staticmethod
    def random_disk(point: Tuple[float, float], radius: int) -> Tuple[float, float]:
        """Get a random point radially around the given point between the radius and 2 * radius"""
        alpha = 2 * pi * random()
        distance = radius * sqrt(3 * random() + 1)
        return (point[0] + distance * cos(alpha), point[1] + distance * sin(alpha))

    def samples(self) -> List[Tuple[int, int]]:
        while self.queue:
            q_index = randint(0, len(self.queue) - 1)
            q_point = self.queue[q_index]
            self.queue[q_index] = self.queue[-1]
            self.queue.pop()

            for _ in range(self.k_points):
                point = self.random_disk(q_point, self.radius)
                if not (
                    0 <= point[0] < self.width and 0 <= point[1] < self.height
                ) or not self.grid.fits(point):
                    continue

                self.queue.append(point)
                grid_x, grid_y = self.grid.coords(point)
                self.grid[grid_x + grid_y * self.grid.width] = point

        return [point for point in self.grid if point is not None]


def cutting_corners(vertices: Points, n: int = 5) -> Points:
    arr = np.array(vertices)
    for _ in range(n):
        narr = np.roll(arr, -2)
        qarr = ((3 / 4) * arr) + ((1 / 4) * narr)
        rarr = ((1 / 4) * arr) + ((3 / 4) * narr)
        arr = np.ravel(np.column_stack((qarr, rarr))).reshape(
            (len(qarr) + len(rarr), 2)
        )
    return list(map(tuple, arr))


# def make_smily_face(center, ctx):
#     ctx.set_line_width(0.01)
#     ctx.set_source_rgb(0, 0, 0)
#     cx, cy = center[0] / WIDTH, center[1] / HEIGHT
#     ctx.move_to(cx + (3 / WIDTH), cy - (2 / HEIGHT))
#     ctx.line_to(cx + (3 / WIDTH), cy - (5 / HEIGHT))
#     ctx.stroke()
#     ctx.move_to(cx - (3 / WIDTH), cy - (2 / HEIGHT))
#     ctx.line_to(cx - (3 / WIDTH), cy - (5 / HEIGHT))
#     ctx.stroke()
#     ctx.arc(cx, cy, 5 / WIDTH, 0, np.pi)
#     ctx.stroke()


def leftmost_rightmost(points: Points) -> Tuple[Point, Point]:
    left = points[0]
    right = points[0]
    for pnt in points:
        x, _ = pnt
        if x < left[0]:
            left = pnt
        elif x > right[0]:
            right = pnt
    return left, right


def rand_params(
    points: np.ndarray,
    max_amp: int = 3,
    min_amp: int = 0,
    max_off: int = 30,
    min_off: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    amps = (max_amp - min_amp) * np.random.random_sample(size=points.shape) + min_amp
    offsets = (max_off - min_off) * np.random.random_sample(size=points.shape) + min_off
    return amps, offsets


# To create a more fluid wave-like motion
# the shape displacement should be the same and
# a phase offset should be applied based on the shapes
# horizontal location, this splits movement roughly
# into 4 waves
def wave_params(
    points: np.ndarray, width: int, amp: int = 3, angle: float = np.pi / 4
) -> Tuple[np.ndarray, np.ndarray]:
    x_amp = amp * np.cos(angle)
    y_amp = amp * np.sin(angle)

    offsets = []
    for pnt in points:
        x, _ = pnt
        offsets.append([x * (width / 4), x * (width / 4)])
    offsets = np.array(offsets)
    amps = np.full(points.shape, (x_amp, y_amp))
    return amps, offsets


class Pattern:
    def __init__(self, out_file, width, height) -> None:
        self.out_file = out_file
        self.width = width
        self.height = height

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

        # dist = PoissonDisk(width, height, 30, 300)
        # self.dist = np.array(dist.samples())

        n_samples = (width * height) // (30 * 30) 
        x_dist = np.random.uniform(0, width, size=(n_samples, 1))
        y_dist = np.random.uniform(0, height, size=(n_samples, 1))
        self.dist = np.column_stack((x_dist, y_dist))

    def draw(
        self,
        centers: np.ndarray,
        outline_color: RGBColor,
        fill_color: RGBColor,
        gradient: Optional[RGBColor] = None,
        rounded: bool = False,
    ):
        ctx = cairo.Context(self.surface)

        # Fill entire image with outline color to cover any gaps
        ctx.set_source_rgb(*outline_color)
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.stroke_preserve()
        ctx.fill()
        ctx.set_line_width(5)

        area = Polygon(
            [(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)]
        )
        polygons = Voronoi(area)
        polygons.create_diagram(points=centers)

        for site, center in zip(polygons.sites, centers):
            vertices = list((vertex.x, vertex.y) for vertex in site.vertices())
            if rounded:
                vertices = cutting_corners(vertices)
            if gradient is not None:
                left, right = leftmost_rightmost(vertices)

            # Draw shape outline and preserve path for fill
            x, y = vertices[0]
            ctx.set_source_rgb(*outline_color)
            ctx.move_to(x, y)
            for point in vertices[1:]:
                x, y = point
                ctx.line_to(x, y)
            ctx.stroke_preserve()

            if gradient is None:
                ctx.set_source_rgb(*fill_color)
            else:
                grad = cairo.LinearGradient(left[0], left[1], right[0], right[1])
                grad.add_color_stop_rgba(
                    0, fill_color[0], fill_color[1], fill_color[2], 1
                )
                grad.add_color_stop_rgba(1, gradient[0], gradient[1], gradient[2], 1)
                ctx.set_source(grad)
            ctx.fill()

            # make_smily_face(center, ctx)

    def make_png(
        self,
        outline_color: RGBColor,
        fill_color: RGBColor,
        gradient: Optional[RGBColor] = None,
        rounded: bool = False,
    ):
        self.draw(
            self.dist, outline_color, fill_color, gradient=gradient, rounded=rounded
        )
        self.surface.write_to_png(self.out_file)

    def make_gif(
        self,
        outline_color: RGBColor,
        fill_color: RGBColor,
        gradient: Optional[RGBColor] = None,
        rounded: bool = True,
        motion_mode: str = "waves",
    ):
        n_frames = 30

        if motion_mode == "waves":
            amps, offsets = wave_params(self.dist, self.width)
        elif motion_mode == "random":
            amps, offsets = rand_params(self.dist)

        for i in range(n_frames):
            new_dist = self.dist + (
                amps * np.sin(2 * np.pi * (1 / n_frames) * (i + offsets)) + (amps / 2)
            )
            self.draw(
                new_dist, outline_color, fill_color, gradient=gradient, rounded=rounded
            )
            self.surface.write_to_png(f"frames/frame{i}.png")

        # ffmpeg applies a dithering effect when compressing images into a
        # gif to account for motion, we can remove that by creating a color palette
        # to use, interesting read on the topic
        # http://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html

        # Also I Didn't want to install the ffmpeg python bindings as it didn't
        # look like it supports all pallette options so here I bare my shame
        # for all to see

        cmd = "ffmpeg -y -i frames/frame0.png -vf palettegen frames/palette.png"
        subprocess.check_output(cmd.split())
        cmd = f"ffmpeg -y -i frames/frame%d.png -i frames/palette.png -r 30 -lavfi paletteuse {self.out_file}"
        subprocess.check_output(cmd.split())


def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    if hex_str.startswith("#"):
        hex_str = hex_str[1:]
    return (
        int(hex_str[0:2], 16) / 256,
        int(hex_str[2:4], 16) / 256,
        int(hex_str[4:6], 16) / 256,
    )


def hex_color(ctx, param, value):
    pattern = "^#?([a-f0-9A-F]{6})$"
    if re.match(pattern, value):
        return hex_to_rgb(value)
    else:
        raise click.BadParameter("")


def size_param(ctx, param, value):
    pattern = "[0-9]+(x[0-9]+)"
    if re.match(pattern, value):
        width, height = map(int, value.split("x"))
        return width, height
    else:
        raise click.BadParameter("")


def gif_or_png(ctx, param, value):
    if value.endswith(".gif"):
        return value, True
    elif value.endswith(".png"):
        return value, False
    else:
        raise click.BadParameter("")


@click.command()
@click.option(
    "--output",
    required=True,
    help="File path for the created image, should be a png or gif",
    callback=gif_or_png,
)
# @click.option(
#     "--motion_mode",
#     default="wave",
#     help="How to handle a shapes movement displacement. Only impacts gifs. Either `random` `uniform` or `waves",
# )
@click.option(
    "--size", default="256x256", help="Image size as WIDTHxHEIGHT", callback=size_param
)
@click.option(
    "--primary_color",
    default="#008888",
    help="Color to fill shapes with as a hex string",
    callback=hex_color,
)
@click.option(
    "--outline_color",
    default="#b8f8b8",
    help="Color to outline shapes with as a hex string",
    callback=hex_color,
)
@click.option(
    "--gradient_color",
    default="#004058",
    help="Secondary color for creating a gradient from the primary color",
    callback=hex_color,
)
def render(output, size, primary_color, outline_color, gradient_color):
    output, is_gif = output
    width, height = size

    pattern = Pattern(output, width, height)
    if is_gif:
        pattern.make_gif(outline_color, primary_color, gradient_color)
    else:
        pattern.make_png(outline_color, primary_color, gradient_color)


if __name__ == "__main__":
    render()
