from numpy import *

import numpy as np

from tkinter import *

from detect_occlusion import does_poly_surround_poly, is_point_in_poly

from collections import Counter

HIDE = 'HIDE'

# cam rotation

theta_x = 0
theta_y = 0.1
theta_z = 0

# cam position

view_x = -1
view_y = -1
view_z = -2

# display surface (relative coords)

e_x = 0
e_y = 0
e_z = 0.1

FOV_factor = 1500

WIDTH = 500
HEIGHT = 500

tk = Tk()
canvas = Canvas(tk, width=WIDTH, height=HEIGHT)
canvas.pack()


class UpdateHandler:
    def __init__(self):
        raise TypeError('dont')

    render_stack = []

    @classmethod
    def add_to_render_stack(cls, polygon):
        cls.render_stack.append(polygon)

    @classmethod
    def remove_from_render_stack(cls, polygon):
        cls.render_stack.remove(polygon)

    @classmethod
    def render(cls):
        polygons = []
        projected = []
        for polygon in cls.render_stack:
            is_behind = True
            for vertex in polygon.vertex_locations:
                if get_relative_coords(vertex)[2] > 0:
                    is_behind = False
                    break
            if is_behind:
                cls.remove_from_render_stack(polygon)
        cls.organize()
        for polygon in cls.render_stack:
            polygons.append(polygon.vertex_locations)
            val = []
            for vertex in polygon.vertex_locations:
                val.append(project(vertex))
            projected.append(val)
        for polygon in cls.render_stack:
            polygon.render()

    @classmethod
    def organize(cls):
        cls.render_stack = binary_space_partition(cls.render_stack)
        cls.render_stack.reverse()


#    @classmethod
#    def is_poly_in_front(cls, polygon):


UpdateHandler.render_stack = []


def project_old(point):
    d = array([[1, 0, 0], [0, cos(theta_x), sin(theta_x)], [0, sin(theta_x), cos(theta_x)]]) @ array(
        [[cos(theta_y), 0, -sin(theta_y)], [0, 1, 0], [sin(theta_y), 0, cos(theta_y)]]) @ array(
        [[cos(theta_z), sin(theta_z), 0], [-sin(theta_z), cos(theta_z), 0], [0, 0, 1]]) @ (
                array([point[0], point[1], point[2]]) - array([view_x, view_y, view_z]))
#    print(d)
    # if the point is behind or level with the viewer, return the closest point instead (good enough)
    if d[2] <= 0:
        return [d[0] + e_x, d[1] + e_y]
    else:
        b = [(e_z / d[2]) * d[0] + e_x, (e_z / d[2]) * d[1] + e_y]
        return b


def project(point):
#    d = array([[1, 0, 0], [0, cos(theta_x), sin(theta_x)], [0, sin(theta_x), cos(theta_x)]]) @ array(
#        [[cos(theta_y), 0, -sin(theta_y)], [0, 1, 0], [sin(theta_y), 0, cos(theta_y)]]) @ array(
#        [[cos(theta_z), sin(theta_z), 0], [-sin(theta_z), cos(theta_z), 0], [0, 0, 1]]) @ (
#                array([point[0], point[1], point[2]]) - array([view_x, view_y, view_z]))
    d = np.array([0, 0, 0])
    X = point[0] - view_x
    Y = point[1] - view_y
    Z = point[2] - view_z
    cx = cos(theta_x)
    sx = sin(theta_x)
    cy = cos(theta_y)
    sy = sin(theta_y)
    cz = cos(theta_z)
    sz = sin(theta_z)
    d[0] = cy*(sz*Y + cz*X) - sy*Z
    d[1] = sx*(cy*Z + sy*(sz*Y + cz*X)) + cx*(cz*Y - sz*X)
    d[2] = cx*(cy*Z + sy*(sz*Y + cz*X)) - sx*(cz*Y - sz*X)
    print(d)
    print(get_relative_coords(point))
    if not (d == get_relative_coords(point)).all():
        print('error')
    # if the point is behind or level with the viewer, return the closest point instead (good enough)
    if d[2] <= 0:
        print('behind')
        return [d[0] + e_x, d[1] + e_y]
    else:
        b = [(e_z / d[2]) * d[0] + e_x, (e_z / d[2]) * d[1] + e_y]
        return b


def get_relative_coords(point):
    d = array([[1, 0, 0], [0, cos(theta_x), sin(theta_x)], [0, sin(theta_x), cos(theta_x)]]) @ array(
        [[cos(theta_x), 0, -sin(theta_x)], [0, 1, 0], [sin(theta_x), 0, cos(theta_x)]]) @ array(
        [[cos(theta_x), sin(theta_x), 0], [-sin(theta_x), cos(theta_x), 0], [0, 0, 1]]) @ (
                array([point[0], point[1], point[2]]) - array([view_x, view_y, view_z]))
    return d


def get_absolute_coords(point):
    pass


def get_frontmost_poly(poly1, poly2):
    # returns a tuple of both polygons, the leading one being the polygon object which should be rendered in front
    # if it doesn't matter, returns None; returns HIDE if the polygon is invisible to the viewer (i.e. it is edge-on)
    # this could be improved by not searching through all the points so many times :(
    poly1points = [get_relative_coords(p) for p in poly1.vertex_locations]
    poly2points = [get_relative_coords(p) for p in poly2.vertex_locations]
    projected1 = [project(p) for p in poly1.vertex_locations]
    projected2 = [project(p) for p in poly2.vertex_locations]
    try:
        if does_poly_surround_poly(projected1, projected2):
            if get_line_intersection(poly1points, poly2points[0])[2] < poly2points[0][2]:
                return (poly1, HIDE)
            else:
                return (poly2, poly1)
        elif does_poly_surround_poly(projected2, projected1):
            if get_line_intersection(poly2points, poly1points[0])[2] < poly1points[0][2]:
                return (poly2, HIDE)
            else:
                return (poly1, poly2)
        else:
            for i in range(len(projected1)):
                if is_point_in_poly(projected1[i], projected2):
                    if get_line_intersection(poly2points, poly1points[i])[2] < poly1points[i][2]:
                        return (poly2, poly1)
                    else:
                        return (poly1, poly2)
            return None
    except TypeError:
        return poly1, HIDE


class Polygon:
    def __init__(self, *args):
        self.vertices = 0
        self.vertex_locations = []
        for a in args:
            self.vertices += 1
            self.vertex_locations.append(a)

    def render(self):
        projected = []
        for v in self.vertex_locations:
            val = project(v)
            projected.append(val[0] * FOV_factor + 250)
            projected.append(val[1] * FOV_factor + 250)
        canvas.create_polygon(projected, outline='black', fill='gray', width=2)

    def get_center(self):
        x = 0
        y = 0
        z = 0
        for v in self.vertex_locations:
            x += v[0]
            y += v[1]
            z += v[2]
        return [x / self.vertices, y / self.vertices, z / self.vertices]

    def get_normal(self):
        first_vertex = np.array(self.vertex_locations[0])
        v1 = np.array(self.vertex_locations[1]) - first_vertex
        v2 = np.array(self.vertex_locations[2]) - first_vertex
        normal = np.cross(v1, v2)
        return normal



def flatten(l):
    return [item for sublist in l for item in sublist]


class Node:
    # simple tree structure for storing polygons for binary_space_partition
    def __init__(self, polys, parent=None, depth=0):
        self.polys = polys
        self.depth = depth
        self.parent = parent
        if self.parent:
            if self.parent.children:
                self.left = self.parent.children[-1]
                self.parent.children[-1].right = self
        self.children = []

    def add_child(self, child_polys):
        self.children.append(Node(child_polys, parent=self, depth=self.depth + 1))
        self.children[-1].parent = self

    def return_ordered_polys(self):
        if self.children:
            return flatten([self.children[i].return_ordered_polys() for i in range(len(self.children))])
        else:
            assert len(self.polys) == 1
            return self.polys

    def __len__(self):
        return len(self.polys)


def binary_space_partition(polys, parent=None):
    # returns a list of polygons in the order they should be rendered by recursively splitting the space along the plane formed by one of the polygons
    # polys is a list of Polygon objects
    root = None
    # if this is the first call to binary_space_partition, create a root node
    if not parent:
        parent = Node(polys)
        root = parent
    slicer = None
    slicer_normal_direction = None
    CULLED = []
    if len(polys) > 1:
        while not slicer:
            # pick a polygon to slice the space with, get the direction of its normal vector relative to the viewer, if the polygon is edge-on, pick another one
            # change this to an algorithm to pick best slicer
            slicer = get_random_slicer(polys)
            slicer_normal_direction = np.dot(slicer.get_normal(), -np.array(get_relative_coords(slicer.vertex_locations[0])))
            if not slicer_normal_direction:
#                print('culling ' + str(slicer))
                CULLED.append(slicer)
                if slicer in UpdateHandler.render_stack:
                    UpdateHandler.remove_from_render_stack(slicer)
                if slicer in polys:
                    polys.remove(slicer)
                if slicer in parent.polys:
                    parent.polys.remove(slicer)
                slicer = None
                if not polys:
                    return None
                continue
            slicer_normal_direction /= abs(slicer_normal_direction)
        # if there are polygons to slice, determine if they're in front of or behind the slicer, and if they intersect the slicer split them in two
        back = []
        front = []
#        print()
#        print("polys:" + str(polys))
        for p in polys:
#            if p in CULLED or slicer in CULLED:
#                print("uh-oh spaghetti-o")
            if p != slicer:
                ret = get_edge_intersections(p, slicer)
                if not ret:
                    # if the polygon doesn't intersect the slicer, determine if it's in front or behind the slicer (loop through the points to make sure the selected point isn't level with the slicer: if all are, then just render it in front (it doesn't matter)
                    for i in range(len(p.vertex_locations)):
                        val = np.dot(slicer.get_normal(), np.array(p.vertex_locations[i]) - np.array(slicer.vertex_locations[0]))
                        if val == 0:
                            if i == len(p.vertex_locations) - 1:
                                front.append(p)
                                break
                            else:
                                continue
                        elif val/abs(val) == slicer_normal_direction:
                            front.append(p)
                            break
                        else:
                            back.append(p)
                            break
                else:
                    # if the polygon intersects the slicer, split it in two, then add the two new polygons to the front or back list
                    new_poly1 = Polygon(*ret[0])
                    new_poly2 = Polygon(*ret[1])
                    try:
                        if np.dot(slicer.get_normal(), np.array(new_poly1.vertex_locations[0]) - np.array(slicer.vertex_locations[0]))/abs(np.dot(slicer.get_normal(), np.array(new_poly1.vertex_locations[0]) - np.array(slicer.vertex_locations[0]))) == slicer_normal_direction:
                            front.append(new_poly1)
                            back.append(new_poly2)
                        else:
                            front.append(new_poly2)
                            back.append(new_poly1)
                    except ZeroDivisionError:
                        pass
            # recursively call binary_space_partition on the front and back lists (if they exist); this should eventually divide the entire list of polygons down to one polygon per node
        if front:
            parent.add_child(front)
            val = binary_space_partition(front, parent.children[-1])
        parent.add_child([slicer])
        if back:
            parent.add_child(back)
            val = binary_space_partition(back, parent.children[-1])
#        print(front, back, slicer)
        #WHY IS THIS SHOWING CULLED POLYGONS IN HERE
#        print(polys)
#        print()
#        assert Counter(front) + Counter(back) + Counter([slicer]) == Counter(polys)
    # if there's only one polygon on a node, don't do anything
    else:
        pass
    if root == parent:
        # if its the outermost layer, it returns the polygons in the order they should be rendered (from front to back)
        return parent.return_ordered_polys()
    else:
        return None


def get_edge_intersections(poly, slicer):
    # finds the 3d points where the edges of poly intersect the plane formed by slicer
    # returns a list of the 2 points
    # returns [] if no intersections, or if the "intersections" are right on the surface of the polygon
    polyedges = [(poly.vertex_locations[i], poly.vertex_locations[i + 1]) for i in range(len(poly.vertex_locations) - 1)]
    first_vertex = np.array(poly.vertex_locations[0])
    v1 = np.array(poly.vertex_locations[1]) - first_vertex
    v2 = np.array(poly.vertex_locations[2]) - first_vertex
    normal = np.cross(v1, v2)
    points = []
    for i in range(len(polyedges)):
        polyedges[i] = (np.array(polyedges[i][0]), np.array(polyedges[i][1]))
        p1dot = np.dot(normal, np.array(polyedges[i][0]) - first_vertex)
        p2dot = np.dot(normal, np.array(polyedges[i][1]) - first_vertex)
        if p1dot * p2dot != 0:
            p1dot /= abs(p1dot)
            p2dot /= abs(p2dot)
            if p1dot != p2dot:
                try:
                    points.append(get_line_intersection(normal, first_vertex, np.array(polyedges[i][1] - polyedges[i][0]), np.array(polyedges[i][0])))
                except TypeError:
                    pass
    assert len(points) == 2 or len(points) == 0
    return points




def get_line_intersection(normal, first_vertex, line_direction, line_point=np.array([0, 0, 0])):
    ndotu = normal.dot(line_direction)
    if abs(ndotu) < 0.0001:
        raise TypeError
    w = line_point-first_vertex
    si = -normal.dot(w) / ndotu
    return w + si * np.array(line_direction) + first_vertex


def split_poly_from_points(poly, point1, point2):
    # point1 and point2 are the points on the edge of the polygon to be split
    # poly is the polygon to be split
    # returns two polygons
    intersect1 = False
    intersect2 = False
    polyedges = [(poly.vertex_locations[i], poly.vertex_locations[i + 1]) for i in range(len(poly.vertex_locations) - 1)]
    polyedges.append((poly.vertex_locations[-1], poly.vertex_locations[0]))
    for i in range(len(polyedges)):
        if is_point_in_segment(np.array(point1), np.array(polyedges[i][0]), np.array(polyedges[i][1])):
            intersect1 = i
        elif is_point_in_segment(np.array(point2), np.array(polyedges[i][0]), np.array(polyedges[i][1])):
            intersect2 = i
    new_poly1 = [point1]
    new_poly1 += [poly.vertex_locations[i % len(poly.vertex_locations)] for i in range(intersect1+1, intersect2)]
    new_poly1 += [point2]
    new_poly2 = [point2]
    new_poly2 += [poly.vertex_locations[i % len(poly.vertex_locations)] for i in range(intersect2+1, intersect1)]
    new_poly2 += [point1]
    return (Polygon(*new_poly1), Polygon(*new_poly2))


def is_point_in_segment(point, seg1, seg2):
    # point is a 3d point (as a numpy array)
    # seg1 and seg2 are the endpoints of the segment
    # returns True if the point is in the segment, False otherwise
    if np.linalg.norm(point - seg1) + np.linalg.norm(point - seg2) == np.linalg.norm(seg1 - seg2):
        return True
    return False


def get_random_slicer(polys):
    return polys[np.random.randint(0, len(polys))]
