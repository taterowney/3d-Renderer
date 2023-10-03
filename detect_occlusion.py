import math
import numpy as np

from sympy import Line3D, Plane, Point3D

HIDE = 1



def get_line_intersection(polypoints, line_direction):
    first_vertex = np.array(polypoints[0])
    v1 = np.array(polypoints[1]) - first_vertex
    v2 = np.array(polypoints[2]) - first_vertex
    normal = np.cross(v1, v2)
    ndotu = normal.dot(line_direction)
    if abs(ndotu) < 0.0001:
        raise TypeError
    w = -first_vertex
    si = -normal.dot(w) / ndotu
    return w + si*np.array(line_direction) + first_vertex





# def find_overlap(poly1, poly2):
#     # function to return whether one polygon should be rendered in front of another
#     # returns None if the polygons do not overlap, otherwise returns the Poly object that should be rendered in front
#     # project the polygons onto the 2d viewplane (get their points and edges)
#     poly1points = [project(p) for p in poly1.vertex_locations]
#     poly2points = [project(p) for p in poly2.vertex_locations]
#     poly1edges = [(poly1points[i], poly1points[i + 1]) for i in range(len(poly1points) - 1)]
#     poly2edges = [(poly2points[i], poly2points[i + 1]) for i in range(len(poly2points) - 1)]
#     poly1edges.append((poly1points[-1], poly1points[0]))
#     poly2edges.append((poly2points[-1], poly2points[0]))
#     # check if the polygons are obscuring each other (From the angle of the viewer)
#     does_overlap = False
#     for point in poly1points:
#         if is_point_in_poly(point, poly2points):
#             does_overlap = True
#             break
#     if not does_overlap:
#         return None
#     else:
#         new_points = []
#         # check if one polygon is completely inside the other
#         if does_poly_surround_poly(poly1points, poly2points):
#             for point in poly2:
#                 if
#         elif does_poly_surround_poly(poly2points, poly1points):
#         else:
#             for i in range(len(poly1points)):
#                 if is_point_in_poly(poly1points[i], poly2points):
#                     if is_point_in_poly(poly1points[(i + 1) % len(poly1points)], poly2points):
#                         pass
#                     new_points.append(poly1points[i])


def does_poly_surround_poly(poly1points, poly2points):
    for point in poly2points:
        if not is_point_in_poly(point, poly1points):
            return False
    return True

def is_point_in_poly(point, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def intersect(edge1, edge2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(edge1[0], edge2[0], edge2[1]) != ccw(edge1[1], edge2[0], edge2[1]) and ccw(edge1[0], edge1[1], edge2[0]) != ccw(edge1[0], edge1[1], edge2[1])

def magnitude(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)