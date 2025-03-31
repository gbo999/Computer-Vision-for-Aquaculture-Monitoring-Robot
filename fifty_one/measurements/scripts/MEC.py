import os
import cv2
import numpy as np
from math import sqrt
from random import randint, shuffle
import fiftyone as fo
import fiftyone.core.labels as fol

# Helper Classes and Functions for Circle Calculation
class Point:
    def __init__(self, X=0, Y=0) -> None:
        self.X = X
        self.Y = Y

class Circle:
    def __init__(self, c=Point(), r=0) -> None:    
        self.C = c
        self.R = r

def dist(a, b):
    return sqrt((a.X - b.X) ** 2 + (a.Y - b.Y) ** 2)

def is_inside(c, p):
    return dist(c.C, p) <= c.R

def get_circle_center(bx, by, cx, cy):
    """
    Calculate the center of the circle based on three points.
    Handle collinear points by returning a fallback.
    """
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx

    # Check for collinear points (D == 0 means the points are collinear)
    if D == 0:
        print("Points are collinear, cannot calculate circle center. Returning fallback.")
        # Fallback: return a midpoint between two points or use another approach.
        return Point(0, 0)  # You can modify this to return a different value based on your use case.
    
    # Otherwise, calculate the circle center
    return Point((cy * B - by * C) / (2 * D), (bx * C - cx * B) / (2 * D))

def circle_from1(A, B):
    C = Point((A.X + B.X) / 2.0, (A.Y + B.Y) / 2.0)
    return Circle(C, dist(A, B) / 2.0)

def circle_from2(A, B, C):
    I = get_circle_center(B.X - A.X, B.Y - A.Y, C.X - A.X, C.Y - A.Y)
    I.X += A.X
    I.Y += A.Y
    return Circle(I, dist(I, A))

def is_valid_circle(c, P):
    for p in P:
        if not is_inside(c, p):
            return False
    return True

def min_circle_trivial(P):
    assert len(P) <= 3
    if not P:
        return Circle()
    elif len(P) == 1:
        return Circle(P[0], 0)
    elif len(P) == 2:
        return circle_from1(P[0], P[1])
    for i in range(3):
        for j in range(i + 1, 3):
            c = circle_from1(P[i], P[j])
            if is_valid_circle(c, P):
                return c
    return circle_from2(P[0], P[1], P[2])

def welzl_helper(P, R, n):
    if n == 0 or len(R) == 3:
        return min_circle_trivial(R)
    idx = randint(0, n - 1)
    p = P[idx]
    P[idx], P[n - 1] = P[n - 1], P[idx]
    d = welzl_helper(P, R.copy(), n - 1)
    if is_inside(d, p):
        return d
    R.append(p)
    return welzl_helper(P, R.copy(), n - 1)

def welzl(P):
    P_copy = P.copy()
    shuffle(P_copy)
    return welzl_helper(P_copy, [], len(P_copy))