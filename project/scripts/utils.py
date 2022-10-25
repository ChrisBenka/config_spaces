from shapely.geometry import Point, LineString, MultiPoint

'''
Shapely has precision issues when checking for overlaps, 
we round to the 4th decimal place to avoid these issues
'''


def round_point(point):
    return Point(round(point.x, 4), round(point.y, 4))


def round_polygon(ls):
    linestring = []
    for point in ls.exterior.coords:
        linestring.append(Point(round(point[0], 4), round(point[1], 4)))
    return MultiPoint(linestring).convex_hull


def round_line_string(ls):
    linestring = []
    for point in ls.boundary.geoms:
        linestring.append(Point(round(point.x, 4), round(point.y, 4)))
    return LineString(linestring)


def midpoint(p1, p2):
    p1 = Point(p1)
    p2 = Point(p2)
    midpt = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
    return tuple([midpt.x, midpt.y])
