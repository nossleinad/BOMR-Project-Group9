"""Constants shared by all modules (vision, global navigation, and later filtering / control).

Conventions (agreed between vision and global navigation):
    - units: millimetres and radians
    - origin at the bottom-left corner of the map, x to the right, y up
    - heading: 0 = along +x, counter-clockwise positive
    - robot position = centre of the robot BODY = midpoint between the blue and green markers
      (the markers are stuck so that their midpoint is the centre of the body)
    - obstacles = list of obstacles, each a list of (x, y) corners in order around the shape
"""

# Map: rectangle whose corners are the CENTRES of the 4 orange beacons (the homography maps
# the beacon centroids to the map corners), NOT the size of the sheet. Measure it!
MAP_W_MM = 976
MAP_H_MM = 413
MAP_SIZE = (MAP_W_MM, MAP_H_MM)

# Thymio body is ~110 x 112 mm. Radius measured from the body centre:
#   56 mm = half the body length/width -> safe when driving straight past an obstacle,
#   but the rear corners (~79 mm from the centre) can clip an obstacle when turning near one.
ROBOT_RADIUS_MM = 56
MARGIN_MM = 10                              # vision / localisation / control errors
PADDING_MM = ROBOT_RADIUS_MM + MARGIN_MM    # obstacles are grown ONCE, by global navigation
WALL_PADDING_MM = 0                         # map borders are not grown

PLANNER = "mix"   # "mix" (shapely + own visibility graph + A*) or "lib" (extremitypathfinder)
