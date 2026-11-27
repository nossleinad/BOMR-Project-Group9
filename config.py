"""Constants shared by all modules (vision, global navigation, and later filtering / control).

Conventions (agreed between vision and global navigation):
    - units: millimetres and radians
    - origin at the bottom-left corner of the map, x to the right, y up
    - heading: 0 = along +x, counter-clockwise positive
    - robot position = centre of the robot BODY = centre of the robot ArUco marker (ID 4),
      stuck at the centre of the body with its printed TOP edge towards the robot's front
    - obstacles = list of obstacles, each a list of (x, y) corners in order around the shape
"""

# Map: rectangle whose corners are the CENTRES of the 4 corner ArUco markers (IDs 0-3, the
# homography maps the marker centres to the map corners), NOT the size of the sheet.
MAP_W_MM = 976
MAP_H_MM = 413
MAP_SIZE = (MAP_W_MM, MAP_H_MM)

# Thymio geometry:
# Thymio body is ~110 x 112 mm. Radius measured from the body centre:
#   56 mm = half the body length/width -> safe when driving straight past an obstacle,
#   but the rear corners (~79 mm from the centre) can clip an obstacle when turning near one.
ROBOT_RADIUS_MM = 56
AXLE_TO_CENTRE_MM = 24.5       # TODO: measure (wheel axle -> body centre / marker)

# Vision / localisation / control errors. 20 mm (not 10) because vision simplifies each obstacle
# outline with epsilon = 2% of its perimeter: the polygon can cut up to that much INTO the
# obstacle (~9 mm for a 150 mm wide round obstacle).
MARGIN_MM = 20
PADDING_MM = ROBOT_RADIUS_MM + MARGIN_MM    # obstacles are grown ONCE, by global navigation
WALL_PADDING_MM = 0                         # map borders are not grown

# Control loop
TS = 0.1                       # s, control period
MAX_UNITS = 400                # motor saturation

# Mission
GOAL_RADIUS_MM = 30.0          # goal reached
KIDNAP_DIST_MM = 150.0         # camera vs filter distance -> kidnapping
