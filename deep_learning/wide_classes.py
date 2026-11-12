"""The wide class list: CURE-TSR signs, LISA lights, GTSRB and the Indian set, with the same sign merged.

The first 17 classes keep their old positions, so CURE-TSR and LISA labels need no change.
Speed limits are one class, because CURE-TSR doesn't record the number.
Signs are merged only when they look alike and mean the same thing.
Indian classes with fewer than 10 usable images, or none left for training, are left out (INDIAN_SKIPPED).
"""
from deep_learning.class_names import CLASS_NAMES

NEW_NAMES = (
    # From GTSRB, and Indian signs that match them
    "end of speed limit 80", "no overtaking for trucks", "right of way at next junction", "priority road",
    "no vehicles", "general caution", "left curve", "right curve", "left reverse bend", "slippery road",
    "road narrows on right", "road work", "traffic signals ahead", "pedestrian crossing", "children crossing",
    "cycle crossing", "ice or snow", "wild animals", "restriction ends", "turn right ahead", "turn left ahead",
    "ahead only", "ahead or right", "ahead or left", "keep right", "keep left", "roundabout",
    "end of no overtaking", "end of no overtaking for trucks",
    # Indian signs only
    "no motor vehicles", "axle load limit", "no bullock and handcart", "no bullock carts",
    "cattle", "minimum speed", "sound horn", "no cycles", "dangerous dip",
    "falling rocks", "gap in median", "guarded level crossing", "height limit",
    "no horn", "left hairpin bend", "length limit", "load limit", "loose gravel", "narrow bridge",
    "narrow road ahead", "pass either side", "no pedestrians", "quay or river bank", "right hairpin bend",
    "right reverse bend", "road widens ahead", "roundabout ahead", "school ahead", "side road left",
    "side road right", "staggered junction", "steep ascent", "steep descent", "no straight ahead",
    "t junction", "no u-turn", "width limit", "y junction",
)
# Clearer names for three of the original signs, now that other datasets share them
RENAMED = {"goods vehicles": "no goods vehicles", "priority": "give way to oncoming", "hump": "hump or rough road"}
WIDE_NAMES = tuple(RENAMED.get(name, name) for name in CLASS_NAMES) + NEW_NAMES
WIDE_CLASSES = len(WIDE_NAMES)


def wide(name):
    return WIDE_NAMES.index(name)


INDIAN_SKIPPED = (
    "BARRIER_AHEAD", "COMPULSARY_TURN_RIGHT", "CROSS_ROAD", "DIRECTION", "FERRY", "HANDCART_PROHIBITED", "TONGA_PROHIBITED",
    "UNGUARDED_LEVEL_CROSSING",
)


GTSRB_TO_WIDE = {
    0: wide("speed limit"), 1: wide("speed limit"), 2: wide("speed limit"), 3: wide("speed limit"),
    4: wide("speed limit"), 5: wide("speed limit"), 6: wide("end of speed limit 80"), 7: wide("speed limit"),
    8: wide("speed limit"), 9: wide("no overtaking"), 10: wide("no overtaking for trucks"),
    11: wide("right of way at next junction"), 12: wide("priority road"), 13: wide("yield"), 14: wide("stop"),
    15: wide("no vehicles"), 16: wide("no goods vehicles"), 17: wide("no entry"), 18: wide("general caution"),
    19: wide("left curve"), 20: wide("right curve"), 21: wide("left reverse bend"), 22: wide("hump or rough road"),
    23: wide("slippery road"), 24: wide("road narrows on right"), 25: wide("road work"),
    26: wide("traffic signals ahead"), 27: wide("pedestrian crossing"), 28: wide("children crossing"),
    29: wide("cycle crossing"), 30: wide("ice or snow"), 31: wide("wild animals"), 32: wide("restriction ends"),
    33: wide("turn right ahead"), 34: wide("turn left ahead"), 35: wide("ahead only"), 36: wide("ahead or right"),
    37: wide("ahead or left"), 38: wide("keep right"), 39: wide("keep left"), 40: wide("roundabout"),
    41: wide("end of no overtaking"), 42: wide("end of no overtaking for trucks"),
}

INDIAN_TO_WIDE = {
    "ALL_MOTOR_VEHICLE_PROHIBITED": wide("no motor vehicles"),
    "AXLE_LOAD_LIMIT": wide("axle load limit"),
    "BULLOCK_AND_HANDCART_PROHIBITED": wide("no bullock and handcart"),
    "BULLOCK_PROHIBITED": wide("no bullock carts"),
    "CATTLE": wide("cattle"),
    "COMPULSARY_AHEAD": wide("ahead only"),
    "COMPULSARY_AHEAD_OR_TURN_LEFT": wide("ahead or left"),
    "COMPULSARY_AHEAD_OR_TURN_RIGHT": wide("ahead or right"),
    "COMPULSARY_CYCLE_TRACK": wide("bicycle"),
    "COMPULSARY_KEEP_LEFT": wide("keep left"),
    "COMPULSARY_KEEP_RIGHT": wide("keep right"),
    "COMPULSARY_MINIMUM_SPEED": wide("minimum speed"),
    "COMPULSARY_SOUND_HORN": wide("sound horn"),
    "COMPULSARY_TURN_LEFT": wide("turn left ahead"),
    "COMPULSARY_TURN_LEFT_AHEAD": wide("turn left ahead"),
    "COMPULSARY_TURN_RIGHT_AHEAD": wide("turn right ahead"),
    "CYCLE_CROSSING": wide("cycle crossing"),
    "CYCLE_PROHIBITED": wide("no cycles"),
    "DANGEROUS_DIP": wide("dangerous dip"),
    "FALLING_ROCKS": wide("falling rocks"),
    "GAP_IN_MEDIAN": wide("gap in median"),
    "GIVE_WAY": wide("yield"),
    "GUARDED_LEVEL_CROSSING": wide("guarded level crossing"),
    "HEIGHT_LIMIT": wide("height limit"),
    "HORN_PROHIBITED": wide("no horn"),
    "HUMP_OR_ROUGH_ROAD": wide("hump or rough road"),
    "LEFT_HAIR_PIN_BEND": wide("left hairpin bend"),
    "LEFT_HAND_CURVE": wide("left curve"),
    "LEFT_REVERSE_BEND": wide("left reverse bend"),
    "LEFT_TURN_PROHIBITED": wide("no left"),
    "LENGTH_LIMIT": wide("length limit"),
    "LOAD_LIMIT": wide("load limit"),
    "LOOSE_GRAVEL": wide("loose gravel"),
    "MEN_AT_WORK": wide("road work"),
    "NARROW_BRIDGE": wide("narrow bridge"),
    "NARROW_ROAD_AHEAD": wide("narrow road ahead"),
    "NO_ENTRY": wide("no entry"),
    "NO_PARKING": wide("no parking"),
    "NO_STOPPING_OR_STANDING": wide("no stopping"),
    "OVERTAKING_PROHIBITED": wide("no overtaking"),
    "PASS_EITHER_SIDE": wide("pass either side"),
    "PEDESTRIAN_CROSSING": wide("pedestrian crossing"),
    "PEDESTRIAN_PROHIBITED": wide("no pedestrians"),
    "PRIORITY_FOR_ONCOMING_VEHICLES": wide("give way to oncoming"),
    "QUAY_SIDE_OR_RIVER_BANK": wide("quay or river bank"),
    "RESTRICTION_ENDS": wide("restriction ends"),
    "RIGHT_HAIR_PIN_BEND": wide("right hairpin bend"),
    "RIGHT_HAND_CURVE": wide("right curve"),
    "RIGHT_REVERSE_BEND": wide("right reverse bend"),
    "RIGHT_TURN_PROHIBITED": wide("no right"),
    "ROAD_WIDENS_AHEAD": wide("road widens ahead"),
    "ROUNDABOUT": wide("roundabout ahead"),
    "SCHOOL_AHEAD": wide("school ahead"),
    "SIDE_ROAD_LEFT": wide("side road left"),
    "SIDE_ROAD_RIGHT": wide("side road right"),
    "SLIPPERY_ROAD": wide("slippery road"),
    "SPEED_LIMIT_5": wide("speed limit"),
    "SPEED_LIMIT_15": wide("speed limit"),
    "SPEED_LIMIT_20": wide("speed limit"),
    "SPEED_LIMIT_30": wide("speed limit"),
    "SPEED_LIMIT_40": wide("speed limit"),
    "SPEED_LIMIT_50": wide("speed limit"),
    "SPEED_LIMIT_60": wide("speed limit"),
    "SPEED_LIMIT_70": wide("speed limit"),
    "SPEED_LIMIT_80": wide("speed limit"),
    "STAGGERED_INTERSECTION": wide("staggered junction"),
    "STEEP_ASCENT": wide("steep ascent"),
    "STEEP_DESCENT": wide("steep descent"),
    "STOP": wide("stop"),
    "STRAIGHT_PROHIBITED": wide("no straight ahead"),
    "TRAFFIC_SIGNAL": wide("traffic signals ahead"),
    "TRUCK_PROHIBITED": wide("no goods vehicles"),
    "TURN_RIGHT": wide("turn right ahead"),
    "T_INTERSECTION": wide("t junction"),
    "U_TURN_PROHIBITED": wide("no u-turn"),
    "WIDTH_LIMIT": wide("width limit"),
    "Y_INTERSECTION": wide("y junction"),
}
