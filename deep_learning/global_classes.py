"""The global class list: the 84 wide classes, plus signs that Mapillary adds from streets on six continents.

The first 84 classes keep their positions, so CURE-TSR, LISA, GTSRB and Indian labels need no change.
Mapillary names each sign as type--meaning--variant, and every variant of one meaning becomes one class,
because the variants are the same sign drawn in different countries' styles.
A new meaning becomes a class only with at least 100 clear training crops. Rarer named signs go to
"other sign", so the model can say a sign isn't one it knows instead of forcing a wrong name.
Plates under signs and look-alikes that would blur a class boundary are left out instead.
"""
from deep_learning.wide_classes import WIDE_NAMES

NEW_NAMES = (
    "chevron left", "chevron right", "crossroads", "railway crossing", "unguarded level crossing",
    "stop ahead", "traffic merges from right", "two way traffic", "winding road", "divided road ends",
    "road narrows on left", "one way, arrow left", "one way, arrow right", "one way, arrow straight",
    "bus stop", "dead end", "hospital", "shared path", "pedestrians only", "trucks crossing", "end of motorway",
    "other sign",
)
RENAMED = {"end of speed limit 80": "end of speed limit"}
GLOBAL_NAMES = tuple(RENAMED.get(name, name) for name in WIDE_NAMES) + NEW_NAMES
GLOBAL_CLASSES = len(GLOBAL_NAMES)
OTHER_SIGN = GLOBAL_NAMES.index("other sign")

# Mapillary meaning (its label without the variant) -> class name
MAPILLARY_MEANINGS = {
    "regulatory--yield": "yield",
    "regulatory--stop": "stop",
    "regulatory--no-entry": "no entry",
    "regulatory--no-parking": "no parking",
    "information--no-parking": "no parking",
    "regulatory--no-stopping": "no stopping",
    "regulatory--no-overtaking": "no overtaking",
    "regulatory--no-overtaking-by-heavy-goods-vehicles": "no overtaking for trucks",
    "regulatory--no-left-turn": "no left",
    "regulatory--no-right-turn": "no right",
    "regulatory--no-u-turn": "no u-turn",
    "regulatory--no-heavy-goods-vehicles": "no goods vehicles",
    "regulatory--no-pedestrians": "no pedestrians",
    "regulatory--no-bicycles": "no cycles",
    "regulatory--no-motor-vehicles": "no motor vehicles",
    "regulatory--no-straight-through": "no straight ahead",
    "regulatory--road-closed-to-vehicles": "no vehicles",
    "regulatory--keep-right": "keep right",
    "regulatory--keep-left": "keep left",
    "regulatory--go-straight": "ahead only",
    "regulatory--go-straight-or-turn-left": "ahead or left",
    "regulatory--go-straight-or-turn-right": "ahead or right",
    "regulatory--turn-left": "turn left ahead",
    "regulatory--turn-left-ahead": "turn left ahead",
    "regulatory--turn-right": "turn right ahead",
    "regulatory--turn-right-ahead": "turn right ahead",
    "regulatory--roundabout": "roundabout",
    "regulatory--pass-on-either-side": "pass either side",
    "regulatory--height-limit": "height limit",
    "regulatory--width-limit": "width limit",
    "regulatory--weight-limit": "load limit",
    "regulatory--bicycles-only": "bicycle",
    "regulatory--mopeds-and-bicycles-only": "bicycle",
    "regulatory--priority-road": "priority road",
    "regulatory--give-way-to-oncoming-traffic": "give way to oncoming",
    "regulatory--end-of-prohibition": "restriction ends",
    "regulatory--end-of-speed-limit-zone": "end of speed limit",
    "regulatory--end-of-maximum-speed-limit-30": "end of speed limit",
    "regulatory--end-of-maximum-speed-limit-70": "end of speed limit",
    "regulatory--shared-path-bicycles-and-pedestrians": "shared path",
    "regulatory--shared-path-pedestrians-and-bicycles": "shared path",
    "regulatory--dual-path-bicycles-and-pedestrians": "shared path",
    "regulatory--dual-path-pedestrians-and-bicycles": "shared path",
    "regulatory--pedestrians-only": "pedestrians only",
    "regulatory--one-way-left": "one way, arrow left",
    "regulatory--one-way-right": "one way, arrow right",
    "regulatory--one-way-straight": "one way, arrow straight",
    "information--parking": "parking",
    "information--pedestrians-crossing": "pedestrian crossing",
    "information--road-bump": "hump or rough road",
    "information--minimum-speed-40": "minimum speed",
    "information--tram-bus-stop": "bus stop",
    "information--bus-stop": "bus stop",
    "information--dead-end": "dead end",
    "information--hospital": "hospital",
    "information--end-of-motorway": "end of motorway",
    "complementary--chevron-left": "chevron left",
    "complementary--chevron-right": "chevron right",
    "warning--horizontal-alignment-left": "chevron left",
    "warning--horizontal-alignment-right": "chevron right",
    "warning--pedestrians-crossing": "pedestrian crossing",
    "warning--children": "children crossing",
    "warning--playground": "children crossing",
    "warning--school-zone": "school ahead",
    "warning--bicycles-crossing": "cycle crossing",
    "warning--wild-animals": "wild animals",
    "warning--kangaloo-crossing": "wild animals",
    "warning--wombat-crossing": "wild animals",
    "warning--domestic-animals": "cattle",
    "warning--other-danger": "general caution",
    "warning--curve-left": "left curve",
    "warning--curve-right": "right curve",
    "warning--turn-left": "left curve",
    "warning--turn-right": "right curve",
    "warning--double-curve-first-left": "left reverse bend",
    "warning--double-curve-first-right": "right reverse bend",
    "warning--double-reverse-curve-right": "right reverse bend",
    "warning--double-turn-first-right": "right reverse bend",
    "warning--hairpin-curve-left": "left hairpin bend",
    "warning--hairpin-curve-right": "right hairpin bend",
    "warning--winding-road-first-left": "winding road",
    "warning--winding-road-first-right": "winding road",
    "warning--slippery-road-surface": "slippery road",
    "warning--road-narrows": "narrow road ahead",
    "warning--road-narrows-left": "road narrows on left",
    "warning--road-narrows-right": "road narrows on right",
    "warning--road-widens": "road widens ahead",
    "warning--road-widens-right": "road widens ahead",
    "warning--roadworks": "road work",
    "warning--flaggers-in-road": "road work",
    "warning--traffic-signals": "traffic signals ahead",
    "warning--road-bump": "hump or rough road",
    "warning--uneven-road": "hump or rough road",
    "warning--uneven-roads-ahead": "hump or rough road",
    "warning--dip": "dangerous dip",
    "warning--roundabout": "roundabout ahead",
    "warning--pass-left-or-right": "pass either side",
    "warning--height-restriction": "height limit",
    "warning--crossroads": "crossroads",
    "warning--crossroads-with-priority-to-the-right": "crossroads",
    "warning--junction-with-a-side-road-perpendicular-left": "side road left",
    "warning--junction-with-a-side-road-acute-left": "side road left",
    "warning--junction-with-a-side-road-perpendicular-right": "side road right",
    "warning--junction-with-a-side-road-acute-right": "side road right",
    "warning--t-roads": "t junction",
    "warning--y-roads": "y junction",
    "warning--offset-roads": "staggered junction",
    "warning--narrow-bridge": "narrow bridge",
    "warning--falling-rocks-or-debris-right": "falling rocks",
    "warning--steep-ascent": "steep ascent",
    "warning--railroad-crossing": "railway crossing",
    "warning--railroad-intersection": "railway crossing",
    "warning--railroad-crossing-with-barriers": "guarded level crossing",
    "warning--railroad-crossing-without-barriers": "unguarded level crossing",
    "warning--stop-ahead": "stop ahead",
    "warning--traffic-merges-right": "traffic merges from right",
    "warning--two-way-traffic": "two way traffic",
    "warning--divided-highway-ends": "divided road ends",
    "warning--trucks-crossing": "trucks crossing",
}
# Left out entirely: plates under signs, unsure labels, and signs that look like a class but mean something else
MAPILLARY_LEFT_OUT_PREFIXES = ("complementary--",)
MAPILLARY_LEFT_OUT = {
    "complementary--chevron-right-unsure",
    "regulatory--no-parking-or-no-stopping",
    "regulatory--no-heavy-goods-vehicles-or-buses",
    "regulatory--no-motorcycles",
    "regulatory--no-mopeds-or-bicycles",
    "regulatory--no-motor-vehicles-except-motorcycles",
    "regulatory--no-buses",
    "regulatory--no-motor-vehicle-trailers",
    "regulatory--no-pedestrians-or-bicycles",
    "regulatory--end-of-bicycles-only",
    "regulatory--end-of-buses-only",
    "regulatory--end-of-no-parking",
    "regulatory--truck-speed-limit-60",
    "warning--accidental-area-unsure",
    "warning--traffic-merges-left",
}


def meaning(label):
    """Drop Mapillary's variant suffix, for example regulatory--stop--g1 -> regulatory--stop."""
    return label.rsplit("--", 1)[0]


def mapillary_class(label):
    """Return the class index for a Mapillary label, or None to leave the crop out."""
    base = meaning(label)
    if base.startswith("regulatory--maximum-speed-limit"):
        return GLOBAL_NAMES.index("speed limit") if base not in MAPILLARY_LEFT_OUT else None
    if base in MAPILLARY_MEANINGS:
        return GLOBAL_NAMES.index(MAPILLARY_MEANINGS[base])
    if base in MAPILLARY_LEFT_OUT or base.startswith(MAPILLARY_LEFT_OUT_PREFIXES):
        return None
    return OTHER_SIGN
