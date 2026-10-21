"""Site / building / floor / room hierarchy ("places").

Stored in config as a flat list of nodes: {"id", "name", "type", "parent", "map"}. Buildings and floors are
optional: a site can hold rooms directly. Every camera is assigned to one place (camera setting "place"); its
room (for counting a person once across the cameras of a room) and its floor plan come from that place's chain.
"""
import uuid

TYPES = ["site", "building", "floor", "room"]
TYPE_LABEL = {"site": "Site", "building": "Building", "floor": "Floor", "room": "Room"}
CHILD_TYPES = {None: ["site"], "site": ["building", "floor", "room"], "building": ["floor", "room"],
               "floor": ["room"], "room": []}
SEP = " › "


def nodes(cfg) -> list[dict]:
    return cfg.setdefault("places", [])


def get(cfg, nid):
    return next((n for n in nodes(cfg) if n["id"] == nid), None) if nid else None


def children(cfg, nid) -> list[dict]:
    return [n for n in nodes(cfg) if n.get("parent") == nid]


def add(cfg, name, ntype, parent=None) -> dict:
    ptype = get(cfg, parent)["type"] if parent else None
    if ntype not in CHILD_TYPES[ptype]:
        raise ValueError(f"A {ntype} cannot be inside a {ptype or 'nothing'}")
    n = {"id": uuid.uuid4().hex[:8], "name": name.strip(), "type": ntype, "parent": parent, "map": ""}
    nodes(cfg).append(n)
    return n


def chain(cfg, nid) -> list[dict]:
    out, seen = [], set()
    n = get(cfg, nid)
    while n and n["id"] not in seen:
        seen.add(n["id"])
        out.append(n)
        n = get(cfg, n.get("parent"))
    return list(reversed(out))


def path_text(cfg, nid, include_room=True) -> str:
    return SEP.join(n["name"] for n in chain(cfg, nid) if include_room or n["type"] != "room")


def walk(cfg, parent=None, depth=0):
    """Depth-first [(depth, node)] in list order."""
    for n in children(cfg, parent):
        yield depth, n
        yield from walk(cfg, n["id"], depth + 1)


def subtree_ids(cfg, nid) -> set:
    out = {nid}
    for _, n in walk(cfg, nid):
        out.add(n["id"])
    return out


def delete(cfg, nid) -> str | None:
    """Delete a place and everything inside it; its cameras move up to the parent. Returns the parent id."""
    n = get(cfg, nid)
    if not n:
        return None
    gone = subtree_ids(cfg, nid)
    cfg["places"] = [x for x in nodes(cfg) if x["id"] not in gone]
    for cc in (cfg.get("cameras") or {}).values():
        if cc.get("place") in gone:
            cc["place"] = n.get("parent") or ""
        if cc.get("map_node") in gone:
            cc["map_node"], cc["map_pos"] = "", None
    return n.get("parent")


def room_of(cfg, nid):
    return next((n for n in reversed(chain(cfg, nid)) if n["type"] == "room"), None)


def map_node(cfg, nid):
    """Nearest place (the camera's own place or above) that has a floor plan."""
    return next((n for n in reversed(chain(cfg, nid)) if n.get("map")), None)


def describe(cfg, nid) -> dict:
    """Derived camera fields: location (path without room), room (id, for counting), room_name, place_path."""
    ch = chain(cfg, nid)
    room = next((n for n in reversed(ch) if n["type"] == "room"), None)
    return {"location": SEP.join(n["name"] for n in ch if n["type"] != "room"),
            "room": room["id"] if room else "", "room_name": room["name"] if room else "",
            "place_path": SEP.join(n["name"] for n in ch),
            "place_short": SEP.join(n["name"] for n in ch[-2:]),
            "site": next((n["name"] for n in ch if n["type"] == "site"), ""),
            "building": next((n["name"] for n in ch if n["type"] == "building"), ""),
            "floor": next((n["name"] for n in ch if n["type"] == "floor"), "")}


def migrate(cfg):
    """Older versions: flat 'locations' (with floor plans) and free-text camera 'location' / 'room'."""
    if cfg.get("places"):
        return
    cams = cfg.get("cameras") or {}
    old_locs = cfg.get("locations") or []
    used = {cc.get("location") for cc in cams.values() if cc.get("location")}
    if not old_locs and not used and not any(cc.get("room") for cc in cams.values()):
        return
    by_name = {}
    for loc in old_locs:
        n = add(cfg, loc["name"], "site")
        n["map"] = loc.get("map", "")
        by_name[loc["name"]] = n
    for name in sorted(used - set(by_name)):
        by_name[name] = add(cfg, name, "site")
    rooms = {}
    for cam, cc in cams.items():
        site = by_name.get(cc.get("location"))
        if cc.get("room"):
            if site is None:
                site = by_name.setdefault("Site", add(cfg, "Site", "site"))
            key = (site["id"], cc["room"])
            if key not in rooms:
                rooms[key] = add(cfg, cc["room"], "room", site["id"])
            cc["place"] = rooms[key]["id"]
        elif site is not None:
            cc["place"] = site["id"]
        if cc.get("map_pos") and site is not None:
            cc["map_node"] = site["id"]
        cc.pop("location", None)
        cc.pop("room", None)
    cfg["locations"] = []


def camera_in(cfg, camera_place, nid) -> bool:
    """Is a camera assigned to camera_place inside the place nid (nid itself or anything below it)?"""
    return not nid or (bool(camera_place) and nid in {n["id"] for n in chain(cfg, camera_place)})


def options(cfg) -> list[tuple[str, str, str]]:
    """(id, indented label, type) for every place in tree order - for drop-down lists."""
    return [(n["id"], "    " * d + n["name"], n["type"]) for d, n in walk(cfg)]
