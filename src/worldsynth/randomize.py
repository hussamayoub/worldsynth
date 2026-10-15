"""Domain-randomization samplers. Pure NumPy so they can be tested without Isaac Sim.

Transforms are 4x4 matrices in column-vector convention (``p_world = T @ p_local``); world z is up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import CameraCfg, ClutterCfg, LightingCfg, PartMaterialCfg

# Colors seen in machine shops: grey/white machine paint, safety colors, dark rubber mats, aluminium, wood.
SURFACE_PALETTE = np.array(
    [
        [0.55, 0.56, 0.57],
        [0.30, 0.31, 0.33],
        [0.12, 0.12, 0.13],
        [0.80, 0.80, 0.78],
        [0.66, 0.68, 0.70],
        [0.18, 0.26, 0.36],
        [0.22, 0.38, 0.28],
        [0.55, 0.42, 0.28],
        [0.75, 0.62, 0.20],
    ]
)
METAL_PALETTE = np.array(
    [
        [0.91, 0.92, 0.92],  # aluminium
        [0.77, 0.78, 0.78],  # steel
        [0.62, 0.62, 0.62],  # cast iron
        [0.95, 0.64, 0.54],  # copper
        [0.91, 0.78, 0.42],  # brass
    ]
)
PAINT_PALETTE = np.array(
    [
        [0.05, 0.05, 0.05],  # black anodized
        [0.10, 0.20, 0.55],  # blue anodized
        [0.60, 0.08, 0.06],  # red
        [0.85, 0.85, 0.82],  # white plastic
        [0.15, 0.40, 0.15],  # PCB green
        [0.85, 0.55, 0.10],  # orange
        [0.35, 0.35, 0.36],  # grey
    ]
)


def uniform(rng: np.random.Generator, r) -> float:
    return float(rng.uniform(r[0], r[1]))


def randint(rng: np.random.Generator, r) -> int:
    return int(rng.integers(r[0], r[1] + 1))


# ---------------------------------------------------------------------------------------------------------------------
# rotations


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quat_to_matrix(q)


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix from a quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) with w >= 0 from a rotation matrix."""
    m = np.asarray(m, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def make_transform(rot: np.ndarray, pos) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


# ---------------------------------------------------------------------------------------------------------------------
# resting poses


def flat_rotations() -> list[np.ndarray]:
    """The 6 rotations that put one of the +-x, +-y, +-z object axes pointing down."""
    rots = []
    for axis in np.eye(3):
        for sign in (1.0, -1.0):
            down = sign * axis
            # rotate `down` onto -z
            target = np.array([0.0, 0.0, -1.0])
            v = np.cross(down, target)
            c = float(np.dot(down, target))
            if np.linalg.norm(v) < 1e-9:
                rots.append(np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0]))
                continue
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rots.append(np.eye(3) + vx + vx @ vx / (1.0 + c))
    return rots


def stable_rotations(vertices: np.ndarray, faces: np.ndarray, max_poses: int = 8, min_prob: float = 0.02):
    """Resting orientations of a rigid part on a plane, from its convex hull (via trimesh).

    Returns:
        A list of rotation matrices and their probabilities. Falls back to :func:`flat_rotations` if the hull cannot
        be computed (e.g. degenerate, flat geometry).
    """
    import trimesh

    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        hull = mesh.convex_hull
        center = mesh.center_mass if mesh.is_watertight and mesh.volume > 1e-12 else hull.center_mass
        transforms, probs = hull.compute_stable_poses(center_mass=center)
    except Exception as e:  # e.g. degenerate geometry, or networkx missing
        logging.getLogger("worldsynth").warning("stable poses failed (%s); using the 6 flat poses", e)
        transforms, probs = [], []
    poses = [(t[:3, :3], float(p)) for t, p in zip(transforms, probs, strict=False) if p >= min_prob][:max_poses]
    if not poses:
        rots = flat_rotations()
        return rots, [1.0 / len(rots)] * len(rots)
    total = sum(p for _, p in poses)
    return [r for r, _ in poses], [p / total for _, p in poses]


# ---------------------------------------------------------------------------------------------------------------------
# placement


@dataclass
class PartFootprint:
    """A part in a fixed resting rotation: what the placer needs to put it on the table."""

    rotation: np.ndarray
    """Object-to-world rotation without the final yaw."""
    min_z: float
    """Lowest vertex z of the rotated part [m]."""
    center_xy: np.ndarray
    """Center of the rotated part's xy bounding box [m]."""
    radius: float
    """Radius of the circle that covers the rotated part's xy footprint [m]."""


def footprint(vertices: np.ndarray, rotation: np.ndarray) -> PartFootprint:
    v = vertices @ rotation.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    center = 0.5 * (lo[:2] + hi[:2])
    radius = float(np.max(np.linalg.norm(v[:, :2] - center, axis=1)))
    return PartFootprint(rotation=rotation, min_z=float(lo[2]), center_xy=center, radius=radius)


def place_on_table(
    rng: np.random.Generator,
    footprints: list[PartFootprint],
    area: tuple[tuple[float, float], tuple[float, float]],
    table_z: float,
    spacing: float,
    max_tries: int = 200,
    occupied: list[tuple[np.ndarray, float]] | None = None,
) -> list[np.ndarray | None]:
    """Place parts on the table top without footprint overlap.

    Each part gets a random yaw about the world z axis and lands with its lowest point on ``table_z``.

    Args:
        occupied: Circles ``(center_xy, radius)`` already taken on the table. Placed parts are appended to it, so
            the same list can be passed to later calls (e.g. parts first, then clutter).

    Returns:
        A world transform per footprint, or ``None`` for parts that did not fit.
    """
    occupied = [] if occupied is None else occupied
    out = []
    for fp in footprints:
        placed = None
        for _ in range(max_tries):
            yaw = rng.uniform(-np.pi, np.pi)
            xy = np.array([rng.uniform(*area[0]), rng.uniform(*area[1])])
            if any(np.linalg.norm(xy - c) < fp.radius + r + spacing for c, r in occupied):
                continue
            rz = rot_z(yaw)
            rot = rz @ fp.rotation
            # rotating about z keeps min_z; shift so the footprint center lands on xy
            offset_xy = rz[:2, :2] @ fp.center_xy
            pos = np.array([xy[0] - offset_xy[0], xy[1] - offset_xy[1], table_z - fp.min_z])
            placed = make_transform(rot, pos)
            occupied.append((xy, fp.radius))
            break
        out.append(placed)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# camera


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World transform of a USD camera (looks along local -z, local +y up) at ``eye`` looking at ``target``."""
    eye, target, up = (np.asarray(a, dtype=float) for a in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight down
        x = np.cross(f, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return make_transform(np.stack([x, y, -f], axis=1), eye)


USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
"""Right-multiply a USD camera transform with this to get the OpenCV camera frame (x right, y down, z forward)."""


def spherical(target, distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Point at ``distance`` from ``target``; azimuth 0 = on the -x side, looking along +x."""
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    d = np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(target, dtype=float) + distance * d


@dataclass
class CameraTrajectory:
    transforms: list[np.ndarray]
    """USD camera world transform per frame."""
    focal_length: float
    """[mm]"""
    params: dict


def sample_camera_trajectory(
    rng: np.random.Generator, cfg: CameraCfg, center, num_frames: int, min_elevation: float = 5.0
) -> CameraTrajectory:
    """Smooth camera path for one clip: spherical coordinates around a jittered target, eased start to end."""
    target = np.asarray(center, dtype=float).copy()
    target[:2] += rng.uniform(-cfg.target_jitter, cfg.target_jitter, size=2)
    d0, az0, el0 = uniform(rng, cfg.distance), uniform(rng, cfg.azimuth), uniform(rng, cfg.elevation)
    dd, daz, del_ = uniform(rng, cfg.distance_sweep), uniform(rng, cfg.azimuth_sweep), uniform(rng, cfg.elevation_sweep)
    focal = uniform(rng, cfg.focal_length)
    transforms = []
    for i in range(num_frames):
        s = 0.0 if num_frames == 1 else i / (num_frames - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in/out: no velocity jump at the clip ends
        el = max(min_elevation, el0 + s * del_)
        eye = spherical(target, max(0.2, d0 + s * dd), az0 + s * daz, el)
        transforms.append(look_at(eye, target))
    params = {
        "target": target.tolist(),
        "distance": [d0, d0 + dd],
        "azimuth": [az0, az0 + daz],
        "elevation": [el0, el0 + del_],
        "focal_length": focal,
    }
    return CameraTrajectory(transforms=transforms, focal_length=focal, params=params)


def rotation_about(axis, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def look_target(t_cam: np.ndarray, plane_z: float, fallback_distance: float = 1.0) -> np.ndarray:
    """Point where a USD camera's optical axis hits the plane ``z = plane_z`` (or a point in front of it)."""
    eye, forward = t_cam[:3, 3], -t_cam[:3, 2]
    if forward[2] < -1e-3:
        t = (plane_z - eye[2]) / forward[2]
        if t > 0:
            return eye + t * forward
    return eye + fallback_distance * forward


def orbit_trajectory(
    t0: np.ndarray, target, num_frames: int, azimuth: float, elevation: float, distance: float
) -> list[np.ndarray]:
    """Move a given camera smoothly around ``target``: ``azimuth`` [deg] about world z, ``elevation`` [deg] up or
    down, ``distance`` [m] closer or farther. Frame 0 is exactly ``t0`` (including any roll)."""
    target = np.asarray(target, dtype=float)
    r0 = t0[:3, 3] - target
    out = []
    for i in range(num_frames):
        s = 0.0 if num_frames == 1 else i / (num_frames - 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in/out, as for sampled cameras
        rot = rotation_about([0, 0, 1], np.deg2rad(s * azimuth))
        side = np.cross(r0, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) > 1e-6:  # positive elevation raises the camera
            rot = rot @ rotation_about(side, np.deg2rad(s * elevation))
        r = rot @ r0
        r *= max(0.05, np.linalg.norm(r0) + s * distance) / np.linalg.norm(r0)
        t = np.eye(4)
        t[:3, :3] = rot @ t0[:3, :3]
        t[:3, 3] = target + r
        out.append(t)
    return out


def sample_orbit_trajectory(
    rng: np.random.Generator, cfg: CameraCfg, t0: np.ndarray, target, num_frames: int, focal_length: float
) -> CameraTrajectory:
    """Trajectory for a fixed (scene) camera, using the ``*_sweep`` ranges of ``cfg``."""
    az, el, dd = uniform(rng, cfg.azimuth_sweep), uniform(rng, cfg.elevation_sweep), uniform(rng, cfg.distance_sweep)
    params = {
        "target": np.asarray(target, dtype=float).tolist(),
        "azimuth_sweep": az,
        "elevation_sweep": el,
        "distance_sweep": dd,
        "focal_length": focal_length,
        "start": np.asarray(t0).tolist(),
    }
    return CameraTrajectory(orbit_trajectory(t0, target, num_frames, az, el, dd), focal_length, params)


def intrinsics(width: int, height: int, focal_length: float, horizontal_aperture: float) -> np.ndarray:
    """Pinhole intrinsics K [px] of a USD camera with square pixels and a centered principal point."""
    fx = focal_length / horizontal_aperture * width
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------------------------------------------------
# appearance


def color_temperature_offset(rng: np.random.Generator, color, amount: float = 0.06) -> list[float]:
    return np.clip(np.asarray(color) + rng.uniform(-amount, amount, size=3), 0.0, 1.0).tolist()


def sample_part_material(rng: np.random.Generator, cfg: PartMaterialCfg) -> dict | None:
    """UsdPreviewSurface inputs for a part, or ``None`` to keep the CAD material."""
    if rng.random() >= cfg.randomize_prob:
        return None
    if rng.random() < cfg.metal_prob:
        color = METAL_PALETTE[rng.integers(len(METAL_PALETTE))]
        return {
            "kind": "metal",
            "diffuseColor": color_temperature_offset(rng, color, 0.04),
            "metallic": uniform(rng, (0.85, 1.0)),
            "roughness": uniform(rng, cfg.metal_roughness),
        }
    color = PAINT_PALETTE[rng.integers(len(PAINT_PALETTE))]
    return {
        "kind": "paint",
        "diffuseColor": color_temperature_offset(rng, color),
        "metallic": uniform(rng, (0.0, 0.3)),
        "roughness": uniform(rng, cfg.paint_roughness),
    }


def sample_surface_material(rng: np.random.Generator, roughness) -> dict:
    color = SURFACE_PALETTE[rng.integers(len(SURFACE_PALETTE))]
    return {
        "diffuseColor": color_temperature_offset(rng, color, 0.08),
        "metallic": uniform(rng, (0.0, 0.2)),
        "roughness": uniform(rng, roughness),
    }


def sample_lights(rng: np.random.Generator, cfg: LightingCfg, area_center) -> dict:
    """Dome and ceiling lights. ``light_height`` is relative to a floor 0.75 m below the placement surface."""
    lights = []
    center = np.asarray(area_center, dtype=float)
    floor_z = center[2] - 0.75 if center.size > 2 else 0.0
    for _ in range(randint(rng, cfg.num_lights)):
        pos = center[:2] + rng.uniform(-1.2, 1.2, size=2)
        lights.append(
            {
                "position": [float(pos[0]), float(pos[1]), floor_z + uniform(rng, cfg.light_height)],
                "size": [uniform(rng, cfg.light_size), uniform(rng, cfg.light_size)],
                "intensity": uniform(rng, cfg.light_intensity),
                "color_temperature": uniform(rng, cfg.light_color_temperature),
            }
        )
    return {
        "dome": {
            "intensity": uniform(rng, cfg.dome_intensity),
            "color_temperature": uniform(rng, cfg.dome_color_temperature),
            "rotation_deg": uniform(rng, (0.0, 360.0)),
        },
        "rect_lights": lights,
    }


def sample_clutter(rng: np.random.Generator, cfg: ClutterCfg) -> list[dict]:
    items = []
    for _ in range(randint(rng, cfg.count)):
        size = uniform(rng, cfg.size)
        shape = str(rng.choice(["cube", "cylinder", "sphere", "cone"]))
        scale = size * rng.uniform(0.4, 1.0, size=3)
        if shape in ("sphere",):
            scale[:] = size / 2.0
        items.append({"shape": shape, "scale": scale.tolist(), "material": None})
    return items
