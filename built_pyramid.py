"""Perception-guided pick: PCA grasp angle, verify, retry, gentle place, pyramid, demolish."""

import asyncio
import math
import struct
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, cast

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, Transform
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.utils import ValueTypes, struct_to_dict

DEFAULT_GRIPPER_NAME = "gripper-1"
DEFAULT_ARM_NAME = "arm"
DEFAULT_VISION_NAME = "detective-to-seg"
DEFAULT_CAMERA_NAME = "cam"

HOME_POSE = Pose(x=327, y=-95, z=426, o_x=0, o_y=0, o_z=-1, theta=180)
PLACE_POSE = Pose(x=303, y=-147, z=140, o_x=0, o_y=0, o_z=-1, theta=180)

HOVER_Z = 250
GRASP_Z = 70
GRIPPER_Z_OFFSET_MM = 105
DESCEND_VIA_ARM = False

MIN_CLOUD_BYTES = 120000
MIN_POINTS = 4000
MAX_ATTEMPTS = 5

THETA_OFFSET = 90.0
JAW_AXIS_OFFSET = 90.0
MIN_ELONGATION = 1.25
AXIS_PROBE_MM = 50.0
FALLBACK_THETA = 0.0

RESCAN_WAIT_S = 1.5
FOREVER_LIMIT = 1000

# ---- Pyramid ----
BLOCK_W = 25.4                      # 1 inch wide
BLOCK_H = 30.0                      # layer height (a bit over block thickness for clearance)
PYRAMID_X = 303.0
PYRAMID_Y = -147.0
PYRAMID_BASE_Z = GRASP_Z
ROW_SPACING = BLOCK_W + 12.0
PLACE_RELEASE_OFFSET = 5.0
PYRAMID_THETA = 180.0
PYRAMID_EXCLUDE_MM = 120.0


def pyramid_slots() -> list:
    """3-2-1 pyramid slots as (x, y, z) world coords, rows along Y."""
    slots = []
    for layer in range(3):
        n = 3 - layer
        z = PYRAMID_BASE_Z + layer * BLOCK_H
        start = -(n - 1) / 2.0
        for i in range(n):
            slots.append((PYRAMID_X, PYRAMID_Y + (start + i) * ROW_SPACING, z))
    return slots


def wrap_symmetric(degrees: float) -> float:
    return ((degrees + 90.0) % 180.0) - 90.0


def parse_pcd_fields(raw: bytes):
    idx = raw.find(b"DATA ")
    if idx < 0:
        return None
    header_end = raw.find(b"\n", idx)
    if header_end < 0:
        return None
    data_parts = raw[idx:header_end].decode("ascii", errors="ignore").split()
    if len(data_parts) < 2 or data_parts[1].strip().lower() != "binary":
        return None
    header = raw[:header_end + 1].decode("ascii", errors="ignore")
    body = raw[header_end + 1:]

    fields = sizes = types = counts = points = None
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(v) for v in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(v) for v in parts[1:]]
        elif key == "POINTS":
            points = int(parts[1])

    if fields is None or sizes is None or types is None:
        return None
    if len(fields) != len(sizes) or len(fields) != len(types):
        return None
    if counts is None:
        counts = [1] * len(fields)
    if len(counts) != len(fields):
        return None
    if "x" not in fields or "y" not in fields:
        return None

    offsets = {}
    offset = 0
    for field, size, count in zip(fields, sizes, counts):
        offsets[field] = offset
        offset += size * count
    stride = offset
    if stride <= 0:
        return None
    if points is None:
        points = len(body) // stride
    points = min(points, len(body) // stride)
    return (body, points, stride, offsets["x"], offsets["y"])


def point_count(raw: bytes) -> int:
    parsed = parse_pcd_fields(raw)
    return parsed[1] if parsed else 0


def pca_axis_camera(raw: bytes) -> Optional[Tuple[float, float, float]]:
    parsed = parse_pcd_fields(raw)
    if parsed is None:
        return None
    body, count, stride, x_offset, y_offset = parsed
    if count < 10:
        return None
    step = max(1, count // 4000) if count > 4000 else 1
    xs, ys = [], []
    for i in range(0, count, step):
        base = i * stride
        try:
            x = struct.unpack_from("<f", body, base + x_offset)[0]
            y = struct.unpack_from("<f", body, base + y_offset)[0]
        except struct.error:
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xs.append(x)
        ys.append(y)
    n = len(xs)
    if n < 10:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = syy = sxy = 0.0
    for x, y in zip(xs, ys):
        dx, dy = x - mean_x, y - mean_y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    midpoint = (sxx + syy) / 2.0
    radius = math.hypot((sxx - syy) / 2.0, sxy)
    lam_major, lam_minor = midpoint + radius, midpoint - radius
    if lam_major <= 0.0:
        return None
    elongation = float("inf") if lam_minor <= 0.0 else math.sqrt(lam_major / lam_minor)
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return math.cos(angle), math.sin(angle), elongation


def grasp_theta_from_world_yaw(world_yaw_deg: float) -> float:
    return wrap_symmetric(THETA_OFFSET - world_yaw_deg - JAW_AXIS_OFFSET)


class MyGenericService(GenericService, EasyResource):

    MODEL: ClassVar[Model] = Model(
        ModelFamily("hackathons", "e6d8f152-eef3-4845-aa20-d9843d66c9ee"),
        "generic-service",
    )

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        attrs = struct_to_dict(config.attributes)
        required_deps = [
            attrs.get("gripper", DEFAULT_GRIPPER_NAME),
            "rdk:service:motion/builtin",
            attrs.get("vision", DEFAULT_VISION_NAME),
        ]
        return required_deps, [attrs.get("arm", DEFAULT_ARM_NAME)]

    @classmethod
    def new(cls, config: ComponentConfig, dependencies) -> Self:
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config: ComponentConfig, dependencies) -> None:
        attrs = struct_to_dict(config.attributes)
        self.gripper_name = attrs.get("gripper", DEFAULT_GRIPPER_NAME)
        self.gripper = cast(Gripper, dependencies[Gripper.get_resource_name(self.gripper_name)])
        self.motion = cast(MotionClient, dependencies[MotionClient.get_resource_name("builtin")])
        self.vision_name = attrs.get("vision", DEFAULT_VISION_NAME)
        self.camera_name = attrs.get("camera", DEFAULT_CAMERA_NAME)
        self.vision = cast(VisionClient, dependencies[VisionClient.get_resource_name(self.vision_name)])
        self.arm_name = attrs.get("arm", DEFAULT_ARM_NAME)
        self.arm = None
        try:
            self.arm = cast(Arm, dependencies[Arm.get_resource_name(self.arm_name)])
        except KeyError:
            pass

    async def move_world(self, pose: Pose) -> None:
        await self.motion.move(
            component_name=self.gripper_name,
            destination=PoseInFrame(reference_frame="world", pose=pose),
        )

    async def move_gripper_to_z(self, pose: Pose) -> None:
        if not DESCEND_VIA_ARM or self.arm is None:
            await self.move_world(pose)
            return
        arm_pose = Pose(x=pose.x, y=pose.y, z=pose.z + GRIPPER_Z_OFFSET_MM,
                        o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta)
        await self.arm.move_to_position(arm_pose)

    async def open_gripper(self) -> None:
        await self.gripper.open()

    async def camera_point_in_world(self, x, y, z, frame_name) -> Pose:
        transform = Transform(
            reference_frame=frame_name,
            pose_in_observer_frame=PoseInFrame(
                reference_frame=self.camera_name,
                pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0),
            ),
        )
        world_pif = await self.motion.get_pose(
            component_name=frame_name,
            destination_frame="world",
            supplemental_transforms=[transform],
        )
        return world_pif.pose

    async def detect_detailed(self):
        objects = await self.vision.get_object_point_clouds(self.camera_name)
        valid, rejected = [], 0
        for obj in objects:
            if not obj.geometries.geometries:
                rejected += 1
                continue
            count = point_count(obj.point_cloud)
            if len(obj.point_cloud) < MIN_CLOUD_BYTES or count < MIN_POINTS:
                rejected += 1
                continue
            try:
                c = obj.geometries.geometries[0].center
                w = await self.camera_point_in_world(c.x, c.y, c.z, "zone-check")
                if (abs(w.x - PYRAMID_X) < PYRAMID_EXCLUDE_MM and
                        abs(w.y - PYRAMID_Y) < PYRAMID_EXCLUDE_MM):
                    rejected += 1
                    continue
            except Exception:
                pass
            valid.append((count, obj))
        if not valid:
            return None, len(objects), rejected
        best = max(valid, key=lambda p: p[0])
        return best[1], len(objects), rejected

    async def detect(self):
        obj, _, _ = await self.detect_detailed()
        return obj

    async def grasp_plan(self, obj):
        geometry = obj.geometries.geometries[0]
        center = geometry.center
        world_center = await self.camera_point_in_world(center.x, center.y, center.z, "detected-object")
        info = {"label": geometry.label}
        axis = pca_axis_camera(obj.point_cloud)
        if axis is None:
            info["pca_status"] = "unusable"
            return world_center, None, info
        ux, uy, elongation = axis
        if elongation < MIN_ELONGATION:
            info["pca_status"] = "degenerate"
            return world_center, None, info
        axis_point = await self.camera_point_in_world(
            center.x + AXIS_PROBE_MM * ux, center.y + AXIS_PROBE_MM * uy,
            center.z, "detected-object-axis")
        dx = axis_point.x - world_center.x
        dy = axis_point.y - world_center.y
        if math.hypot(dx, dy) < 1e-6:
            info["pca_status"] = "vertical"
            return world_center, None, info
        world_yaw = math.degrees(math.atan2(dy, dx))
        theta = grasp_theta_from_world_yaw(world_yaw)
        info["pca_status"] = "ok"
        info["world_yaw"] = world_yaw
        return world_center, theta, info

    async def look(self) -> dict:
        await self.move_world(HOME_POSE)
        obj = await self.detect()
        if obj is None:
            return {"found": False}
        wc, theta, info = await self.grasp_plan(obj)
        return {"found": True, "label": info.get("label"),
                "wx": wc.x, "wy": wc.y, "wz": wc.z,
                "pca_status": info.get("pca_status"), "grasp_theta": theta}

    async def inspect(self) -> dict:
        await self.move_world(HOME_POSE)
        obj = await self.detect()
        if obj is None:
            return {"found": False}
        wc, theta, info = await self.grasp_plan(obj)
        result = {"found": True, "label": info.get("label"),
                  "wx": wc.x, "wy": wc.y, "wz": wc.z, "grasp_theta": theta}
        try:
            dims = obj.geometries.geometries[0].box.dims_mm
            result["box_x"], result["box_y"], result["box_z"] = dims.x, dims.y, dims.z
        except Exception:
            pass
        return result

    async def verify_grabbed(self, grab_reported) -> dict:
        try:
            status = await self.gripper.is_holding_something()
            holding = bool(status.is_holding_something)
            print(f"VERIFY: holding={holding}")
            return {"grabbed": holding, "verified_by": "is_holding_something"}
        except Exception:
            pass
        if grab_reported is not None:
            return {"grabbed": bool(grab_reported), "verified_by": "grab_return"}
        await self.move_world(HOME_POSE)
        best, seen, _ = await self.detect_detailed()
        if best is not None:
            return {"grabbed": False, "verified_by": "vision"}
        if seen == 0:
            return {"grabbed": True, "verified_by": "vision"}
        return {"grabbed": False, "verified_by": "vision-ambiguous"}

    async def pick_from_table(self) -> dict:
        """Detect a block anywhere, pick, verify. Does not place."""
        await self.move_world(HOME_POSE)
        obj = await self.detect()
        if obj is None:
            return {"ok": True, "grabbed": False, "found": False}
        wc, theta, info = await self.grasp_plan(obj)
        if theta is None:
            theta = FALLBACK_THETA
        print(f"PICK {info.get('label')}: x={wc.x:.0f} y={wc.y:.0f} angle={theta:.0f} deg")
        hover = Pose(x=wc.x, y=wc.y, z=HOVER_Z,
                     o_x=HOME_POSE.o_x, o_y=HOME_POSE.o_y, o_z=HOME_POSE.o_z, theta=theta)
        await self.move_world(hover)
        await self.open_gripper()
        grasp = Pose(x=wc.x, y=wc.y, z=GRASP_Z,
                     o_x=HOME_POSE.o_x, o_y=HOME_POSE.o_y, o_z=HOME_POSE.o_z, theta=theta)
        await self.move_gripper_to_z(grasp)
        try:
            grab_reported = await self.gripper.grab()
        except Exception as exc:
            print(f"grab() raised: {exc}")
            grab_reported = None
        await self.move_world(hover)
        result = {"ok": True, "found": True, "theta": theta, "label": info.get("label")}
        result.update(await self.verify_grabbed(grab_reported))
        return result

    async def place_at(self, x: float, y: float, z: float) -> None:
        """Place the held block at world x,y,z, releasing just above."""
        over = Pose(x=x, y=y, z=HOVER_Z,
                    o_x=HOME_POSE.o_x, o_y=HOME_POSE.o_y, o_z=HOME_POSE.o_z, theta=PYRAMID_THETA)
        await self.move_world(over)
        low = Pose(x=x, y=y, z=z + PLACE_RELEASE_OFFSET,
                   o_x=HOME_POSE.o_x, o_y=HOME_POSE.o_y, o_z=HOME_POSE.o_z, theta=PYRAMID_THETA)
        await self.move_gripper_to_z(low)
        await self.open_gripper()
        await self.move_world(over)

    async def place(self) -> None:
        await self.place_at(PLACE_POSE.x, PLACE_POSE.y, GRASP_Z)
        await self.move_world(HOME_POSE)

    async def pick_and_place(self) -> dict:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result = await self.pick_from_table()
            if not result.get("found"):
                return {"ok": True, "grabbed": False, "reason": "no block"}
            if result.get("grabbed"):
                await self.place()
                return {"ok": True, "grabbed": True, "attempts": attempt}
            await self.open_gripper()
            await self.move_world(HOME_POSE)
        return {"ok": True, "grabbed": False, "reason": "gave up"}

    async def retrieve(self) -> dict:
        """One block: wait and re-scan if none seen, then pick and place."""
        no_block = 0
        while no_block < MAX_ATTEMPTS:
            await self.move_world(HOME_POSE)
            if await self.detect() is None:
                no_block += 1
                print(f"No block seen (scan {no_block}); waiting.")
                await asyncio.sleep(RESCAN_WAIT_S)
                continue
            return await self.pick_and_place()
        return {"ok": True, "grabbed": False, "reason": "no block found"}

    async def build_pyramid(self) -> dict:
        """Build a 3-2-1 pyramid from blocks found anywhere on the table."""
        slots = pyramid_slots()
        placed = 0

        for idx, (sx, sy, sz) in enumerate(slots):
            print(f"=== Slot {idx+1}/{len(slots)} at y={sy:.0f} z={sz:.0f} ===")
            grabbed = False
            waits = 0
            last = {}

            while not grabbed:
                await self.move_world(HOME_POSE)
                if await self.detect() is None:
                    waits += 1
                    print(f"No block available; waiting ({waits}).")
                    if waits >= FOREVER_LIMIT:
                        return {"ok": False, "placed": placed, "reason": "out of blocks"}
                    await asyncio.sleep(RESCAN_WAIT_S)
                    continue

                for attempt in range(1, MAX_ATTEMPTS + 1):
                    last = await self.pick_from_table()
                    if not last.get("found"):
                        break
                    if last.get("grabbed"):
                        grabbed = True
                        break
                    print(f"Grab missed ({attempt}); retrying.")
                    await self.open_gripper()
                    await self.move_world(HOME_POSE)

            print(f"Placing {last.get('label', 'block')} "
                  f"(angle {last.get('theta', 0):.0f} deg) in pyramid.")
            await self.place_at(sx, sy, sz)
            placed += 1
            print(f"Placed block {placed}/{len(slots)}.")

        await self.move_world(HOME_POSE)
        return {"ok": True, "placed": placed}

    async def demolish(self) -> dict:
        """Sweep through the pyramid and knock it down."""
        y_start = PYRAMID_Y - 3 * ROW_SPACING
        y_end = PYRAMID_Y + 3 * ROW_SPACING
        sweep_z = PYRAMID_BASE_Z + BLOCK_H

        await self.gripper.grab()   # close jaws to make a solid sweeper
        start = Pose(x=PYRAMID_X, y=y_start, z=sweep_z,
                     o_x=HOME_POSE.o_x, o_y=HOME_POSE.o_y, o_z=HOME_POSE.o_z,
                     theta=PYRAMID_THETA)
        await self.move_world(start)
        end = Pose(x=PYRAMID_X, y=y_end, z=sweep_z,
                   o_x=HOME_POSE.o_x, o_y=HOME_POSE.o_y, o_z=HOME_POSE.o_z,
                   theta=PYRAMID_THETA)
        await self.move_world(end)
        await self.open_gripper()
        await self.move_world(HOME_POSE)
        return {"ok": True, "demolished": True}

    async def run_forever(self) -> dict:
        picked, empty = 0, 0
        while empty < FOREVER_LIMIT:
            await self.move_world(HOME_POSE)
            if await self.detect() is None:
                empty += 1
                await asyncio.sleep(RESCAN_WAIT_S)
                continue
            empty = 0
            if (await self.pick_and_place()).get("grabbed"):
                picked += 1
        return {"ok": True, "picked": picked}

    async def do_command(self, command, *, timeout=None, **kwargs):
        action = command.get("action")
        if action == "home":
            await self.move_world(HOME_POSE)
            return {"ok": True}
        if action == "look":
            return await self.look()
        if action == "inspect":
            return await self.inspect()
        if action == "pick-once":
            return await self.pick_from_table()
        if action == "retrieve":
            return await self.retrieve()
        if action == "run-forever":
            return await self.run_forever()
        if action == "build-pyramid":
            return await self.build_pyramid()
        if action == "demolish":
            return await self.demolish()
        return {"error": f"unknown action: {action}"}
