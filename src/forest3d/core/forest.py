"""Forest world generation with procedural model placement."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from stl import mesh

from forest3d.config.schema import DensityConfig

logger = logging.getLogger("forest3d.forest")


class WorldPopulator:
    """Procedurally generate forest worlds with intelligent model placement.

    Places models on terrain using zone weighting, distance constraints,
    and natural clustering patterns. Handles cross-category collision
    avoidance and scale-aware spacing.
    """

    # Scale ranges for each model category
    SCALE_RANGES = {
        "tree": (0.8, 1.5),
        "rock": (0.25, 1.0),
        "bush": (0.3, 1.0),
        "grass": (0.2, 0.6),
        "sand": (1.0, 2.5),
    }

    # Minimum distances between models of same category
    MIN_DISTANCES = {
        "tree": 8.0,
        "bush": 2.0,
        "rock": 4.0,
        "grass": 0.5,
        "sand": 3.0,
    }

    # Cross-category minimum distances
    # Keys are tuples of (category1, category2), order doesn't matter
    CROSS_CATEGORY_DISTANCES = {
        ("tree", "tree"): 8.0,
        ("tree", "bush"): 1.5,
        ("tree", "rock"): 2.0,
        ("tree", "grass"): 0.5,
        ("tree", "sand"): 4.0,
        ("bush", "bush"): 2.0,
        ("bush", "rock"): 1.5,
        ("bush", "grass"): 0.3,
        ("bush", "sand"): 2.0,
        ("rock", "rock"): 4.0,
        ("rock", "grass"): 0.5,
        ("rock", "sand"): 2.0,
        ("grass", "grass"): 0.5,
        ("grass", "sand"): 0.5,
        ("sand", "sand"): 3.0,
    }

    # Zone weights (edge vs center preference)
    ZONE_WEIGHTS = {
        "tree": {"edge": 0.2, "center": 0.8},
        "rock": {"edge": 0.2, "center": 0.8},
        "bush": {"edge": 0.4, "center": 0.6},
        "grass": {"edge": 0.5, "center": 0.5},
        "sand": {"edge": 0.7, "center": 0.3},
    }

    def __init__(
            self,
            base_path: Path,
            progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """Initialize the world populator.

        Args:
            base_path: Project base path containing models/ and worlds/.
            progress_callback: Optional callback for progress updates (percent, message).

        Raises:
            FileNotFoundError: If required paths don't exist.
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.worlds_path = self.base_path / "worlds"
        self.progress_callback = progress_callback

        # Store (x, y, z, scale) for each placed model
        self.placed_models: Dict[str, List[Tuple[float, float, float, float]]] = {
            "tree": [],
            "bush": [],
            "rock": [],
            "grass": [],
            "sand": [],
        }

        self._verify_paths()
        self.model_variants = self._get_model_variants()

    def _verify_paths(self) -> None:
        """Verify all required paths exist."""
        required_paths = [
            self.models_path / "ground",
            self.worlds_path,
            ]

        # Check for at least one model category
        categories = ["tree", "rock", "bush", "grass", "sand"]
        has_category = False
        for cat in categories:
            if (self.models_path / cat).exists():
                has_category = True
                required_paths.append(self.models_path / cat)

        if not has_category:
            logger.warning("No model categories found in models directory")

        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))

        if self.models_path / "ground" not in [Path(p) for p in missing_paths]:
            # Ground is required
            pass
        elif missing_paths:
            missing_str = "\n  - ".join(missing_paths)
            raise FileNotFoundError(f"Required paths not found:\n  - {missing_str}")

        # Create worlds directory if it doesn't exist
        self.worlds_path.mkdir(parents=True, exist_ok=True)

    def _get_model_variants(self) -> Dict[str, List[str]]:
        """Get available variants for each model category."""
        variants = {}
        categories = ["tree", "bush", "rock", "grass", "sand"]

        for category in categories:
            category_path = self.models_path / category
            if category_path.exists():
                variants[category] = []
                for d in category_path.iterdir():
                    if d.is_dir() and not d.name.startswith("."):
                        variants[category].append(d.name)
                if variants[category]:
                    logger.info(f"Found {len(variants[category])} variants for {category}")

        return variants

    def _get_model_vertices(self, category: str, variant: str) -> np.ndarray:
        """Read vertex positions from a model's glTF file.

        Returns:
            (N, 3) array of vertex positions in local model coordinates.
        """
        import json
        import struct

        glb_path = self.models_path / category / variant / "mesh" / f"{variant}.glb"
        if not glb_path.exists():
            raise FileNotFoundError(f"Model glTF not found: {glb_path}")

        with open(glb_path, "rb") as f:
            f.read(12)  # GLB header
            chunks = []
            while True:
                chunk_len_bytes = f.read(4)
                if len(chunk_len_bytes) < 4:
                    break
                chunk_len = struct.unpack("<I", chunk_len_bytes)[0]
                f.read(4)  # chunk type
                chunks.append(f.read(chunk_len))

        gltf = json.loads(chunks[0].decode())
        bin_data = chunks[1]

        # Find the first POSITION accessor
        pos_accessor = None
        for acc in gltf.get("accessors", []):
            if acc.get("type") == "VEC3" and "min" in acc:
                pos_accessor = acc
                break

        if pos_accessor is None:
            raise ValueError(f"No POSITION accessor found in {glb_path}")

        # Get buffer view
        bv = gltf["bufferViews"][pos_accessor["bufferView"]]
        offset = bv.get("byteOffset", 0)
        count = pos_accessor["count"]

        # Read FLOAT (componentType 5126) vertices
        verts = np.frombuffer(
            bin_data, dtype=np.float32, count=count * 3, offset=offset
        ).reshape(count, 3).copy()

        return verts

    def _get_terrain_mesh(self) -> mesh.Mesh:
        """Get terrain mesh for height sampling."""
        mesh_path = self.models_path / "ground" / "mesh" / "terrain.stl"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Terrain mesh not found at: {mesh_path}")
        return mesh.Mesh.from_file(str(mesh_path))

    def _build_triangle_index(self, terrain_mesh: mesh.Mesh) -> None:
        """Build a KD-tree of triangle centroids for fast spatial lookup.

        The height/normal sampling methods need the closest triangle to a
        given (x, y) point. A brute-force scan over all triangles is O(n)
        per call — with millions of faces and many sampling calls per
        model, generation stalls. The KD-tree reduces lookup to O(log n).

        Caches the tree and triangle centroids on the instance so it is
        built only once per generation run.
        """
        from scipy.spatial import cKDTree

        centroids = terrain_mesh.vectors.mean(axis=1)
        self._terrain_centroids = centroids
        self._terrain_kdtree = cKDTree(centroids[:, :2])

    def _find_closest_triangle(self, terrain_mesh: mesh.Mesh, x: float, y: float):
        """Find the closest terrain triangle to (x, y) using the KD-tree."""
        if not hasattr(self, "_terrain_kdtree"):
            self._build_triangle_index(terrain_mesh)
        _, idx = self._terrain_kdtree.query([x, y])
        return terrain_mesh.vectors[idx]

    def _get_random_variant(self, category: str) -> Optional[str]:
        """Get random variant for category."""
        variants = self.model_variants.get(category, [])
        if not variants:
            return None
        return np.random.choice(variants)

    def _get_cross_distance(self, cat1: str, cat2: str) -> float:
        """Get minimum distance between two categories.

        Args:
            cat1: First category name.
            cat2: Second category name.

        Returns:
            Minimum required distance between the two categories.
        """
        # Try both orderings since we store only one direction
        if (cat1, cat2) in self.CROSS_CATEGORY_DISTANCES:
            return self.CROSS_CATEGORY_DISTANCES[(cat1, cat2)]
        elif (cat2, cat1) in self.CROSS_CATEGORY_DISTANCES:
            return self.CROSS_CATEGORY_DISTANCES[(cat2, cat1)]
        else:
            # Fallback to average of individual minimum distances
            return (self.MIN_DISTANCES.get(cat1, 1.0) + self.MIN_DISTANCES.get(cat2, 1.0)) / 2

    def _check_distance_to_placed(
            self, x: float, y: float, category: str, scale: float = 1.0
    ) -> bool:
        """Check if position is far enough from ALL placed models.

        Args:
            x: X coordinate of proposed position.
            y: Y coordinate of proposed position.
            category: Category of model being placed.
            scale: Scale factor of model being placed.

        Returns:
            True if position is valid (far enough from all models), False otherwise.
        """
        for other_category, positions in self.placed_models.items():
            # Get base minimum distance between these categories
            base_distance = self._get_cross_distance(category, other_category)

            for px, py, pz, p_scale in positions:
                # Calculate actual distance
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)

                # Adjust required distance based on both models' scales
                # Use average of scales, clamped to reasonable range
                scale_factor = (max(scale, 0.5) + max(p_scale, 0.5)) / 2
                required_dist = base_distance * scale_factor

                if dist < required_dist:
                    return False

        return True

    def _sample_terrain_height(self, terrain_mesh: mesh.Mesh, x: float, y: float) -> float:
        """Sample terrain height at given x, y coordinates.

        Args:
            terrain_mesh: The terrain mesh to sample from.
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Interpolated Z height at the given position.
        """
        point = np.array([x, y])
        closest_tri = self._find_closest_triangle(terrain_mesh, x, y)

        # Use barycentric interpolation for more accurate height
        # Fallback to mean if interpolation fails
        try:
            v0, v1, v2 = closest_tri

            # Calculate barycentric coordinates
            denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if abs(denom) < 1e-10:
                return np.mean(closest_tri[:, 2])

            w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
            w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
            w2 = 1 - w0 - w1

            # Interpolate height
            z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
            return z
        except Exception:
            return np.mean(closest_tri[:, 2])

    def _check_terrain_support(
            self,
            terrain_mesh: mesh.Mesh,
            x: float,
            y: float,
            scale: float,
            sample_radius: float = 2.0,
            max_height_diff: float = 1.5,
    ) -> bool:
        """Check that the terrain fully supports a model's footprint.

        Samples terrain heights at the center and around a circle of the
        model's footprint radius. If the height variation across the
        footprint exceeds a threshold (scaled by model size), the terrain
        has an edge or cliff and part of the model would float — reject.

        Args:
            terrain_mesh: Terrain mesh for height sampling.
            x: X coordinate of the model center.
            y: Y coordinate of the model center.
            scale: Model scale factor.
            sample_radius: Base radius (meters) for footprint sampling,
                multiplied by scale.
            max_height_diff: Base maximum allowed height difference (meters)
                across the footprint, multiplied by scale.

        Returns:
            True if the terrain fully supports the footprint, False otherwise.
        """
        radius = sample_radius * scale
        threshold = max_height_diff * scale

        # Sample at center and 8 points around the footprint
        sample_points = [(x, y)]
        for i in range(8):
            angle = i * (2 * np.pi / 8)
            sx = x + radius * np.cos(angle)
            sy = y + radius * np.sin(angle)
            sample_points.append((sx, sy))

        heights = [
            self._sample_terrain_height(terrain_mesh, sx, sy)
            for sx, sy in sample_points
        ]

        height_range = max(heights) - min(heights)
        return height_range <= threshold

    def _sample_terrain_normal(
            self, terrain_mesh: mesh.Mesh, x: float, y: float
    ) -> Tuple[float, float, float]:
        """Sample terrain surface normal at given x, y coordinates.

        Finds the closest triangle to (x, y) and returns its face normal,
        oriented to point up (positive Z component).

        Args:
            terrain_mesh: The terrain mesh to sample from.
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Tuple of (nx, ny, nz) unit normal vector at the terrain surface.
        """
        closest_tri = self._find_closest_triangle(terrain_mesh, x, y)

        v0, v1, v2 = closest_tri
        normal = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(normal)
        if length > 0:
            normal /= length

        if normal[2] < 0:
            normal = -normal

        return float(normal[0]), float(normal[1]), float(normal[2])

    @staticmethod
    def _slope_align_angles(
            nx: float, ny: float, nz: float, yaw: float
    ) -> Tuple[float, float, float]:
        """Compute (roll, pitch, yaw) so the model's Z axis aligns with the
        terrain surface normal, then rotates around the local Z by yaw.

        Gazebo uses ZYX intrinsic Euler angles (R = Rz * Ry * Rx). A naive
        roll/pitch from the normal breaks alignment when yaw is non-zero,
        because yaw rotates around the world Z, not the slope normal. We
        instead build R = R_align @ Rz(yaw) — which keeps the up axis locked
        to the normal for any yaw — then decompose back to ZYX Euler.

        Args:
            nx, ny, nz: Terrain surface normal components.
            yaw: Desired rotation around the local (slope-aligned) Z axis.

        Returns:
            Tuple of (roll, pitch, yaw) in radians for Gazebo's pose element.
        """
        n = np.array([nx, ny, nz], dtype=float)
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n

        # R_align = Ry(pitch0) @ Rx(roll0) maps [0,0,1] to n
        roll0 = -np.arcsin(np.clip(n[1], -1, 1))
        pitch0 = np.arctan2(n[0], n[2])

        cr, sr = np.cos(roll0), np.sin(roll0)
        cp, sp = np.cos(pitch0), np.sin(pitch0)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # R = Ry(pitch0) @ Rx(roll0) @ Rz(yaw)
        R00 = cp * cy + sp * sr * sy
        R10 = cr * sy
        R20 = -sp * cy + cp * sr * sy
        R21 = sp * sy + cp * sr * cy
        R22 = cp * cr

        # Extract ZYX Euler from R = Rz(yaw') @ Ry(pitch') @ Rx(roll')
        pitch_out = np.arcsin(np.clip(-R20, -1, 1))
        roll_out = np.arctan2(R21, R22)
        yaw_out = np.arctan2(R10, R00)

        return float(roll_out), float(pitch_out), float(yaw_out)

    def _compute_z_offset(
            self,
            category: str,
            variant: str,
            scale: float,
            roll: float,
            pitch: float,
            yaw: float,
    ) -> float:
        """Compute the Z offset to seat a model's lowest point on the terrain.

        Reads the actual mesh vertices (not just the bounding box) and
        transforms them by the slope-aligned rotation to find the true
        lowest point. Adds a penetration margin so the rock sinks slightly
        into the terrain — ensuring contact for robot navigation rather
        than floating with a gap.

        Args:
            category: Model category (e.g. "rock").
            variant: Model variant name.
            scale: Model scale factor.
            roll, pitch, yaw: Rotation angles in radians.

        Returns:
            Z offset (meters) to subtract from the placement Z.
        """
        vert_cache_attr = "_vert_cache"
        if not hasattr(self, vert_cache_attr):
            setattr(self, vert_cache_attr, {})
        cache = getattr(self, vert_cache_attr)
        key = f"{category}/{variant}"
        if key not in cache:
            try:
                cache[key] = self._get_model_vertices(category, variant)
            except (FileNotFoundError, ValueError):
                return 0.0

        verts = cache[key]

        # Build rotation matrix from ZYX Euler (Gazebo convention)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ])

        # Rotate all vertices and find the true minimum Z
        rotated = verts @ R.T
        min_z = float(np.min(rotated[:, 2])) * scale
        max_z = float(np.max(rotated[:, 2])) * scale
        rock_height = max_z - min_z

        # Penetration margin: sink the rock into the terrain by 15% of its
        # height so there is no gap. The robot can drive over the embedded
        # part; a small overlap with the terrain is better than floating.
        penetration = rock_height * 0.15

        return min_z + penetration

    def _get_random_position(
            self,
            terrain_mesh: mesh.Mesh,
            category: str,
            scale: float = 1.0,
            margin: float = 2.0
    ) -> Optional[Tuple[float, float, float]]:
        """Get random position with intelligent placement.

        Args:
            terrain_mesh: Terrain mesh for bounds and height sampling.
            category: Model category being placed.
            scale: Scale of the model (affects distance requirements).
            margin: Minimum distance from terrain edges.

        Returns:
            Tuple of (x, y, z) if valid position found, None otherwise.
        """
        bounds = terrain_mesh.vectors.reshape(-1, 3)
        min_x, max_x = np.min(bounds[:, 0]) + margin, np.max(bounds[:, 0]) - margin
        min_y, max_y = np.min(bounds[:, 1]) + margin, np.max(bounds[:, 1]) - margin

        max_attempts = 100  # Increased attempts for better placement

        for _ in range(max_attempts):
            x, y = None, None
            is_edge = np.random.random() < self.ZONE_WEIGHTS[category]["edge"]

            if category == "sand":
                if is_edge:
                    edge = np.random.choice(["top", "bottom", "left", "right"])
                    if edge in ["top", "bottom"]:
                        x = np.random.uniform(min_x + margin, max_x - margin)
                        y = max_y - margin if edge == "top" else min_y + margin
                        y += np.random.uniform(-1, 1)
                    else:
                        x = max_x - margin if edge == "right" else min_x + margin
                        x += np.random.uniform(-1, 1)
                        y = np.random.uniform(min_y + margin, max_y - margin)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            elif category == "tree":
                if self.placed_models["tree"] and np.random.random() < 0.7:
                    # Cluster near existing trees
                    base_idx = np.random.randint(len(self.placed_models["tree"]))
                    base_tree = self.placed_models["tree"][base_idx]
                    base_scale = base_tree[3]

                    # Adjust cluster radius based on scales
                    min_cluster_dist = self.MIN_DISTANCES["tree"] * max(scale, base_scale)
                    max_cluster_dist = min_cluster_dist * 2

                    radius = np.random.uniform(min_cluster_dist, max_cluster_dist)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = base_tree[0] + radius * np.cos(angle)
                    y = base_tree[1] + radius * np.sin(angle)
                else:
                    # Place in open area, avoiding sand
                    for _ in range(10):
                        x = np.random.uniform(min_x + margin, max_x - margin)
                        y = np.random.uniform(min_y + margin, max_y - margin)

                        # Check distance from sand areas
                        sand_clear = all(
                            np.sqrt((x - sx) ** 2 + (y - sy) ** 2) >
                            self._get_cross_distance("tree", "sand") * max(scale, s_scale)
                            for sx, sy, _, s_scale in self.placed_models["sand"]
                        )
                        if sand_clear:
                            break
                    else:
                        continue

            elif category == "rock":
                if is_edge:
                    edge = np.random.choice(["top", "bottom", "left", "right"])
                    edge_variance = np.random.uniform(-2, 2)
                    if edge in ["top", "bottom"]:
                        x = np.random.uniform(min_x + margin, max_x - margin)
                        y = (max_y - margin if edge == "top" else min_y + margin) + edge_variance
                    else:
                        x = (max_x - margin if edge == "right" else min_x + margin) + edge_variance
                        y = np.random.uniform(min_y + margin, max_y - margin)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            elif category == "bush":
                if np.random.random() < 0.6 and self.placed_models["tree"]:
                    # Place near trees
                    base_idx = np.random.randint(len(self.placed_models["tree"]))
                    base_tree = self.placed_models["tree"][base_idx]
                    base_scale = base_tree[3]

                    # Bushes cluster closer to trees but not too close
                    min_dist = self._get_cross_distance("bush", "tree") * max(scale, base_scale)
                    radius = np.random.uniform(min_dist, min_dist + 3.0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = base_tree[0] + radius * np.cos(angle)
                    y = base_tree[1] + radius * np.sin(angle)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            elif category == "grass":
                # Grass can go almost anywhere but prefers areas with trees/bushes
                if np.random.random() < 0.5 and (self.placed_models["tree"] or self.placed_models["bush"]):
                    # Place near vegetation
                    all_vegetation = self.placed_models["tree"] + self.placed_models["bush"]
                    base = all_vegetation[np.random.randint(len(all_vegetation))]
                    radius = np.random.uniform(1.0, 5.0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = base[0] + radius * np.cos(angle)
                    y = base[1] + radius * np.sin(angle)
                else:
                    x = np.random.uniform(min_x + margin, max_x - margin)
                    y = np.random.uniform(min_y + margin, max_y - margin)

            else:
                x = np.random.uniform(min_x + margin, max_x - margin)
                y = np.random.uniform(min_y + margin, max_y - margin)

            # Validate position
            if x is None or y is None:
                continue

            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue

            if not self._check_distance_to_placed(x, y, category, scale):
                continue

            # For rocks, check that the terrain fully supports the footprint
            if category == "rock":
                if not self._check_terrain_support(terrain_mesh, x, y, scale):
                    continue

            # Sample height from terrain
            z = self._sample_terrain_height(terrain_mesh, x, y)

            # Category-specific height adjustments
            if category == "sand":
                z += np.random.uniform(-0.2, 0)
            elif category == "grass":
                z += np.random.uniform(-0.05, 0.05)
            elif category == "rock":
                z += np.random.uniform(-0.1, 0.1)
            else:
                z += np.random.uniform(-0.08, 0.08)

            # Store position with scale
            self.placed_models[category].append((x, y, z, scale))
            return x, y, z

        # No valid position found after max attempts
        return None

    def _add_scene_settings(self, world: ET.Element) -> None:
        """Add scene settings for proper PBR lighting.

        Args:
            world: World XML element to add settings to.
        """
        scene = ET.SubElement(world, "scene")
        ET.SubElement(scene, "ambient").text = "0.4 0.4 0.4 1"
        ET.SubElement(scene, "background").text = "0.7 0.8 0.9 1"

    def _add_extra_lighting(self, world: ET.Element) -> None:
        """Add extra lighting for forest scenes (ambient and point lights).

        Args:
            world: World XML element to add lights to.
        """
        # Add ambient directional light (softer fill light)
        ambient = ET.SubElement(world, "light", {"name": "ambient", "type": "directional"})
        ET.SubElement(ambient, "cast_shadows").text = "false"
        ET.SubElement(ambient, "pose").text = "0 0 10 0 0 0"
        ET.SubElement(ambient, "diffuse").text = "0.6 0.6 0.6 1"
        ET.SubElement(ambient, "specular").text = "0.1 0.1 0.1 1"
        ET.SubElement(ambient, "direction").text = "0.1 0.1 -0.9"

        # Add point light with proper attenuation structure
        point = ET.SubElement(world, "light", {"name": "point_light", "type": "point"})
        ET.SubElement(point, "cast_shadows").text = "false"
        ET.SubElement(point, "pose").text = "0 0 15 0 0 0"
        ET.SubElement(point, "diffuse").text = "0.3 0.3 0.3 1"
        ET.SubElement(point, "specular").text = "0.05 0.05 0.05 1"
        point_attenuation = ET.SubElement(point, "attenuation")
        ET.SubElement(point_attenuation, "range").text = "50"
        ET.SubElement(point_attenuation, "constant").text = "0.5"
        ET.SubElement(point_attenuation, "linear").text = "0.01"
        ET.SubElement(point_attenuation, "quadratic").text = "0.001"

    def create_forest_world(
            self,
            density_config: Optional[Dict[str, int]] = None,
    ) -> Path:
        """Create forest world with placed models.

        Args:
            density_config: Dict of category -> count. Uses defaults if None.

        Returns:
            Path to created world file.
        """
        from forest3d.utils.sdf import create_world_base, write_world_file

        # Reset placed models
        for category in self.placed_models:
            self.placed_models[category] = []

        # Use provided config or defaults
        if density_config is None:
            config = DensityConfig()
            density_config = {
                "tree": config.tree,
                "bush": config.bush,
                "rock": config.rock,
                "grass": config.grass,
                "sand": config.sand,
            }

        # Create world with shared base (plugins, physics, gravity, sun)
        world_elem, world = create_world_base("forest_world")

        # Add scene settings for proper PBR lighting
        self._add_scene_settings(world)

        # Add extra lighting for forest scenes
        self._add_extra_lighting(world)

        # Add terrain
        terrain = ET.SubElement(world, "include")
        ET.SubElement(terrain, "uri").text = "model://ground"
        ET.SubElement(terrain, "name").text = "terrain"
        ET.SubElement(terrain, "pose").text = "0 0 0 0 0 0"

        terrain_mesh = self._get_terrain_mesh()

        # Process categories in specific order (larger/important first)
        category_order = ["sand", "rock", "tree", "bush", "grass"]
        total_models = sum(density_config.get(c, 0) for c in category_order)
        models_placed = 0
        models_failed = 0

        for category in category_order:
            if category not in density_config or category not in self.model_variants:
                continue

            count = density_config[category]
            if count == 0:
                continue

            logger.info(f"Adding {count} {category} models...")
            category_placed = 0
            category_failed = 0

            for i in range(count):
                try:
                    variant = self._get_random_variant(category)
                    if not variant:
                        logger.warning(f"No variants available for {category}")
                        continue

                    # Generate scale FIRST (needed for distance calculations)
                    scale = np.random.uniform(*self.SCALE_RANGES[category])

                    # Get position considering scale
                    position = self._get_random_position(terrain_mesh, category, scale)

                    if position is None:
                        category_failed += 1
                        models_failed += 1
                        logger.debug(f"Could not find valid position for {category}_{i}")
                        continue

                    x, y, z = position

                    # Category-specific rotations
                    if category == "sand":
                        roll = pitch = 0
                        yaw = np.random.uniform(0, 2 * np.pi)
                    elif category == "tree":
                        # Slight tilt for natural look
                        roll = np.random.uniform(-0.05, 0.05)
                        pitch = np.random.uniform(-0.05, 0.05)
                        yaw = np.random.uniform(0, 2 * np.pi)
                    elif category == "rock":
                        # Align rocks to terrain slope: sample the surface
                        # normal and compute roll/pitch so the rock lies flat
                        # on the slope instead of floating horizontally.
                        nx, ny, nz = self._sample_terrain_normal(terrain_mesh, x, y)
                        yaw = np.random.uniform(0, 2 * np.pi)
                        roll, pitch, yaw = self._slope_align_angles(
                            nx, ny, nz, yaw
                        )
                        # Small jitter for natural variation
                        roll += np.random.uniform(-0.1, 0.1)
                        pitch += np.random.uniform(-0.1, 0.1)

                        # Offset Z so the rock's lowest point sits on the
                        # terrain surface after rotation. Without this, the
                        # rock rotates around its center and the uphill edge
                        # floats above the terrain.
                        z -= self._compute_z_offset(category, variant, scale, roll, pitch, yaw)
                    elif category == "bush":
                        roll = np.random.uniform(-0.03, 0.03)
                        pitch = np.random.uniform(-0.03, 0.03)
                        yaw = np.random.uniform(0, 2 * np.pi)
                    else:  # grass
                        roll = pitch = 0
                        yaw = np.random.uniform(0, 2 * np.pi)

                    # Add model to world
                    include = ET.SubElement(world, "include")
                    ET.SubElement(include, "uri").text = f"model://{category}/{variant}"
                    ET.SubElement(include, "name").text = f"{category}_{i}"
                    ET.SubElement(include, "pose").text = f"{x:.4f} {y:.4f} {z:.4f} {roll:.4f} {pitch:.4f} {yaw:.4f}"
                    ET.SubElement(include, "scale").text = f"{scale:.3f} {scale:.3f} {scale:.3f}"

                    category_placed += 1
                    models_placed += 1

                    if self.progress_callback and total_models > 0:
                        progress = int((models_placed / total_models) * 100)
                        self.progress_callback(progress, f"Placing {category}...")

                except Exception as e:
                    logger.warning(f"Failed to add {category} model: {e}")
                    category_failed += 1
                    models_failed += 1
                    continue

            logger.info(f"  {category}: placed {category_placed}/{count} (failed: {category_failed})")

        # Save the world file
        output_path = self.worlds_path / "forest_world.world"
        write_world_file(world_elem, output_path)

        logger.info(f"World file created at: {output_path}")
        logger.info(f"Total models placed: {models_placed}/{total_models} (failed: {models_failed})")
        logger.info("Models placed by category:")
        for category in category_order:
            if category in self.placed_models:
                logger.info(f"  - {category}: {len(self.placed_models[category])}")

        return output_path

    def get_model_statistics(self) -> Dict:
        """Get statistics about placed models.

        Returns:
            Dictionary containing placement statistics.
        """
        stats = {
            "total_models": sum(len(models) for models in self.placed_models.values()),
            "by_category": {
                category: len(models) for category, models in self.placed_models.items()
            },
            "variants_available": {
                category: len(variants) for category, variants in self.model_variants.items()
            },
        }

        # Add scale statistics per category
        stats["scale_stats"] = {}
        for category, models in self.placed_models.items():
            if models:
                scales = [m[3] for m in models]
                stats["scale_stats"][category] = {
                    "min": float(np.min(scales)),
                    "max": float(np.max(scales)),
                    "mean": float(np.mean(scales)),
                }

        return stats

    def get_placement_density_map(self, resolution: int = 50) -> Dict[str, np.ndarray]:
        """Generate density maps for visualization/debugging.

        Args:
            resolution: Grid resolution for density calculation.

        Returns:
            Dictionary of category -> 2D density array.
        """
        density_maps = {}

        # Get bounds from placed models
        all_positions = []
        for models in self.placed_models.values():
            all_positions.extend([(m[0], m[1]) for m in models])

        if not all_positions:
            return density_maps

        all_positions = np.array(all_positions)
        min_x, max_x = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        min_y, max_y = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])

        for category, models in self.placed_models.items():
            if not models:
                density_maps[category] = np.zeros((resolution, resolution))
                continue

            density = np.zeros((resolution, resolution))
            for x, y, z, scale in models:
                # Convert to grid coordinates
                gx = int((x - min_x) / (max_x - min_x + 1e-6) * (resolution - 1))
                gy = int((y - min_y) / (max_y - min_y + 1e-6) * (resolution - 1))
                gx = np.clip(gx, 0, resolution - 1)
                gy = np.clip(gy, 0, resolution - 1)
                density[gy, gx] += 1

            density_maps[category] = density

        return density_maps