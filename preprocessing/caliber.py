import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


GEOMETRY = {'neighborhood_radius': 5.0, 'minimum_points': 3, 'minimum_pca_ratio': 3.0, 'sampling_step': 0.25, 'boundary_level': 0.5, 'percentile': 99.0, 'connectivity': 8}


def local_caliber(mask, g):
    mask = np.asarray(mask, dtype=bool)
    if g["connectivity"] != 8:
        raise ValueError("This implementation uses 8-connectivity")

    labels, components = ndi.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    skeleton = skeletonize(mask)
    raw = np.zeros(mask.shape, dtype=np.float32)
    valid = np.zeros(mask.shape, dtype=bool)
    samples = []

    step = float(g["sampling_step"])
    level = float(g["boundary_level"])
    ray = np.arange(
        0.0, np.hypot(*mask.shape) + step, step
    )
    height, width = mask.shape

    for component_id in range(1, components + 1):
        region = labels == component_id
        points = np.argwhere(skeleton & region)
        if len(points) < g["minimum_points"]:
            continue

        tree = cKDTree(points)
        region_float = region.astype(np.float32)
        good_points = []
        good_widths = []

        for point in points:
            neighbors = points[
                tree.query_ball_point(
                    point, g["neighborhood_radius"]
                )
            ]
            if len(neighbors) < g["minimum_points"]:
                continue

            eigenvalues, eigenvectors = np.linalg.eigh(
                np.cov(neighbors.T)
            )
            if eigenvalues[-1] <= 1e-8:
                continue
            ratio = eigenvalues[-1] / max(
                eigenvalues[0], 1e-8
            )
            if ratio < g["minimum_pca_ratio"]:
                continue

            # Coordinates here are (row, column).
            tangent = eigenvectors[:, -1]
            normal = np.array(
                [-tangent[1], tangent[0]], dtype=np.float64
            )
            distances = []

            for sign in (-1, 1):
                coordinates = (
                    point[:, None]
                    + sign * normal[:, None] * ray
                )
                in_image = (
                    (coordinates[0] >= 0)
                    & (coordinates[0] <= height - 1)
                    & (coordinates[1] >= 0)
                    & (coordinates[1] <= width - 1)
                )
                q = coordinates[:, in_image]

                # Bilinear interpolation of the binary component.
                values = ndi.map_coordinates(
                    region_float, q,
                    order=1, mode="constant",
                    cval=0.0, prefilter=False,
                )
                crossings = np.flatnonzero(values < level)
                if not len(crossings):
                    # No observed boundary before leaving the image.
                    break

                j = int(crossings[0])
                if j == 0:
                    break

                # Interpolate the threshold crossing within this step.
                fraction = (
                    (values[j - 1] - level)
                    / (values[j - 1] - values[j])
                )
                distance = ray[j - 1] + step * fraction
                distances.append(float(distance))

            if len(distances) != 2:
                continue

            caliber = sum(distances)
            if not np.isfinite(caliber) or caliber <= 0:
                continue

            good_points.append(point)
            good_widths.append(caliber)

        if not good_points:
            # Entire component has no reliable centerline sample.
            continue

        vessel_pixels = np.argwhere(region)
        _, indices = cKDTree(
            np.asarray(good_points)
        ).query(vessel_pixels)

        widths = np.asarray(good_widths, dtype=np.float32)
        rr, cc = vessel_pixels.T

        # Assignment is strictly restricted to this component.
        raw[rr, cc] = widths[indices]
        valid[rr, cc] = True
        samples.extend(good_widths)

    return raw, valid, np.asarray(samples, dtype=np.float32)
