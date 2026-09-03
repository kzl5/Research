import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def points_on_sphere(center, radius=1.0, n_points=8, lat_range_deg=None):
    """Return n_points located at `radius` distance from `center` using the Fibonacci sphere.

    If `lat_range_deg` is provided as (min_deg, max_deg) it limits points to that latitude
    band around the equator (latitude measured from the equator, 0 deg = equator).
    """
    center = np.asarray(center, dtype=float)

    # Compute z-bounds from latitude band (latitude measured from equator)
    if lat_range_deg is not None:
        if len(lat_range_deg) != 2:
            raise ValueError('lat_range_deg must be (min_deg, max_deg)')
        min_lat_deg, max_lat_deg = float(lat_range_deg[0]), float(lat_range_deg[1])
        # convert lat (deg from equator) to z = sin(lat_rad)
        zmin = np.sin(np.deg2rad(min_lat_deg))
        zmax = np.sin(np.deg2rad(max_lat_deg))
        if zmin > zmax:
            zmin, zmax = zmax, zmin
    else:
        zmin, zmax = -1.0, 1.0

    # Fibonacci sphere adapted to restrict z to [zmin, zmax]
    n = n_points
    if n == 1:
        dirs = np.array([[0.0, 0.0, 1.0]])
    else:
        i = np.arange(0, n, dtype=float)
        # uniformly sample z within [zmin, zmax]
        z = zmin + (zmax - zmin) * (i + 0.5) / n
        phi = np.arccos(z)
        theta = np.pi * (1 + 5**0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        dirs = np.stack([x, y, z], axis=1)

    points = center + radius * dirs
    return points


# def set_axes_equal(ax):
#     """Set 3D plot axes to equal scale (works around Matplotlib limitation)."""
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()

#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)

#     plot_radius = 0.5 * max([x_range, y_range, z_range])

#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# def plot_waypoints(points, center=(0, 0, 0), arrows=True, save='waypoints.png'):
#     """Plot 3D points and optionally arrows from center to each point. Saves figure to `save`."""
#     points = np.asarray(points)
#     center = np.asarray(center)

#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='C0', s=60)
#     ax.scatter([center[0]], [center[1]], [center[2]], c='C3', s=80, marker='x', label='center')

#     if arrows:
#         # arrows from center to points
#         U = points[:, 0] - center[0]
#         V = points[:, 1] - center[1]
#         W = points[:, 2] - center[2]
#         ax.quiver(center[0], center[1], center[2], U, V, W, length=1.0, normalize=False, color='gray', linewidth=0.8)

#     # Label points
#     for i, p in enumerate(points):
#         ax.text(p[0], p[1], p[2], f'{i}', color='black')

#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_zlabel('Z (m)')
#     ax.set_title(f'Waypoints (radius={np.linalg.norm(points[0]-center):.2f} m from center)')

#     set_axes_equal(ax)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save, dpi=200)
#     print(f'Saved plot to {save}')
#     try:
#         plt.show(block=True)
#     except TypeError:
#         plt.show()



# Example: 12 waypoints 1m away from origin
center = np.array([0.0, 0.0, 0.0])
radius = 1.0
n_points = 36

# pts = points_on_sphere(center=center, radius=radius, n_points=n_points)
# print('Waypoints:\n', np.round(pts, 4))

# plot_waypoints(pts, center=center, arrows=True, save='waypoints.png')

# Example: restrict points to a latitude band around the equator (±45 degrees)
band_pts = points_on_sphere(center=center, radius=radius, n_points=n_points, lat_range_deg=(-45, 45))
print('\nBand waypoints (±45° from equator):\n', np.round(band_pts, 4))
plot_waypoints(band_pts, center=center, arrows=True, save='waypoints_band.png')
