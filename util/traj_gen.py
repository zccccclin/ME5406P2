import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import splprep, splev

def generate_ellipse_trajectory(start_coord, a, b, n_points):
    # Generate ellipse points in 2D
    t = np.linspace(0, 2*np.pi, n_points)
    x = -a * np.cos(t)
    y = b * np.sin(t)

    # Add z coordinate for 3D
    z = np.zeros_like(x) + start_coord[2]

    # Combine x, y, z coordinates into a 3D point cloud
    points_3d = np.column_stack([x, y, z])

    # Use scipy's spline interpolation to generate a smooth trajectory
    tck, u = splprep(points_3d.T, s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), n_points)
    points_3d_smooth = splev(u_new, tck, der=0)

    points_3d_smooth = np.array(points_3d_smooth)
    # Shift the ellipse to start at the desired coordinate
    points_3d_shifted = points_3d_smooth - points_3d_smooth[:,0].reshape(-1,1) + start_coord.reshape(-1,1)

    return points_3d_shifted

def generate_rectangular_trajectory(start_coord, length, width, n_points):
    height = start_coord[2]

    side1_n_points = int(n_points * length / (length + width)) // 2
    side2_n_points = int(n_points * width / (length + width)) // 2  
    # Generate x, y coordinates
    x1 = np.linspace(start_coord[0], start_coord[0] + length / 2, num=side1_n_points // 2)
    y1 = np.zeros_like(x1) + start_coord[1]
    z1 = np.zeros_like(x1) + start_coord[2]
    side1 = np.column_stack([x1, y1, z1])
    y2 = np.linspace(start_coord[1], start_coord[1] + width, num=side2_n_points)
    x2 = np.zeros_like(y2) + start_coord[0] + length / 2
    z2 = np.zeros_like(y2) + start_coord[2]
    side2 = np.column_stack([x2, y2, z2])
    x3 = np.linspace(start_coord[0] + length / 2, start_coord[0] - length / 2, num=side1_n_points)
    y3 = np.zeros_like(x3) + start_coord[1] + width 
    z3 = np.zeros_like(x3) + start_coord[2]
    side3 = np.column_stack([x3, y3, z3])
    y4 = np.linspace(start_coord[1] + width, start_coord[1], num=side2_n_points)
    x4 = np.zeros_like(y4) + start_coord[0] - length / 2
    z4 = np.zeros_like(y4) + start_coord[2]
    side4 = np.column_stack([x4, y4, z4])
    x5 = np.linspace(start_coord[0] - length / 2, start_coord[0] , num=side1_n_points // 2)
    y5 = np.zeros_like(x5) + start_coord[1]
    z5 = np.zeros_like(x5) + start_coord[2]
    side5 = np.column_stack([x5, y5, z5])
    points_3d = np.concatenate([side1, side2, side3, side4, side5])
    points_3d = points_3d.T
    print(points_3d.shape)
    return points_3d



# Define ellipse parameters, number of points, and starting coordinate
a = np.random.uniform(0.05, 0.35)
b = np.random.uniform(0.05, 0.35)
c = np.random.uniform(0.1, 0.5)
n_points = 2000
start_coord = np.array([0.3, 0, 1])

# Generate ellipse trajectory starting at start_coord
points_3d = generate_ellipse_trajectory(start_coord, a, b, n_points)
#points_3d = generate_rectangular_trajectory(start_coord, a, b, n_points)
print(points_3d)
# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot ellipse points
ax.scatter(points_3d[0], points_3d[1], points_3d[2], label='Trajectory', s=1)

# Plot starting coordinate as a red sphere
ax.scatter(start_coord[0], start_coord[1], start_coord[2], s=100, color='r')

# Set plot limits and labels
ax.set_xlim(0.3, 0.7+0.3)
ax.set_ylim(-0.6, 0.6)
ax.set_zlim(0.8, 1.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
