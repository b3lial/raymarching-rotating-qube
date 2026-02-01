import pyglet
try:
    import numpy as np
except ImportError:
    print("Using system numpy")
    import sys
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
    import numpy as np

# Window dimensions
WIDTH = 320
HEIGHT = 255


class Ray:
    """
    Represents a parametrized ray in 3D space.

    A ray is defined as: P(t) = origin + t * direction
    where t >= 0
    """
    def __init__(self, origin, direction):
        """
        Args:
            origin: 3D position vector (Ortsvector) - where the ray starts
            direction: 3D direction vector (Richtungsvector) - direction the ray travels
        """
        self.origin = np.array(origin, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)

    def at(self, t):
        """
        Get the point along the ray at parameter t.

        Args:
            t: Distance along the ray (t >= 0)

        Returns:
            3D point at position origin + t * direction
        """
        return self.origin + t * self.direction

    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction})"


class Cube:
    """
    Represents a cube in world space.

    In object space, the cube is centered at (0, 0, 0) and extends from
    min_corner to max_corner.
    """
    def __init__(self, world_position, color=None, min_corner=None, max_corner=None):
        """
        Args:
            world_position: 3D position of the cube center in world space
            color: RGB color of the cube (default: [255, 0, 0] red)
            min_corner: Minimum corner in object space (default: [-1, -1, -1])
            max_corner: Maximum corner in object space (default: [1, 1, 1])
        """
        self.world_position = np.array(world_position, dtype=np.float32)
        self.color = np.array(color if color is not None else [255, 0, 0], dtype=np.uint8)

        # Object space bounds (centered at origin)
        if min_corner is None:
            min_corner = [-1.0, -1.0, -1.0]
        if max_corner is None:
            max_corner = [1.0, 1.0, 1.0]

        self.min_corner = np.array(min_corner, dtype=np.float32)
        self.max_corner = np.array(max_corner, dtype=np.float32)

        # Calculate half extents (distance from center to edge)
        self.half_extents = (self.max_corner - self.min_corner) / 2.0

    def __repr__(self):
        return f"Cube(world_position={self.world_position}, min={self.min_corner}, max={self.max_corner})"

# Global variables (initialized in main)
window = None
write_buffer = None
read_buffer = None
cube = None
rotation_angle = 0.0


def rotation_matrix_y(angle_degrees):
    """
    Build a 3x3 rotation matrix around the Y-axis.

    The rotation matrix for angle θ around Y is:
        [ cos(θ)   0   sin(θ) ]
        [   0      1     0    ]
        [ -sin(θ)  0   cos(θ) ]

    Args:
        angle_degrees: Rotation angle in degrees

    Returns:
        3x3 numpy rotation matrix
    """
    # Convert degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Calculate sin and cos
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)

    # Build the rotation matrix
    matrix = np.array([
        [ cos_a, 0.0, sin_a],
        [   0.0, 1.0,   0.0],
        [-sin_a, 0.0, cos_a]
    ], dtype=np.float32)

    return matrix


def world_to_object_space(ray, object):
    """
    Transform a ray from world space to an object's object space.

    In object space, the object is centered at (0, 0, 0).
    In world space, the object is centered at object.world_position.

    Args:
        ray: Ray in world space
        object: The object whose object space we're transforming to

    Returns:
        Ray in object space (where object is centered at origin)
    """
    # Transform origin: subtract the object's world position
    # This translates the ray so the object center becomes the origin
    object_space_origin = ray.origin - object.world_position

    # Transform direction: for pure translation, direction remains unchanged
    # (If we had rotation/scale, we'd need to apply inverse transform here)
    object_space_direction = ray.direction

    # Create and return the transformed ray
    return Ray(origin=object_space_origin, direction=object_space_direction)


def ray_aabb_intersection(ray, aabb_min, aabb_max):
    """
    Ray-AABB (Axis-Aligned Bounding Box) intersection using the slab method.

    Tests if a ray intersects an axis-aligned box and returns the intersection distances.
    The slab method treats the AABB as the intersection of three slabs (one per axis).

    Args:
        ray: Ray to test (in the same coordinate space as the AABB)
        aabb_min: Minimum corner of the AABB (3D point)
        aabb_max: Maximum corner of the AABB (3D point)

    Returns:
        (t_near, t_far) if intersection occurs, where:
            t_near: distance along ray to entry point
            t_far: distance along ray to exit point
        None if no intersection
    """
    # Initialize to extreme values
    t_min = -np.inf
    t_max = np.inf

    # Test intersection with each slab (x, y, z)
    for axis in range(3):
        # Get the ray direction component for this axis
        direction_component = ray.direction[axis]

        # Get the ray origin component for this axis
        origin_component = ray.origin[axis]

        # Check if ray is parallel to this slab (direction ≈ 0)
        if abs(direction_component) < 1e-8:
            # Ray is parallel to slab
            # Check if ray origin is outside the slab bounds
            if origin_component < aabb_min[axis] or origin_component > aabb_max[axis]:
                # Ray is parallel and outside - no intersection
                return None
            # Ray is parallel and inside - continue to next axis
        else:
            # Calculate intersection distances with the two planes of this slab
            # t1 = intersection with min plane, t2 = intersection with max plane
            t1 = (aabb_min[axis] - origin_component) / direction_component
            t2 = (aabb_max[axis] - origin_component) / direction_component

            # Ensure t1 is the near plane and t2 is the far plane
            if t1 > t2:
                t1, t2 = t2, t1

            # Update the overall intersection interval
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            # Check if the slabs don't overlap
            if t_min > t_max:
                return None

    # Check if intersection is behind the ray origin
    if t_max < 0:
        return None

    # Return the intersection distances
    # t_min is the entry point, t_max is the exit point
    return (t_min, t_max)


def pixel_to_normalized_coords(pixel_x, pixel_y):
    """
    Convert pixel coordinates to normalized coordinates [-1, 1] with aspect ratio correction.

    Formula:
        u = ((x + 0.5) / W) * 2 - 1
        v = 1 - ((y + 0.5) / H) * 2

    Args:
        pixel_x: Pixel x-coordinate (0 to WIDTH-1)
        pixel_y: Pixel y-coordinate (0 to HEIGHT-1)

    Returns:
        (u, v) in range approximately [-1, 1]
    """
    # Step 1: Add 0.5 to center the sample in the pixel
    x_centered = pixel_x + 0.5
    y_centered = pixel_y + 0.5

    # Step 2: Divide by width/height to get [0, 1] range (approximately)
    x_unit = x_centered / WIDTH
    y_unit = y_centered / HEIGHT

    # Step 3: Map to [-1, 1] range
    u = x_unit * 2.0 - 1.0

    # Step 4: For v, invert the y-axis
    v = 1.0 - y_unit * 2.0

    # Step 5: Apply aspect ratio correction to u
    # This ensures that a circle appears as a circle, not an ellipse
    aspect_ratio = WIDTH / HEIGHT
    u = u * aspect_ratio

    return u, v


def update_framebuffer():
    """Update the framebuffer data. Currently just black, will contain ray marching later."""
    global write_buffer

    # Camera position
    camera_origin = [0.0, 0.0, 0.0]

    # Build rotation matrix once per frame (same for all pixels)
    inverse_rotation = rotation_matrix_y(-rotation_angle)

    # Iterate over every pixel
    for pixel_y in range(HEIGHT):
        for pixel_x in range(WIDTH):
            # Get normalized coordinates for this pixel
            u, v = pixel_to_normalized_coords(pixel_x, pixel_y)

            # The screen plane is at z=1
            # Point on screen for this pixel is (u, v, 1)
            screen_point = [u, v, 1.0]

            # Ray direction from camera to screen point
            # Since camera is at origin, direction = screen_point - camera_origin = screen_point
            ray_direction = np.array(screen_point, dtype=np.float32)

            # Normalize the ray direction to have length 1
            # Length (magnitude) = sqrt(x^2 + y^2 + z^2)
            length = np.sqrt(np.sum(ray_direction ** 2))
            ray_direction_normalized = ray_direction / length

            # Create ray for this pixel with normalized direction
            ray = Ray(origin=camera_origin, direction=ray_direction_normalized)

            # Convert ray into object space of cube so we can calculate
            # whether they intersect
            object_space_ray = world_to_object_space(ray, cube)

            # Rotate the ray in object space (inverse rotation)
            # To make the cube appear to rotate, we rotate the ray in the opposite direction
            object_space_ray.origin = inverse_rotation @ object_space_ray.origin
            object_space_ray.direction = inverse_rotation @ object_space_ray.direction

            # Check whether a ray hits the cube
            result = ray_aabb_intersection(object_space_ray, cube.min_corner, cube.max_corner)
            if result:
                t_near, t_far = result

                # Use t_near (entry distance) for depth-based shading
                # Closer surfaces are brighter, farther surfaces are dimmer
                # t_near is the distance from the camera to the cube surface
                max_distance = 10.0
                intensity = 1.0 - (t_near / max_distance)
                intensity = max(0.0, min(1.0, intensity))

                # Apply intensity to the cube's color
                red = int(cube.color[0] * intensity)
                green = int(cube.color[1] * intensity)
                blue = int(cube.color[2] * intensity)

                write_buffer[pixel_y, pixel_x] = [red, green, blue]
            else:
                # Ray misses the cube - black background
                write_buffer[pixel_y, pixel_x] = [0, 0, 0]


def on_draw():
    """Called when the window needs to be redrawn."""
    global window, read_buffer

    window.clear()

    # Convert numpy array to pyglet image
    # Flip vertically because OpenGL origin is bottom-left
    image_data = pyglet.image.ImageData(
        WIDTH,
        HEIGHT,
        'RGB',
        read_buffer.tobytes(),
        pitch=-WIDTH * 3  # Negative pitch to flip vertically
    )

    # Draw the image
    image_data.blit(0, 0)


def update(dt):
    """Called every frame for animations/updates."""
    global write_buffer, read_buffer, rotation_angle

    # Update the write buffer with new ray marching data
    update_framebuffer()

    # Swap buffers atomically
    write_buffer, read_buffer = read_buffer, write_buffer

    # Increment rotation angle by 1 degree per frame
    rotation_angle = (rotation_angle + 1.0) % 360.0


def initialize():
    """Initialize all global resources."""
    global window, write_buffer, read_buffer, cube

    # Create window
    window = pyglet.window.Window(width=WIDTH, height=HEIGHT, caption="Ray Marching")

    # Double buffering: one buffer for drawing, one for updating
    write_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    read_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Create a cube in world space
    # Object space: from (-1, -1, -1) to (1, 1, 1)
    # World space: positioned at (0, 0, 5)
    cube = Cube(world_position=[0.0, 0.0, 3.0])

    # Register event handlers
    window.event(on_draw)

    # Schedule update function to be called every frame
    pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS


def main():
    """Main entry point for the ray marching application."""
    print("Starting ray marching window...")
    print(f"Window size: {WIDTH}x{HEIGHT}")

    # Initialize all resources
    initialize()

    # Start the application
    pyglet.app.run()


if __name__ == '__main__':
    main()
