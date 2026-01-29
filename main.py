import pyglet
try:
    import numpy as np
except ImportError:
    print("Using system numpy")
    import sys
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
    import numpy as np

# Window dimensions
WIDTH = 800
HEIGHT = 600

# Create window
window = pyglet.window.Window(width=WIDTH, height=HEIGHT, caption="Ray Marching")

# Double buffering: one buffer for drawing, one for updating
write_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
read_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def update_framebuffer():
    """Update the framebuffer data. Currently just black, will contain ray marching later."""
    global write_buffer

    # Compute your ray marching here, writing to write_buffer
    # Example: Draw a red square
    write_buffer[100:200, 100:200] = [255, 0, 0]  # Red square

    # Later, this is where you'll loop through each pixel and compute the ray marching:
    # for y in range(HEIGHT):
    #     for x in range(WIDTH):
    #         color = ray_march(x, y)
    #         write_buffer[y, x] = color


@window.event
def on_draw():
    """Called when the window needs to be redrawn."""
    global read_buffer

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
    global write_buffer, read_buffer

    # Update the write buffer with new ray marching data
    update_framebuffer()

    # Swap buffers atomically
    write_buffer, read_buffer = read_buffer, write_buffer


# Schedule update function to be called every frame
pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS


if __name__ == '__main__':
    print("Starting ray marching window...")
    print(f"Window size: {WIDTH}x{HEIGHT}")
    pyglet.app.run()
