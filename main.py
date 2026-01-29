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

# Create a black framebuffer (RGBA format)
framebuffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def update_framebuffer():
    """Update the framebuffer data. Currently just black, will contain ray marching later."""
    # For now, framebuffer stays black
    # Later this will be where we compute ray marching results
    pass


@window.event
def on_draw():
    """Called when the window needs to be redrawn."""
    window.clear()

    # Update framebuffer
    update_framebuffer()

    # Convert numpy array to pyglet image
    # Flip vertically because OpenGL origin is bottom-left
    image_data = pyglet.image.ImageData(
        WIDTH,
        HEIGHT,
        'RGB',
        framebuffer.tobytes(),
        pitch=-WIDTH * 3  # Negative pitch to flip vertically
    )

    # Draw the image
    image_data.blit(0, 0)


def update(dt):
    """Called every frame for animations/updates."""
    pass


# Schedule update function to be called every frame
pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS


if __name__ == '__main__':
    print("Starting ray marching window...")
    print(f"Window size: {WIDTH}x{HEIGHT}")
    pyglet.app.run()
