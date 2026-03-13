import pyglet

WIDTH, HEIGHT = 800, 600
window = pyglet.window.Window(WIDTH, HEIGHT, "Infinity Grid")

HORIZON = HEIGHT * 0.42   # lower horizon = more floor space
VP_X    = WIDTH  * 0.5
NUM_V   = 40
NUM_H   = 20
DECAY   = 0.75            # geometric spacing: controls how fast lines bunch at horizon
SPEED   = 1.5             # lines scrolled per second
# vertical lines span beyond screen width so edges are fully covered
X_MIN   = -WIDTH * 0.5
X_MAX   =  WIDTH * 1.5

COLOR_LINE = (200, 200, 200)
COLOR_DIM  = (60, 60, 60)

scroll_t = 0.0

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

@window.event
def on_draw():
    window.clear()
    batch = pyglet.graphics.Batch()
    lines = []

    # Horizontal lines — geometric depth spacing + scroll
    for i in range(NUM_H + 1):
        eff_i = i - scroll_t
        y = HORIZON - HORIZON * (DECAY ** eff_i)
        if y < 0 or y >= HORIZON:
            continue
        t = y / HORIZON   # 0 = near viewer, 1 = horizon
        color = lerp_color(COLOR_LINE, COLOR_DIM, t)
        x_left  = X_MIN * (1 - t) + VP_X * t
        x_right = X_MAX * (1 - t) + VP_X * t
        lines.append(pyglet.shapes.Line(x_left, y, x_right, y, color=color, batch=batch))

    # Vertical lines — span beyond screen width, all converge to vanishing point
    for i in range(NUM_V + 1):
        x_bottom = X_MIN + (X_MAX - X_MIN) * i / NUM_V
        lines.append(pyglet.shapes.Line(x_bottom, 0, VP_X, HORIZON, color=COLOR_LINE, batch=batch))

    # Horizon glow — fading band above and below the horizon
    GLOW_COLOR  = (120, 200, 255)
    GLOW_STEPS  = 40
    GLOW_SPREAD = 80   # pixels the glow extends above/below horizon

    for g in range(GLOW_STEPS):
        t_glow = g / GLOW_STEPS
        alpha  = int(160 * (1 - t_glow) ** 2)
        c = (*GLOW_COLOR, alpha)
        offset = int(GLOW_SPREAD * t_glow)
        lines.append(pyglet.shapes.Line(0, HORIZON + offset, WIDTH, HORIZON + offset, color=c, batch=batch))
        lines.append(pyglet.shapes.Line(0, HORIZON - offset, WIDTH, HORIZON - offset, color=c, batch=batch))

    batch.draw()

def update(dt):
    global scroll_t
    scroll_t = (scroll_t + dt * SPEED) % 1.0

pyglet.clock.schedule_interval(update, 1 / 60)

if __name__ == '__main__':
    pyglet.app.run()
