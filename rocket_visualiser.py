# rocket_visualiser.py
#
# Required packages — install with:
#   pip install pyserial pygame numpy

import sys
import threading
import math
from collections import deque

import numpy as np
import pygame
import serial

# ─── Serial configuration ─────────────────────────────────────────────────────
PORT = 'COM3'
BAUD = 115200

# ─── Buffer ───────────────────────────────────────────────────────────────────
HISTORY = 1500  # 500 Hz × 3 s

data = {
    'gx':    deque(maxlen=HISTORY),
    'gy':    deque(maxlen=HISTORY),
    'gz':    deque(maxlen=HISTORY),
    'pitch': deque(maxlen=HISTORY),
    'roll':  deque(maxlen=HISTORY),
    'yaw':   deque(maxlen=HISTORY),
}

latest = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
          'gx': 0.0, 'gy': 0.0, 'gz': 0.0}

# ─── Serial reader thread ─────────────────────────────────────────────────────
def serial_reader():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException:
        print(f"\nCould not open {PORT}. Check your COM port setting at the "
              f"top of this file and verify the ESP32 is connected.")
        sys.exit(1)

    print(f"Opened {PORT} at {BAUD} baud.")
    while True:
        try:
            raw = ser.readline().decode('ascii', errors='ignore').strip()
        except Exception:
            continue
        if not raw or not (raw[0].isdigit() or raw[0] == '-'):
            continue
        parts = raw.split(',')
        if len(parts) != 7:
            continue
        try:
            gx, gy, gz, pitch, roll, yaw = (float(p) for p in parts[:6])
        except ValueError:
            continue
        data['gx'].append(gx);      data['gy'].append(gy);    data['gz'].append(gz)
        data['pitch'].append(pitch); data['roll'].append(roll); data['yaw'].append(yaw)
        latest['gx'] = gx;  latest['gy'] = gy;  latest['gz'] = gz
        latest['pitch'] = pitch; latest['roll'] = roll; latest['yaw'] = yaw

threading.Thread(target=serial_reader, daemon=True).start()

# ─── Pygame setup ─────────────────────────────────────────────────────────────
pygame.init()
W, H = 1200, 680
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption('Rocket IMU Visualiser')
clock = pygame.time.Clock()

BG     = (13, 13, 18)
WHITE  = (220, 220, 220)
GREY   = (80,  80,  90)
DGREY  = (40,  40,  48)
RED    = (220,  60,  60)
GREEN  = ( 60, 200,  80)
BLUE   = ( 60, 120, 220)
ORANGE = (220, 140,  40)
PURPLE = (160,  60, 210)
CYAN   = ( 40, 200, 200)
YELLOW = (220, 200,  50)
DARK   = ( 28,  28,  36)

font_lg = pygame.font.SysFont('consolas', 26, bold=True)
font_md = pygame.font.SysFont('consolas', 16)
font_sm = pygame.font.SysFont('consolas', 12)

# ─── 3-D rocket geometry (body frame: nose points along +Y, right = +X, up = -Z)
# All coordinates in arbitrary units; the view is scaled to fit the panel.
# We define faces as lists of vertex indices so we can sort by depth (painter's
# algorithm) and shade each face by how much it faces the light source.

def build_rocket_mesh():
    """
    Returns (verts, faces).
    verts : (N,3) float32 array in body frame
    faces : list of (indices_tuple, base_color)
    """
    verts = []
    faces = []

    def v(*xyz):
        verts.append(xyz)
        return len(verts) - 1

    # ── Body cylinder ─────────────────────────────────────────────────────
    R   = 0.18   # body radius
    y0  = -1.0   # bottom of body
    y1  =  0.7   # top of body (nose cone base)
    N   = 12     # sides

    bot_ring = []
    top_ring = []
    for i in range(N):
        a = 2 * math.pi * i / N
        x, z = R * math.cos(a), R * math.sin(a)
        bot_ring.append(v(x, y0, z))
        top_ring.append(v(x, y1, z))

    for i in range(N):
        j = (i + 1) % N
        faces.append(((bot_ring[i], bot_ring[j], top_ring[j], top_ring[i]),
                      (70, 70, 160)))

    # ── Nose cone ─────────────────────────────────────────────────────────
    tip = v(0, 1.5, 0)
    for i in range(N):
        j = (i + 1) % N
        faces.append(((top_ring[i], top_ring[j], tip),
                      (200, 120, 30)))

    # ── Three fins equally spaced at 120° ─────────────────────────────────
    fin_colors = [(40, 180, 60), (40, 160, 80), (40, 200, 50)]
    for k in range(3):
        a    = 2 * math.pi * k / 3
        outx = math.cos(a) * 0.65
        outz = math.sin(a) * 0.65
        innx = math.cos(a) * R
        innz = math.sin(a) * R

        p0 = v(innx, y0 + 0.45, innz)   # root top
        p1 = v(innx, y0,        innz)   # root bottom
        p2 = v(outx, y0 - 0.05, outz)   # tip
        faces.append(((p0, p1, p2), fin_colors[k]))

    return np.array(verts, dtype=np.float32), faces

ROCKET_VERTS, ROCKET_FACES = build_rocket_mesh()

# ─── Rotation matrices ────────────────────────────────────────────────────────
def rot_matrix(pitch_deg, roll_deg, yaw_deg):
    p, r, y = math.radians(pitch_deg), math.radians(roll_deg), math.radians(yaw_deg)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    cy, sy = math.cos(y), math.sin(y)

    Ry = np.array([[ cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]],  np.float32)
    Rz = np.array([[cy,-sy, 0],[sy, cy, 0], [0,   0,  1]], np.float32)
    return Rz @ Ry @ Rx

# ─── Perspective projection ───────────────────────────────────────────────────
# Camera sits on the +Z axis looking toward the origin.
CAM_DIST = 4.5   # camera distance from origin
FOV_SCALE = 260  # pixels per unit at unit distance — controls zoom

def project(pt3d, cx, cy):
    """Project a 3-D point to 2-D screen coords (perspective divide)."""
    x, y, z = pt3d
    denom = CAM_DIST - z
    if denom < 0.1:
        denom = 0.1
    sx = cx + x / denom * FOV_SCALE * (CAM_DIST / 3)
    sy = cy - y / denom * FOV_SCALE * (CAM_DIST / 3)
    return sx, sy

def face_depth(face_verts_3d):
    """Average Z of face vertices — used for painter's algorithm sort."""
    return sum(v[2] for v in face_verts_3d) / len(face_verts_3d)

def face_normal_dot(face_verts_3d):
    """
    Dot product of face normal with light direction (0,1,-1 normalised).
    Used for simple diffuse shading so faces facing the light are brighter.
    """
    v0 = np.array(face_verts_3d[0])
    v1 = np.array(face_verts_3d[1])
    v2 = np.array(face_verts_3d[2])
    n  = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n)
    if ln < 1e-6:
        return 0.5
    n = n / ln
    light = np.array([0.3, 0.7, -0.6], np.float32)
    light /= np.linalg.norm(light)
    return float(np.clip(np.dot(n, light), 0, 1))

def draw_rocket_3d(surface, rect, pitch_deg, roll_deg, yaw_deg):
    """
    Rotate the rocket mesh, project to 2-D, sort faces back-to-front,
    shade each face, and draw it — all in pure numpy + pygame.
    """
    rx, ry, rw, rh = rect
    cx = rx + rw // 2
    cy = ry + rh // 2

    # Background panel
    pygame.draw.rect(surface, DARK, rect)

    R = rot_matrix(pitch_deg, roll_deg, yaw_deg)
    # Rotate all vertices at once: (N,3) @ (3,3)^T = (N,3)
    rotated = (R @ ROCKET_VERTS.T).T   # shape (N, 3)

    # Build list of (depth, 2d_screen_pts, base_color, shading)
    draw_list = []
    for vert_indices, base_color in ROCKET_FACES:
        pts3d = [rotated[i] for i in vert_indices]
        depth = face_depth(pts3d)
        shade = 0.35 + 0.65 * face_normal_dot(pts3d)  # ambient + diffuse
        pts2d = [project(p, cx, cy) for p in pts3d]
        draw_list.append((depth, pts2d, base_color, shade))

    # Painter's algorithm: draw back faces first
    draw_list.sort(key=lambda x: x[0])

    for depth, pts2d, base_color, shade in draw_list:
        fill = tuple(int(c * shade) for c in base_color)
        edge = tuple(min(255, int(c * shade * 1.4)) for c in base_color)
        ipts = [(int(p[0]), int(p[1])) for p in pts2d]
        if len(ipts) >= 3:
            pygame.draw.polygon(surface, fill, ipts)
            pygame.draw.polygon(surface, edge, ipts, 1)

    # World-frame axis arrows (fixed in world space, not rotated)
    def draw_axis(direction, color, label):
        d = np.array(direction, np.float32)
        scale = 0.9
        tip3d = d * scale
        sx, sy = project(tip3d, cx, cy)
        ox, oy = project([0, 0, 0], cx, cy)
        pygame.draw.line(surface, color, (int(ox), int(oy)), (int(sx), int(sy)), 2)
        surface.blit(font_sm.render(label, True, color),
                     (int(sx) + 3, int(sy) - 6))

    draw_axis([1, 0, 0], RED,    'X')
    draw_axis([0, 1, 0], GREEN,  'Y')
    draw_axis([0, 0, -1], BLUE,  'Z')  # Z toward viewer in camera frame

    # Border
    pygame.draw.rect(surface, GREY, rect, 1)


# ─── Scrolling line graph ─────────────────────────────────────────────────────
def draw_graph(surface, rect, series, colors, labels):
    x, y, w, h = rect
    pygame.draw.rect(surface, DARK, rect)
    pygame.draw.rect(surface, GREY, rect, 1)
    mid_y = y + h // 2
    pygame.draw.line(surface, (55, 55, 55), (x, mid_y), (x + w, mid_y), 1)

    # Snapshot and length-match all series
    arrays = [np.array(list(s)) for s in series]
    n = min((len(a) for a in arrays), default=0)
    if n < 2:
        return
    arrays = [a[-n:] for a in arrays]

    all_vals = np.concatenate(arrays)
    peak = max(abs(all_vals).max(), 1.0)

    def to_px(val):
        return int(mid_y - (val / peak) * (h // 2 - 4))

    for arr, color in zip(arrays, colors):
        pts = [(x + int(i * w / (n - 1)),
                max(y + 1, min(y + h - 1, to_px(v))))
               for i, v in enumerate(arr)]
        pygame.draw.lines(surface, color, False, pts, 1)

    for i, (label, color) in enumerate(zip(labels, colors)):
        surface.blit(font_sm.render(label, True, color), (x + 5 + i * 65, y + 3))


# ─── Attitude indicator ───────────────────────────────────────────────────────
def draw_attitude(surface, cx, cy, r, pitch_deg, roll_deg):
    clip_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
    cr, cc = r, r  # centre within the clip surface

    roll_rad = math.radians(roll_deg)
    pitch_px = int(pitch_deg / 90.0 * r)

    # Sky fill
    pygame.draw.circle(clip_surf, (25, 60, 130), (cr, cc), r)

    # Ground fill — draw a large rect rotated by roll offset by pitch
    # Build horizon polygon: half circle of ground colour
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    pdx = -sin_r * pitch_px
    pdy =  cos_r * pitch_px
    perp = np.array([-sin_r, cos_r])
    horiz_pts = []
    for deg in range(0, 361, 6):
        a = math.radians(deg)
        px = cr + int(math.cos(a) * r)
        py = cc + int(math.sin(a) * r)
        vec = np.array([px - (cr + pdx), py - (cc + pdy)])
        if np.dot(vec, perp) > 0:
            horiz_pts.append((px, py))
    if len(horiz_pts) >= 3:
        pygame.draw.polygon(clip_surf, (110, 75, 35), horiz_pts)

    # Horizon line
    p1 = (int(cr - cos_r * r + pdx), int(cc - sin_r * r + pdy))
    p2 = (int(cr + cos_r * r + pdx), int(cc + sin_r * r + pdy))
    pygame.draw.line(clip_surf, YELLOW, p1, p2, 2)

    # Clip to circle mask
    mask = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (cr, cc), r)
    clip_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    surface.blit(clip_surf, (cx - r, cy - r))

    # Overlay: fixed aircraft symbol
    pygame.draw.line(surface, WHITE, (cx - 22, cy), (cx + 22, cy), 2)
    pygame.draw.line(surface, WHITE, (cx, cy - 5), (cx, cy + 5), 2)
    pygame.draw.circle(surface, GREY, (cx, cy), r, 1)


# ─── Main loop ────────────────────────────────────────────────────────────────
# Layout:
#   [  3-D rocket (left, wide)  ] [ numbers ] [ graphs (right) ]
ROCKET_PANEL = (0, 0, 420, H)
NUM_X  = 430
GRAPH_X = 730
GRAPH_W = W - GRAPH_X - 8

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    screen.fill(BG)

    p = latest['pitch']
    r = latest['roll']
    y = latest['yaw']

    # ── 3-D rocket ────────────────────────────────────────────────────────
    draw_rocket_3d(screen, ROCKET_PANEL, p, r, y)

    lbl = font_md.render('3D ORIENTATION', True, GREY)
    screen.blit(lbl, (ROCKET_PANEL[2] // 2 - lbl.get_width() // 2, 8))

    # ── Attitude indicator (bottom of 3D panel) ───────────────────────────
    draw_attitude(screen, 210, H - 72, 58, p, r)
    screen.blit(font_sm.render('attitude', True, GREY), (182, H - 11))

    # ── Numeric readout ───────────────────────────────────────────────────
    screen.blit(font_md.render('ANGLES', True, GREY), (NUM_X, 12))
    for i, (name, val, color) in enumerate([
        ('PITCH', p, ORANGE),
        ('ROLL ', r, PURPLE),
        ('YAW  ', y, CYAN),
    ]):
        yt = 42 + i * 68
        screen.blit(font_md.render(name, True, color), (NUM_X, yt))
        screen.blit(font_lg.render(f'{val:+8.2f}°', True, WHITE), (NUM_X, yt + 20))

    screen.blit(font_md.render('GYRO (°/s)', True, GREY), (NUM_X, 262))
    for i, (name, val, color) in enumerate([
        ('gx', latest['gx'], RED),
        ('gy', latest['gy'], GREEN),
        ('gz', latest['gz'], BLUE),
    ]):
        yt = 290 + i * 60
        screen.blit(font_md.render(name, True, color), (NUM_X, yt))
        screen.blit(font_lg.render(f'{val:+7.3f}', True, WHITE), (NUM_X, yt + 20))

    screen.blit(font_sm.render(f'{clock.get_fps():.0f} fps', True, DGREY),
                (NUM_X, H - 18))

    # ── Scrolling graphs ──────────────────────────────────────────────────
    screen.blit(font_md.render('GYRO RATES', True, GREY), (GRAPH_X, 8))
    draw_graph(screen, (GRAPH_X, 30, GRAPH_W, 190),
               [data['gx'], data['gy'], data['gz']],
               [RED, GREEN, BLUE], ['gx', 'gy', 'gz'])

    screen.blit(font_md.render('ANGLES', True, GREY), (GRAPH_X, 232))
    draw_graph(screen, (GRAPH_X, 254, GRAPH_W, 190),
               [data['pitch'], data['roll'], data['yaw']],
               [ORANGE, PURPLE, CYAN], ['pitch', 'roll', 'yaw'])

    # Mini gx/gy noise scatter
    screen.blit(font_md.render('NOISE SCATTER', True, GREY), (GRAPH_X, 454))
    sr = pygame.Rect(GRAPH_X, 474, GRAPH_W, H - 482)
    pygame.draw.rect(screen, DARK, sr)
    pygame.draw.rect(screen, GREY, sr, 1)
    if len(data['gx']) > 5:
        sg = np.array(list(data['gx']))[-200:]
        sh = np.array(list(data['gy']))[-200:]
        pk = max(abs(sg).max(), abs(sh).max(), 0.05)
        scx, scy = sr.centerx, sr.centery
        srx2, sry2 = sr.width // 2 - 4, sr.height // 2 - 4
        pygame.draw.line(screen, DGREY, (scx, sr.top), (scx, sr.bottom), 1)
        pygame.draw.line(screen, DGREY, (sr.left, scy), (sr.right, scy), 1)
        for gxv, gyv in zip(sg, sh):
            ppx = int(scx + gxv / pk * srx2)
            ppy = int(scy - gyv / pk * sry2)
            if sr.collidepoint(ppx, ppy):
                screen.set_at((ppx, ppy), CYAN)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
