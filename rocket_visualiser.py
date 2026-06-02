# rocket_visualiser.py
#
# Required packages — install with:
#   pip install pyserial pygame numpy

import sys
import threading
import time
import math
from collections import deque

import numpy as np
import pygame
import serial

# ─── Serial configuration ─────────────────────────────────────────────────────
PORT = 'COM3'
BAUD = 115200

# ─── Buffer: 3 seconds of history for the scrolling graph ────────────────────
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

        data['gx'].append(gx);    data['gy'].append(gy);    data['gz'].append(gz)
        data['pitch'].append(pitch); data['roll'].append(roll); data['yaw'].append(yaw)
        latest['gx'] = gx;  latest['gy'] = gy;  latest['gz'] = gz
        latest['pitch'] = pitch; latest['roll'] = roll; latest['yaw'] = yaw

threading.Thread(target=serial_reader, daemon=True).start()

# ─── Pygame setup ─────────────────────────────────────────────────────────────
pygame.init()
W, H = 1100, 620
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption('Rocket IMU Visualiser')
clock = pygame.time.Clock()

BG      = (15, 15, 20)
WHITE   = (220, 220, 220)
GREY    = (80, 80, 90)
RED     = (220, 60, 60)
GREEN   = (60, 200, 80)
BLUE    = (60, 120, 220)
ORANGE  = (220, 140, 40)
PURPLE  = (160, 60, 210)
CYAN    = (40, 200, 200)
YELLOW  = (220, 200, 50)
DARK    = (30, 30, 38)

font_lg = pygame.font.SysFont('consolas', 28, bold=True)
font_md = pygame.font.SysFont('consolas', 18)
font_sm = pygame.font.SysFont('consolas', 13)

# ─── Draw the rocket silhouette rotated by roll, tilted by pitch ──────────────
def draw_rocket(surface, cx, cy, pitch_deg, roll_deg):
    """
    Simple 2D rocket:
      - roll  rotates the whole shape
      - pitch tilts it (shown as additional rotation so the nose tips forward)
    Combined angle = roll + pitch so both axes are visible at once.
    """
    angle = math.radians(-(roll_deg + pitch_deg * 0.5))
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rot(x, y):
        return (cx + x * cos_a - y * sin_a,
                cy + x * sin_a + y * cos_a)

    # Body
    body_pts = [rot(-14, -70), rot(14, -70), rot(14, 55), rot(-14, 55)]
    pygame.draw.polygon(surface, (80, 80, 160), body_pts)
    pygame.draw.polygon(surface, (130, 130, 220), body_pts, 2)

    # Nose cone
    nose_pts = [rot(-14, -70), rot(14, -70), rot(0, -130)]
    pygame.draw.polygon(surface, ORANGE, nose_pts)
    pygame.draw.polygon(surface, YELLOW, nose_pts, 2)

    # Fins (three: left, right, bottom)
    for fx, fy_root, fx2, fy2 in [
        (-14, 30,  -55, 75),   # left fin
        ( 14, 30,   55, 75),   # right fin
        (  0, 55,    0, 95),   # bottom nozzle
    ]:
        fin_pts = [rot(fx, fy_root - 30), rot(fx, fy_root), rot(fx2, fy2)]
        pygame.draw.polygon(surface, GREEN, fin_pts)
        pygame.draw.polygon(surface, (100, 220, 100), fin_pts, 1)

    # Centre dot
    pygame.draw.circle(surface, WHITE, (int(cx), int(cy)), 3)


# ─── Draw a scrolling line graph ──────────────────────────────────────────────
def draw_graph(surface, rect, series, colors, labels, y_range=None):
    x, y, w, h = rect
    pygame.draw.rect(surface, DARK, rect)
    pygame.draw.rect(surface, GREY, rect, 1)

    # Zero line
    mid_y = y + h // 2
    pygame.draw.line(surface, (60, 60, 60), (x, mid_y), (x + w, mid_y), 1)

    n = min(len(s) for s in series)
    if n < 2:
        return

    # Snapshot
    arrays = [np.array(list(s))[-n:] for s in series]

    if y_range is None:
        all_vals = np.concatenate(arrays)
        peak = max(abs(all_vals.max()), abs(all_vals.min()), 1.0)
    else:
        peak = y_range

    def to_px(val):
        return int(mid_y - (val / peak) * (h // 2 - 4))

    for arr, color in zip(arrays, colors):
        pts = []
        for i, v in enumerate(arr):
            px = x + int(i * w / (n - 1))
            py = max(y + 1, min(y + h - 1, to_px(v)))
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, 1)

    # Legend
    for i, (label, color) in enumerate(zip(labels, colors)):
        t = font_sm.render(label, True, color)
        surface.blit(t, (x + 5 + i * 70, y + 4))


# ─── Attitude indicator (artificial horizon) ─────────────────────────────────
def draw_attitude(surface, cx, cy, r, pitch_deg, roll_deg):
    """Circle showing roll (horizon line angle) and pitch (horizon offset)."""
    pygame.draw.circle(surface, DARK, (cx, cy), r)
    pygame.draw.circle(surface, GREY, (cx, cy), r, 2)

    roll_rad  = math.radians(roll_deg)
    pitch_px  = int(pitch_deg / 90.0 * r)   # map ±90° to ±r pixels

    # Horizon line rotated by roll
    dx = math.cos(roll_rad) * r
    dy = math.sin(roll_rad) * r
    # Shift perpendicular to horizon by pitch
    pdx = -math.sin(roll_rad) * pitch_px
    pdy =  math.cos(roll_rad) * pitch_px

    p1 = (int(cx - dx + pdx), int(cy - dy + pdy))
    p2 = (int(cx + dx + pdx), int(cy + dy + pdy))

    # Sky above horizon
    pygame.draw.circle(surface, (30, 60, 120), (cx, cy), r - 2)
    # Clip horizon fill — draw ground polygon
    ground_pts = []
    for deg in range(0, 361, 5):
        a = math.radians(deg)
        px = cx + int(math.cos(a) * r)
        py = cy + int(math.sin(a) * r)
        # Check which side of the horizon line
        nx = -math.sin(roll_rad)
        ny =  math.cos(roll_rad)
        rel_x = px - (cx + pdx)
        rel_y = py - (cy + pdy)
        if rel_x * nx + rel_y * ny > 0:
            ground_pts.append((px, py))

    # Simpler: just draw the horizon line and label
    pygame.draw.line(surface, YELLOW, p1, p2, 2)

    # Centre cross
    pygame.draw.line(surface, WHITE, (cx - 20, cy), (cx + 20, cy), 2)
    pygame.draw.line(surface, WHITE, (cx, cy - 5), (cx, cy + 5), 2)

    # Clip to circle
    pygame.draw.circle(surface, GREY, (cx, cy), r, 2)


# ─── Main loop ────────────────────────────────────────────────────────────────
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

    # ── Left panel: rocket silhouette ─────────────────────────────────────
    panel_w = 280
    pygame.draw.rect(screen, DARK, (0, 0, panel_w, H))

    label = font_md.render('ORIENTATION', True, GREY)
    screen.blit(label, (panel_w // 2 - label.get_width() // 2, 12))

    draw_rocket(screen, panel_w // 2, H // 2 - 20, p, r)

    # Attitude indicator (small, bottom of left panel)
    draw_attitude(screen, panel_w // 2, H - 90, 60, p, r)
    screen.blit(font_sm.render('attitude', True, GREY),
                (panel_w // 2 - 28, H - 22))

    # ── Centre panel: big numeric readout ────────────────────────────────
    cx = panel_w + 20
    screen.blit(font_md.render('ANGLES (deg)', True, GREY), (cx, 12))

    for i, (name, val, color) in enumerate([
        ('PITCH', p, ORANGE),
        ('ROLL ', r, PURPLE),
        ('YAW  ', y, CYAN),
    ]):
        yt = 50 + i * 70
        screen.blit(font_md.render(name, True, color), (cx, yt))
        txt = font_lg.render(f'{val:+8.2f}°', True, WHITE)
        screen.blit(txt, (cx, yt + 22))

    screen.blit(font_md.render('GYRO RATES (°/s)', True, GREY), (cx, 270))
    for i, (name, val, color) in enumerate([
        ('gx', latest['gx'], RED),
        ('gy', latest['gy'], GREEN),
        ('gz', latest['gz'], BLUE),
    ]):
        yt = 300 + i * 52
        screen.blit(font_md.render(name, True, color), (cx, yt))
        txt = font_lg.render(f'{val:+8.4f}', True, WHITE)
        screen.blit(txt, (cx, yt + 20))

    # FPS
    fps_txt = font_sm.render(f'display {clock.get_fps():.0f} fps', True, GREY)
    screen.blit(fps_txt, (cx, H - 22))

    # ── Right panel: scrolling graphs ─────────────────────────────────────
    gx = panel_w + 260
    gw = W - gx - 10

    screen.blit(font_md.render('GYRO RATES', True, GREY), (gx, 12))
    draw_graph(screen, (gx, 35, gw, 180),
               [data['gx'], data['gy'], data['gz']],
               [RED, GREEN, BLUE], ['gx', 'gy', 'gz'])

    screen.blit(font_md.render('ANGLES', True, GREY), (gx, 228))
    draw_graph(screen, (gx, 250, gw, 180),
               [data['pitch'], data['roll'], data['yaw']],
               [ORANGE, PURPLE, CYAN], ['pitch', 'roll', 'yaw'])

    screen.blit(font_md.render('GYRO XY SCATTER', True, GREY), (gx, 443))
    # Mini scatter: last 200 gx vs gy samples — shows noise cloud shape
    scatter_rect = pygame.Rect(gx, 465, gw, H - 475)
    pygame.draw.rect(screen, DARK, scatter_rect)
    pygame.draw.rect(screen, GREY, scatter_rect, 1)
    if len(data['gx']) > 5:
        sc_gx = np.array(list(data['gx']))[-200:]
        sc_gy = np.array(list(data['gy']))[-200:]
        peak  = max(abs(sc_gx).max(), abs(sc_gy).max(), 0.05)
        scx, scy = scatter_rect.centerx, scatter_rect.centery
        srx, sry = scatter_rect.width // 2 - 4, scatter_rect.height // 2 - 4
        pygame.draw.line(screen, (50,50,50), (scx, scatter_rect.top),
                         (scx, scatter_rect.bottom), 1)
        pygame.draw.line(screen, (50,50,50), (scatter_rect.left, scy),
                         (scatter_rect.right, scy), 1)
        for gxv, gyv in zip(sc_gx, sc_gy):
            px = int(scx + gxv / peak * srx)
            py = int(scy - gyv / peak * sry)
            if scatter_rect.collidepoint(px, py):
                screen.set_at((px, py), CYAN)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
