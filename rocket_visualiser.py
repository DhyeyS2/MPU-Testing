# rocket_visualiser.py
#
# Required packages — install with:
#   pip install pyserial matplotlib numpy
#
# Reads MPU6050 gyroscope + integrated angle data from an ESP32 over serial USB
# and renders a live 3D rocket orientation visualiser plus scrolling time-series
# graphs of gyro rates and integrated angles.

import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection
import serial
import serial.tools.list_ports

# ─── Serial configuration ─────────────────────────────────────────────────────
# Change PORT to match your system. On Windows: 'COM3', 'COM4', etc.
# On Linux/Mac: '/dev/ttyUSB0', '/dev/ttyACM0', '/dev/cu.usbserial-*' etc.
PORT = 'COM3'
BAUD = 115200

# ─── Data buffer configuration ────────────────────────────────────────────────
# 500 Hz × 5 seconds = 2500 samples maximum kept in memory.
SAMPLE_RATE   = 500
HISTORY_SECS  = 5
MAX_DEQUE_LEN = SAMPLE_RATE * HISTORY_SECS  # 2500

# ─── Thread-safe data store ────────────────────────────────────────────────────
# Each deque stores one scalar per sample in time order.
# maxlen causes old samples to be automatically discarded, giving a sliding window.
data = {
    'gx':    deque(maxlen=MAX_DEQUE_LEN),
    'gy':    deque(maxlen=MAX_DEQUE_LEN),
    'gz':    deque(maxlen=MAX_DEQUE_LEN),
    'pitch': deque(maxlen=MAX_DEQUE_LEN),
    'roll':  deque(maxlen=MAX_DEQUE_LEN),
    'yaw':   deque(maxlen=MAX_DEQUE_LEN),
    'dt_us': deque(maxlen=MAX_DEQUE_LEN),
    'time':  deque(maxlen=MAX_DEQUE_LEN),   # wall-clock time of each sample (s)
}

# Latest single values — written by reader thread, read by animation thread.
# Reading/writing a Python float is atomic on CPython, so a simple variable
# is safe here; we don't need a Lock for these scalars.
latest = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
          'gx': 0.0, 'gy': 0.0, 'gz': 0.0}

# Noise statistics updated once per second by the reader thread.
noise_stats = {'gx_std': 0.0, 'gy_std': 0.0, 'gz_std': 0.0}

t0 = time.time()  # session start time for relative timestamps


# ─── Serial reader thread ─────────────────────────────────────────────────────
def serial_reader(port: str, baud: int) -> None:
    """
    Runs as a daemon thread.  Opens the serial port, reads lines, parses them,
    and pushes values into the deques.  Because this is a daemon thread it is
    automatically killed when the main thread (visualiser) exits.
    """
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException:
        print(f"\nCould not open {port}. Check your COM port setting at the "
              f"top of this file and verify the ESP32 is connected.")
        sys.exit(1)

    print(f"Opened {port} at {baud} baud.  Waiting for ESP32 READY...")

    noise_window = 100  # samples used for live std-dev calculation
    last_noise_update = time.time()

    while True:
        try:
            raw = ser.readline().decode('ascii', errors='ignore').strip()
        except Exception:
            continue

        if not raw:
            continue

        # Echo every line to the terminal so raw numbers are visible.
        print(raw)

        # Skip status lines from the firmware (CALIBRATING / READY).
        if not raw[0].lstrip('-').replace('.', '', 1).isdigit() and \
                not raw[0] in '0123456789-':
            continue

        parts = raw.split(',')
        if len(parts) != 7:
            continue

        try:
            gx, gy, gz, pitch, roll, yaw, dt_us = (
                float(parts[0]), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]), float(parts[5]),
                int(parts[6])
            )
        except ValueError:
            continue

        now = time.time() - t0

        data['gx'].append(gx)
        data['gy'].append(gy)
        data['gz'].append(gz)
        data['pitch'].append(pitch)
        data['roll'].append(roll)
        data['yaw'].append(yaw)
        data['dt_us'].append(dt_us)
        data['time'].append(now)

        latest['gx']    = gx
        latest['gy']    = gy
        latest['gz']    = gz
        latest['pitch'] = pitch
        latest['roll']  = roll
        latest['yaw']   = yaw

        # Update noise statistics once per second.
        if time.time() - last_noise_update >= 1.0:
            if len(data['gx']) >= noise_window:
                arr_gx = np.array(list(data['gx']))[-noise_window:]
                arr_gy = np.array(list(data['gy']))[-noise_window:]
                arr_gz = np.array(list(data['gz']))[-noise_window:]
                noise_stats['gx_std'] = float(np.std(arr_gx))
                noise_stats['gy_std'] = float(np.std(arr_gy))
                noise_stats['gz_std'] = float(np.std(arr_gz))
            last_noise_update = time.time()


# ─── Rotation matrices ────────────────────────────────────────────────────────
def Rx(angle_deg: float) -> np.ndarray:
    """Rotation matrix about X axis (roll)."""
    a = np.radians(angle_deg)
    return np.array([
        [1,          0,           0],
        [0,  np.cos(a), -np.sin(a)],
        [0,  np.sin(a),  np.cos(a)],
    ])

def Ry(angle_deg: float) -> np.ndarray:
    """Rotation matrix about Y axis (pitch)."""
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [         0, 1,         0],
        [-np.sin(a), 0, np.cos(a)],
    ])

def Rz(angle_deg: float) -> np.ndarray:
    """Rotation matrix about Z axis (yaw)."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [        0,          0, 1],
    ])

def combined_rotation(pitch_deg: float, roll_deg: float,
                       yaw_deg: float) -> np.ndarray:
    """
    Combined rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Applied to column vectors: p_rotated = R @ p_body.
    """
    return Rz(yaw_deg) @ Ry(pitch_deg) @ Rx(roll_deg)


# ─── Rocket geometry (body frame, rocket points along +Z) ────────────────────
def make_rocket_geometry():
    """
    Returns a dictionary of named point arrays in body-frame coordinates.
    The rocket points up along the +Z axis, centred at the origin.

    Body dimensions (arbitrary units that fill the ±2 axes view nicely):
      Body cylinder: radius 0.2, from z = -1.0 (bottom) to z = 0.8 (top)
      Nose cone:     from z = 0.8 (base) to z = 1.5 (tip)
      Fins:          triangles at the base, three fins 120° apart
    """
    body_r    = 0.2
    body_bot  = -1.0
    body_top  =  0.8
    nose_tip  =  1.5
    n_circle  = 36  # segments for circle approximation

    theta = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)

    # ── Body cylinder ─────────────────────────────────────────────────────
    # Circle of points at the bottom rim
    bottom_ring = np.column_stack([
        body_r * np.cos(theta),
        body_r * np.sin(theta),
        np.full(n_circle, body_bot),
    ])
    # Circle of points at the top rim
    top_ring = np.column_stack([
        body_r * np.cos(theta),
        body_r * np.sin(theta),
        np.full(n_circle, body_top),
    ])
    # Vertical lines connecting corresponding points
    vert_lines = []
    for i in range(0, n_circle, 6):  # every 6th line to keep plot light
        vert_lines.append((bottom_ring[i], top_ring[i]))

    # ── Nose cone ─────────────────────────────────────────────────────────
    nose_base_ring = top_ring.copy()
    nose_tip_pt    = np.array([0.0, 0.0, nose_tip])
    nose_lines = [(nose_base_ring[i], nose_tip_pt) for i in range(0, n_circle, 6)]

    # ── Fins — three triangular polygons 120° apart ───────────────────────
    # Each fin: root chord along the body, sweeping back and outward.
    fin_root_top_z  = body_bot + 0.5   # upper attachment point on body
    fin_root_bot_z  = body_bot         # lower attachment point on body
    fin_tip_z       = body_bot - 0.1   # fin tip slightly below body bottom
    fin_span        = 0.7              # distance from body axis to fin tip

    fins = []
    for angle_deg in [0, 120, 240]:
        a = np.radians(angle_deg)
        cx, cy = np.cos(a), np.sin(a)  # fin pointing direction

        p0 = np.array([body_r * cx,    body_r * cy,    fin_root_top_z])  # root top
        p1 = np.array([body_r * cx,    body_r * cy,    fin_root_bot_z])  # root bot
        p2 = np.array([fin_span * cx,  fin_span * cy,  fin_tip_z])       # tip

        fins.append(np.array([p0, p1, p2]))  # triangle vertices

    return {
        'bottom_ring': bottom_ring,
        'top_ring':    top_ring,
        'vert_lines':  vert_lines,
        'nose_lines':  nose_lines,
        'fins':        fins,
    }

GEOM = make_rocket_geometry()


def rotate_geom(geom: dict, R: np.ndarray) -> dict:
    """Apply rotation matrix R to every point in the geometry dictionary."""
    def rot(pts):
        return (R @ pts.T).T

    rotated = {}
    rotated['bottom_ring'] = rot(geom['bottom_ring'])
    rotated['top_ring']    = rot(geom['top_ring'])
    rotated['vert_lines']  = [(rot(np.array([a]))[0], rot(np.array([b]))[0])
                               for a, b in geom['vert_lines']]
    rotated['nose_lines']  = [(rot(np.array([a]))[0], rot(np.array([b]))[0])
                               for a, b in geom['nose_lines']]
    rotated['fins']        = [rot(tri) for tri in geom['fins']]
    return rotated


# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor='#0d0d0d')
fig.canvas.manager.set_window_title('Rocket IMU Visualiser')

# Left: 3D rocket view
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_facecolor('#0d0d0d')

# Right top: gyro rates time series
ax_gyro = fig.add_subplot(2, 2, 2)
ax_gyro.set_facecolor('#111111')

# Right bottom: integrated angles time series
ax_angle = fig.add_subplot(2, 2, 4)
ax_angle.set_facecolor('#111111')

plt.tight_layout(pad=2.0)


def draw_world_axes(ax):
    """Draw fixed XYZ world-frame reference arrows in the 3D axes."""
    length = 1.8
    ax.quiver(0, 0, 0, length, 0, 0, color='red',   linewidth=1.5,
              arrow_length_ratio=0.1, label='X')
    ax.quiver(0, 0, 0, 0, length, 0, color='lime',  linewidth=1.5,
              arrow_length_ratio=0.1, label='Y')
    ax.quiver(0, 0, 0, 0, 0, length, color='cyan',  linewidth=1.5,
              arrow_length_ratio=0.1, label='Z')
    ax.text(length + 0.1, 0, 0, 'X', color='red',  fontsize=9)
    ax.text(0, length + 0.1, 0, 'Y', color='lime', fontsize=9)
    ax.text(0, 0, length + 0.1, 'Z', color='cyan', fontsize=9)


def plot_rocket(ax, rgeom):
    """Clear the 3D axes and redraw the rocket with rotated geometry."""
    ax.cla()
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
    ax.set_xlabel('X', color='white', labelpad=4)
    ax.set_ylabel('Y', color='white', labelpad=4)
    ax.set_zlabel('Z', color='white', labelpad=4)
    ax.tick_params(colors='#444444', labelsize=7)
    ax.set_title('Rocket Orientation (3D)', color='white', pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_world_axes(ax)

    # ── Body rings ────────────────────────────────────────────────────────
    br = rgeom['bottom_ring']
    tr = rgeom['top_ring']
    ax.plot(np.append(br[:, 0], br[0, 0]),
            np.append(br[:, 1], br[0, 1]),
            np.append(br[:, 2], br[0, 2]),
            color='#aaaaff', linewidth=0.8)
    ax.plot(np.append(tr[:, 0], tr[0, 0]),
            np.append(tr[:, 1], tr[0, 1]),
            np.append(tr[:, 2], tr[0, 2]),
            color='#aaaaff', linewidth=0.8)

    # ── Vertical lines ────────────────────────────────────────────────────
    for (a, b) in rgeom['vert_lines']:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color='#7777cc', linewidth=0.7)

    # ── Nose cone ─────────────────────────────────────────────────────────
    for (a, b) in rgeom['nose_lines']:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color='#ffaa44', linewidth=0.8)

    # ── Fins ──────────────────────────────────────────────────────────────
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fin_polys = [tri.tolist() for tri in rgeom['fins']]
    fin_coll  = Poly3DCollection(fin_polys, alpha=0.6,
                                  facecolor='#44cc44', edgecolor='#88ff88',
                                  linewidth=0.6)
    ax.add_collection3d(fin_coll)

    # ── Attitude text ─────────────────────────────────────────────────────
    p  = latest['pitch']
    r  = latest['roll']
    y  = latest['yaw']
    ax.text2D(0.02, 0.97,
              f"Pitch: {p:+7.2f}°\nRoll:  {r:+7.2f}°\nYaw:   {y:+7.2f}°",
              transform=ax.transAxes, color='white', fontsize=9,
              verticalalignment='top',
              bbox=dict(facecolor='#222222', alpha=0.7, edgecolor='none'))


# ─── Animation update function ────────────────────────────────────────────────
def update(frame):
    pitch = latest['pitch']
    roll  = latest['roll']
    yaw   = latest['yaw']

    # ── 3D rocket plot ────────────────────────────────────────────────────
    R      = combined_rotation(pitch, roll, yaw)
    rgeom  = rotate_geom(GEOM, R)
    plot_rocket(ax3d, rgeom)

    # ── Snapshot all deques atomically to avoid race-condition length mismatches
    # The reader thread can append between individual np.array() calls, so we
    # convert everything in one go and then trim to the shortest length.
    t_arr     = np.array(data['time'])
    gx_arr    = np.array(data['gx'])
    gy_arr    = np.array(data['gy'])
    gz_arr    = np.array(data['gz'])
    pitch_arr = np.array(data['pitch'])
    roll_arr  = np.array(data['roll'])
    yaw_arr   = np.array(data['yaw'])

    n = min(len(t_arr), len(gx_arr), len(gy_arr), len(gz_arr),
            len(pitch_arr), len(roll_arr), len(yaw_arr))
    t_arr     = t_arr[:n]
    gx_arr    = gx_arr[:n]
    gy_arr    = gy_arr[:n]
    gz_arr    = gz_arr[:n]
    pitch_arr = pitch_arr[:n]
    roll_arr  = roll_arr[:n]
    yaw_arr   = yaw_arr[:n]

    # ── Gyro rate time series ─────────────────────────────────────────────
    ax_gyro.cla()
    ax_gyro.set_facecolor('#111111')

    if n > 1:
        t_rel = t_arr - t_arr[-1]   # time relative to "now", so x scrolls left
        ax_gyro.plot(t_rel, gx_arr, color='#ff4444', linewidth=0.8, label='gx')
        ax_gyro.plot(t_rel, gy_arr, color='#44ff44', linewidth=0.8, label='gy')
        ax_gyro.plot(t_rel, gz_arr, color='#4488ff', linewidth=0.8, label='gz')
        ax_gyro.axhline(0, color='#555555', linewidth=0.6)
        ax_gyro.set_xlim(-HISTORY_SECS, 0)
        ax_gyro.autoscale(axis='y', tight=False)

    ax_gyro.set_xlabel('Time (s)', color='white', fontsize=8)
    ax_gyro.set_ylabel('deg/s', color='white', fontsize=8)
    ax_gyro.set_title('Gyro Rates', color='white', fontsize=9)
    ax_gyro.tick_params(colors='#888888', labelsize=7)
    ax_gyro.legend(loc='upper left', fontsize=7, framealpha=0.3,
                   labelcolor='white', facecolor='#222222')

    # ── Live noise statistics text box ────────────────────────────────────
    stats_text = (
        f"Noise σ (last {100} samples)\n"
        f"  gx: {noise_stats['gx_std']:.4f} °/s\n"
        f"  gy: {noise_stats['gy_std']:.4f} °/s\n"
        f"  gz: {noise_stats['gz_std']:.4f} °/s"
    )
    ax_gyro.text(0.98, 0.97, stats_text,
                 transform=ax_gyro.transAxes, color='#cccccc', fontsize=7,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='#1a1a1a', alpha=0.85, edgecolor='#444444',
                           boxstyle='round,pad=0.4'))

    # ── Integrated angle time series ──────────────────────────────────────
    ax_angle.cla()
    ax_angle.set_facecolor('#111111')

    if n > 1:
        t_rel = t_arr - t_arr[-1]
        ax_angle.plot(t_rel, pitch_arr, color='#ff8800', linewidth=0.8,
                      label='pitch')
        ax_angle.plot(t_rel, roll_arr,  color='#cc44ff', linewidth=0.8,
                      label='roll')
        ax_angle.plot(t_rel, yaw_arr,   color='#00cccc', linewidth=0.8,
                      label='yaw')
        ax_angle.axhline(0, color='#555555', linewidth=0.6)
        ax_angle.set_xlim(-HISTORY_SECS, 0)
        ax_angle.autoscale(axis='y', tight=False)

    ax_angle.set_xlabel('Time (s)', color='white', fontsize=8)
    ax_angle.set_ylabel('degrees', color='white', fontsize=8)
    ax_angle.set_title('Integrated Angles (drift visible here)', color='white',
                        fontsize=9)
    ax_angle.tick_params(colors='#888888', labelsize=7)
    ax_angle.legend(loc='upper left', fontsize=7, framealpha=0.3,
                    labelcolor='white', facecolor='#222222')

    # Style the figure background.
    fig.patch.set_facecolor('#0d0d0d')

    return []


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Start serial reader as a background daemon thread.
    reader = threading.Thread(target=serial_reader, args=(PORT, BAUD),
                               daemon=True)
    reader.start()

    # FuncAnimation drives the display at 20 Hz (interval = 50 ms).
    # blit=False is required because we are redrawing 3D axes.
    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)

    plt.show()
