/*
 * rocket_firmware.ino
 * Gyro-only TVC orientation — MPU6050 + ESP32
 *
 * Required libraries (copy from github.com/jrowberg/i2cdevlib):
 *   I2Cdev, MPU6050  →  Arduino libraries directory
 *
 * MPU6050 wiring to ESP32:
 *   VCC → 3.3V   GND → GND   SDA → GPIO 21   SCL → GPIO 22   AD0 → GND
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * TWO-PHASE OPERATION: PAD MODE vs FLIGHT MODE
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * PAD MODE (before launch):
 *   - Runtime bias learning is ACTIVE.
 *     The bias estimate continuously tracks temperature drift so that by the
 *     time the rocket launches the bias is as accurate as it can possibly be.
 *   - Integration is GATED when stationary.
 *     Noise below the stationary threshold does not accumulate into the angle.
 *   - The rocket sits on the pad for potentially minutes. Without this gating
 *     even 0.01 °/s of residual bias would add up to visible angle error before
 *     launch and the TVC controller would try to "correct" a phantom lean.
 *
 * FLIGHT MODE (after launch detected):
 *   - Runtime bias learning is FROZEN at the last pad value.
 *     During flight the rocket is genuinely rotating. If we kept learning we
 *     would mistake real rotation for bias and subtract out the signal we need.
 *   - Stationary gating is DISABLED.
 *     The rocket may fly nearly straight (low angular rates) when the TVC loop
 *     is working well. We must still integrate those small rates accurately.
 *   - The bias estimate from the pad phase is the best we have and is held
 *     for the duration of the flight.
 *
 * LAUNCH DETECTION — gyro magnitude threshold:
 *   Without an accelerometer the cleanest launch signal available is the gyro
 *   itself. When the rocket leaves the launch rod it will experience a clear
 *   disturbance from rod-exit wobble and initial atmosphere. The gyro magnitude
 *   will spike well above the bench noise floor.
 *   Threshold = 15 °/s. A value this high cannot be reached by bench vibration
 *   but will be reached immediately at rod exit on any practical launch.
 *   Once triggered, flight mode is latched permanently — it never reverts.
 *   This avoids any false reversion if the rocket happens to fly very straight.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY ±500 °/s RANGE
 * ─────────────────────────────────────────────────────────────────────────────
 * The C6 motor produces ~255 °/s² angular acceleration against this rocket's
 * MMOI. An uncorrected lean of just 1° would produce over 250 °/s rotation
 * within one second. ±250 °/s would clip immediately. ±500 °/s gives headroom.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SERIAL OUTPUT FORMAT
 * ─────────────────────────────────────────────────────────────────────────────
 * One CSV line per loop iteration at 500 Hz:
 *   gx,gy,gz,pitch,roll,yaw,dt_us,flight
 * flight = 0 on pad, 1 in flight — lets the visualiser show mode clearly.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

// ─── Constants ────────────────────────────────────────────────────────────────

// Gyro sensitivity at ±500 °/s = 65.5 LSB/(°/s).
const float GYRO_SENSITIVITY    = 65.5f;

// PAD MODE: stationary detection threshold (°/s).
// Set ~3× your observed noise floor (read from the visualiser noise stats box).
// Typical noise floor for MPU6050 at 500 Hz after DLPF ≈ 0.05–0.15 °/s σ.
// 0.5 °/s threshold is safely above noise but well below any real movement.
const float STATIONARY_THRESH   = 0.5f;

// PAD MODE: how fast the bias tracks temperature drift while stationary.
// 0.001 = takes ~1000 stationary samples (2 s at 500 Hz) to fully update.
// Faster tracking is fine on the pad because we have unlimited time.
const float BIAS_LEARN_RATE     = 0.001f;

// LAUNCH DETECT: gyro vector magnitude threshold to switch to flight mode.
// 15 °/s cannot be reached by bench vibration but is easily exceeded at rod exit.
// Lower this if your launch rod is very smooth; raise it if bench vibration
// is triggering false launches (watch the visualiser flight indicator).
const float LAUNCH_THRESH_DPS   = 15.0f;

// EMA smoothing on gyro output.
// 0.6 = responsive with mild smoothing. The hardware DLPF already did the
// heavy lifting; this just takes the edge off sample-to-sample jitter.
const float GYRO_EMA_ALPHA      = 0.6f;

// Multi-sample averaging per loop iteration.
const int   NUM_AVG_SAMPLES     = 2;

// 500 Hz loop.
const unsigned long LOOP_PERIOD_US = 2000UL;

// Initial calibration.
const int   CAL_WARMUP_MS       = 3000;   // thermal soak before collecting
const int   CAL_SAMPLES         = 5000;   // ~1.7 s of I2C reads

const int   BAUD_RATE           = 115200;

// ─── Globals ─────────────────────────────────────────────────────────────────

MPU6050 imu;

float biasX = 0.0f, biasY = 0.0f, biasZ = 0.0f;
float filtGx = 0.0f, filtGy = 0.0f, filtGz = 0.0f;
float pitch = 0.0f, roll = 0.0f, yaw = 0.0f;

bool  inFlight = false;   // latched true at launch detection, never reverts

unsigned long lastTime = 0;

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(BAUD_RATE);
    Wire.begin();
    Wire.setClock(400000);
    imu.initialize();

    if (!imu.testConnection()) {
        Serial.println("MPU6050 FAILED");
        while (true) {}
    }

    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    imu.setDLPFMode(MPU6050_DLPF_BW_42);
    imu.setRate(0);

    // ── Initial bias calibration ──────────────────────────────────────────
    Serial.println("CALIBRATING");
    delay(CAL_WARMUP_MS);

    long sumX = 0, sumY = 0, sumZ = 0;
    for (int i = 0; i < CAL_SAMPLES; i++) {
        int16_t rx, ry, rz;
        imu.getRotation(&rx, &ry, &rz);
        sumX += rx; sumY += ry; sumZ += rz;
    }
    biasX = (float)sumX / CAL_SAMPLES;
    biasY = (float)sumY / CAL_SAMPLES;
    biasZ = (float)sumZ / CAL_SAMPLES;

    Serial.println("READY");
    lastTime = micros();
}

// ─── loop() ──────────────────────────────────────────────────────────────────
void loop() {
    while ((micros() - lastTime) < LOOP_PERIOD_US) {}
    unsigned long now = micros();
    float dt = (now - lastTime) * 1e-6f;
    lastTime = now;

    // ── Average raw samples ───────────────────────────────────────────────
    long rawSumX = 0, rawSumY = 0, rawSumZ = 0;
    for (int s = 0; s < NUM_AVG_SAMPLES; s++) {
        int16_t rx, ry, rz;
        imu.getRotation(&rx, &ry, &rz);
        rawSumX += rx; rawSumY += ry; rawSumZ += rz;
    }
    float avgX = (float)rawSumX / NUM_AVG_SAMPLES;
    float avgY = (float)rawSumY / NUM_AVG_SAMPLES;
    float avgZ = (float)rawSumZ / NUM_AVG_SAMPLES;

    // ── De-bias ───────────────────────────────────────────────────────────
    float gx = (avgX - biasX) / GYRO_SENSITIVITY;
    float gy = (avgY - biasY) / GYRO_SENSITIVITY;
    float gz = (avgZ - biasZ) / GYRO_SENSITIVITY;

    // ── Launch detection (one-way latch) ──────────────────────────────────
    // Check gyro vector magnitude. Once latched, never un-latches.
    if (!inFlight) {
        float mag = sqrtf(gx*gx + gy*gy + gz*gz);
        if (mag > LAUNCH_THRESH_DPS) {
            inFlight = true;
            // Freeze bias at current best estimate — no more learning.
        }
    }

    // ── Pad mode: bias learning + stationary gating ───────────────────────
    bool integrate = true;

    if (!inFlight) {
        float absGx = fabsf(gx);
        float absGy = fabsf(gy);
        float absGz = fabsf(gz);

        bool stationary = (absGx < STATIONARY_THRESH &&
                           absGy < STATIONARY_THRESH &&
                           absGz < STATIONARY_THRESH);

        if (stationary) {
            // Refine bias toward current raw reading.
            biasX = (1.0f - BIAS_LEARN_RATE) * biasX + BIAS_LEARN_RATE * avgX;
            biasY = (1.0f - BIAS_LEARN_RATE) * biasY + BIAS_LEARN_RATE * avgY;
            biasZ = (1.0f - BIAS_LEARN_RATE) * biasZ + BIAS_LEARN_RATE * avgZ;
            // Recompute with updated bias.
            gx = (avgX - biasX) / GYRO_SENSITIVITY;
            gy = (avgY - biasY) / GYRO_SENSITIVITY;
            gz = (avgZ - biasZ) / GYRO_SENSITIVITY;
            // Do not integrate noise into the angle while sitting on pad.
            integrate = false;
        }
    }

    // ── EMA smooth ────────────────────────────────────────────────────────
    filtGx = GYRO_EMA_ALPHA * gx + (1.0f - GYRO_EMA_ALPHA) * filtGx;
    filtGy = GYRO_EMA_ALPHA * gy + (1.0f - GYRO_EMA_ALPHA) * filtGy;
    filtGz = GYRO_EMA_ALPHA * gz + (1.0f - GYRO_EMA_ALPHA) * filtGz;

    // ── Integrate ─────────────────────────────────────────────────────────
    if (integrate) {
        pitch += filtGx * dt;
        roll  += filtGy * dt;
        yaw   += filtGz * dt;
    }

    // ── Serial output ─────────────────────────────────────────────────────
    // Format: gx,gy,gz,pitch,roll,yaw,dt_us,flight
    Serial.print(filtGx, 4); Serial.print(',');
    Serial.print(filtGy, 4); Serial.print(',');
    Serial.print(filtGz, 4); Serial.print(',');
    Serial.print(pitch,  4); Serial.print(',');
    Serial.print(roll,   4); Serial.print(',');
    Serial.print(yaw,    4); Serial.print(',');
    Serial.print((unsigned long)(dt * 1e6f)); Serial.print(',');
    Serial.println(inFlight ? 1 : 0);
}
