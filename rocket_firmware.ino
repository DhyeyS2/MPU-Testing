/*
 * rocket_firmware.ino
 * MPU6050 gyroscope + accelerometer fusion for rocket TVC flight computer
 *
 * Required libraries — install by copying from github.com/jrowberg/i2cdevlib:
 *   I2Cdev
 *   MPU6050
 * Both folders go into your Arduino libraries directory.
 *
 * MPU6050 wiring to ESP32:
 *   VCC → 3.3V
 *   GND → GND
 *   SDA → GPIO 21
 *   SCL → GPIO 22
 *   AD0 → GND  (I2C address 0x68)
 *   INT → not connected
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY GYRO-ONLY DRIFTS AND HOW THE COMPLEMENTARY FILTER FIXES IT
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * A gyroscope measures rotation rate (°/s). To get angle you integrate:
 *   angle += rate * dt
 * Any tiny error in rate — even 0.01 °/s of residual bias — accumulates
 * forever. After 60 seconds that is 0.6° of error just from bias. Temperature
 * changes shift the bias further. This is gyro drift and it cannot be removed
 * by better calibration alone.
 *
 * An accelerometer measures the direction of gravity. When the sensor is
 * stationary, gravity always points straight down, so you can compute the
 * absolute pitch and roll angle from the accelerometer readings using atan2.
 * This angle has NO long-term drift — gravity is always there as a reference.
 * However, the accelerometer also picks up vibration and any linear
 * acceleration (e.g. the rocket accelerating upward), so it is noisy and
 * wrong during rapid movement.
 *
 * The complementary filter exploits both sensors' strengths:
 *   angle = CF_ALPHA * (angle + gyro_rate * dt)      <- gyro integration
 *         + (1 - CF_ALPHA) * accel_angle             <- accel absolute reference
 *
 * CF_ALPHA = 0.98 means:
 *   - 98% of the angle update comes from the gyro (fast, responsive, no noise)
 *   - 2% per step nudges the angle back toward what the accelerometer says
 * Over time the 2% correction prevents drift from accumulating. During fast
 * movement the accelerometer reading is ignored almost entirely (only 2%).
 *
 * This is the same principle used in every commercial flight controller
 * (ArduPilot, Betaflight, etc.) before more sophisticated Kalman/Madgwick
 * filters are applied. For bench testing and slow manoeuvres it works
 * extremely well with almost no computational cost.
 *
 * IMPORTANT: The complementary filter corrects pitch and roll only.
 * Yaw cannot be corrected this way because gravity has no yaw component —
 * you cannot tell which way you are facing just by measuring "down".
 * Yaw will still drift slowly. A magnetometer (compass) is needed to fix yaw.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY FAST MOVEMENT CAUSED LARGE ERRORS
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * At ±250 °/s full-scale range, if you physically rotate the sensor faster
 * than 250 °/s the gyro output register clips at its maximum value (32767).
 * The integration then accumulates a wrong rate for however long the motion
 * lasts, producing a permanent offset in the integrated angle.
 *
 * Fix: switch to ±500 °/s range. This halves the resolution from
 * 131 LSB/(°/s) to 65.5 LSB/(°/s), but prevents clipping during fast
 * hand movements (which easily exceed 250 °/s). For a TVC rocket the
 * complementary filter correction also helps pull the angle back to truth
 * after any transient saturation.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

// ─── Tunable constants ────────────────────────────────────────────────────────

// Complementary filter coefficient.
// 0.98 = trust gyro 98%, nudge toward accelerometer 2% every step.
// Increase toward 1.0 to follow gyro more (more drift, less accel noise).
// Decrease toward 0.9 to correct drift faster (more accel noise in output).
const float CF_ALPHA = 0.98f;

// Gyro EMA smoothing (applied before the complementary filter).
const float GYRO_ALPHA = 0.4f;

// Gyro sensitivity for ±500 °/s range = 65.5 LSB/(°/s).
// (Changed from ±250 to prevent clipping on fast hand movements.)
const float GYRO_SENSITIVITY = 65.5f;

// Accelerometer sensitivity for ±2g range = 16384 LSB/g.
const float ACCEL_SENSITIVITY = 16384.0f;

// Number of raw gyro samples averaged per loop iteration.
const int NUM_AVG_SAMPLES = 2;

// 500 Hz loop target.
const unsigned long LOOP_PERIOD_US = 2000UL;

// Calibration settings.
const int CAL_SAMPLES   = 3000;
const int CAL_WARMUP_MS = 2000;

const int BAUD_RATE = 115200;

// ─── Globals ─────────────────────────────────────────────────────────────────

MPU6050 imu;

// Gyro bias in raw LSB (measured during calibration at rest).
float gyroBiasX = 0.0f, gyroBiasY = 0.0f, gyroBiasZ = 0.0f;

// EMA filter state for gyro.
float filtGx = 0.0f, filtGy = 0.0f, filtGz = 0.0f;

// Final fused angles in degrees.
float pitch = 0.0f, roll = 0.0f, yaw = 0.0f;

unsigned long lastTime = 0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

void readRawGyro(int16_t &rx, int16_t &ry, int16_t &rz) {
    imu.getRotation(&rx, &ry, &rz);
}

void readRawAccel(int16_t &ax, int16_t &ay, int16_t &az) {
    imu.getAcceleration(&ax, &ay, &az);
}

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(BAUD_RATE);
    Wire.begin();
    Wire.setClock(400000);
    imu.initialize();

    if (!imu.testConnection()) {
        Serial.println("MPU6050 connection FAILED — check wiring");
        while (true) {}
    }

    // ±500 °/s — wider range prevents clipping during fast movements.
    // Sensitivity = 65.5 LSB/(°/s) instead of 131 LSB/(°/s) at ±250.
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);

    // ±2g accelerometer range — maximum sensitivity, fine for orientation.
    imu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);

    // 42 Hz hardware DLPF — removes motor vibration above 42 Hz.
    imu.setDLPFMode(MPU6050_DLPF_BW_42);

    // Maximum output rate (1000 Hz at DLPF_BW_42).
    imu.setRate(0);

    // ── Calibration ───────────────────────────────────────────────────────
    // Collect gyro bias only. Accel bias is not needed because the
    // complementary filter uses accel for absolute angle, not rate.
    // Keep the sensor perfectly still during this phase.
    Serial.println("CALIBRATING");
    delay(CAL_WARMUP_MS);

    long sumX = 0, sumY = 0, sumZ = 0;
    for (int i = 0; i < CAL_SAMPLES; i++) {
        int16_t rx, ry, rz;
        readRawGyro(rx, ry, rz);
        sumX += rx; sumY += ry; sumZ += rz;
    }
    gyroBiasX = (float)sumX / CAL_SAMPLES;
    gyroBiasY = (float)sumY / CAL_SAMPLES;
    gyroBiasZ = (float)sumZ / CAL_SAMPLES;

    // Initialise pitch and roll from accelerometer so the display starts at
    // the correct physical orientation immediately — no spin-up period.
    int16_t ax, ay, az;
    readRawAccel(ax, ay, az);
    float axg = ax / ACCEL_SENSITIVITY;
    float ayg = ay / ACCEL_SENSITIVITY;
    float azg = az / ACCEL_SENSITIVITY;
    pitch = atan2(-axg, sqrt(ayg * ayg + azg * azg)) * 180.0f / M_PI;
    roll  = atan2( ayg, azg)                          * 180.0f / M_PI;
    yaw   = 0.0f;  // no absolute yaw reference without magnetometer

    Serial.println("READY");
    lastTime = micros();
}

// ─── loop() ──────────────────────────────────────────────────────────────────
void loop() {
    while ((micros() - lastTime) < LOOP_PERIOD_US) {}
    unsigned long now = micros();
    float dt = (now - lastTime) * 1e-6f;
    lastTime = now;

    // ── Read and average gyro samples ─────────────────────────────────────
    long rawSumX = 0, rawSumY = 0, rawSumZ = 0;
    for (int s = 0; s < NUM_AVG_SAMPLES; s++) {
        int16_t rx, ry, rz;
        readRawGyro(rx, ry, rz);
        rawSumX += rx; rawSumY += ry; rawSumZ += rz;
    }
    float gx = ((float)rawSumX / NUM_AVG_SAMPLES - gyroBiasX) / GYRO_SENSITIVITY;
    float gy = ((float)rawSumY / NUM_AVG_SAMPLES - gyroBiasY) / GYRO_SENSITIVITY;
    float gz = ((float)rawSumZ / NUM_AVG_SAMPLES - gyroBiasZ) / GYRO_SENSITIVITY;

    // ── EMA smooth the gyro rates ─────────────────────────────────────────
    filtGx = GYRO_ALPHA * gx + (1.0f - GYRO_ALPHA) * filtGx;
    filtGy = GYRO_ALPHA * gy + (1.0f - GYRO_ALPHA) * filtGy;
    filtGz = GYRO_ALPHA * gz + (1.0f - GYRO_ALPHA) * filtGz;

    // ── Read accelerometer ─────────────────────────────────────────────────
    int16_t ax, ay, az;
    readRawAccel(ax, ay, az);
    float axg = ax / ACCEL_SENSITIVITY;
    float ayg = ay / ACCEL_SENSITIVITY;
    float azg = az / ACCEL_SENSITIVITY;

    // Compute absolute pitch and roll from gravity direction.
    // atan2 returns radians; convert to degrees.
    // These formulas assume the sensor is near-stationary; they are wrong
    // during heavy linear acceleration (e.g. rocket boost phase), but the
    // complementary filter's 0.98 weighting already de-weights them then.
    float accelPitch = atan2(-axg, sqrt(ayg * ayg + azg * azg)) * 180.0f / M_PI;
    float accelRoll  = atan2( ayg, azg)                          * 180.0f / M_PI;

    // ── Complementary filter ──────────────────────────────────────────────
    // For pitch and roll: blend gyro integration with accel absolute angle.
    // For yaw: gyro integration only (no accel correction possible).
    pitch = CF_ALPHA * (pitch + filtGx * dt) + (1.0f - CF_ALPHA) * accelPitch;
    roll  = CF_ALPHA * (roll  + filtGy * dt) + (1.0f - CF_ALPHA) * accelRoll;
    yaw  += filtGz * dt;

    // ── Serial output ─────────────────────────────────────────────────────
    // Format unchanged: gx,gy,gz,pitch,roll,yaw,dt_us
    Serial.print(filtGx, 4); Serial.print(',');
    Serial.print(filtGy, 4); Serial.print(',');
    Serial.print(filtGz, 4); Serial.print(',');
    Serial.print(pitch,  4); Serial.print(',');
    Serial.print(roll,   4); Serial.print(',');
    Serial.print(yaw,    4); Serial.print(',');
    Serial.println((unsigned long)(dt * 1e6f));
}
