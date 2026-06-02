/*
 * rocket_firmware.ino
 * Gyro-only orientation for rocket TVC — no accelerometer
 *
 * Required libraries (copy from github.com/jrowberg/i2cdevlib):
 *   I2Cdev, MPU6050  →  Arduino libraries directory
 *
 * MPU6050 wiring to ESP32:
 *   VCC → 3.3V   GND → GND   SDA → GPIO 21   SCL → GPIO 22   AD0 → GND
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY NO ACCELEROMETER
 * ─────────────────────────────────────────────────────────────────────────────
 * Accelerometers measure specific force: gravity PLUS linear acceleration.
 * During rocket boost both are present simultaneously and cannot be separated.
 * Using the accelerometer for tilt correction during flight produces completely
 * wrong angles. Gyro-only integration is the correct choice for the flight
 * phase of a rocket.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THE DRIFT PROBLEM AND HOW WE SOLVE IT WITHOUT ACCELEROMETER
 * ─────────────────────────────────────────────────────────────────────────────
 * Gyro drift has one root cause: the zero-rate offset (bias) is not perfectly
 * constant. It shifts slowly with temperature after calibration ends.
 *
 * Solution — runtime bias tracking:
 *   Every loop we check if all three gyro axes are reading below a small
 *   threshold (e.g. 1.5 °/s). If yes, the sensor is stationary and the
 *   current gyro reading IS the bias error, not real rotation. We use that
 *   to slowly update our bias estimate with a leaky integrator:
 *     bias = (1 - LEARN) * bias + LEARN * raw_reading
 *   LEARN = 0.0005 means the bias estimate moves very slowly — it takes
 *   ~2000 stationary samples (4 seconds at 500 Hz) to fully converge.
 *   This is slow enough that real slow rotation is not mistaken for bias,
 *   but fast enough to track temperature drift over a 30-second pre-launch.
 *
 *   The moment movement is detected the bias update freezes and the last
 *   good estimate is held for the duration of the motion.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY ±500 °/s RANGE
 * ─────────────────────────────────────────────────────────────────────────────
 * At ±250 °/s, flipping the sensor by hand (which easily exceeds 300 °/s)
 * clips the register at its max value. The integration then accumulates a
 * wrong rate for however long the clip lasts, producing a permanent jump in
 * the angle that never corrects itself.
 * ±500 °/s doubles the range while halving resolution (65.5 LSB/°/s vs 131).
 * The noise increase is small compared to eliminating hard clipping errors.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

// ─── Tunable constants ────────────────────────────────────────────────────────

// ±500 °/s range — prevents clipping on fast movements.
// Sensitivity = 65.5 LSB/(°/s).
const float GYRO_SENSITIVITY   = 65.5f;

// Stationary detection threshold in °/s.
// If all three axes read below this, assume the sensor is not rotating.
// Set just above your observed noise floor (check the visualiser noise box).
// Too high → bias tracks during slow real rotation (bad).
// Too low  → bias never updates because noise exceeds threshold (bad).
const float STATIONARY_THRESH  = 1.5f;

// How fast the runtime bias estimate tracks temperature drift.
// 0.0005 means roughly 2000 stationary samples to fully update.
// At 500 Hz that is ~4 seconds of stillness to fully correct a step change.
const float BIAS_LEARN_RATE    = 0.0005f;

// EMA smoothing on the gyro output — applied AFTER bias removal.
// 0.5 is a bit more responsive than before; raise toward 1.0 for even less
// smoothing if you need maximum speed (at the cost of more noise).
const float GYRO_EMA_ALPHA     = 0.5f;

// Multi-sample averaging per loop iteration.
const int   NUM_AVG_SAMPLES    = 2;

// 500 Hz loop.
const unsigned long LOOP_PERIOD_US = 2000UL;

// Initial calibration — longer warm-up and more samples for better accuracy.
const int   CAL_WARMUP_MS      = 3000;   // 3 s thermal soak
const int   CAL_SAMPLES        = 5000;   // ~1.5 s of data at I2C speed

const int   BAUD_RATE          = 115200;

// ─── Globals ─────────────────────────────────────────────────────────────────

MPU6050 imu;

// Bias in raw LSB — updated both at startup and at runtime when stationary.
float biasX = 0.0f, biasY = 0.0f, biasZ = 0.0f;

// EMA state.
float filtGx = 0.0f, filtGy = 0.0f, filtGz = 0.0f;

// Integrated angles.
float pitch = 0.0f, roll = 0.0f, yaw = 0.0f;

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

    // ±500 °/s — GYRO_CONFIG register, FS_SEL = 0b01.
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);

    // 42 Hz DLPF — removes vibration above 42 Hz.
    imu.setDLPFMode(MPU6050_DLPF_BW_42);

    // Maximum output rate = 1000 Hz.
    imu.setRate(0);

    // ── Initial bias calibration ──────────────────────────────────────────
    // Keep the sensor perfectly still.
    // Longer warm-up lets the chip reach stable temperature.
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
    // Precise 500 Hz timing.
    while ((micros() - lastTime) < LOOP_PERIOD_US) {}
    unsigned long now = micros();
    float dt = (now - lastTime) * 1e-6f;
    lastTime = now;

    // ── Average multiple raw samples ──────────────────────────────────────
    long rawSumX = 0, rawSumY = 0, rawSumZ = 0;
    for (int s = 0; s < NUM_AVG_SAMPLES; s++) {
        int16_t rx, ry, rz;
        imu.getRotation(&rx, &ry, &rz);
        rawSumX += rx; rawSumY += ry; rawSumZ += rz;
    }
    float avgX = (float)rawSumX / NUM_AVG_SAMPLES;
    float avgY = (float)rawSumY / NUM_AVG_SAMPLES;
    float avgZ = (float)rawSumZ / NUM_AVG_SAMPLES;

    // ── Runtime bias tracking ─────────────────────────────────────────────
    // Remove current bias estimate before checking the threshold.
    float debiasedX = (avgX - biasX) / GYRO_SENSITIVITY;
    float debiasedY = (avgY - biasY) / GYRO_SENSITIVITY;
    float debiasedZ = (avgZ - biasZ) / GYRO_SENSITIVITY;

    bool stationary = (fabsf(debiasedX) < STATIONARY_THRESH &&
                       fabsf(debiasedY) < STATIONARY_THRESH &&
                       fabsf(debiasedZ) < STATIONARY_THRESH);

    if (stationary) {
        // Sensor is still — slowly pull bias toward the current raw reading.
        // This compensates for temperature drift between calibration and now.
        biasX = (1.0f - BIAS_LEARN_RATE) * biasX + BIAS_LEARN_RATE * avgX;
        biasY = (1.0f - BIAS_LEARN_RATE) * biasY + BIAS_LEARN_RATE * avgY;
        biasZ = (1.0f - BIAS_LEARN_RATE) * biasZ + BIAS_LEARN_RATE * avgZ;
    }

    // Recompute with updated bias.
    float gx = (avgX - biasX) / GYRO_SENSITIVITY;
    float gy = (avgY - biasY) / GYRO_SENSITIVITY;
    float gz = (avgZ - biasZ) / GYRO_SENSITIVITY;

    // ── EMA smooth ────────────────────────────────────────────────────────
    filtGx = GYRO_EMA_ALPHA * gx + (1.0f - GYRO_EMA_ALPHA) * filtGx;
    filtGy = GYRO_EMA_ALPHA * gy + (1.0f - GYRO_EMA_ALPHA) * filtGy;
    filtGz = GYRO_EMA_ALPHA * gz + (1.0f - GYRO_EMA_ALPHA) * filtGz;

    // ── Integrate — but zero out tiny noise when stationary ───────────────
    // When stationary, force the integrated rate contribution to exactly zero.
    // This stops noise from slowly adding up into the angle while you sit still.
    if (stationary) {
        // Do not integrate — hold current angle.
    } else {
        pitch += filtGx * dt;
        roll  += filtGy * dt;
        yaw   += filtGz * dt;
    }

    // ── Serial output ─────────────────────────────────────────────────────
    Serial.print(filtGx, 4); Serial.print(',');
    Serial.print(filtGy, 4); Serial.print(',');
    Serial.print(filtGz, 4); Serial.print(',');
    Serial.print(pitch,  4); Serial.print(',');
    Serial.print(roll,   4); Serial.print(',');
    Serial.print(yaw,    4); Serial.print(',');
    Serial.println((unsigned long)(dt * 1e6f));
}
