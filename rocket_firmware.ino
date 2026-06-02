/*
 * rocket_firmware.ino
 * MPU6050 gyroscope data acquisition for rocket TVC flight computer
 *
 * Required libraries — install by copying from github.com/jrowberg/i2cdevlib:
 *   I2Cdev
 *   MPU6050
 * Both folders go into your Arduino libraries directory
 * (typically ~/Documents/Arduino/libraries/ on Windows/Mac or ~/Arduino/libraries/ on Linux).
 *
 * MPU6050 wiring to ESP32:
 *   VCC → 3.3V      (the MPU6050 runs on 3.3V; do NOT connect to 5V)
 *   GND → GND
 *   SDA → GPIO 21   (ESP32 default I2C data line)
 *   SCL → GPIO 22   (ESP32 default I2C clock line)
 *   AD0 → GND       (pulls the I2C address select pin low → address 0x68)
 *   INT → not connected for this project (we poll rather than use interrupts)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * NOISE REDUCTION PIPELINE — why every stage matters
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Stage 1 — Hardware DLPF (Digital Low Pass Filter) inside the MPU6050
 *   The MPU6050 contains a hardware filter on the gyro signal path.
 *   Setting DLPF_BW_42 configures a 42 Hz cutoff frequency.
 *   A rocket's actual rotation is slow — well below 10 Hz even for an agressive
 *   TVC maneuver. Motor vibration, on the other hand, is typically 50–500 Hz.
 *   By cutting off at 42 Hz we eliminate most vibration noise before it ever
 *   reaches the digital output register. This is free noise reduction that costs
 *   us nothing because the signal we care about is entirely below 42 Hz.
 *   Register affected: CONFIG (0x1A), bits [2:0] = DLPF_CFG.
 *   At DLPF_BW_42 the gyro internal sample rate becomes 1000 Hz.
 *
 * Stage 2 — Multi-sample averaging in firmware
 *   Random electronic noise is uncorrelated sample-to-sample. If you average
 *   N samples of random noise, the noise amplitude shrinks by sqrt(N). So
 *   averaging 2 samples gives ~1.41× noise reduction, 4 samples gives 2× etc.
 *   We read 2 raw samples back-to-back as fast as I2C allows and average the
 *   raw integer counts before converting to float. This keeps integer arithmetic
 *   exact until the final conversion step and gives us a free ~1.41× improvement.
 *   The effective output rate becomes 1000 / 2 = 500 Hz, which is more than
 *   fast enough for a TVC loop (100–500 Hz is typical for model rocketry).
 *
 * Stage 3 — Software Exponential Moving Average (EMA) filter
 *   Even after stages 1 and 2, a small amount of sample-to-sample jitter
 *   remains. The EMA is a single-pole IIR low-pass filter:
 *     filtered = alpha * new_reading + (1 - alpha) * previous_filtered
 *   alpha = 1.0 means no smoothing (output = raw input).
 *   alpha = 0.0 means infinite smoothing (output never changes).
 *   At alpha = 0.4 and 500 Hz, the 3 dB cutoff of the EMA is roughly
 *     f_cutoff ≈ (alpha / (2π)) * Fs ≈ (0.4 / 6.28) * 500 ≈ 32 Hz
 *   which adds a soft roll-off below the hardware DLPF, further suppressing
 *   any noise that leaked through while keeping phase lag small enough for
 *   a 500 Hz control loop.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "I2Cdev.h"
#include "MPU6050.h"
#include "Wire.h"

// ─── Tunable constants ────────────────────────────────────────────────────────

// EMA smoothing factor (0 < alpha ≤ 1).
// Increase toward 1.0 for less smoothing / faster response.
// Decrease toward 0.0 for more smoothing / slower response.
// 0.4 is a good starting point for a 500 Hz TVC loop.
const float ALPHA = 0.4f;

// Number of raw samples averaged per control loop iteration.
// More samples = lower noise but lower effective output rate.
// 2 samples → effective 500 Hz, noise reduced by ~1.41×.
const int   NUM_AVG_SAMPLES = 2;

// Target loop rate in microseconds (1 000 000 / 500 = 2000 µs per iteration).
const unsigned long LOOP_PERIOD_US = 2000UL;

// Calibration: number of samples and warm-up delay.
const int   CAL_SAMPLES       = 3000;
const int   CAL_WARMUP_MS     = 2000;

// Serial baud rate — must match the Python visualiser constant.
const int   BAUD_RATE         = 115200;

// ─── Global objects ───────────────────────────────────────────────────────────

MPU6050 imu;  // Default I2C address 0x68 (AD0 tied to GND)

// Calibration bias — mean gyro output while stationary, in raw LSB counts.
// Subtracted from every reading so that true zero rotation → 0 LSB.
float biasX = 0.0f, biasY = 0.0f, biasZ = 0.0f;

// EMA filter state — initialised to zero, updated every loop iteration.
float filtGx = 0.0f, filtGy = 0.0f, filtGz = 0.0f;

// Integrated angles in degrees, accumulated from gyro rates × dt.
float pitch = 0.0f, roll = 0.0f, yaw = 0.0f;

// Timing: time of last loop iteration start in microseconds.
unsigned long lastTime = 0;

// ─── Helper: read one raw gyro sample ─────────────────────────────────────────
// Fills the three int16 references with raw register counts from GYRO_XOUT,
// GYRO_YOUT, GYRO_ZOUT (registers 0x43–0x48).
void readRawGyro(int16_t &rx, int16_t &ry, int16_t &rz) {
    imu.getRotation(&rx, &ry, &rz);
}

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(BAUD_RATE);

    // Initialise I2C bus.  Wire.begin() uses GPIO 21 (SDA) and 22 (SCL) by
    // default on the ESP32.  The MPU6050 supports up to 400 kHz fast-mode.
    Wire.begin();
    Wire.setClock(400000);  // 400 kHz reduces I2C read time → faster averaging loop

    // Initialise the MPU6050.  imu.initialize() writes to the PWR_MGMT_1
    // register (0x6B) to wake the device from sleep and select the internal
    // 8 MHz oscillator (or PLL if available).
    imu.initialize();

    if (!imu.testConnection()) {
        Serial.println("MPU6050 connection FAILED — check wiring");
        while (true) {}  // halt; nothing useful we can do
    }

    // ── Gyroscope full-scale range ─────────────────────────────────────────
    // Sets GYRO_CONFIG register (0x1B), bits [4:3] = FS_SEL = 0b00.
    // ±250 °/s → sensitivity = 131.0 LSB/(°/s).
    // Narrower range = finer resolution = lower quantisation noise per LSB.
    // For a TVC rocket that rotates at most ~90 °/s this range is plenty.
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);

    // ── Digital Low Pass Filter ────────────────────────────────────────────
    // Sets CONFIG register (0x1A), bits [2:0] = DLPF_CFG = 0b011.
    // Gyro bandwidth = 42 Hz, delay = 4.8 ms.
    // This is the single most effective noise-reduction step in the pipeline.
    imu.setDLPFMode(MPU6050_DLPF_BW_42);

    // ── Output data rate ───────────────────────────────────────────────────
    // Sets SMPLRT_DIV register (0x19).
    // Formula: ODR = Gyro_Rate / (1 + SMPLRT_DIV).
    // At DLPF_BW_42, Gyro_Rate = 1000 Hz.
    // SMPLRT_DIV = 0 → ODR = 1000 / (1 + 0) = 1000 Hz.
    // We want the fastest possible register update rate so our averaging loop
    // is pulling fresh data every ~1 ms.
    imu.setRate(0);

    // ── Bias calibration ──────────────────────────────────────────────────
    // Give the sensor 2 seconds to reach thermal equilibrium after power-on.
    // The gyro zero-rate offset drifts with temperature; collecting bias data
    // before the sensor stabilises will produce an inaccurate offset estimate.
    Serial.println("CALIBRATING");
    delay(CAL_WARMUP_MS);

    long sumX = 0, sumY = 0, sumZ = 0;
    for (int i = 0; i < CAL_SAMPLES; i++) {
        int16_t rx, ry, rz;
        readRawGyro(rx, ry, rz);
        sumX += rx;
        sumY += ry;
        sumZ += rz;
        // No explicit delay — the 400 kHz I2C transaction takes ~300 µs,
        // so we naturally sample at roughly 3000 Hz during calibration.
    }
    biasX = (float)sumX / CAL_SAMPLES;
    biasY = (float)sumY / CAL_SAMPLES;
    biasZ = (float)sumZ / CAL_SAMPLES;

    Serial.println("READY");

    lastTime = micros();
}

// ─── loop() ──────────────────────────────────────────────────────────────────
void loop() {
    // ── Precise 500 Hz timing ──────────────────────────────────────────────
    // Spin-wait until LOOP_PERIOD_US has elapsed since the last iteration.
    // Using micros() is more precise than delay() which has millisecond
    // resolution and cannot account for the time spent inside the loop body.
    while ((micros() - lastTime) < LOOP_PERIOD_US) {}
    unsigned long now = micros();
    float dt = (now - lastTime) * 1e-6f;  // actual timestep in seconds
    lastTime = now;

    // ── Multi-sample averaging ─────────────────────────────────────────────
    // Read NUM_AVG_SAMPLES raw samples as quickly as I2C allows and sum them.
    // Averaging in integer arithmetic avoids floating-point rounding on every
    // intermediate step.
    long rawSumX = 0, rawSumY = 0, rawSumZ = 0;
    for (int s = 0; s < NUM_AVG_SAMPLES; s++) {
        int16_t rx, ry, rz;
        readRawGyro(rx, ry, rz);
        rawSumX += rx;
        rawSumY += ry;
        rawSumZ += rz;
    }

    // ── Convert averaged raw counts to degrees per second ─────────────────
    // Sensitivity at ±250 °/s range = 131.0 LSB/(°/s)  (from datasheet Table 1)
    // Subtract calibration bias before dividing so the bias stays in LSB space.
    const float SENSITIVITY = 131.0f;
    float gx = ((float)rawSumX / NUM_AVG_SAMPLES - biasX) / SENSITIVITY;
    float gy = ((float)rawSumY / NUM_AVG_SAMPLES - biasY) / SENSITIVITY;
    float gz = ((float)rawSumZ / NUM_AVG_SAMPLES - biasZ) / SENSITIVITY;

    // ── Software EMA filter ────────────────────────────────────────────────
    // filtered = alpha * new + (1 - alpha) * previous
    // This is a first-order IIR low-pass filter.  Each new sample shifts the
    // output 40% toward the measurement and retains 60% of the previous state.
    filtGx = ALPHA * gx + (1.0f - ALPHA) * filtGx;
    filtGy = ALPHA * gy + (1.0f - ALPHA) * filtGy;
    filtGz = ALPHA * gz + (1.0f - ALPHA) * filtGz;

    // ── Gyroscope integration ─────────────────────────────────────────────
    // Integrate angular rate over dt to obtain accumulated angle.
    // This is the simplest (Euler) integration method: angle += rate * dt.
    // It accumulates drift over time because gyros have a slowly-varying bias
    // that calibration only removes at one temperature snapshot.  For a short
    // flight (< 30 s) and a well-calibrated sensor this is acceptable.
    pitch += filtGx * dt;
    roll  += filtGy * dt;
    yaw   += filtGz * dt;

    // ── Serial output ──────────────────────────────────────────────────────
    // Format: gx,gy,gz,pitch,roll,yaw,dt_us
    // dt is sent as integer microseconds so the PC can verify loop timing.
    Serial.print(filtGx, 4); Serial.print(',');
    Serial.print(filtGy, 4); Serial.print(',');
    Serial.print(filtGz, 4); Serial.print(',');
    Serial.print(pitch,  4); Serial.print(',');
    Serial.print(roll,   4); Serial.print(',');
    Serial.print(yaw,    4); Serial.print(',');
    Serial.println((unsigned long)(dt * 1e6f));
}
