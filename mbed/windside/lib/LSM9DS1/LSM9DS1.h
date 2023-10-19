// Most of the Credit goes to jimblom
// Modifications by Allen Wild
#ifndef _LSM9DS1_H__
#define _LSM9DS1_H__

#include "mbed.h"

/////////////////////////////////////////
// LSM9DS1 Accel/Gyro (XL/G) Registers //
/////////////////////////////////////////
#define ACT_THS             0x04
#define ACT_DUR             0x05
#define INT_GEN_CFG_XL      0x06
#define INT_GEN_THS_X_XL    0x07
#define INT_GEN_THS_Y_XL    0x08
#define INT_GEN_THS_Z_XL    0x09
#define INT_GEN_DUR_XL      0x0A
#define REFERENCE_G         0x0B
#define INT1_CTRL           0x0C
#define INT2_CTRL           0x0D
#define WHO_AM_I_XG         0x0F
#define CTRL_REG1_G         0x10
#define CTRL_REG2_G         0x11
#define CTRL_REG3_G         0x12
#define ORIENT_CFG_G        0x13
#define INT_GEN_SRC_G       0x14
#define OUT_TEMP_L          0x15
#define OUT_TEMP_H          0x16
#define STATUS_REG_0        0x17
#define OUT_X_L_G           0x18
#define OUT_X_H_G           0x19
#define OUT_Y_L_G           0x1A
#define OUT_Y_H_G           0x1B
#define OUT_Z_L_G           0x1C
#define OUT_Z_H_G           0x1D
#define CTRL_REG4           0x1E
#define CTRL_REG5_XL        0x1F
#define CTRL_REG6_XL        0x20
#define CTRL_REG7_XL        0x21
#define CTRL_REG8           0x22
#define CTRL_REG9           0x23
#define CTRL_REG10          0x24
#define INT_GEN_SRC_XL      0x26
#define STATUS_REG_1        0x27
#define OUT_X_L_XL          0x28
#define OUT_X_H_XL          0x29
#define OUT_Y_L_XL          0x2A
#define OUT_Y_H_XL          0x2B
#define OUT_Z_L_XL          0x2C
#define OUT_Z_H_XL          0x2D
#define FIFO_CTRL           0x2E
#define FIFO_SRC            0x2F
#define INT_GEN_CFG_G       0x30
#define INT_GEN_THS_XH_G    0x31
#define INT_GEN_THS_XL_G    0x32
#define INT_GEN_THS_YH_G    0x33
#define INT_GEN_THS_YL_G    0x34
#define INT_GEN_THS_ZH_G    0x35
#define INT_GEN_THS_ZL_G    0x36
#define INT_GEN_DUR_G       0x37

///////////////////////////////
// LSM9DS1 Magneto Registers //
///////////////////////////////
#define OFFSET_X_REG_L_M    0x05
#define OFFSET_X_REG_H_M    0x06
#define OFFSET_Y_REG_L_M    0x07
#define OFFSET_Y_REG_H_M    0x08
#define OFFSET_Z_REG_L_M    0x09
#define OFFSET_Z_REG_H_M    0x0A
#define WHO_AM_I_M          0x0F
#define CTRL_REG1_M         0x20
#define CTRL_REG2_M         0x21
#define CTRL_REG3_M         0x22
#define CTRL_REG4_M         0x23
#define CTRL_REG5_M         0x24
#define STATUS_REG_M        0x27
#define OUT_X_L_M           0x28
#define OUT_X_H_M           0x29
#define OUT_Y_L_M           0x2A
#define OUT_Y_H_M           0x2B
#define OUT_Z_L_M           0x2C
#define OUT_Z_H_M           0x2D
#define INT_CFG_M           0x30
#define INT_SRC_M           0x30
#define INT_THS_L_M         0x32
#define INT_THS_H_M         0x33

////////////////////////////////
// LSM9DS1 WHO_AM_I Responses //
////////////////////////////////
#define WHO_AM_I_AG_RSP     0x68
#define WHO_AM_I_M_RSP      0x3D

// Possible I2C addresses for the accel/gyro and mag
#define LSM9DS1_AG_I2C_ADDR(sa0) ((sa0) ? 0xD6 : 0xD4)
#define LSM9DS1_M_I2C_ADDR(sa1) ((sa1) ? 0x3C : 0x38)

/**
 * LSM9DS1 Class - driver for the 9 DoF IMU
 */
class LSM9DS1
{
public:

    /// gyro_scale defines the possible full-scale ranges of the gyroscope:
    enum gyro_scale
    {
        G_SCALE_245DPS = 0x0 << 3,     // 00 << 3: +/- 245 degrees per second
        G_SCALE_500DPS = 0x1 << 3,     // 01 << 3: +/- 500 dps
        G_SCALE_2000DPS = 0x3 << 3     // 11 << 3: +/- 2000 dps
    };

    /// gyro_odr defines all possible data rate/bandwidth combos of the gyro:
    enum gyro_odr
    {                               // ODR (Hz) --- Cutoff
        G_POWER_DOWN     = 0x00,    //  0           0
        G_ODR_15_BW_0    = 0x20,    //  14.9        0
        G_ODR_60_BW_16   = 0x40,    //  59.5        16
        G_ODR_119_BW_14  = 0x60,    //  119         14
        G_ODR_119_BW_31  = 0x61,    //  119         31
        G_ODR_238_BW_14  = 0x80,    //  238         14
        G_ODR_238_BW_29  = 0x81,    //  238         29
        G_ODR_238_BW_63  = 0x82,    //  238         63
        G_ODR_238_BW_78  = 0x83,    //  238         78
        G_ODR_476_BW_21  = 0xA0,    //  476         21
        G_ODR_476_BW_28  = 0xA1,    //  476         28
        G_ODR_476_BW_57  = 0xA2,    //  476         57
        G_ODR_476_BW_100 = 0xA3,    //  476         100
        G_ODR_952_BW_33  = 0xC0,    //  952         33
        G_ODR_952_BW_40  = 0xC1,    //  952         40
        G_ODR_952_BW_58  = 0xC2,    //  952         58
        G_ODR_952_BW_100 = 0xC3     //  952         100
    };

    /// accel_scale defines all possible FSR's of the accelerometer:
    enum accel_scale
    {
        A_SCALE_2G, // 00: +/- 2g
        A_SCALE_16G,// 01: +/- 16g
        A_SCALE_4G, // 10: +/- 4g
        A_SCALE_8G  // 11: +/- 8g
    };

    /// accel_oder defines all possible output data rates of the accelerometer:
    enum accel_odr
    {
        A_POWER_DOWN,   // Power-down mode (0x0)
        A_ODR_10,       // 10 Hz (0x1)
        A_ODR_50,       // 50 Hz (0x2)
        A_ODR_119,      // 119 Hz (0x3)
        A_ODR_238,      // 238 Hz (0x4)
        A_ODR_476,      // 476 Hz (0x5)
        A_ODR_952       // 952 Hz (0x6)
    };

    // accel_bw defines all possible bandwiths for low-pass filter of the accelerometer:
    enum accel_bw
    {
        A_BW_AUTO_SCALE = 0x0,  // Automatic BW scaling (0x0)
        A_BW_408 = 0x4,         // 408 Hz (0x4)
        A_BW_211 = 0x5,         // 211 Hz (0x5)
        A_BW_105 = 0x6,         // 105 Hz (0x6)
        A_BW_50 = 0x7           // 50 Hz (0x7)
    };

    /// mag_scale defines all possible FSR's of the magnetometer:
    enum mag_scale
    {
        M_SCALE_4GS,    // 00: +/- 4Gs
        M_SCALE_8GS,    // 01: +/- 8Gs
        M_SCALE_12GS,   // 10: +/- 12Gs
        M_SCALE_16GS,   // 11: +/- 16Gs
    };

    /// mag_odr defines all possible output data rates of the magnetometer:
    enum mag_odr
    {
        M_ODR_0625, // 0.625 Hz (0x00)
        M_ODR_125,  // 1.25 Hz  (0x01)
        M_ODR_25,   // 2.5 Hz   (0x02)
        M_ODR_5,    // 5 Hz     (0x03)
        M_ODR_10,   // 10       (0x04)
        M_ODR_20,   // 20 Hz    (0x05)
        M_ODR_40,   // 40 Hz    (0x06)
        M_ODR_80    // 80 Hz    (0x07)
    };

    // We'll store the gyro, accel, and magnetometer readings in a series of
    // public class variables. Each sensor gets three variables -- one for each
    // axis. Call readGyro(), readAccel(), and readMag() first, before using
    // these variables!
    // These values are the RAW signed 16-bit readings from the sensors.
    int16_t gx_raw, gy_raw, gz_raw; // x, y, and z axis readings of the gyroscope
    int16_t ax_raw, ay_raw, az_raw; // x, y, and z axis readings of the accelerometer
    int16_t mx_raw, my_raw, mz_raw; // x, y, and z axis readings of the magnetometer
    int16_t temperature_raw;

    // floating-point values of scaled data in real-world units
    float gx, gy, gz;
    float ax, ay, az;
    float mx, my, mz;
    float temperature_c, temperature_f; // temperature in celcius and fahrenheit

    
    /**  LSM9DS1 -- LSM9DS1 class constructor
    *  The constructor will set up a handful of private variables, and set the
    *  communication mode as well.
    *  Input:
    *   - interface = Either MODE_SPI or MODE_I2C, whichever you're using
    *               to talk to the IC.
    *   - xgAddr = If MODE_I2C, this is the I2C address of the accel/gyro.
    *               If MODE_SPI, this is the chip select pin of the accel/gyro (CS_A/G)
    *   - mAddr = If MODE_I2C, this is the I2C address of the mag.
    *               If MODE_SPI, this is the cs pin of the mag (CS_M)
    */
    LSM9DS1(I2C& p_i2c, uint8_t xgAddr = LSM9DS1_AG_I2C_ADDR(1), uint8_t mAddr = LSM9DS1_M_I2C_ADDR(1));
    
    /**  begin() -- Initialize the gyro, accelerometer, and magnetometer.
    *  This will set up the scale and output rate of each sensor. It'll also
    *  "turn on" every sensor and every axis of every sensor.
    *  Input:
    *   - gScl = The scale of the gyroscope. This should be a gyro_scale value.
    *   - aScl = The scale of the accelerometer. Should be a accel_scale value.
    *   - mScl = The scale of the magnetometer. Should be a mag_scale value.
    *   - gODR = Output data rate of the gyroscope. gyro_odr value.
    *   - aODR = Output data rate of the accelerometer. accel_odr value.
    *   - mODR = Output data rate of the magnetometer. mag_odr value.
    *  Output: The function will return an unsigned 16-bit value. The most-sig
    *       bytes of the output are the WHO_AM_I reading of the accel/gyro. The
    *       least significant two bytes are the WHO_AM_I reading of the mag.
    *  All parameters have a defaulted value, so you can call just "begin()".
    *  Default values are FSR's of: +/- 245DPS, 4g, 2Gs; ODRs of 119 Hz for 
    *  gyro, 119 Hz for accelerometer, 80 Hz for magnetometer.
    *  Use the return value of this function to verify communication.
    */
    uint16_t begin(gyro_scale gScl = G_SCALE_245DPS, 
                accel_scale aScl = A_SCALE_2G, mag_scale mScl = M_SCALE_4GS,
                gyro_odr gODR = G_ODR_119_BW_14, accel_odr aODR = A_ODR_119, 
                mag_odr mODR = M_ODR_80);
    
    /**  readGyro() -- Read the gyroscope output registers.
    *  This function will read all six gyroscope output registers.
    *  The readings are stored in the class' gx_raw, gy_raw, and gz_raw variables. Read
    *  those _after_ calling readGyro().
    */
    void readGyro();
    
    /**  readAccel() -- Read the accelerometer output registers.
    *  This function will read all six accelerometer output registers.
    *  The readings are stored in the class' ax_raw, ay_raw, and az_raw variables. Read
    *  those _after_ calling readAccel().
    */
    void readAccel();
    
    /**  readMag() -- Read the magnetometer output registers.
    *  This function will read all six magnetometer output registers.
    *  The readings are stored in the class' mx_raw, my_raw, and mz_raw variables. Read
    *  those _after_ calling readMag().
    */
    void readMag();
    
    /**  readTemp() -- Read the temperature output register.
    *  This function will read two temperature output registers.
    *  The combined readings are stored in the class' temperature variables. Read
    *  those _after_ calling readTemp().
    */
    void readTemp();
    
    /**  setGyroScale() -- Set the full-scale range of the gyroscope.
    *  This function can be called to set the scale of the gyroscope to 
    *  245, 500, or 2000 degrees per second.
    *  Input:
    *   - gScl = The desired gyroscope scale. Must be one of three possible
    *       values from the gyro_scale enum.
    */
    void setGyroScale(gyro_scale gScl);
    
    /**  setAccelScale() -- Set the full-scale range of the accelerometer.
    *  This function can be called to set the scale of the accelerometer to
    *  2, 4, 8, or 16 g's.
    *  Input:
    *   - aScl = The desired accelerometer scale. Must be one of five possible
    *       values from the accel_scale enum.
    */
    void setAccelScale(accel_scale aScl);
    
    /**  setMagScale() -- Set the full-scale range of the magnetometer.
    *  This function can be called to set the scale of the magnetometer to
    *  4, 8, 12, or 16 Gs.
    *  Input:
    *   - mScl = The desired magnetometer scale. Must be one of four possible
    *       values from the mag_scale enum.
    */
    void setMagScale(mag_scale mScl);
    
    /**  setGyroODR() -- Set the output data rate and bandwidth of the gyroscope
    *  Input:
    *   - gRate = The desired output rate and cutoff frequency of the gyro.
    *       Must be a value from the gyro_odr enum (check above).
    */
    void setGyroODR(gyro_odr gRate);
    
    /**  setAccelODR() -- Set the output data rate of the accelerometer
    *  Input:
    *   - aRate = The desired output rate of the accel.
    *       Must be a value from the accel_odr enum (check above).
    */
    void setAccelODR(accel_odr aRate);
    
    /**  setMagODR() -- Set the output data rate of the magnetometer
    *  Input:
    *   - mRate = The desired output rate of the mag.
    *       Must be a value from the mag_odr enum (check above).
    */
    void setMagODR(mag_odr mRate);


private:    
    /**  xgAddress and mAddress store the I2C address
    *  for each sensor.
    */
    uint8_t xgAddress, mAddress;
    
    // I2C bus
    I2C &i2c;

    /**  gScale, aScale, and mScale store the current scale range for each 
    *  sensor. Should be updated whenever that value changes.
    */
    gyro_scale gScale;
    accel_scale aScale;
    mag_scale mScale;
    
    /**  gRes, aRes, and mRes store the current resolution for each sensor. 
    *  Units of these values would be DPS (or g's or Gs's) per ADC tick.
    *  This value is calculated as (sensor scale) / (2^15).
    */
    float gRes, aRes, mRes;
    
    /**  initGyro() -- Sets up the gyroscope to begin reading.
    *  This function steps through all three gyroscope control registers.
    */
    void initGyro();
    
    /**  initAccel() -- Sets up the accelerometer to begin reading.
    *  This function steps through all accelerometer related control registers.
    */
    void initAccel();
    
    /**  initMag() -- Sets up the magnetometer to begin reading.
    *  This function steps through all magnetometer-related control registers.
    */
    void initMag();
    
    /**  calcgRes() -- Calculate the resolution of the gyroscope.
    *  This function will set the value of the gRes variable. gScale must
    *  be set prior to calling this function.
    */
    void calcgRes();
    
    /**  calcmRes() -- Calculate the resolution of the magnetometer.
    *  This function will set the value of the mRes variable. mScale must
    *  be set prior to calling this function.
    */
    void calcmRes();
    
    /**  calcaRes() -- Calculate the resolution of the accelerometer.
    *  This function will set the value of the aRes variable. aScale must
    *  be set prior to calling this function.
    */
    void calcaRes();
};

#endif // _LSM9DS1_H //
