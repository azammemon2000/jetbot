#!/usr/bin/env python3
import os, fcntl

I2C_BUS = '/dev/i2c-1'
I2C_SLAVE = 0x0703
BAT_ADDR = 0x1b

fd = os.open(I2C_BUS, os.O_RDWR)
try:
    fcntl.ioctl(fd, I2C_SLAVE, BAT_ADDR)
    os.write(fd, bytes([0x00]))
    data = os.read(fd, 2)
    raw = (data[0] << 8) + data[1]
    vol = raw * 13.3 / 1023.0
    print("Battery Telemetry ADC (0x1b):")
    print("  Raw ADC: %d" % raw)
    print("  Voltage: %.2f V" % vol)
    if vol >= 11.5:
        print("  Status: HIGH (Normal)")
    elif vol >= 10.5:
        print("  Status: MEDIUM")
    elif vol >= 9.9:
        print("  Status: LOW (Charge soon)")
    else:
        print("  Status: EMPTY / CUTOFF (Motors disabled)")
finally:
    os.close(fd)
