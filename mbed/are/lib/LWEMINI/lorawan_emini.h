#ifndef LORAWAN_EMINI_H_
#define LORAWAN_EMINI_H_

#include "mbed.h"

class LW_Emini
{
public:
    bool joined = false;
    uint8_t init();
    int16_t send_message(uint8_t *payload, uint8_t payload_length);
};

#endif