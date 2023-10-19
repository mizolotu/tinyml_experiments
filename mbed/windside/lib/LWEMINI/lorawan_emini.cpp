#include "lorawan_emini.h"
#include "lorawan/LoRaWANInterface.h"
#include "lorawan/system/lorawan_data_structures.h"
#include "events/EventQueue.h"
#include "lora_radio_helper.h"
#include "trace_helper.h"

extern LW_Emini lwemini;

#define CONFIRMED_MSG_RETRY_COUNTER 3
#define MAX_NUMBER_OF_EVENTS 10

using namespace events;

static EventQueue ev_queue(MAX_NUMBER_OF_EVENTS *EVENTS_EVENT_SIZE);

uint8_t rx_buffer[255]; // LORAMAC_PHY_MAXPAYLOAD
static void lora_event_handler(lorawan_event_t event);
static LoRaWANInterface lorawan(radio);
static lorawan_app_callbacks_t callbacks;

uint8_t LW_Emini::init()
{
    setup_trace();
    lorawan_status_t retcode;
    if (lorawan.initialize(&ev_queue) != LORAWAN_STATUS_OK)
    {
        printf("\r\n LoRa initialization failed! \r\n");
        return -1;
    }
    printf("\r\n Mbed LoRaWANStack initialized \r\n");
    callbacks.events = mbed::callback(lora_event_handler);
    lorawan.add_app_callbacks(&callbacks);
    if (lorawan.set_confirmed_msg_retries(CONFIRMED_MSG_RETRY_COUNTER) != LORAWAN_STATUS_OK)
    {
        printf("\r\n set_confirmed_msg_retries failed! \r\n\r\n");
        return -1;
    }
    printf("\r\n CONFIRMED message retries : %d \r\n",
           CONFIRMED_MSG_RETRY_COUNTER);
    if (lorawan.enable_adaptive_datarate() != LORAWAN_STATUS_OK)
    {
        printf("\r\n enable_adaptive_datarate failed! \r\n");
        return -1;
    }
    printf("\r\n Adaptive data  rate (ADR) - Enabled \r\n");
    retcode = lorawan.connect();
    if (retcode == LORAWAN_STATUS_OK ||
        retcode == LORAWAN_STATUS_CONNECT_IN_PROGRESS)
    {
    }
    else
    {
        printf("\r\n Connection error, code = %d \r\n", retcode);
        return -1;
    }
    printf("\r\n Connection - In Progress ...\r\n");
    ev_queue.dispatch_forever();
    return 1;
}

int16_t LW_Emini::send_message(uint8_t *payload, uint8_t payload_length)
{
    int16_t retcode;

    retcode = lorawan.send(MBED_CONF_LORA_APP_PORT, payload, payload_length,
                           MSG_UNCONFIRMED_FLAG);

    if (retcode < 0)
    {
        retcode == LORAWAN_STATUS_WOULD_BLOCK ? printf("send - WOULD BLOCK\r\n")
                                              : printf("\r\n send() - Error code %d \r\n", retcode);

        if (retcode == LORAWAN_STATUS_WOULD_BLOCK)
        {
            return retcode;
            // retry in 3 seconds
            //if (MBED_CONF_LORA_DUTY_CYCLE_ON)
            //{
                //ev_queue.call_in(3000, send_message(payload,payload_length));
                //ThisThread::sleep_for(3s);
                //send_message(payload, payload_length);
            //}
        }
        return 0;
    }

    printf("\r\n %d bytes scheduled for transmission \r\n", retcode);
    // memset(tx_buffer, 0, sizeof(tx_buffer)); //should we get rid of this
    return retcode;
}

void receive_message()
{
    uint8_t port;
    int flags;
    int16_t retcode = lorawan.receive(rx_buffer, sizeof(rx_buffer), port, flags);

    if (retcode < 0)
    {
        printf("\r\n receive() - Error code %d \r\n", retcode);
        return;
    }

    printf(" RX Data on port %u (%d bytes): ", port, retcode);
    for (uint8_t i = 0; i < retcode; i++)
    {
        printf("%02x ", rx_buffer[i]);
    }
    printf("\r\n");

    memset(rx_buffer, 0, sizeof(rx_buffer));
}

void lora_event_handler(lorawan_event_t event)
{
    switch (event)
    {
    case CONNECTED:
        lwemini.joined = true;
        printf("\r\n Connection - Successful \r\n");
        break;
    case DISCONNECTED:
        lwemini.joined = false;
        ev_queue.break_dispatch();
        printf("\r\n Disconnected Successfully \r\n");
        break;
    case TX_DONE:
        printf("\r\n Message Sent to Network Server \r\n");
        break;
    case TX_TIMEOUT:
    case TX_ERROR:
    case TX_CRYPTO_ERROR:
    case TX_SCHEDULING_ERROR:
        printf("\r\n Transmission Error - EventCode = %d \r\n", event);
        break;
    case RX_DONE:
        printf("\r\n Received message from Network Server \r\n");
        receive_message();
        break;
    case RX_TIMEOUT:
    case RX_ERROR:
        printf("\r\n Error in reception - Code = %d \r\n", event);
        break;
    case JOIN_FAILURE:
        printf("\r\n OTAA Failed - Check Keys \r\n");
        break;
    case UPLINK_REQUIRED:
        printf("\r\n Uplink required by NS \r\n");
        break;
    default:
        MBED_ASSERT("Unknown Event");
    }
}