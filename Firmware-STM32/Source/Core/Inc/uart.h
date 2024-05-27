#ifndef	__UART_H__
#define __UART_H__

#include "stm32f1xx_hal.h"
#include "stdint.h"
#include "stdlib.h"
#include "string.h"
#include "status.h"

void uart_handle(uint8_t datarx);
void uart_init(UART_HandleTypeDef *huart1);
void uart_proce(uint8_t flags);
#endif

