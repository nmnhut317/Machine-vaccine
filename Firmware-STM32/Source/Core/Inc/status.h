#ifndef __STATUS__
#define __STATUS__

#include "stm32f1xx_hal.h"

/*  **	**	**	**	*/
#define PORTA 							GPIOA

#define DIR_PIN 						GPIO_PIN_1
#define STEP_PIN  					GPIO_PIN_2
#define M0 									GPIO_PIN_5
#define M1 									GPIO_PIN_4
#define M2	  							GPIO_PIN_3

#define SWITCH_START_PUMP   GPIO_PIN_6		/*INPUT*/
#define SWITCH_STOP_PUMP    GPIO_PIN_7		/*INPUT*/

#define CLAMP     					GPIO_PIN_12		/*OUTPUT*/
#define BLOCK   						GPIO_PIN_11		/*OUTPUT*/

/*  **	**	**	**	*/
#define PORTB 							GPIOB

#define Starts 							GPIO_PIN_11 	/*INPUT*/
#define Stops  							GPIO_PIN_10	/*INPUT*/

#define INPUT    						GPIO_PIN_1		/*INPUT*/
#define OUTPUT  						GPIO_PIN_0		/*INPUT*/

#define SWITCH_STOP_PUSH    GPIO_PIN_8		/*INPUT*/

#define XILANH_PUMP_ENA   	GPIO_PIN_3		/*OUTPUT*/
#define IN1_XL_PUMP    			GPIO_PIN_4		/*OUTPUT*/
#define IN2_XL_PUMP   			GPIO_PIN_5		/*OUTPUT*/

#define XILANH_PUSH_ENB   	GPIO_PIN_12		/*OUTPUT*/
#define IN3_XL_PUSH    			GPIO_PIN_13		/*OUTPUT*/
#define IN4_XL_PUSH  				GPIO_PIN_14		/*OUTPUT*/

#define PRESSED  0


void start_itinerary(uint32_t mm);
void mode1(int DOR, int ST );
void mode2(int DOR, int ST);
void control_step(int DOR ,int ST);
void microDelay (uint16_t delay);
void msDelay(uint16_t ms);
void ACT_PUMP(uint16_t ms, uint8_t rotate);
void ACT_PUSH(uint16_t ms, uint8_t rotate);

#endif
