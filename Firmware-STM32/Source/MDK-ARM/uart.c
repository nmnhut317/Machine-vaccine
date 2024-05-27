#include "uart.h"
#include "flash.h"

UART_HandleTypeDef *_huart;
extern uint32_t Data_flash;
extern TIM_HandleTypeDef htim1;

static uint8_t length;
uint8_t databuff[5];
volatile uint8_t flag = 0;
volatile uint8_t dir = 0;
volatile uint16_t position = 0;
volatile uint32_t timebuf = 0;
volatile uint32_t tempTime = 0;

volatile uint8_t positionPrevious = 0;
volatile uint8_t positionCurrent = 0;
volatile uint8_t set = 0;

const uint32_t Overflow	= 10000000U;

static char** StringTok(uint8_t* data)
{
	char** ptr = malloc(10*sizeof(char*));
	for(uint8_t i=0; i<10; i++)
	{
		ptr[i] = malloc(20*sizeof(char));
	}
	
	int n = 0;
	char *token = strtok((char*)data, " ");
	
	while(token != NULL)
	{
		strcpy(ptr[n++], token);
		token = strtok(NULL, " "); 
	}	
	
	return ptr;
}


void uart_proce(uint8_t flags)
{	
	if(flags)
	{
		positionCurrent = (uint32_t)atoi((char*)databuff);
		
		memset(databuff, '\0', sizeof(databuff));
		length = 0;
		flag = 0;	
		
		if (positionCurrent > positionPrevious)
		{
			position = positionCurrent - positionPrevious;
			control_step(1, position); /* 0 tien, 1 lui*/			
		}
		
		else if (positionCurrent < positionPrevious)
		{
			position = abs(positionCurrent - positionPrevious);
			control_step(0, position); /* 0 tien, 1 lui*/			
		}
		
		else
		{
			control_step(0, 0);
		}			
		
		positionPrevious =  positionCurrent;
		
		
		/* wait until when the INPUT gets signal */
		TIM1->CNT = 0;
		while(HAL_GPIO_ReadPin(PORTB, INPUT)) 
		{
			if(TIM1->CNT == 0)
			{
				tempTime = 0;
			}
			
			timebuf += (TIM1->CNT - tempTime);
			tempTime = TIM1->CNT;
			
			if(timebuf >= Overflow)
			{
				timebuf = 0;
				tempTime = 0;
				break;
			}
		}
		
//		while(HAL_GPIO_ReadPin(PORTB,INPUT));
		msDelay(500);
		HAL_GPIO_WritePin(PORTA, CLAMP, GPIO_PIN_SET);
		ACT_PUSH(500, 1);
		msDelay(200);
		ACT_PUMP(700, 0);
		ACT_PUSH(500, 0);		
		ACT_PUMP(700, 1);
		HAL_GPIO_WritePin(PORTA, CLAMP, GPIO_PIN_RESET);
		msDelay(500);
		HAL_GPIO_WritePin(PORTA, BLOCK, GPIO_PIN_SET);
		
//		while(HAL_GPIO_ReadPin(PORTB, OUTPUT));	
		TIM1->CNT = 0;
		while(HAL_GPIO_ReadPin(PORTB, OUTPUT)) /* wait until when the INPUT gets signal */
		{
			if(TIM1->CNT == 0)
			{
				tempTime = 0;
			}
			
			timebuf += (TIM1->CNT - tempTime);
			tempTime = TIM1->CNT;
			
			if(timebuf >= Overflow)
			{
				timebuf = 0;
				tempTime = 0;
				break;
			}
		}
		msDelay(500);		
		HAL_GPIO_WritePin(PORTA, BLOCK, GPIO_PIN_RESET);				
	}
}

void uart_handle(uint8_t datarx)
{
	if(datarx == '\n')
	{
		flag = 1;
	}
	
	/* process when receive data fish wrong side */
	else if (datarx == '\r')
	{
		//todo
		HAL_GPIO_WritePin(PORTA, BLOCK, GPIO_PIN_SET);
//		while(HAL_GPIO_ReadPin(PORTB, OUTPUT));	
		TIM1->CNT = 0;
		while(HAL_GPIO_ReadPin(PORTB, OUTPUT)) /* wait until when the INPUT gets signal */
		{
			if(TIM1->CNT == 0)
			{
				tempTime = 0;
			}
			
			timebuf += (TIM1->CNT - tempTime);
			tempTime = TIM1->CNT;
			
			if(timebuf >= Overflow)
			{
				timebuf = 0;
				tempTime = 0;
				break;
			}
		}		
		msDelay(500);		
		HAL_GPIO_WritePin(PORTA, BLOCK, GPIO_PIN_RESET);		
	}
	
	else if (datarx == 'e')
	{
	  //todo
	}
	
	else 
	{
		databuff[length++] = datarx;
	}
}


void uart_init(UART_HandleTypeDef *huart1)
{
	_huart = huart1;
}

