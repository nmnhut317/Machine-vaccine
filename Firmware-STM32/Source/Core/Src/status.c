#include "status.h" 

int a = 0;
extern TIM_HandleTypeDef htim1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim3;
int stepDelay = 500 ;// 1000us more delay means less speed

void microDelay (uint16_t delay)
{
  __HAL_TIM_SET_COUNTER(&htim1, 0);
  while (__HAL_TIM_GET_COUNTER(&htim1) < delay);
}

void msDelay(uint16_t ms)
{
	for(uint16_t i=0; i<ms; i++)
	{
		microDelay(1000);
	}
}

//void microDelayTim2 (uint16_t delay)
//{
//	 __HAL_TIM_SET_COUNTER(&htim2, 0);
//  while (__HAL_TIM_GET_COUNTER(&htim2) < delay);
//}



void mode1(int DOR , int RT)
{    
	 
//		lcd_put_cur(1,0);
//		lcd_send_string ("Full Step");

	  HAL_GPIO_WritePin(PORTA, M0, 0);
	  HAL_GPIO_WritePin(PORTA, M1, 0);
	  HAL_GPIO_WritePin(PORTA, M2, 0);
    HAL_GPIO_WritePin(PORTA, DIR_PIN, DOR);
    for(int x=0; x<(200*RT); x=x+1)
    {
      a++;
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay);
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
			{
//				lcd_clear ();
//				lcd_put_cur(1,0);
//				lcd_send_string ("STOP");
				break;
			}

		}
}
void mode2(int DOR , int ST )
 {    
	  HAL_GPIO_WritePin(PORTA, M0, 1);
	  HAL_GPIO_WritePin(PORTA, M1, 1);
	  HAL_GPIO_WritePin(PORTA, M2, 1);
		int x;
    HAL_GPIO_WritePin(PORTA, DIR_PIN, DOR);
    for(x=0; x<800*ST; x=x+1)
    {
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay/32);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay/32);
    }
}
 
void control_step(int DOR , int ST)
{
	  HAL_GPIO_WritePin(PORTA, M0, 1);
	  HAL_GPIO_WritePin(PORTA, M1, 1);
	  HAL_GPIO_WritePin(PORTA, M2, 1);
    HAL_GPIO_WritePin(PORTA, DIR_PIN, DOR);
	
    for(int x=0; x<800*ST ; x=x+1)
    {
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay/32);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay/32);
			
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
				{
					mode2(1,3);
					break;			
				}
			if (HAL_GPIO_ReadPin(PORTB,Starts) == PRESSED )
				{
					mode2(0,3);
					break;			
				}				
	 }
}


void TIM_XL_PUMP(uint16_t ms)
{
	HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_RESET);
	
	for(uint16_t i=0; i<ms; i++)
	{
		HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_SET);
		microDelay(1000);
	}
	
	HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_RESET);
}


void ACT_PUMP(uint16_t ms, uint8_t rotate)
{
	HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_RESET);
	if(rotate)
	{
		HAL_GPIO_WritePin(PORTB, IN1_XL_PUMP, GPIO_PIN_SET);
		HAL_GPIO_WritePin(PORTB, IN2_XL_PUMP, GPIO_PIN_RESET);
		
		HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_SET);
		while(HAL_GPIO_ReadPin(PORTA, SWITCH_STOP_PUMP));	
	}
	else
	{
		HAL_GPIO_WritePin(PORTB, IN1_XL_PUMP, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(PORTB, IN2_XL_PUMP, GPIO_PIN_SET);		
		
		HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_SET);
		while(HAL_GPIO_ReadPin(PORTA, SWITCH_START_PUMP));	
	}
	HAL_GPIO_WritePin(PORTB, XILANH_PUMP_ENA, GPIO_PIN_RESET);
}



void TIM_XL_PUSH(uint16_t ms)
{
	HAL_GPIO_WritePin(PORTB, XILANH_PUSH_ENB, GPIO_PIN_RESET);
	
	for(uint16_t i=0; i<ms; i++)
	{
		HAL_GPIO_WritePin(PORTB, XILANH_PUSH_ENB, GPIO_PIN_SET);
		microDelay(1000);
	}
	
	HAL_GPIO_WritePin(PORTB, XILANH_PUSH_ENB, GPIO_PIN_RESET);
}


void ACT_PUSH(uint16_t ms, uint8_t rotate)
{
	HAL_GPIO_WritePin(PORTB, XILANH_PUSH_ENB, GPIO_PIN_RESET);	
	if(rotate)
	{
		HAL_GPIO_WritePin(PORTB, IN4_XL_PUSH, GPIO_PIN_SET);
		HAL_GPIO_WritePin(PORTB, IN3_XL_PUSH, GPIO_PIN_RESET);
		
		HAL_GPIO_WritePin(PORTB, XILANH_PUSH_ENB, GPIO_PIN_SET);
		while(HAL_GPIO_ReadPin(PORTB, SWITCH_STOP_PUSH));
		HAL_GPIO_WritePin(PORTB, XILANH_PUSH_ENB, GPIO_PIN_RESET);		
	}
	else
	{
		HAL_GPIO_WritePin(PORTB, IN4_XL_PUSH, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(PORTB, IN3_XL_PUSH, GPIO_PIN_SET);
		TIM_XL_PUSH(ms);		
	}
}

void start_itinerary(uint32_t mm)
{
	  HAL_GPIO_WritePin(PORTA, M0, 1);
	  HAL_GPIO_WritePin(PORTA, M1, 1);
	  HAL_GPIO_WritePin(PORTA, M2, 1);
    HAL_GPIO_WritePin(PORTA, DIR_PIN, 0);
	
    for(int x=0; x<800*mm ; x=x+1)
    {
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay/32);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay/32);
			
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
				{
					break;			
				}
	 }	
}

