#include "button.h"

#define PORTB GPIOB
#define PORTA GPIOA
#define Starts GPIO_PIN_11
#define Stops  GPIO_PIN_10
#define REV    GPIO_PIN_1
#define MOD1   GPIO_PIN_0
#define MOD2   GPIO_PIN_7
#define MOD3   GPIO_PIN_6
#define PRESSED  0 

int status ;
int ST = 20 ;
char datas[20] ;
void button()
{   
	 
		if (HAL_GPIO_ReadPin(PORTB,Starts) == PRESSED && HAL_GPIO_ReadPin(PORTB,REV) == PRESSED )
		{
    status = 4;
		lcd_clear ();
		}
		if(status==4)
		{
			while(1){
			lcd_put_cur(0,5);
			lcd_send_string ("SETTING");
			if (HAL_GPIO_ReadPin(PORTB,Starts) == PRESSED )
		   {
				 ST++;
				 sprintf(datas,"%d",ST);
				 lcd_put_cur(1,0);
			   lcd_send_string ("Rotation:");
				 lcd_put_cur(1,9);
			   lcd_send_string ("    ");
				 lcd_put_cur(1,9);
			   lcd_send_string (datas);
				 HAL_Delay(500);	
			 }
			if (HAL_GPIO_ReadPin(PORTB,REV) == PRESSED )
		   {
				 ST--;
				 sprintf(datas,"%d",ST);
				 lcd_put_cur(1,0);
			   lcd_send_string ("Rotation:");
				 lcd_put_cur(1,9);
			   lcd_send_string ("    ");
				 lcd_put_cur(1,9);
			   lcd_send_string (datas);
				 HAL_Delay(500);	
			 }
			 
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
		   { 
				 status = 0;
				 break;
				 	
			 }
		}
	}
	if (HAL_GPIO_ReadPin(PORTB,Starts) == PRESSED )
		{
			lcd_clear ();
			lcd_put_cur(1,0);
			lcd_send_string ("START");
			if (status ==1 ) mode1(0,ST);
			if (status ==2 ) mode2(0,ST);
			if (status ==3 ) control_step(0,ST);
			
		}

  if (HAL_GPIO_ReadPin(PORTB,REV) == PRESSED )
		{
			lcd_clear ();
			lcd_put_cur(1,0);
			lcd_send_string ("REV");
			if (status ==1 ) mode1(1,ST);
			if (status ==2 ) mode2(1,ST);
			if (status ==3 ) control_step(1,ST);
		}
			/////
	if (HAL_GPIO_ReadPin(PORTB,MOD1) == PRESSED )
		{
			lcd_clear ();
			lcd_put_cur(1,0);
			lcd_send_string ("MOD1");
			status = 1 ;
		}
  if (HAL_GPIO_ReadPin(PORTA,MOD2) == PRESSED )
		{
			lcd_clear ();
			lcd_put_cur(1,0);
			lcd_send_string ("MOD2");
			status = 2 ;
		}
  if (HAL_GPIO_ReadPin(PORTA,MOD3) == PRESSED )
		{
			lcd_clear ();
			lcd_put_cur(1,0);
			lcd_send_string ("MOD3");
			status = 3 ;
		}
		
}
