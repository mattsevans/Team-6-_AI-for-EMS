#include <TinyWireM.h>      // I2C library for ATtiny
#include <Tiny4kOLED.h>     // OLED library

#define BQ_ADDR 0x55
#define SOC_CMD 0x2C  // Relative State of Charge (SoC)

uint16_t readWord(uint8_t reg) {
  TinyWireM.beginTransmission(BQ_ADDR);
  TinyWireM.write(reg);
  if (TinyWireM.endTransmission(false)) return 0xFFFF;

  TinyWireM.requestFrom(BQ_ADDR, (uint8_t)2);
  if (TinyWireM.available() < 2) return 0xFFFF;
  uint8_t lsb = TinyWireM.read();
  uint8_t msb = TinyWireM.read();
  return (msb << 8) | lsb;
}

void setup() {
  TinyWireM.begin();
  oled.begin();
  oled.clear();
  oled.on();

  oled.setFont(FONT8X16);
  oled.setCursor(10, 0);        // centered a bit
  oled.print("EMS ASSISTANT");
}

void loop() {
  uint16_t soc = readWord(SOC_CMD);
  if (soc == 0xFFFF) soc = 0;   // if no data

  oled.setCursor(25, 2);        // line 2 (middle)
  oled.print("CHG: ");
  oled.print(soc);
  oled.print("%   ");            // erase old numbers

  delay(1000);
}

