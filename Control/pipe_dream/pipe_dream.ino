// This example shows a simple way to control the
// Motoron Motor Controller using I2C.
//
// The motors will stop but automatically recover if:
// - Motor power (VIN) is interrupted, or
// - A temporary motor fault occurs, or
// - A command timeout occurs.
//
// The motors will stop until you power cycle or reset your
// Arduino if:
// - The Motoron experiences a reset.
//
// If a latched motor fault occurs, the motors
// experiencing the fault will stop until you power cycle motor
// power (VIN) or cause the motors to coast.

#include <Motoron.h>
#include <WiFi.h>
#include <WiFiUdp.h>

MotoronI2C mc;

// WiFi AP settings
const char *ssid = "pipedreamAP";  // WiFi network name
const char *password = "pipe_dream"; // WiFi password (8+ characters)

// UDP settings
WiFiUDP udp;
const int udpPort = 4210;
IPAddress laptopIP;

char packetBuffer[255];

int motor_speed = 600;

void setup()
{
  Serial.begin(115200);

  // Start WiFi in AP mode
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);
  delay(20);
  Serial.println("Access Point Started");
  Serial.print("Specified IP Address: ");
  Serial.println(WiFi.softAPIP()); // Print ESP32 AP IP specified above

  // Start UDP listener
  udp.begin(udpPort);
  Serial.println("Waiting for UDP packets...");

  Wire.begin();
  // Reset the controller to its default settings, then disable
  // CRC.  The bytes for each of these commands are shown here
  // in case you want to implement them on your own without
  // using the library.
  mc.reinitialize();    // Bytes: 0x96 0x74
  mc.disableCrc();      // Bytes: 0x8B 0x04 0x7B 0x43

  // Clear the reset flag, which is set after the controller
  // reinitializes and counts as an error.
  mc.clearResetFlag();  // Bytes: 0xA9 0x00 0x04

  // By default, the Motoron is configured to stop the motors if
  // it does not get a motor control command for 1500 ms.  You
  // can uncomment a line below to adjust this time or disable
  // the timeout feature.
  // mc.setCommandTimeoutMilliseconds(1000);
  // mc.disableCommandTimeout();

  // Configure motor 1
  mc.setMaxAcceleration(1, 140);
  mc.setMaxDeceleration(1, 300);
  // Configure motor 2
  mc.setMaxAcceleration(2, 140);
  mc.setMaxDeceleration(2, 300);

}

void loop()
{

  Serial.println(WiFi.softAPIP());

  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (len > 0) {
      packetBuffer[len] = '\0';  // Null-terminate string
    }

    String command = String(packetBuffer);
    Serial.print("Received: ");
    Serial.println(command);

    if (command == "UP") {
      // Handle UP
      Serial.println("Moving forward");
      mc.setSpeed(1, -motor_speed);
      mc.setSpeed(2, motor_speed);
    } else if (command == "DOWN") {
      Serial.println("Moving backward");
      mc.setSpeed(1, motor_speed);
      mc.setSpeed(2, -motor_speed);
    } else if (command == "LEFT") {
      Serial.println("Turning left");
      mc.setSpeed(1, -motor_speed);
      mc.setSpeed(2, 0);
    } else if (command == "RIGHT") {
      Serial.println("Turning right");
      mc.setSpeed(1, 0);
      mc.setSpeed(2, motor_speed);
    } else if (command == "STOP"){
      mc.setSpeed(1, 0);
      mc.setSpeed(2, 0);
    } else {
      Serial.println("Waiting for command...");
    }
  }

  // if (millis() & 2048)
  // {
  //   mc.setSpeed(1, 800);
  // }
  // else
  // {
  //   mc.setSpeed(1, -800);
  // }

}
