# Appendix

## User Information List

Periodically update user passwords to avoid the risks caused by using the same password for a long time.

**Table 1** User list

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|root|Used for deploying RecSDK-Torch.|User-defined|Run the `passwd` command to change it.|
|HwHiAiUser|User for installing drivers and running the demo.|User-defined|Run the `passwd` command to change it.|

**Base Image Users in the Example Dockerfile on Debian**

|User|Description|Initial Password|Password Change Method|
|--|--|----|--|
|root|-|None|-|
|bin|-|None|-|
|daemon|-|None|-|
|lp|-|None|-|
|sync|-|None|-|
|mail|-|None|-|
|games|-|None|-|
|nobody|-|None|-|
|sys|-|None|-|
|man|-|None|-|
|news|-|None|-|
|uucp|-|None|-|
|proxy|-|None|-|
|www-data|-|None|-|
|backup|-|None|-|
|list|-|None|-|
|irc|-|None|-|
|_apt|-|None|-|

**Users in the RecSDK-Torch Component Container on Debian**

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|systemd-network|-|None|-|
|systemd-timesync|-|None|-|
|messagebus|-|None|-|
|sshd|-|None|-|
