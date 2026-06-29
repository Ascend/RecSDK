# Appendix

## User Information List

Periodically update user passwords to avoid the risks caused by using the same password for a long time.

**Table 1** User list

|User|Description|Initial Password|Password Change Method|
|--|---|---|--|
|root|Used for deploying Rec SDK TensorFlow.|User-defined|Run the `passwd` command to change it.|
|HwHiAiUser|User for installing drivers and running the demo.|User-defined|Run the `passwd` command to change it.|

**Base Image Users in the Example Dockerfile on CentOS**

|User|Initial Password|Password Change Method|
|--|--|--|
|root|None|-|
|bin|None|-|
|daemon|None|-|
|adm|None|-|
|lp|None|-|
|sync|None|-|
|shutdown|None|-|
|halt|None|-|
|mail|None|-|
|operator|None|-|
|games|None|-|
|ftp|None|-|
|nobody|None|-|
|systemd-network|None|-|
|dbus|None|-|

**Users in the RecSDK-TensorFlow Component Container on CentOS**

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|sshd|-|None|-|
