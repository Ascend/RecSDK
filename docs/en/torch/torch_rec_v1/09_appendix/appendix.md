# Appendix

## User Information List

Periodically update user passwords to avoid the risks caused by using the same password for a long time.

**Table 1** User list

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|root|Used for deploying Rec SDK Torch.|User-defined|Run the `passwd` command to change it.|
|HwHiAiUser|User for installing drivers and running the demo.|User-defined|Run the `passwd` command to change it.|

**Base Image Users in the Example Dockerfile on openEuler**

|User|Initial Password|Password Change Method|
|--|---|--|
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
|unbound|None|-|
|tss|None|-|

**Users in the Rec SDK Torch Component Container on openEuler**

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|systemd-coredump|-|None|-|
|systemd-network|-|None|-|
|systemd-resolve|-|None|-|
|sshd|-|None|-|
|dbus|-|None|-|
|HwHiAiUser|Used for running the driver .run package.|None|-|

**Base Image Users in the Example Dockerfile on Debian**

|User|Initial Password|Password Change Method|
|--|--|--|
|root|None|-|
|bin|None|-|
|daemon|None|-|
|lp|None|-|
|sync|None|-|
|mail|None|-|
|games|None|-|
|nobody|None|-|
|sys|None|-|
|man|None|-|
|news|None|-|
|uucp|None|-|
|proxy|None|-|
|www-data|None|-|
|backup|None|-|
|list|None|-|
|irc|None|-|
|_apt|None|-|

**Users in the Rec SDK Torch Component Container on Debian**

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|systemd-network|-|None|-|
|systemd-timesync|-|None|-|
|messagebus|-|None|-|
|sshd|-|None|-|

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

**Users in the Rec SDK Torch Component Container on CentOS**

|User|Description|Initial Password|Password Change Method|
|--|--|--|--|
|sshd|-|None|-|
