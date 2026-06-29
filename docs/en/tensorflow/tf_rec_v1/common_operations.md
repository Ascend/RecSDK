# Common Operations

## (Optional) Installing the On-Chip Memory Dynamic Expansion Operator Package

If you need to use the on-chip memory dynamic expansion function, use the following steps to compile and install the on-chip memory dynamic expansion operator package.

**Procedure**

1. Run the following command to set CANN environment variables in the container.

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. Download [Rec SDK](https://gitcode.com/Ascend/RecSDK/tree/develop/cust_op).
3. Decompress the package and go to the `cust_op/ascendc_op/ai_core_op/cust_op_by_addr` directory. For Atlas A2/A3 servers, go to the v220 directory. For Atlas A5 servers, go to the c310 directory. Run the following command to compile and install the on-chip memory dynamic expansion operator package:

    ```bash
    bash run.sh
    ```

## Viewing Rec SDK TensorFlow Installation and Uninstallation Records

Rec SDK TensorFlow is provided as a wheel package. The system history records installation and uninstallation logs.

**Viewing installation and uninstallation history**

When you log out of the system or exit a container (Rec SDK TensorFlow is typically installed and run in a container), the system saves the command history to the `~/.bash_history` file. You can check the `.bash_history` file directly to find Rec SDK TensorFlow installation and uninstallation records.

**Modifying the number of saved history records**

In Linux systems, the `history` command typically saves the latest 1,000 commands by default. To modify the number of saved commands, for example, to keep only 200 commands, modify the `HISTSIZE` environment variable in the `/etc/profile` file. Use the following methods:

- Use an editor (such as Vim) to modify the file.
- Run the `sed` command to modify the file. The command is as follows:

    `sed -i 's/^HISTSIZE=number/HISTSIZE=newNumber/' /etc/profile`, where *number* represents the original number of commands and *newNumber* represents the new number of commands. For example, to change the number of saved commands from 1,000 to 200:

    ```bash
    sed -i 's/^HISTSIZE=1000/HISTSIZE=200/' /etc/profile
    ```

After the modification, run `source /etc/profile` to make the environment variable take effect.

**Modifying timestamps in the command history file**

To record timestamps in the command history file, add the following configuration to `/etc/profile`:

**HISTTIMEFORMAT='%F %T '**

After adding the configuration, run `source /etc/profile` to make the environment variable take effect. After timestamps are added, the `history` command result is as follows:

```bash
2023-08-18 10:01:57 pip3 install mx_rec-5.0.T104-py3-none-linux_x86_64.whl --force-reinstall
2023-08-18 10:01:57 pip3 install mx_rec-5.0.T104-py3-none-linux_x86_64.whl --force-reinstall
2023-08-18 10:04:37 history | grep "pip3 install"
2023-08-18 10:10:17 history | grep "pip3 install"
```

In addition, to record command history in a custom file, set the `HISTFILE` environment variable in `/etc/profile`, and run source /etc/profile for the change to take effect. For example:

```bash
HISTDIR=~/log/RecSDK   # Configure the file for saving command history.
HISTFILE="$HISTDIR/RecSDK.log"
mkdir -p $HISTDIR
chmod 750 $HISTDIR
touch $HISTFILE
chmod 640 $HISTFILE
USER_IP=`who -u am i 2>/dev/null| awk '{print $NF}'|sed -e 's/[()]//g'`
if [ -z $USER_IP ]
then
  USER_IP=`hostname`
fi
export HISTTIMEFORMAT="%F %T $USER_IP:`whoami` "   # command history display format: time, IP address, username, command
PROMPT_COMMAND=' { date "+%Y-%m-%d %T - $(history 1 | { read x cmd; echo "$cmd"; })"; } >> $HISTFILE'    # Write the command history to the configured file in real time.
```

The log file path is `~/log/RecSDK`. Ensure that the drive space is sufficient and the log file permissions are set to 640.
