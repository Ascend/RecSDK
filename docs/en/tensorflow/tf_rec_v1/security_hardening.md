# Security Hardening

## Security Requirements

### File Permissions

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues.

Software code or programs downloaded from external sources may pose risks. You must guarantee the security of their functions.

### Hadoop Installation Permissions

If you need to use the Hadoop distributed file system, after installation, set the permissions on `libhdfs.so` in the installation directory to no more than `640`. This prevents tampering with the shared object package and reduces the risk of command injection.

## Hardening Precautions

The security hardening measures listed in this document provide basic recommendations. You should re-evaluate the network security posture of the entire system based on specific service requirements. When necessary, consult industry best practices and security experts.

## OS Security Hardening

### Firewall Configuration

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation.

### Setting umask

Set the host umask to `027` or more restrictive on the host and in containers to enhance file security.

To set umask to `027`:

1. Log in as `root` and edit `/etc/profile`.

    ```bash
    vim /etc/profile
    ```

2. Append `umask 027` to the end of the file. Save the file and exit.
3. Run the following command to apply the configuration:

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening

Differences between official Docker images and the host OS may result in a mismatch between user definitions. This can lead to the creation of ownerless files during system or container operation.

You can find ownerless files on the host or in containers by running `find / -nouser -o -nogroup`. To mitigate security risks, create corresponding users and groups based on file UIDs and GIDs, or adjust existing UIDs and GIDs to match, thereby ensuring every file has a valid owner.

### Port Scanning

Monitor ports listening on all interfaces and identify unnecessary ports for immediate closure. You are advised to disable insecure services, such as Telnet and FTP. For details, see the related documents of the OS in use.

### Anti-DoS Protection

Protect the system against Denial of Service (DoS) attacks by implementing IP address restriction and rate limiting. Recommended methods include using the Linux iptables firewall and optimizing sysctl parameters. For details, see related documents.

### Prevention of MPI Listening on `0.0.0.0`

- The background `mpirun` process that MPI starts during initialization has an unauthenticated port issue and may be vulnerable to attacks.

    Mitigation: After MPI starts, add firewall rules to block external network connections to the ports on which `mpirun` and `orted` (an internal MPI component) listen. After the run finishes, clear the firewall rules.

    - Add firewall rules.

        ```bash
        #Define variables. Modify them as needed.
        CHAIN_NAME="RECSDK_BLOCK"  # Rule name
        INTERFACE="eth0"           # NIC to restrict
        PORT="8080"                # Port to restrict

        #Clear old rules. Delete rules in the chain first, then delete the chain.
        # 1. Remove the reference to the custom chain from the INPUT chain.
        sudo iptables -D INPUT -p tcp -j ${CHAIN_NAME} 2>/dev/null
        # 2. Clear all rules in the custom chain.
        sudo iptables -F ${CHAIN_NAME} 2>/dev/null
        # 3. Delete the custom chain.
        sudo iptables -X ${CHAIN_NAME} 2>/dev/null

        #Create new rules.
        # 4. Create a new custom chain.
        sudo iptables -t filter -N ${CHAIN_NAME}
        # 5. Insert the custom chain as the first rule in the INPUT chain.
        sudo iptables -I INPUT 1 -p tcp -j ${CHAIN_NAME}
        # 6. Add a drop rule to the custom chain.
        sudo iptables -t filter -I ${CHAIN_NAME} 1 -i ${INTERFACE} -p tcp --dport ${PORT} -j DROP
        ```

    - Clear firewall rules.

        ```bash
        sudo iptables -D INPUT -p tcp -j ${CHAIN_NAME} 2>/dev/null
        sudo iptables -F ${CHAIN_NAME} 2>/dev/null
        sudo iptables -X ${CHAIN_NAME} 2>/dev/null
        ```

- The default MPI configuration listens on `0.0.0.0` when it starts multiple Python training processes.

    Mitigation: Find the `etc` folder in the MPI configuration path, and add the following two configuration items to the end of `openmpi-mca-params.conf`.

    ```bash
    btl_tcp_if_include = lo
    btl_tcp_if_exclude = docker0
    ```

    >[!NOTE]
    >It is recommended that you use MPI only in containers and rely on network namespaces for isolation. This ensures that the host has no service listening on `0.0.0.0` and that other users on the host cannot connect to unauthenticated OpenMPI ports.

## Device Security Hardening

After the device OS allocates huge page memory, it clears that memory by default. This also reduces performance. If you need high-performance mode, you can manually configure the system not to clear the memory. However, disabling this operation may expose kernel data and user data.

To disable huge page clearing:

1. Log in to every device that participates in training.
2. On every participating device, run `echo 0 > /proc/sys/sharepool/sharepool_clear_hugepage`.

    You can use other APIs to confirm whether the change succeeded.
