# Security Hardening

## Security Requirements

**File Permissions**

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues.

Software code or programs downloaded from external sources may pose risks. You must guarantee the security of their functions.

## Hardening Precautions

The security hardening measures listed in this document provide basic recommendations. You should re-evaluate the network security posture of the entire system based on specific service requirements. When necessary, consult industry best practices and security experts.

## OS Security Hardening

### Firewall Configuration

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation.

### Setting umask

Set the host umask to `0027` or more restrictive on the host and in containers to enhance file security.

To set umask to `0027`:

1. Log in as `root` and edit `/etc/profile`.

    ```bash
    vim /etc/profile
    ```

2. Append `umask 0027` to the end of the file. Save the file and exit.
3. Run the following command to apply the configuration:

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening

Differences between official Docker images and the host OS may result in a mismatch between user definitions. This can lead to the creation of ownerless files during system or container operation.

You can find ownerless files on the host or in containers by running `find / -nouser -o -nogroup`. To mitigate security risks, create corresponding users and groups based on file UIDs and GIDs, or adjust existing UIDs and GIDs to match, thereby ensuring every file has a valid owner.

### Port scanning

Monitor ports listening on all interfaces and identify unnecessary ports for immediate closure. You are advised to disable insecure services, such as Telnet and FTP. For details, see the related documents of the OS in use.

### Anti-DoS Protection

Protect the system against Denial of Service (DoS) attacks by implementing IP address restriction and rate limiting. Recommended methods include using the Linux iptables firewall and optimizing sysctl parameters. For details, see related documents.

## Device Security Hardening

After the device OS allocates huge page memory, it clears that memory by default. This also reduces performance. If you need high-performance mode, you can manually configure the system not to clear the memory. However, disabling this operation may expose kernel data and user data.

To disable huge page clearing:

1. Log in to every device that participates in training.
2. On every participating device, run `echo 0 > /proc/sys/sharepool/sharepool_clear_hugepage`.

    You can use other APIs to confirm whether the change succeeded.
