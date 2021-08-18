### Setting up wired Ethernet connection 

If used in conditions without wireless Internet connection, the Raspberry Pi imaging component can communicate with the Jetson Nano deep learning component using wired Ethernet. There are a wide variety of ways to achieve this, such as using a combination of `inotifywait`, `watch`, and `scp`.

First, set a static IP for Jetson Nano

* Open `/etc/default/networking`
* Set `CONFIGURE_INTERFACES=no` to skip interface configuration on boot
* Open `/etc/network/interfaces`
* Modifying the following settings:

```
auto eth0
iface eth0 inet static
	address XXX.XXX.X.X
	netmask XXX.XXX.X.X
	gateway XXX.XXX.X.X
```

* Reboot and check with `ifup eth0`

Next, set a static IP for Raspberry Pi 

* Open `/etc/dhcpcd.conf`
* Scroll to the bottom of the file and add the following:

```
interface eth0
static ip_address=XXX.XXX.X.X
static routers=XXX.XXX.X.X
static domain_name_servers=XXX.XXX.X.X
```

* Reboot

