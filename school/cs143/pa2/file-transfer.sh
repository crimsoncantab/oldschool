#!/bin/sh

sudo nohup tcpdump -i wlan0 tcp port 22 -vv > tcpdump_output.txt &
#   (where <interface> is replaced with the network 
#   interface that your machine is using)
scp HRO_Mozart35_Mvmt4.mp3 harvardcs143@troll.iis.sinica.edu.tw:loren_HRO_Mozart35_Mvmt4.mp3 
#   using the password "Fall2010pa2" when prompted.
sudo killall tcpdump
