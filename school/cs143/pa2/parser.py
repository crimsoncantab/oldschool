#!/usr/bin/python

import math
from time import strftime
import sys
import datetime
import re
from datetime import datetime
from datetime import timedelta

tcp1 = re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d{6}) (\S*) \(tos (0x[0-9a-f]+), ttl (\d+), id (\d+), offset (\d+), flags \[(.*?)\], proto ([^,]+), length (\d+)\)$')
tcp2 = re.compile(r'^\s*(\S*)\.([^\. ]+) > (\S*)\.([^\.:]+): Flags \[(.*?)\], (cksum \S+ \((.*?)\), )?seq (\d+(:(\d+))?), (ack (\d+), )?win (\d+),')

#CLIENT_HOST = 'noether.eecs.harvard.edu'
CLIENT_HOST = 'dhcp-0095359592-e7-94.client.student.harvard.edu'
SERVER_HOST = 'troll.iis.sinica.edu.tw'
TIMESTAMP_FORMAT = '%H:%M:%S.%f'

TIMESTAMP_1 = 1
PACKET_TYPE_1 = 2
SERVICE_TYPE_1 = 3
TTL_1 = 4
ID_1 = 5
OFFSET_1 = 6
FLAGS_1 = 7
PROTOCOL_1 = 8
LEN_1 = 9

ADDR_S_2 = 1
PORT_S_2 = 2
ADDR_D_2 = 3
PORT_D_2 = 4
CTRL_BITS_2 = 5
CKSUM_STAT_2 = 7
SEQ_2 = 8
SEQ_END_2 = 10
ACK_2 = 12
AD_WIN_2 = 13


#num_1 = 0
#num_2 = 0
#no_match = 0
DELAY_MS = 0
#DELAY_MS = 200
#DELAY_MS = 400
DELAY = timedelta(milliseconds=DELAY_MS)

class Packet:
    def __init__(self, line_1, line_2):
        self.line_1 = line_1
        self.line_2 = line_2
        if self.line_2.group(ADDR_S_2)==CLIENT_HOST:
            self.timestamp = datetime.strptime(line_1.group(TIMESTAMP_1), TIMESTAMP_FORMAT) - DELAY
        else:
            self.timestamp = datetime.strptime(line_1.group(TIMESTAMP_1), TIMESTAMP_FORMAT)

    def get_timestamp(self):
        return self.timestamp

    def get_flags(self):
        return self.line_2.group(CTRL_BITS_2)

    def get_src(self):
        if self.from_client():
            return CLIENT_HOST
        else:
            return SERVER_HOST

    def get_dest(self):
        if not self.from_client():
            return CLIENT_HOST
        else:
            return SERVER_HOST

    def get_seq(self):
        if self.line_2.group(SEQ_END_2) is None:
            return int(self.line_2.group(SEQ_2))
        else:
            return int(self.line_2.group(SEQ_END_2))-1

    def get_ack(self):
        return int(self.line_2.group(ACK_2))


    def from_client(self):
        return self.line_2.group(ADDR_S_2)==CLIENT_HOST

    def get_payload_size(self):
        return int(self.line_1.group(LEN_1))


fstring = {'P':'PSH','S':'SYN','F':'FIN','.':'ACK'}
def flags_to_string(flags):
    ret = ""
    for c in flags:
        ret += fstring[c] + '/'
    return ret[0:-1]

def host_to_string(host):
#    print host
#    print CLIENT_HOST
    if host==CLIENT_HOST:
        return 'client'
    else:
        return 'server'


packets = []
line_1 = ''

#load data
for line in sys.stdin:
    m = tcp1.match(line)
    if m:
        line_1 = m
    else:
        m2 = tcp2.match(line)
        if m2:
            packets.append(Packet(line_1, m2))
        else:
            print line,

#question 1
#print 'opening connection'
#for i in [0,1,2]:
#    packet = packets[i]
#    print host_to_string(packet.get_src()),'>',
#    print host_to_string(packet.get_dest()),
#    print flags_to_string(packet.get_flags()),
#    print packet.get_timestamp()
#
#diff = packets[2].get_timestamp() - packets[0].get_timestamp()
#print 'total time:', diff
#print
#print 'closing connection'
#fin = 0
#
#for i, packet in enumerate(packets):
#    if fin == 0 and packet.get_flags().find('F') != -1:
#        fin = i
#    if fin != 0:
##        print packet.get_src()
#        print host_to_string(packet.get_src()),'>',
#        print host_to_string(packet.get_dest()),
#        print flags_to_string(packet.get_flags()),
#        print packet.get_timestamp()
#
#diff = packets[len(packets)-1].get_timestamp() - packets[fin].get_timestamp()
#print 'total time:', diff

#question 2
#bucket_size = timedelta(milliseconds=500)
#next_timestamp = packets[0].get_timestamp() + bucket_size
#interval = .5
#sum = 0
#for packet in packets:
#    if packet.get_timestamp() > next_timestamp:
#        kb = float(sum) / 1024
#        print repr(interval), "%.2f" % kb
#        interval += .5
#        next_timestamp += bucket_size
#    if packet.from_client():
#        sum += packet.get_payload_size()
#
#kb = float(sum) / 1024
#print repr(interval+5), "%.2f" % kb

#question 3
#def stats_rtt_bucket(bucket):
#    if len(bucket) == 0:
#        return 0,0
#    millis = map(lambda rtt: (rtt.microseconds / 1000), bucket)
#    avg = math.fsum(millis)/ len(millis)
#    variance = 0
#    for rtt in millis:
#        variance += (rtt - avg) **2
#    return avg, variance ** .5
#
#
#
#bucket_width = timedelta(milliseconds=500)
#next_timestamp = packets[0].get_timestamp() + bucket_width
#interval = .5
#cur_bucket = []
#rtt_avg_sum = 0
#num = 0
#for i, packet in enumerate(packets):
#    if packet.get_timestamp() > next_timestamp:
#        avg, std = stats_rtt_bucket(cur_bucket)
#        if avg != 0:
#            rtt_avg_sum += avg
#            num += 1
#            print interval, avg, avg - (std/2), avg + (std/2)
#        interval += .5
#        next_timestamp += bucket_width
#        cur_bucket = []
#    if packet.from_client():
#        expect_ack = packet.get_seq() + 1
#        for packet2 in packets[i+1:]:
#            if not packet2.from_client() and packet2.get_ack() == expect_ack:
#                cur_bucket.append(packet2.get_timestamp() - packet.get_timestamp())

#print rtt_avg_sum / num

#   |
#   |
#   |
#   |
#   V

#question 4
#rtt_avg = .278265
#bucket_width = timedelta(seconds=rtt_avg)
#next_timestamp = packets[0].get_timestamp() + bucket_width
#interval = rtt_avg
#sum = 0
#for packet in packets:
#    if packet.get_timestamp() > next_timestamp:
#        kb = float(sum) / 1024
#        print interval, kb/rtt_avg
#        sum = 0
#        interval += rtt_avg
#        next_timestamp += bucket_width
#    if packet.from_client():
#        sum += packet.get_payload_size()
#
#kb = float(sum) / 1024
#print interval, kb/rtt_avg


#question 5
ack = 0
seq = 0
time_0 = packets[0].get_timestamp()

for packet in packets[2:]:
    if packet.from_client():
        print 'seq', seq
        seq = max(seq, packet.get_seq())
    else:
        print 'ack', ack
        ack = max(ack, packet.get_ack())
    time_delta = packet.get_timestamp() - time_0
    print time_delta.seconds + (time_delta.microseconds / 1000000.), max(0, seq - ack), ack, seq

#question 6
#cur_ack = -1
#num_dup = 0
#time_0 = packets[0].get_timestamp()
#
#for packet in packets[2:]:
#    if not packet.from_client():
#        next_ack = packet.get_ack()
#        if next_ack == cur_ack:
#            num_dup += 1
#            if num_dup >= 3:
#                time_delta = packet.get_timestamp() - time_0
#                print time_delta.seconds + (time_delta.microseconds / 1000000.), next_ack
#        else:
#            num_dup = 0
#            cur_ack = next_ack
#

#cur_seq = -1
#cur_ack = -1
#num_dup = 0
#for packet in packets[2:]:
#    if not packet.from_client():
#        next_ack = packet.get_ack()
#        if next_ack == cur_ack:
#            num_dup += 1
#            if num_dup >= 3:
#                time_delta = packet.get_timestamp() - time_0
#                print 'ack', time_delta.seconds + (time_delta.microseconds / 1000000.), next_ack
#        else:
#            num_dup = 0
#            cur_ack = next_ack
#    if packet.from_client():
#        next_seq = packet.get_seq()
#        if next_seq <= cur_seq:
#            time_delta = packet.get_timestamp() - time_0
#            print 'seq', time_delta.seconds + (time_delta.microseconds / 1000000.), next_seq
##            print packet.get_timestamp(),time_delta.seconds + (time_delta.microseconds / 1000000.), 0
#        else:
#            cur_seq = next_seq


#question 7
#def stats_rtt_bucket(bucket):
#    if len(bucket) == 0:
#        return 0,0
#    millis = map(lambda rtt: (rtt.microseconds / 1000), bucket)
#    avg = math.fsum(millis)/ len(millis)
#    variance = 0
#    for rtt in millis:
#        variance += (rtt - avg) **2
#    return avg, variance ** .5
#
#
#print DELAY_MS,
#
#bucket_width = timedelta(milliseconds=500)
#next_timestamp = packets[0].get_timestamp() + bucket_width
#interval = .5
#cur_bucket = []
#rtt_avg_sum = 0
#num = 0
#for i, packet in enumerate(packets):
#    if packet.get_timestamp() > next_timestamp:
#        avg, std = stats_rtt_bucket(cur_bucket)
#        if avg != 0:
#            rtt_avg_sum += avg
#            num += 1
##        print interval, avg, avg - (std/2), avg + (std/2)
#        interval += .5
#        next_timestamp += bucket_width
#        cur_bucket = []
#    if packet.from_client():
#        expect_ack = packet.get_seq() + 1
#        for packet2 in packets[i+1:]:
#            if not packet2.from_client() and packet2.get_ack() == expect_ack:
#                cur_bucket.append(packet2.get_timestamp() - packet.get_timestamp())
#
#rtt_avg = rtt_avg_sum / num / 1000
#print rtt_avg,
#bucket_width = timedelta(seconds=rtt_avg)
#next_timestamp = packets[0].get_timestamp() + bucket_width
#interval = rtt_avg
#sum = 0
#rate_sum = 0
#rate_num = 0
#for packet in packets:
#    if packet.get_timestamp() > next_timestamp:
#        kb = float(sum) / 1024
##        print interval, kb/rtt_avg
#        rate_sum += kb/rtt_avg
#        rate_num +=1
#        sum = 0
#        interval += rtt_avg
#        next_timestamp += bucket_width
#    if packet.from_client():
#        sum += packet.get_payload_size()
#
#print rate_sum/rate_num,