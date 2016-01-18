set terminal png
set xlabel "Time (s)"
#set style data lines lw 3

set nokey

set output "q2.png"
set title "Cumulative Data Transferred"
set ylabel "Data (KB)"
plot "q2.txt" u 1:2 with lines

set output "q3.png"
set title "Average RTT"
set ylabel "RTT (ms)"
plot "q3.txt" u 1:2:3:4 with errorbars

set output "q4.png"
set title "Transfer Rate"
set ylabel "Rate (KB/s)"
plot "q4.txt" u 1:2 with lines

set output "q5a.png"
set title "Value of CWND"
set ylabel "Data in flight (B)"
plot "q5.txt" u 1:2 with lines

set key left top

set output "q5b.png"
set title "Difference between ACK and SEQ"
set ylabel "ACK/SEQ value (B)"
plot "q5.txt" u 1:3 with lines title "ACK", "q5.txt" u 1:4 with lines title "SEQ"

set output "q6.png"
set title "Packet Loss"
set style line 1 lt rgb "red" 
set style line 2 lt rgb "blue" 
plot "q5.txt" u 1:3 with lines title "ACK", "q5.txt" u 1:4 with lines title "SEQ", "q6d.txt" u 1:2 title "DUP ACK" ls 2, "q6t.txt" u 1:2 title "RTO" ls 1

set output "q7a.png"
set title "Emulating Larger RTT"
set ylabel "Transfer rate (KB/s)"
plot "out1rate" u 1:2 with lines title "No Delay", "out2rate" u 1:2 with lines title "200ms Delay", "out3rate" u 1:2 with lines title "400ms Delay"

set nokey

set output "q7b.png"
set title "Emulating Larger RTT"
set ylabel "Average Throughput (ms)"
set xlabel "Average RTT (ms)"
plot "ps7-2.txt" u 2:3 with lines
