set datafile separator "\t"
set terminal png size 900,400
set title "Firing neurons"
set xlabel "Time"
set ylabel "Neuron"
set xdata
plot "firing.log" using 1:2 title 'Firing'
