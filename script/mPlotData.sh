#!/bin/bash
# Author: Jinhui Song; Date: 2018.12.18
# This script is used to run the statistic data from mbox. One run
# can plot one figure of one result (e.g. data rate, llr) with several 
# senders. One should at least input data prefix and # sender to specify
# the type of result and number of sender.
# 
# Usage: ~/scpt/mPlotData.sh -f [figure prefix] -d [data prefix] -n [# sender] -r [ range a:b ]
#                            -s [size of figure (s/m/l)] -o [if output figure (y/n)] -h
# 
# Change log:   	2019.1.24           Modify to adapt different kinds of statistics, retain unit only for data rate.
# 	                2019.1.31           Add option size to control the output figure size.
#                   2019.2.7            Complete the help message, polish size option and add feh option.
#                   2019.2.13           Reduce the width of size large to draw a long figure.
#                   2019.2.21           Add topic option: RTT and congestion window.
#                   2019.9.16           Add offset of the flow index to only display cross traffic flows.

# Last update:      2019.12.5           Update new ns3 path in new system.

# predefined parameter
now=$(date +"%T")
figPre="mboxStat"
dataPre="DataRate"
topic="Data Rate"
sFlag="m"
sizeStr="800, 600"
oFlag="n"
num=2
offset=0
gsc="gscript"
StaFolder="MboxStatistics"
FigFolder="MboxFig"
range=""
xtic=5
NS3PATH='$(pwd)/../BBR_test/ns-3.27'

# parse input option
while getopts f:d:n:m:p:r:s:o:x:h option        # ':' to give an argument after an option!
do
case ${option}
in
f) figPre=${OPTARG};;
d) dataPre=${OPTARG};;
n) num=${OPTARG};;
m) offset=${OPTARG};;
p) NS3PATH=${OPTARG};;
r) range=${OPTARG};;
s) sFlag=${OPTARG};;
o) oFlag=${OPTARG};;
x) xtic=${OPTARG};;
h) echo "Usage: ./mPlotData.sh 
    -f [figure prefix] -d [data prefix] -n [sender number] -m [offset of sender index] -p [path]
    -r [range a:b] -s [size of figure (s/m/l)] -o [if output figure (y/n)] -x [xtics] -h "
    exit 1;;
esac
done

# set size string
# echo "size: "$sFlag
if [ $sFlag = "s" ]
then
    sizeStr="480, 360"
elif [ $sFlag = "m" ]
then
    sizeStr="800, 600"
else
    sizeStr="1120, 600"
fi

# set file names
declare -a data
cnt=0
while [ $cnt -lt $num ];        # need testing
do
    data[$cnt]="${dataPre}_$(($cnt+$offset)).dat"
    # echo ${data[cnt]}
    let cnt=cnt+1
done
fig="${figPre}_$now.png"

cd "$NS3PATH/$StaFolder"
echo "../${FigFolder}/${fig}"

# set the topic name in figure title

unit=" "
case $dataPre in
    "DataRate"*) 
        topic="Data Rate"
        unit=" /kbps";;
    "SLR"*) topic="Slort-term Loss Rate";;
    "LLR"*) topic="Long-term Loss Rate";;
    "Dwnd"*) topic="Total Drop Window";;
    "Rwnd"*) topic="Receive Window of Mbox";;
    "Rtt"*) topic="Round Trip Time (RTT)";;
    "CongWnd"*) topic="TCP Congestion Window";;
    "txRate"*) topic="TX Rate of each flow"
        unit=" /kbps";;
    "QueueSize"*) topic="Queue size";;
    "TcpWnd"*) topic="TCP window";;
    *) topic="Something interesting";;
esac

# gnuplot
nl=$'\n'

gnuStr="set output \"../${FigFolder}/${fig}\"
set terminal png size ${sizeStr}
set autoscale
set xlabel \"time /s\"
set ylabel \"${topic} ${unit}\"
set title \"${topic} of $num senders\"
set xtics ${xtic}
set grid
plot [${range}] "
cnt=0
while [ $cnt -lt $num ];
do
    if [ $cnt -ne $(($num-1)) ]
    then
        gnuStr+="\"${data[$cnt]}\" using 1:2 title 'flow $(($cnt+$offset))' with linespoints, \\$nl"
    else
        gnuStr+="\"${data[$cnt]}\" using 1:2 title 'flow $(($cnt+$offset))' with linespoints"
    fi
    let cnt+=1
done

echo "$gnuStr" > $gsc
gnuplot "$gsc"
if [ $oFlag = "y" ]
then
    feh ../$FigFolder/$fig            # comment when need to scan the parameters
fi
