## NS Simulation Toolkit
This repo is used to manage the ns-3 simulation and data generation. It can be used to quickly install and configure ns-3, install BRITE topology generator, start multiple simulation runs, and collect and visualize the data (data rate, RTT, LLR, etc).

## Usage
After clone the repo, run

```
./install.sh
```

to install BRITE topology generator and ns-3. After it finishes, BRITE and ns-3 should be ready. Run the following command to test ns-3:

```
cd BBR_test/ns-3.27
./waf --run "scratch/brite-for-all --PrintHelp"
```

You should see something like:

```
Waf: Entering directory `/home/sapphire/Documents/NS_Simulation_Toolkit/BBR_test/ns-3.27/build'
Waf: Leaving directory `/home/sapphire/Documents/NS_Simulation_Toolkit/BBR_test/ns-3.27/build'
Build commands will be stored in build/compile_commands.json
'build' finished successfully (1.180s)
brite-for-all [Program Arguments] [General Arguments]

Program Arguments:
    --v:            Enable verbose [2]
    --mid:          Run ID (4 digit) [1179]
    --tid:          Topology stream ID [9563]
    --nNormal:      Number of normal flows [3]
    --nCross:       Number of cross traffic [0]
    --nDsForEach:   Downstream flow number for each destination [0]
    --normalRate:   Rate of normal flow [100000000]
    --crossRate:    Rate of cross traffic [5000000]
    --edgeRate:     Rate of edge link (in Mbps only) [8000]
    --dsCrossRate:  Rate of downstream flow [30000000]
    --tStop:        Time to stop simulation [2]
    --confFile:     path of BRITE configure file [brite_conf/TD_CustomWaxman.conf]

General Arguments:
    --PrintGlobals:              Print the list of globals.
    --PrintGroups:               Print the list of groups.
    --PrintGroup=[group]:        Print all TypeIds of group.
    --PrintTypeIds:              Print all TypeIds.
    --PrintAttributes=[typeid]:  Print all attributes of typeid.
    --PrintHelp:                 Print this help message.
```
Then ns-3 is ready to use. One example of typical brite simulation commands is:
```
./waf --run "scratch/brite-for-all -nNormal=2 -nCross=0 -nDsForEach=2 -mid=9563 -confFile=brite_conf/TD_inter=1G_intra=100M_Waxman.conf"
```
which generates 2 normal flows, 0 main cross traffic, 2 downstream cross traffic for each with mid 9563 (run id) and Brite topology specified in ```brite_conf/TD_inter=1G_intra=100M_Waxman.conf```. Check the program arguments for more detail.

## Data Generation and Visualization
The ns-3 program generates raw data in ```BBR_test/ns-3.27/MboxStatistics```. The general form of the name is ```[data type]_[mid]_[flow No.].dat```, e.g. ```DataRate_9563_0.dat``` (data rate of flow 0 in run 9563). To plot a certain raw data, use the following commands:
```
cd scripts
./mPlotData.sh -d DataRate_7001 -f test -s l -o y -n 3
```
which plots the data rates of the first 3 flows in run 7001, and generate the figures with prefix "test". Check ```mPlotData.sh``` for more options.

## Parameter Scan
To start a parameter scan, i.e. multiple runs with some parameters changing in a range, we can use ```multiRun.py```. Try the following commands
```
cd scripts
python3 multiRun.py
```
which will show the usage of multiRun.py. Typical use is like
```
python3 multiRun.py -nNormal 3:1:5 -nCross 0:3:3
```
which will run the simulations with nNormal 3, 4, 5 and nCross = 0, 3, i.e. 6 runs. Data (only AckLatency and RttLlr), figures (of data rate) and logs are collected in ```BBR_test/ns-3.27/results_[current-time]```. 
