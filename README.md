## NS Simulation Toolkit
This repo is used to manage the ns-3 simulation and data generation. It can be used to quickly install and configure ns-3, install BRITE topology generator, start multiple simulation runs, and collect and visualize the data (data rate, RTT, LLR, etc).

## Prerequisite
Tested on Ubuntu 16.04, the following packages are needed: `make`, and `hg`. Besides, please make sure you install the dependencies of PyViz listed [here](https://www.nsnam.org/wiki/PyViz) to enable PyViz visualizer.

```
sudo apt-get install make hg
sudo apt-get install python-dev python-pygraphviz python-kiwi python-pygoocanvas \
                     python-gnome2 python-gnomedesktop python-rsvg
```

Note that different Ubuntu version might not provide the python packages above.

## Usage
After cloning the repo, run

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

You can use ```--visualize``` option when running waf to show the topology created by BRITE and visualize the simulated traffic (which I find quite useful!). The example is:
```
./waf --run "scratch/brite-for-all -nNormal=1 -nCross=0 -edgeRate=100 -nDsForEach=2 -mid=9563 -confFile=brite_conf/TD_inter=1G_intra=1G_Waxman.conf" --visualize
```

Note that the topology won't change if you don't change the tid and configure file of BRITE. If you want to randomly generated the topology with the same BRITE configuration, you can run with different tid.

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

## Troubleshooting
1. Brite error
```
./scratch/brite-for-all.cc:48:10: fatal error: ns3/brite-module.h: No such file or directory
```
It indicates Brite is not installed correctly. Please make sure you have hg and make installed and run
```
hg clone http://code.nsnam.org/jpelkey3/BRITE
cd BRITE
make
```
Make sure there's `libbrite.so` in your BRITE folder, and then you can configure and build ns-3.

2. PyViz not enabled
2.1
```
assert failed. cond="uid != 0", msg="Assert in TypeId::LookupByName: ns3:VisualSimulatorImpl not found", file=../src/core/model/type-id.cc, line=827
```
It indicates PyViz is not enabled and please make sure you have all the PyViz dependencies installed and reconfigure ns-3.

2.2 
I already have the requirement installed, but still
```
PyViz visualizer              : not enabled (Missing python modules: gtk, goocanvas, pygraphviz)
```

Please double check the Python version scanned by `./waf configure`, and ensure the prerequiste packeges can be imported for the specific version of python.
Current PyViz is only tested for python2.7


3. Multiple exit

```
assert failed. cond="m_ecmpRootExits.size () <= 1", msg="Assumed there is at most one exit from the root to this vertex", file=../src/internet/model/global-route-manager-impl.cc, line=316
```
This will sometimes happen and you can change ```tid``` to get another random topology to get it work.

4. Insufficient leaves

```
assert failed. cond="bth.GetNLeafNodesForAs (i) >= nNormal + nCross", msg="-> AS 0 doesn't have enough leaves, please change configure file!", file=../scratch/brite-for-all.cc, line=122
```
It happens when the number of leaves generated is less than the number of flows we want (cannot be guaranteed since topology is generated by the configure file like ```TD_CustomWaxman.conf```). You can either decrease the number of flows you want or modify ```Number of nodes in graph``` in the configure file (which infers the number of nodes in one AS).

## Reference
\[1\] NS-3: a Discrete-Event Network Simulator. http://www.nsnam.org/, Accessed in 2019.

\[2\] Medina, A., Lakhina, A., Matta, I., & Byers, J. (2001, August). BRITE: An approach to universal topology generation. In MASCOTS 2001, Proceedings Ninth International Symposium on Modeling, Analysis and Simulation of Computer and Telecommunication Systems (pp. 346-353). IEEE.

\[3\] PyViz - Nsnam. https://www.nsnam.org/wiki/PyViz, Accessed on 2020.