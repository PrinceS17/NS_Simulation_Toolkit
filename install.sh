# Install ns-3.27 with customized modules and BRITE generator in root directory.

hg clone http://code.nsnam.org/jpelkey3/BRITE
cd BRITE
make
cd ..

# TODO: may need recompile waf on new platform before using it
git clone https://github.com/PrinceS17/BBR_test.git
cd BBR_test/ns-3.27
CXXFLAGS="-Wall" ./waf configure --with-brite=../../BRITE --enable-examples --enable-sudo --visualize
./waf build
mkdir MboxStatistics MboxFig

