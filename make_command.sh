cmake -DCMAKE_BUILD_TYPE=Debug -DGAUXC_ENABLE_OPENMP=off -DGAUXC_ENABLE_MPI=off ..
make -j 4
make DESTDIR=../install2 install
touch /Users/aodongliu/Softwares/GauXC-devel/cmake/gauxc-exchcxx.cmake
