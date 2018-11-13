#!/bin/bash
POME=$PWD
export MATLAB_ROOT=/usr/local/matlab/
mkdir -p $POME/lib
mkdir -p $POME/exe
mkdir -p $POME/src/filters
mkdir -p $POME/src/ckpt

if [ "$1" == "luajit" ]; then
    cd $POME/lib
    if [ ! -f luajit-rocks ]; then
	git clone https://github.com/torch/luajit-rocks.git
    fi
    cd $POME/lib/luajit-rocks
    rm -rf build .git
    mkdir build
    cd $POME/lib/luajit-rocks/build
    cmake .. -DCMAKE_INSTALL_PREFIX=$POME/exe/luajit 
    make install
fi

LUAROCKS=$POME/exe/luajit/bin/luarocks

if [ "$1" == "basic" ]; then
    cd $POME/lib
    if [ ! -f cwrap ]; then
	git clone https://github.com/torch/cwrap.git
    fi
    cd $POME/lib/cwrap
    $LUAROCKS make rocks/cwrap-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f paths ]; then
	git clone https://github.com/torch/paths.git
    fi
    cd $POME/lib/paths

    $LUAROCKS make rocks/paths-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f torch7 ]; then
	git clone https://github.com/torch/torch7.git
    fi
    cd $POME/lib/torch7
    $LUAROCKS make rocks/torch-scm-1.rockspec #1
    rm -rf build .git

    cd $POME/lib
    if [ ! -f luaffifb ]; then
	git clone https://github.com/facebook/luaffifb.git
    fi
    cd $POME/lib/luaffifb
    $LUAROCKS make luaffi-scm-1.rockspec
    rm -rf build .git
fi

if [ "$1" == "nn" ]; then
    cd $POME/lib
    if [ ! -f nn ]; then
        git clone https://github.com/torch/nn.git
    fi
    cd $POME/lib/nn
    $LUAROCKS make rocks/nn-scm-1.rockspec
    cd $POME/lib
    if [ ! -f nngraph ]; then
	git clone https://github.com/torch/nngraph.git
    fi
    cd $POME/lib/nngraph
    $LUAROCKS make nngraph-scm-1.rockspec
    rm -rf build .git
fi

if [ "$1" == "extra" ]; then
    cd $POME/lib
    if [ ! -f optim ]; then
        git clone https://github.com/torch/optim.git
    fi
    cd $POME/lib/optim
    $LUAROCKS make optim-1.0.5-0.rockspec
    rm -rf build .git
    
    cd $POME/lib
    if [ ! -f sys ]; then
	git clone https://github.com/torch/sys
    fi
    cd $POME/lib/sys
    $LUAROCKS make sys-1.0-0.rockspec
    rm -rf build .git
    
    cd $POME/lib
    if [ ! -f xlua ]; then
	git clone https://github.com/torch/xlua.git
    fi
    cd $POME/lib/xlua
    $LUAROCKS make xlua-1.0-0.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f sundown-ffi ]; then
	git clone https://github.com/torch/sundown-ffi.git
    fi
    cd $POME/lib/sundown-ffi
    $LUAROCKS make rocks/sundown-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f dok ]; then
        git clone https://github.com/torch/dok.git
    fi
    cd $POME/lib/dok
    $LUAROCKS make rocks/dok-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f image ]; then
	git clone https://github.com/torch/image.git
    fi
    cd $POME/lib/image
    $LUAROCKS make image-1.1.alpha-0.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f fftw3-ffi ]; then
	git clone https://github.com/soumith/fftw3-ffi.git
    fi
    cd $POME/lib/fftw3-ffi
    $LUAROCKS make rocks/fftw3-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f torch-signal ]; then
	git clone https://github.com/soumith/torch-signal.git
    fi
    cd $POME/lib/torch-signal
    $LUAROCKS make rocks/signal-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f gnuplot ]; then
	git clone https://github.com/torch/gnuplot.git
    fi
    cd $POME/lib/gnuplot
    $LUAROCKS make rocks/gnuplot-scm-1.rockspec
    rm -rf build .git
fi

if [ "$1" == "mattorch" ]; then 
    cd $POME/lib
    if [ ! -f mattorch ]; then
	git clone https://github.com/clementfarabet/lua---mattorch.git
    fi
    cd $POME/lib/lua---mattorch
    $LUAROCKS make mattorch-1.0-0.rockspec
    rm -rf build .git
fi

if [ "$1" == "cuda" ]; then
    cd $POME/lib
    if [ ! -f cutorch ]; then
	git clone https://github.com/torch/cutorch.git
    fi
    cd $POME/lib/cutorch
    $LUAROCKS make rocks/cutorch-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f cunn ]; then
        git clone https://github.com/torch/cunn.git
    fi
    cd $POME/lib/cunn
    $LUAROCKS make rocks/cunn-scm-1.rockspec
    rm -rf build .git

    cd $POME/lib
    if [ ! -f cudnn.torch ]; then
        git clone https://github.com/soumith/cudnn.torch.git
    fi
    cd $POME/lib/cudnn.torch
    $LUAROCKS make cudnn-scm-1.rockspec
    rm -rf build .git
fi

if [ "$1" == "scat" ]; then
    cd $POME/lib/scatwave-synth
    $LUAROCKS make scatwave-scm-1.rockspec
    rm -rf build .git
fi

if [ "$1" == "randomkit" ]; then
    cd $POME/lib
    if [ ! -f torch-randomkit ]; then
        git clone https://github.com/deepmind/torch-randomkit.git
    fi
    cd $POME/lib/torch-randomkit
    $LUAROCKS make rocks/randomkit-scm-1.rockspec
    rm -rf build
    mv .git .git_
fi
 
echo "7 options: luajit,basic,nn,extra,scat,randomkit,cuda,mattorch"
echo "to export=LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$POME/exe/luajit/lib/lua/5.1"
