sudo apt-get install libsnappy-dev
pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl --progress-bar off
pip install -e .[dev] --progress-bar off
pip install python-snappy pyarrow -I --progress-bar off

# Install dev version of skorch
pip install git+https://github.com/dnouri/skorch@24ac0d1392306da2337174eba206446fab7b179c
