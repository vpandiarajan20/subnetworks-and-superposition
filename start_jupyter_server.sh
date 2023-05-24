unset XDG_RUNTIME_DIR
#interact -n 20 -t 01:00:00 -m 10g
module load python/3.8.12_gcc8.3 cuda/11.7.1 openssl/1.1.1
source env/bin/activate
ipnip=$(hostname -i)
ipnport=8889
echo "Paste the following command onto your local computer:"
echo "ssh -N -L ${ipnport}:${ipnip}:${ipnport} $USER@sshcampus.ccv.brown.edu"
output = $(jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip)