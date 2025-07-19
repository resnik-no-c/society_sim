
🎯 All Commands in Sequence:
pkill -f python3
cd society_sim
git pull origin main
nohup python3 -u constraint_simulation_v3.py -n 100 -m > simulation.log 2>&
tail -f simulation.log |

🎛️ Command Options:
bash# Quick runs for testing
python3 -u constraint_simulation_v2.py -n 10 --single-thread

# Medium-scale study  
nohup python3 -u constraint_simulation_v2.py -n 200 -m > simulation.log 2>&1 &

# Large-scale research
nohup python3 -u constraint_simulation_v2.py -n 1000 --multiprocessing > simulation.log 2>&1 &

# Help and options
python3 constraint_simulation_v2.py --help

# Compressing results from the past 30 minutes
find . -type f -mmin -30 -print0 | tar --null -czvf recent_files_$(date +%Y%m%d_%H%M).tar.gz --files-from -
