
🎯 All Commands in Sequence:
pkill -f python3
cd society_sim
git pull origin main
nohup python3 -u constraint_simulation_v3.py -n 500 -m > simulation.log 2>&1 &
tail -f simulation.log |

🎛️ Command Options:
bash# Quick runs for testing
python3 -u constraint_simulation_v2.py -n 10 --single-thread

# Compressing results from the past 30 minutes
find . -type f -mmin -120 -print0 | tar --null -czvf recent_files_$(date +%Y%m%d_%H%M).tar.gz --files-from -
