🚀 Complete Step-by-Step Guide:
Step 1: Kill Current Process & Pull Latest Code
bash# Kill any running simulations
pkill -f python3

# Navigate to repository and pull latest changes
cd society_sim
git pull origin main
Step 2: Run in Background with Custom Parameters
bash# Start simulation in background with custom settings (unbuffered output)
# Example: 500 runs with multiprocessing enabled
nohup python3 -u constraint_simulation_v2.py -n 500 -m > simulation.log 2>&1 &

# Alternative examples:
# nohup python3 -u constraint_simulation_v2.py -n 1000 --multiprocessing > simulation.log 2>&1 &
# nohup python3 -u constraint_simulation_v2.py --num-runs 50 --single-thread > simulation.log 2>&1 &
# nohup python3 -u constraint_simulation_v2.py > simulation.log 2>&1 &  # (default: 100 runs)
Step 3: Stream Log with Timestamps
bash# Watch the log file with live timestamps
tail -f simulation.log | ts '[%H:%M:%S]'
🎯 All Commands in Sequence:
bashpkill -f python3
cd society_sim
git pull origin main
nohup python3 -u constraint_simulation_v3.py -n 100 --design random -m > simulation.log 2>&1 &
tail -f simulation.log | ts '[%H:%M:%S]'
📊 What You'll See:
[18:52:15] 🔬 Enhanced Constraint Cascade Simulation - Mass Parameter Exploration
[18:52:15] ⚙️  Experiment Configuration:
[18:52:15]    🔢 Number of simulations: 500
[18:52:15]    🖥️  Multiprocessing: ✅ ENABLED
[18:52:15] 🚀 Starting enhanced mass experiment with 500 simulations...
[18:52:16] 🔧 Using 8 CPU cores for parallel processing...
[18:54:22] ⏳ Progress: 50/500 (10.0%) | Rate: 1.2 sim/sec | ETA: 375.0s
🎛️ Command Options:
bash# Quick runs for testing
python3 -u constraint_simulation_v2.py -n 10 --single-thread

# Medium-scale study  
nohup python3 -u constraint_simulation_v2.py -n 200 -m > simulation.log 2>&1 &

# Large-scale research
nohup python3 -u constraint_simulation_v2.py -n 1000 --multiprocessing > simulation.log 2>&1 &

# Help and options
python3 constraint_simulation_v2.py --help