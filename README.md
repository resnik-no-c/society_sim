# Enhanced Constraint Cascade Simulation v3

**A streamlined, high-performance agent-based model exploring cooperation dynamics under stress with realistic social mechanics.**

## 🎯 Overview

The Enhanced Constraint Cascade Simulation v3 models how cooperation emerges and persists in populations facing external shocks and internal social dynamics. This version features a **streamlined 13-parameter interface** that replaces the legacy 18-parameter system, providing improved interpretability and performance while preserving all analytical capabilities.

### Key Features

- **🚀 Streamlined Interface**: 13 focused parameters replacing complex legacy system
- **⚡ Enhanced Performance**: Optimized simulation engine with improved scalability  
- **📊 Rich Analytics**: Comprehensive metrics including Maslow hierarchy tracking
- **🏘️ Multi-Group Dynamics**: Realistic in-group/out-group social mechanics
- **🔄 Adaptive Logging**: Variable frequency logging optimized for different metrics
- **📈 Progress Tracking**: Aggregate progress reporting showing true work completion
- **💾 Complete Data Preservation**: All original metrics and outputs maintained

## 🔧 v3 Improvements

### Streamlined Parameter System
- **Focused Design**: 13 high-impact parameters vs. 18 legacy parameters
- **Better Interpretability**: Clear parameter roles and ranges
- **Validated Sampling**: Built-in parameter validation and safe defaults
- **Performance Gains**: Reduced computational overhead

### Enhanced Mechanics
- **Realistic Trust Development**: Gradual trust building with group biases
- **Stress Model**: Acute/chronic stress dynamics with community buffering  
- **Intervention Events**: Cross-group mixing with configurable frequency
- **Social Diffusion**: Network-wide trust smoothing effects
- **Turnover Dynamics**: Birth/replacement mechanics with inheritance

### Improved Logging
- **Maslow Needs**: Logged every 4 rounds (annually in simulation time)
- **Social Diffusion**: Logged every 8 rounds for efficiency
- **Network Topology**: Logged every 4 rounds for analysis
- **Core Metrics**: Per-round logging maintained for diagnostics

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM recommended for large experiments
- **CPU**: Multi-core processor recommended for parallel execution
- **Storage**: 1GB+ free space for results

### Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy
```

**Core Dependencies:**
- `numpy` - Numerical computations
- `pandas` - Data analysis and CSV handling
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting (optional)
- `scipy` - Statistical analysis (optional)

## 🚀 Quick Start

### Basic Usage
```bash
# Run single simulation test
python3 constraint_simulation_v3.py --test smoke

# Run 50 simulations (default)
python3 constraint_simulation_v3.py

# Run 150 simulations with multiprocessing
python3 constraint_simulation_v3.py --runs 150 --multiprocessing
```

### Background Execution
```bash
# Run large experiment in background
nohup python3 -u constraint_simulation_v3.py --runs 150 --multiprocessing > simulation.log 2>&1 &

# Monitor progress
tail -f simulation.log
```

## 📊 Command Line Options

### Basic Options
```bash
python3 constraint_simulation_v3.py [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--runs N` | Number of simulations to run | 50 |
| `--multiprocessing` | Use parallel processing | Auto-detect |
| `--single-thread` | Force single-threaded execution | False |

### Testing Options
| Option | Description |
|--------|-------------|
| `--test quick` | Fast unit tests |
| `--test smoke` | Single 30-round simulation test |
| `--test batch` | 5 simulations with 60 rounds each |

### Advanced Options
| Option | Description | Default |
|--------|-------------|---------|
| `--sweep basic` | Run parameter sweep | - |
| `--params N` | Number of parameter configs for sweep | 50 |

## 🎛️ Parameter System

### v3 Core Parameters (13 Parameters)

| Parameter | Description | Range/Options | Impact |
|-----------|-------------|---------------|---------|
| `shock_interval_years` | Mean years between external shocks | [2, 5, 10, 20] | System stability |
| `homophily_bias` | Preference for same-group interactions | [0.0, 0.8] | Group segregation |
| `num_groups` | Number of identity groups | [1, 2, 3] | Social complexity |
| `out_group_trust_bias` | Trust scaling for out-group interactions | [0.8, 1.2] | Inter-group relations |
| `out_group_penalty` | Extra penalty for out-group betrayals | [1.1, 1.5] | Group conflict intensity |
| `intervention_interval` | Rounds between cross-group events | [10, 15, 20, 25] | Social integration |
| `intervention_scale` | Fraction of agents in mixing events | [0.05, 0.30] | Integration effectiveness |
| `event_bonus` | Payoff multiplier during events | [1.5, 2.5] | Cooperation incentives |
| `base_trust_delta` | Trust change magnitude per interaction | [0.05, 0.20] | Trust development speed |
| `group_trust_bias` | In-group vs out-group trust scaling | [1.2, 2.0] | Group preference strength |
| `resilience_profile` | Decision threshold + noise | threshold: [0.1, 0.4]<br>noise: [0.0, 0.15] | Individual variation |
| `turnover_rate` | Birth/replacement rate per round | [0.02, 0.05] | Population dynamics |
| `social_diffusion` | Network-wide trust smoothing | [0.0, 0.10] | Social influence |

### Legacy Parameters (Preserved)
- Population settings (initial: 200, max: 800)
- Maslow hierarchy parameters
- Constraint thresholds and recovery mechanics
- Trust thresholds and relationship limits

## 📈 Output Files

### Primary Outputs

#### 1. Incremental CSV (`simulation_results_incremental.csv`)
Real-time results appended as simulations complete.

**Key Columns:**
- **v3 Parameters**: All 13 core parameters plus legacy settings
- **Outcomes**: Cooperation rates, population dynamics, extinction events
- **Maslow Metrics**: Initial/final needs, changes across hierarchy levels
- **Group Dynamics**: In/out-group interactions, trust asymmetry, segregation
- **Derived Metrics**: Shock frequency, trust sensitivity, social cohesion

#### 2. Individual Results (`sim_XXXX_result.pkl/json`)
Complete simulation objects with full state information.

#### 3. Analysis Outputs
- `v3_streamlined_simulation_results.csv` - Complete dataset
- `v3_summary_stats.csv` - Statistical summaries  
- `v3_experiment_summary.txt` - Human-readable report
- `v3_streamlined_analysis.png` - Visualization plots

### Key Metrics Explained

#### Cooperation Metrics
- **`final_cooperation_rate`**: Proportion of agents using cooperative strategy
- **`cooperation_resilience`**: Cooperation rate after system stress
- **`total_mutual_coop`**: Count of successful cooperative interactions

#### Population Dynamics  
- **`population_growth`**: Peak population / initial population
- **`population_stability`**: Coefficient of variation in later rounds
- **`total_births/deaths`**: Demographic transitions

#### Trust & Social Capital
- **`avg_trust_level`**: Mean trust across all relationships
- **`trust_asymmetry`**: In-group trust - out-group trust
- **`social_cohesion_factor`**: Social diffusion × average trust

#### Stress & Resilience
- **`total_shock_events`**: External stress events experienced
- **`avg_system_stress`**: Mean stress level across simulation
- **`total_defections`**: Strategy switches from cooperation to selfishness

## 🎮 Simulation Mechanics

### Core Dynamics

#### 1. Shock System
- **Timing**: Exponential distribution with mean = `shock_interval_years`
- **Magnitude**: Pareto distribution (bounded 0.1-2.0)
- **Effects**: Increases acute stress, affects cooperation decisions

#### 2. Interaction Engine
- **Frequency**: ~5% of population per round (12 max per agent)
- **Partner Selection**: Homophily bias influences group preference
- **Cooperation**: Based on resilience threshold + trust levels
- **Trust Updates**: Scaled by group bias and base delta

#### 3. Group Dynamics
- **Identity**: Agents belong to 1-3 groups (A, B, C)
- **Inheritance**: Children inherit group from random parent
- **Bias Effects**: In-group interactions favored, out-group penalties applied
- **Interventions**: Cross-group mixing events at regular intervals

#### 4. Population Evolution
- **Births**: Triggered by mutual cooperation, limited by turnover rate
- **Deaths**: Natural aging plus stress-induced mortality
- **Strategy Changes**: Cooperation ↔ selfishness based on constraint levels

### Time Scale
- **1 Round** = ~3 months (quarterly)  
- **300 Rounds** = 75 simulation years
- **4 Rounds** = 1 year (for annual logging)

## 📊 Analysis Examples

### Basic Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('simulation_results_incremental.csv')

# Cooperation vs shock frequency
plt.scatter(df['shock_frequency'], df['final_cooperation_rate'])
plt.xlabel('Shock Frequency (1/years)')
plt.ylabel('Final Cooperation Rate') 
plt.title('Shock Impact on Cooperation')
plt.show()

# Group dynamics
multi_group = df[df['num_groups'] > 1]
plt.scatter(multi_group['homophily_bias'], multi_group['trust_asymmetry'])
plt.xlabel('Homophily Bias')
plt.ylabel('Trust Asymmetry (In-group - Out-group)')
plt.show()
```

### Performance Analysis
```python
# Identify high-performing configurations
top_coop = df.nlargest(10, 'final_cooperation_rate')
print("Top cooperation parameters:")
print(top_coop[['shock_interval_years', 'resilience_threshold', 
                'social_diffusion', 'final_cooperation_rate']])

# Resilience analysis  
stable_societies = df[df['population_stability'] < 0.1]
print(f"Stable societies: {len(stable_societies)}/{len(df)} ({100*len(stable_societies)/len(df):.1f}%)")
```

## ⚡ Performance Guidelines

### Recommended Configurations

#### Small Experiments (< 50 simulations)
- **Single-threaded**: Fine for testing and development
- **Memory**: 2GB sufficient
- **Time**: ~30-60 minutes

#### Medium Experiments (50-200 simulations) 
- **Multiprocessing**: Recommended  
- **Memory**: 4-8GB recommended
- **Cores**: 4-8 cores optimal
- **Time**: 1-4 hours

#### Large Experiments (200+ simulations)
- **Background execution**: Use `nohup` 
- **Memory**: 8GB+ recommended
- **Cores**: 8+ cores for best performance
- **Time**: 4+ hours

### Optimization Tips

1. **Use Multiprocessing**: 3-5x speedup on multi-core systems
2. **Background Execution**: Prevents interruption of long runs
3. **Monitor Progress**: Use `tail -f simulation.log` to track status
4. **Disk Space**: Reserve 1GB+ for large experiments
5. **Parameter Bounds**: Extreme parameters may slow simulation

## 🐛 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Update pip and install dependencies
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy
```

#### Memory Issues
- Reduce number of parallel workers
- Use `--single-thread` for memory-constrained systems
- Monitor memory usage: `htop` or `top`

#### Slow Performance
- Enable multiprocessing: `--multiprocessing`
- Reduce simulation count for testing
- Check CPU utilization during execution

#### Parameter Validation Errors
```
AssertionError: Invalid intervention_interval: 0
```
**Solution**: This is normal for single-group scenarios. The simulation will use safe defaults.

#### Import Errors
```bash
# Install missing dependencies
pip install pandas numpy matplotlib

# For visualization features (optional)
pip install seaborn scipy
```

### Error Recovery

The simulation includes comprehensive error handling:
- **Parameter Validation**: Automatic fallback to safe defaults
- **Simulation Crashes**: Emergency result generation
- **File I/O Errors**: Graceful degradation with warnings
- **Memory Issues**: Automatic cleanup and continuation

### Getting Help

1. **Run Tests**: Start with `--test smoke` to verify installation
2. **Check Logs**: Review `simulation.log` for detailed error messages  
3. **Parameter Issues**: Use single-group defaults (`num_groups=1`)
4. **Performance**: Start with small runs (`--runs 10`) to test timing

## 📚 Research Applications

### Use Cases

#### Social Science Research
- **Cooperation Evolution**: How cooperation emerges under stress
- **Group Dynamics**: In-group/out-group bias effects
- **Social Capital**: Trust network formation and maintenance
- **Resilience**: Community response to external shocks

#### Policy Analysis  
- **Intervention Design**: Optimal cross-group mixing strategies
- **Shock Response**: Preparation for economic/social disruptions
- **Community Building**: Social cohesion enhancement methods
- **Inequality**: Group-based disparities in outcomes

#### Computational Studies
- **Parameter Sensitivity**: Which factors most influence cooperation
- **Phase Transitions**: Critical thresholds for cooperation collapse
- **Network Effects**: Social diffusion and contagion dynamics
- **Evolutionary Dynamics**: Strategy change and adaptation

### Validation Features

- **Realistic Parameters**: Based on empirical social science research
- **Multiple Scales**: Individual, group, and population-level dynamics  
- **Temporal Patterns**: Long-term evolution over 75 simulation years
- **Robustness Testing**: Comprehensive parameter sweeps and edge cases

## 🔬 Technical Details

### Architecture
- **Agent-Based Model**: Individual agents with heterogeneous traits
- **Event-Driven**: Shocks, interventions, and demographic events
- **Network Dynamics**: Relationship formation and trust evolution
- **Multi-Level**: Individual psychology + group identity + system stress

### Validation
- **Unit Tests**: Core mechanics verification
- **Integration Tests**: End-to-end simulation validation  
- **Performance Tests**: Scalability and timing benchmarks
- **Parameter Validation**: Automatic bounds checking and safe defaults

### Data Integrity
- **Complete Preservation**: All original metrics maintained
- **Incremental Saving**: Results saved as completed (crash recovery)
- **Multiple Formats**: CSV, JSON, and pickle outputs
- **Comprehensive Logging**: Detailed execution traces

## 📄 Citation

If you use this simulation in research, please cite:

```
Enhanced Constraint Cascade Simulation v3: A Streamlined Agent-Based Model 
for Cooperation Dynamics Under Social Stress. 2025.
```

## 📜 License

This simulation is provided for research and educational purposes. See documentation for full license terms.

---

**Version**: 3.0  
**Last Updated**: 2025  
**Compatibility**: Python 3.8+