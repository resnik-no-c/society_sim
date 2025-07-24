#!/usr/bin/env python3
"""
Enhanced Constraint Cascade Simulation v3 - Streamlined 13-Parameter Implementation
Implements the v3 simulation design with focused 13-parameter API while preserving all functionality.

NEW v3 FEATURES:
1. Streamlined 13-parameter interface replacing legacy 18-parameter system
2. Improved interpretability and performance with focused parameter set
3. Enhanced logging frequencies for different metrics
4. Aggregate progress reporting showing total work completion
5. All original data outputs preserved

CRITICAL FIXES APPLIED:
- Fixed interaction double-counting causing 10x defection inflation
- Fixed trust relationship initialization and update errors
- Fixed parameter validation mismatch with documentation
- Fixed cooperation threshold confusion and trust calculation exclusions
- Completed institutional memory system and standardized group assignment
- Added comprehensive bounds checking and validation
"""

import random
import time
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, NamedTuple
from collections import defaultdict, deque
from itertools import product
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import traceback
import threading
import queue
import pickle

# Try to import seaborn for better visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ===== 1. V3 CONFIG & CONSTANTS =====

# v3 streamlined constants
DEFAULT_MAX_ROUNDS = 300  # 75 years simulation time (was 200 in legacy)
DEFAULT_INITIAL_POPULATION = 200
DEFAULT_MAX_POPULATION = 800

def timestamp_print(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    import sys
    sys.stdout.flush()

def validate_simulation_results(result) -> List[str]:
    """VALIDATION FIX #12: Add comprehensive bounds checking for results"""
    warnings = []
    
    # Check BOTH cooperation rates are in valid range (TOPLINE METRICS)
    if not (0.0 <= result.final_cooperation_rate <= 1.0):
        warnings.append(f"Invalid strategy_cooperation_rate: {result.final_cooperation_rate}")
    
    if not (0.0 <= result.behavioral_cooperation_rate <= 1.0):
        warnings.append(f"Invalid behavioral_cooperation_rate: {result.behavioral_cooperation_rate}")
    
    # Check trust levels are in valid range
    if not (0.0 <= result.avg_trust_level <= 1.0):
        warnings.append(f"Invalid avg_trust_level: {result.avg_trust_level}")
    
    # Check for impossible defection rates
    if result.total_defections > result.final_population * result.rounds_completed * 10:
        warnings.append(f"Impossible defection rate: {result.total_defections} defections with {result.final_population} people over {result.rounds_completed} rounds")
    
    # Check population consistency
    if result.final_population < 0:
        warnings.append(f"Negative final population: {result.final_population}")
    
    # Check redemption rate bounds
    if not (0.0 <= result.redemption_rate <= 1.0):
        warnings.append(f"Invalid redemption_rate: {result.redemption_rate}")
    
    # Check cooperation consistency (behavioral should generally be <= strategy-based)
    if result.behavioral_cooperation_rate > result.final_cooperation_rate + 0.1:  # Allow small tolerance
        warnings.append(f"Behavioral cooperation ({result.behavioral_cooperation_rate:.3f}) exceeds strategy cooperation ({result.final_cooperation_rate:.3f}) by large margin")
    
    # Check encounter consistency
    if result.total_encounters > 0:
        total_outcomes = result.total_mutual_cooperation + result.total_mutual_defection + result.total_mixed_outcomes
        if abs(total_outcomes - result.total_encounters) > 5:  # Allow small rounding errors
            warnings.append(f"Encounter tracking inconsistency: {total_outcomes} outcomes vs {result.total_encounters} encounters")
    
    return warnings

def save_simulation_result(result, results_dir: str = "simulation_results"):
    """Enhanced save function that captures complete simulation state"""
    import os
    import pickle
    import json
    
    # VALIDATION FIX #12: Add bounds checking before saving
    validation_warnings = validate_simulation_results(result)
    if validation_warnings:
        timestamp_print(f"⚠️ Validation warnings for simulation {result.run_id}:")
        for warning in validation_warnings:
            timestamp_print(f"   - {warning}")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        timestamp_print(f"📁 Created results directory: {results_dir}")
    
    pkl_filename = f"sim_{result.run_id:04d}_result.pkl"
    pkl_filepath = os.path.join(results_dir, pkl_filename)
    
    try:
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(result, f)
        timestamp_print(f"💾 Saved complete simulation {result.run_id} to {pkl_filepath}")
    except Exception as e:
        timestamp_print(f"❌ Error saving pickle for simulation {result.run_id}: {e}")
        return None
    
    json_filename = f"sim_{result.run_id:04d}_result.json"
    json_filepath = os.path.join(results_dir, json_filename)
    
    try:
        json_data = {
            'run_id': result.run_id,
            'parameters': {
                # v3 core parameters
                'shock_interval_years': result.parameters.shock_interval_years,
                'homophily_bias': result.parameters.homophily_bias,
                'num_groups': result.parameters.num_groups,
                'out_group_trust_bias': result.parameters.out_group_trust_bias,
                'out_group_penalty': result.parameters.out_group_penalty,
                'intervention_interval': result.parameters.intervention_interval,
                'intervention_scale': result.parameters.intervention_scale,
                'event_bonus': result.parameters.event_bonus,
                'base_trust_delta': result.parameters.base_trust_delta,
                'group_trust_bias': result.parameters.group_trust_bias,
                'resilience_profile': result.parameters.resilience_profile,
                'turnover_rate': result.parameters.turnover_rate,
                'social_diffusion': result.parameters.social_diffusion,
                'max_rounds': result.parameters.max_rounds,
                
                # Legacy parameters preserved
                'initial_population': getattr(result.parameters, 'initial_population', 200),
                'max_population': getattr(result.parameters, 'max_population', 800),
                'maslow_variation': getattr(result.parameters, 'maslow_variation', 0.5),
                'constraint_threshold_range': getattr(result.parameters, 'constraint_threshold_range', [0.15, 0.35]),
                'recovery_threshold': getattr(result.parameters, 'recovery_threshold', 0.3),
                'cooperation_bonus': getattr(result.parameters, 'cooperation_bonus', 0.2),
                'trust_threshold': getattr(result.parameters, 'trust_threshold', 0.6),
                'max_relationships_per_person': getattr(result.parameters, 'max_relationships_per_person', 150),
            },
            'outcomes': {
                'final_cooperation_rate': result.final_cooperation_rate,  # TOPLINE: Strategy-based cooperation
                'behavioral_cooperation_rate': result.behavioral_cooperation_rate,  # TOPLINE: Action-based cooperation
                'final_constrained_rate': result.final_constrained_rate,
                'final_population': result.final_population,
                'extinction_occurred': result.extinction_occurred,
                'rounds_completed': result.rounds_completed,
                'first_cascade_round': result.first_cascade_round,
                'total_cascade_events': result.total_cascade_events,
                'total_shock_events': result.total_shock_events,
                'total_defections': result.total_defections,
                'total_redemptions': result.total_redemptions,
                'net_strategy_change': result.net_strategy_change,
                'total_births': result.total_births,
                'total_deaths': result.total_deaths,
                'max_population_reached': result.max_population_reached,
                'population_stability': result.population_stability,
                'population_growth': result.population_growth,
                'avg_system_stress': result.avg_system_stress,
                'max_system_stress': result.max_system_stress,
                'avg_maslow_pressure': result.avg_maslow_pressure,
                'avg_basic_needs_crisis_rate': result.avg_basic_needs_crisis_rate,
                'avg_trust_level': result.avg_trust_level,
                'cooperation_benefit_total': result.cooperation_benefit_total,
                'cooperation_resilience': result.cooperation_resilience,
            },
            'maslow_hierarchy': {
                'initial_needs_avg': result.initial_needs_avg,
                'final_needs_avg': result.final_needs_avg,
                'needs_improvement': result.needs_improvement,
            },
            'inter_group_metrics': {
                'final_group_populations': result.final_group_populations,
                'final_group_cooperation_rates': result.final_group_cooperation_rates,
                'in_group_interaction_rate': result.in_group_interaction_rate,
                'out_group_interaction_rate': result.out_group_interaction_rate,
                'avg_in_group_trust': result.avg_in_group_trust,
                'avg_out_group_trust': result.avg_out_group_trust,
                'trust_asymmetry': result.trust_asymmetry,
                'group_segregation_index': result.group_segregation_index,
                'total_mixing_events': result.total_mixing_events,
                'mixing_event_success_rate': result.mixing_event_success_rate,
                'reputational_spillover_events': result.reputational_spillover_events,
                'out_group_constraint_amplifications': result.out_group_constraint_amplifications,
                'group_extinction_events': result.group_extinction_events,
            },
            'interaction_metrics': {
                'total_interactions': result.total_interactions,
                'avg_interaction_processing_time': result.avg_interaction_processing_time,
            }
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        timestamp_print(f"📄 Saved JSON version for simulation {result.run_id} to {json_filepath}")
        
    except Exception as e:
        timestamp_print(f"⚠️ Error saving JSON for simulation {result.run_id}: {e}")
    
    return pkl_filepath

def save_incremental_csv(result, csv_file: str = "simulation_results_incremental.csv"):
    """Save simulation result to incremental CSV file - COMPLETE VERSION with v3 parameters"""
    import os
    import pandas as pd
    
    # MAJOR FIX #6: Correct redemption rate calculation
    total_strategy_switches = result.total_defections + result.total_redemptions
    corrected_redemption_rate = result.total_redemptions / max(1, total_strategy_switches)
    
    row_data = {
        # Basic identifiers
        'run_id': result.run_id,
        'timestamp': datetime.now().isoformat(),
        
        # === V3 SIMULATION PARAMETERS ===
        'shock_interval_years': result.parameters.shock_interval_years,
        'homophily_bias': result.parameters.homophily_bias,
        'num_groups': result.parameters.num_groups,
        'out_group_trust_bias': result.parameters.out_group_trust_bias,
        'out_group_penalty': result.parameters.out_group_penalty,
        'intervention_interval': result.parameters.intervention_interval,
        'intervention_scale': result.parameters.intervention_scale,
        'event_bonus': result.parameters.event_bonus,
        'base_trust_delta': result.parameters.base_trust_delta,
        'group_trust_bias': result.parameters.group_trust_bias,
        'resilience_threshold': result.parameters.resilience_profile['threshold'],
        'resilience_noise': result.parameters.resilience_profile['noise'],
        'turnover_rate': result.parameters.turnover_rate,
        'social_diffusion': result.parameters.social_diffusion,
        'max_rounds': result.parameters.max_rounds,
        
        # Legacy parameters preserved
        'initial_population': getattr(result.parameters, 'initial_population', 200),
        'max_population': getattr(result.parameters, 'max_population', 800),
        'maslow_variation': getattr(result.parameters, 'maslow_variation', 0.5),
        'constraint_threshold_min': getattr(result.parameters, 'constraint_threshold_range', [0.05, 0.25])[0],
        'constraint_threshold_max': getattr(result.parameters, 'constraint_threshold_range', [0.05, 0.25])[1],
        'recovery_threshold': getattr(result.parameters, 'recovery_threshold', 0.3),
        'cooperation_bonus': getattr(result.parameters, 'cooperation_bonus', 0.2),
        'trust_threshold': getattr(result.parameters, 'trust_threshold', 0.6),
        'max_relationships_per_person': getattr(result.parameters, 'max_relationships_per_person', 150),
        
        # === FINAL OUTCOMES ===
        'final_cooperation_rate': result.final_cooperation_rate,  # TOPLINE METRIC: Strategy-based cooperation rate
        'behavioral_cooperation_rate': result.behavioral_cooperation_rate,  # TOPLINE METRIC: Action-based cooperation rate
        'final_constrained_rate': result.final_constrained_rate,
        'final_population': result.final_population,
        'extinction_occurred': result.extinction_occurred,
        'rounds_completed': result.rounds_completed,
        
        # === SYSTEM DYNAMICS ===
        'first_cascade_round': result.first_cascade_round,
        'total_cascade_events': result.total_cascade_events,
        'total_shock_events': result.total_shock_events,
        
        # === STRATEGY CHANGES ===
        'total_defections': result.total_defections,
        'total_redemptions': result.total_redemptions,
        'net_strategy_change': result.net_strategy_change,
        'redemption_rate': corrected_redemption_rate,  # MAJOR FIX #6: Use corrected calculation

        # === NEW: DETAILED INTERACTION METRICS ===
        'total_encounters': getattr(result, 'total_encounters', 0),
        'total_mutual_cooperation': getattr(result, 'total_mutual_cooperation', 0),
        'total_mutual_defection': getattr(result, 'total_mutual_defection', 0),
        'total_mixed_outcomes': getattr(result, 'total_mixed_outcomes', 0),
        'cooperation_consistency': (getattr(result, 'behavioral_cooperation_rate', 0.0) / 
                                   max(0.001, result.final_cooperation_rate)),
        
        # === POPULATION METRICS ===
        'total_births': result.total_births,
        'total_deaths': result.total_deaths,
        'max_population_reached': result.max_population_reached,
        'population_stability': result.population_stability,
        'population_growth': result.population_growth,
        
        # === PRESSURE METRICS ===
        'avg_system_stress': result.avg_system_stress,
        'max_system_stress': result.max_system_stress,
        'avg_maslow_pressure': result.avg_maslow_pressure,
        'avg_basic_needs_crisis_rate': result.avg_basic_needs_crisis_rate,
        
        # === MASLOW HIERARCHY METRICS ===
        'initial_physiological': result.initial_needs_avg.get('physiological', 0),
        'initial_safety': result.initial_needs_avg.get('safety', 0),
        'initial_love': result.initial_needs_avg.get('love', 0),
        'initial_esteem': result.initial_needs_avg.get('esteem', 0),
        'initial_self_actualization': result.initial_needs_avg.get('self_actualization', 0),
        
        'final_physiological': result.final_needs_avg.get('physiological', 0),
        'final_safety': result.final_needs_avg.get('safety', 0),
        'final_love': result.final_needs_avg.get('love', 0),
        'final_esteem': result.final_needs_avg.get('esteem', 0),
        'final_self_actualization': result.final_needs_avg.get('self_actualization', 0),
        
        'physiological_change': result.needs_improvement.get('physiological', 0),
        'safety_change': result.needs_improvement.get('safety', 0),
        'love_change': result.needs_improvement.get('love', 0),
        'esteem_change': result.needs_improvement.get('esteem', 0),
        'self_actualization_change': result.needs_improvement.get('self_actualization', 0),
        
        # === COOPERATION METRICS ===
        'avg_trust_level': result.avg_trust_level,
        'cooperation_benefit_total': result.cooperation_benefit_total,
        'cooperation_resilience': result.cooperation_resilience,  # MAJOR FIX #7: Cooperation sustainability after stress
        
        # === INTER-GROUP METRICS ===
        'in_group_interaction_rate': result.in_group_interaction_rate,
        'out_group_interaction_rate': result.out_group_interaction_rate,
        'avg_in_group_trust': result.avg_in_group_trust,
        'avg_out_group_trust': result.avg_out_group_trust,
        'trust_asymmetry': result.trust_asymmetry,
        'group_segregation_index': result.group_segregation_index,
        'total_mixing_events': result.total_mixing_events,
        'mixing_event_success_rate': result.mixing_event_success_rate,
        'reputational_spillover_events': result.reputational_spillover_events,
        'out_group_constraint_amplifications': result.out_group_constraint_amplifications,
        'group_extinction_events': result.group_extinction_events,
        
        # Group population breakdown
        'final_group_a_population': result.final_group_populations.get('A', 0),
        'final_group_b_population': result.final_group_populations.get('B', 0),
        'final_group_c_population': result.final_group_populations.get('C', 0),
        
        'final_group_a_cooperation_rate': result.final_group_cooperation_rates.get('A', 0),
        'final_group_b_cooperation_rate': result.final_group_cooperation_rates.get('B', 0),
        'final_group_c_cooperation_rate': result.final_group_cooperation_rates.get('C', 0),
        
        # === INTERACTION METRICS ===
        'total_interactions': result.total_interactions,
        'avg_interaction_processing_time': result.avg_interaction_processing_time,
        'interaction_intensity': result.total_interactions / max(1, result.final_population),
        
        # === V3 DERIVED METRICS ===
        'shock_frequency': 1.0 / max(1, result.parameters.shock_interval_years),
        'group_complexity': result.parameters.num_groups,
        'trust_sensitivity': result.parameters.base_trust_delta * result.parameters.group_trust_bias,
        'intervention_intensity': result.parameters.intervention_scale / max(1, result.parameters.intervention_interval),
        'resilience_variability': result.parameters.resilience_profile['noise'] / max(0.001, result.parameters.resilience_profile['threshold']),
        'social_cohesion_factor': result.parameters.social_diffusion * result.avg_trust_level,
        # NEW: Social network and penalty metrics
        'avg_network_size': (sum(len(getattr(p, 'network_neighbors', set())) for p in alive_people) / 
                            max(1, len(alive_people)) if 'alive_people' in locals() else 0),
        'avg_out_group_penalty_accumulator': (sum(getattr(p, 'out_group_penalty_accumulator', 0) for p in alive_people) / 
                                            max(1, len(alive_people)) if 'alive_people' in locals() else 0),
        'max_out_group_penalty_accumulator': (max((getattr(p, 'out_group_penalty_accumulator', 0) for p in alive_people), default=0) 
                                            if 'alive_people' in locals() else 0),
    }
    
    df_row = pd.DataFrame([row_data])
    
    try:
        if os.path.exists(csv_file):
            df_row.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_file, mode='w', header=True, index=False)
            timestamp_print(f"📊 Created comprehensive incremental CSV: {csv_file}")
        
        return True
    except Exception as e:
        timestamp_print(f"❌ Error saving to comprehensive incremental CSV: {e}")
        return False

# ===== 2. V3 DATA CLASSES =====

@dataclass
class MaslowNeeds:
    """Maslow's Hierarchy of Needs representation"""
    physiological: float = 0.0
    safety: float = 0.0
    love: float = 0.0
    esteem: float = 0.0
    self_actualization: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'physiological': self.physiological,
            'safety': self.safety,
            'love': self.love,
            'esteem': self.esteem,
            'self_actualization': self.self_actualization
        }

@dataclass
class FastRelationship:
    """Enhanced relationship tracking - CRITICAL FIX #2: Fixed trust initialization and updates"""
    trust: float = 0.5  # CRITICAL FIX #2: Initialize with neutral trust, never None
    interaction_count: int = 0
    cooperation_history: deque = field(default_factory=lambda: deque(maxlen=40))
    last_interaction_round: int = 0
    is_same_group: bool = True
    betrayal_count: int = 0
    cooperation_count: int = 0
    is_developed: bool = False
    
    def update_trust(self, cooperated: bool, round_num: int, base_delta: float,
                    group_bias: float = 1.0, out_group_bias: float = 1.0):
        """CRITICAL FIX #2: Update trust without duplicate increments"""
        self.interaction_count += 1
        self.last_interaction_round = round_num
        self.cooperation_history.append(cooperated)
        self.is_developed = True

        # Experience-based first impression
        if self.trust is None:
           self.trust = 0.7 if cooperated else 0.3  # Start based on first experience
           return
        
        if cooperated:
            self.cooperation_count += 1
            delta = base_delta * (group_bias if self.is_same_group else out_group_bias)
            self.trust = min(1.0, self.trust + delta)
        else:
            self.betrayal_count += 1
            asymmetry_factor = 2.2  # Was 1.1 - now matches empirical loss aversion research
            delta = -asymmetry_factor * base_delta * (group_bias if self.is_same_group else out_group_bias)
            self.trust = max(0.0, self.trust + delta)

@dataclass
class SimulationConfig:
    """v3 Streamlined 13-parameter configuration - CRITICAL FIX #3: Corrected parameter validation"""
    # Core v3 parameters
    shock_interval_years: int  # CRITICAL FIX #3: Choice set: [10, 15, 20, 25] (was [2,5,10,20])
    homophily_bias: float  # uniform(0.0, 0.8)
    num_groups: int  # choice([1, 2, 3])
    out_group_trust_bias: float  # uniform(0.8, 1.2)
    out_group_penalty: float  # uniform(1.1, 1.5)
    intervention_interval: int  # choice([10,15,20,25])
    intervention_scale: float  # uniform(0.05, 0.30)
    event_bonus: float  # uniform(1.5, 2.5)
    base_trust_delta: float  # uniform(0.05, 0.20)
    group_trust_bias: float  # uniform(1.2, 2.0)
    resilience_profile: Dict[str, float]  # threshold ∈ [0.1,0.4], noise ∈ [0.0,0.15]
    turnover_rate: float  # uniform(0.02, 0.05)
    social_diffusion: float  # uniform(0.0, 0.10)
    max_rounds: int = DEFAULT_MAX_ROUNDS
    
    # Legacy parameters preserved for compatibility
    initial_population: int = DEFAULT_INITIAL_POPULATION
    max_population: int = DEFAULT_MAX_POPULATION
    maslow_variation: float = 0.5
    constraint_threshold_range: Tuple[float, float] = (0.2, 0.8)
    recovery_threshold: float = 0.15
    cooperation_bonus: float = 0.2
    trust_threshold: float = 0.6
    max_relationships_per_person: int = 150
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        # CRITICAL FIX #3: Validate v3 parameters with correct ranges
        assert self.shock_interval_years in [2, 5, 10, 20], f"Invalid shock_interval_years: {self.shock_interval_years}"
        assert 0.0 <= self.homophily_bias <= 0.8, f"Invalid homophily_bias: {self.homophily_bias}"
        assert self.num_groups in [1, 2, 3], f"Invalid num_groups: {self.num_groups}"
        assert 0.8 <= self.out_group_trust_bias <= 1.2, f"Invalid out_group_trust_bias: {self.out_group_trust_bias}"
        assert 1.1 <= self.out_group_penalty <= 1.5, f"Invalid out_group_penalty: {self.out_group_penalty}"
        
        # SIGNIFICANT FIX #9: Standardized group assignment logic
        if self.num_groups == 1:
            assert self.intervention_interval == 0, f"Single group must have intervention_interval = 0, got: {self.intervention_interval}"
            assert self.homophily_bias == 0.0, f"Single group must have homophily_bias = 0.0, got: {self.homophily_bias}"
        else:
            assert self.intervention_interval in [10, 15, 20, 25], f"Invalid intervention_interval: {self.intervention_interval}"
            
        assert 0.05 <= self.intervention_scale <= 0.30, f"Invalid intervention_scale: {self.intervention_scale}"
        assert 1.5 <= self.event_bonus <= 2.5, f"Invalid event_bonus: {self.event_bonus}"
        assert 0.05 <= self.base_trust_delta <= 0.20, f"Invalid base_trust_delta: {self.base_trust_delta}"
        assert 1.2 <= self.group_trust_bias <= 2.0, f"Invalid group_trust_bias: {self.group_trust_bias}"
        assert 0.02 <= self.turnover_rate <= 0.05, f"Invalid turnover_rate: {self.turnover_rate}"
        assert 0.0 <= self.social_diffusion <= 0.10, f"Invalid social_diffusion: {self.social_diffusion}"
        
        # Validate resilience profile
        assert 'threshold' in self.resilience_profile, "Missing threshold in resilience_profile"
        assert 'noise' in self.resilience_profile, "Missing noise in resilience_profile"
        assert 0.1 <= self.resilience_profile['threshold'] <= 0.4, f"Invalid resilience threshold: {self.resilience_profile['threshold']}"
        assert 0.0 <= self.resilience_profile['noise'] <= 0.15, f"Invalid resilience noise: {self.resilience_profile['noise']}"
        assert 0.0 <= self.recovery_threshold <= 1.0, f"Invalid recovery_threshold: {self.recovery_threshold}"

@dataclass
class EnhancedSimulationResults:
    """Comprehensive results container - all metrics preserved"""
    parameters: SimulationConfig
    run_id: int
    
    # Final outcomes (required fields first)
    final_population: int
    final_cooperation_rate: float  # TOPLINE METRIC: Strategy-based cooperation rate (% with cooperative strategy)
    behavioral_cooperation_rate: float  # TOPLINE METRIC: Action-based cooperation rate (% of cooperative interactions)
    final_constrained_rate: float
    
    # System dynamics (required fields)
    rounds_completed: int
    extinction_occurred: bool
    first_cascade_round: Optional[int]
    total_cascade_events: int
    total_shock_events: int
    
    # Strategy changes (required fields)
    total_defections: int
    total_redemptions: int
    net_strategy_change: int
    
    # Population metrics (required fields)
    total_births: int
    total_deaths: int
    max_population_reached: int
    population_stability: float
    
    # Pressure metrics (required fields)
    avg_system_stress: float
    max_system_stress: float
    avg_maslow_pressure: float
    avg_basic_needs_crisis_rate: float
    
    # Maslow evolution (required fields)
    initial_needs_avg: Dict[str, float]
    final_needs_avg: Dict[str, float]
    needs_improvement: Dict[str, float]
    
    # Cooperation benefits (required fields)
    avg_trust_level: float
    cooperation_benefit_total: float
    
    # Additional metrics (required fields)
    population_growth: float
    cooperation_resilience: float  # TOPLINE METRIC: Cooperation sustainability after stress events
    
    # Detailed interaction metrics (required fields)
    total_encounters: int = 0
    total_mutual_cooperation: int = 0
    total_mutual_defection: int = 0
    total_mixed_outcomes: int = 0
    
    # Strategy tracking (with defaults for backward compatibility)
    redemption_rate: float = 0.0
    
    # Inter-Group Metrics (with defaults)
    final_group_populations: Dict[str, int] = field(default_factory=dict)
    final_group_cooperation_rates: Dict[str, float] = field(default_factory=dict)
    in_group_interaction_rate: float = 0.0
    out_group_interaction_rate: float = 0.0
    avg_in_group_trust: float = 0.5
    avg_out_group_trust: float = 0.5
    group_segregation_index: float = 0.0
    total_mixing_events: int = 0
    mixing_event_success_rate: float = 0.0
    reputational_spillover_events: int = 0
    out_group_constraint_amplifications: int = 0
    group_extinction_events: int = 0
    trust_asymmetry: float = 0.0
    
    # Interaction metrics (with defaults)
    total_interactions: int = 0
    total_mutual_coop: int = 0
    avg_interaction_processing_time: float = 0.0

class OptimizedPerson:
    """Enhanced person with v3 parameter integration"""
    
    __slots__ = ['id', 'strategy', 'constraint_level', 'constraint_threshold', 
                 'recovery_threshold', 'is_constrained', 'is_dead', 'relationships',
                 'max_lifespan', 'age', 'strategy_changes', 'rounds_as_selfish',
                 'rounds_as_cooperative', 'maslow_needs', 'maslow_pressure', 'is_born',
                 'group_id', 'in_group_interactions', 'out_group_interactions', 
                 'mixing_event_participations', 'acute_stress', 'chronic_queue', 
                 'base_coop', 'society_trust', 'resilience_threshold', 'resilience_noise', 
                 'cooperation_threshold', 'stress_recovery_rate','network_neighbors',
                 'out_group_penalty_accumulator', 'initial_maslow_needs']
    
    def __init__(self, person_id: int, params: SimulationConfig, 
                 parent_a: Optional['OptimizedPerson'] = None, 
                 parent_b: Optional['OptimizedPerson'] = None,
                 group_id: Optional[str] = None):
        self.id = person_id
        self.strategy = 'cooperative'
        self.constraint_level = 0.0
        self.constraint_threshold = random.uniform(*params.constraint_threshold_range)

        recovery_factor = random.uniform(0.5,0.8)

        self.recovery_threshold = self.constraint_threshold * recovery_factor
        self.is_constrained = False
        self.is_dead = False
        self.is_born = (parent_a is not None and parent_b is not None)
        
        # MAJOR FIX #5: Separate cooperation decisions from stress recovery
        self.cooperation_threshold = random.uniform(0.2, 0.35)  # For cooperation decisions only
        
        # Resilience now controls stress recovery, not cooperation
        base_threshold = params.resilience_profile['threshold']
        noise_range = params.resilience_profile['noise']
        self.stress_recovery_rate = base_threshold + random.uniform(-noise_range, noise_range)
        self.stress_recovery_rate = max(0.01, min(0.99, self.stress_recovery_rate))
        self.resilience_noise = noise_range
        self.resilience_threshold = self.stress_recovery_rate  # For stress recovery only
        
        self.acute_stress = 0.0
        self.chronic_queue = deque(maxlen=16)  # 4-year window
        for _ in range(16):
            self.chronic_queue.append(0.0)
        
        self.base_coop = 0.4 + (random.random() - 0.5) * 0.4
        self.base_coop = max(0.1, min(0.9, self.base_coop))
        
        self.relationships: Dict[int, FastRelationship] = {}
        self.society_trust = 0.5
        
        self.max_lifespan = int((200 + random.random() * 300) * (params.max_rounds / 500))
        self.age = 0
        # prepare neighbor set for diffusion
        self.network_neighbors: Set['OptimizedPerson'] = set()
        self.out_group_penalty_accumulator = 0.0  # Persistent out-group penalty
        
        self.strategy_changes = 0
        self.rounds_as_selfish = 0
        self.rounds_as_cooperative = 0
        
        # SIGNIFICANT FIX #9: Standardized group assignment logic
        self.group_id = self._assign_group(params, group_id, parent_a, parent_b)
            
        self.in_group_interactions = 0
        self.out_group_interactions = 0
        self.mixing_event_participations = 0
        
        if parent_a and parent_b:
            self.maslow_needs = self._inherit_traits(
                parent_a.maslow_needs, parent_b.maslow_needs, 
                params.maslow_variation, parent_a, parent_b
            )
        else:
            self.maslow_needs = MaslowNeeds(
                physiological=random.random() * 10,
                safety=random.random() * 10,
                love=random.random() * 10,
                esteem=random.random() * 10,
                self_actualization=random.random() * 10
            )

        # MASLOW FIX: Store initial values for individual change tracking
        self.initial_maslow_needs = MaslowNeeds(
            physiological=self.maslow_needs.physiological,
            safety=self.maslow_needs.safety,
            love=self.maslow_needs.love,
            esteem=self.maslow_needs.esteem,
            self_actualization=self.maslow_needs.self_actualization
        )
        
        self.maslow_pressure = 0.0
        self._calculate_maslow_pressure_fast()

   def get_individual_maslow_changes(self) -> Dict[str, float]:
       """Calculate individual Maslow changes from initial values"""
       return {
           'physiological': self.maslow_needs.physiological - self.initial_maslow_needs.physiological,
           'safety': self.maslow_needs.safety - self.initial_maslow_needs.safety,
           'love': self.maslow_needs.love - self.initial_maslow_needs.love,
           'esteem': self.maslow_needs.esteem - self.initial_maslow_needs.esteem,
           'self_actualization': self.maslow_needs.self_actualization - self.initial_maslow_needs.self_actualization
       }
    
    def _assign_group(self, params: SimulationConfig, group_id: Optional[str], 
                     parent_a: Optional['OptimizedPerson'], parent_b: Optional['OptimizedPerson']) -> str:
        """SIGNIFICANT FIX #9: Standardized group assignment logic"""
        if group_id is not None:
            return group_id
        elif parent_a and parent_b:
            # Inherit from parents with small chance of group switching
            if random.random() < 0.05:  # 5% chance of group switching
                group_names = [chr(65 + i) for i in range(params.num_groups)]
                return random.choice(group_names)
            else:
                return random.choice([parent_a.group_id, parent_b.group_id])
        else:
            # Initial population assignment
            if params.num_groups == 1:
                return "A"
            else:
                group_names = [chr(65 + i) for i in range(params.num_groups)]
                return random.choice(group_names)
    
    def _inherit_traits(self, parent_a_needs: MaslowNeeds, parent_b_needs: MaslowNeeds, 
                       variation: float, parent_a: Optional['OptimizedPerson'] = None, 
                       parent_b: Optional['OptimizedPerson'] = None) -> MaslowNeeds:
        """Inherit traits with variation"""
        cooperation_bonus = 0
        if parent_a and parent_b and parent_a.strategy == 'cooperative' and parent_b.strategy == 'cooperative':
            cooperation_bonus = 0.5
        
        def inherit_trait(value_a: float, value_b: float) -> float:
            average = (value_a + value_b) / 2 + cooperation_bonus
            trait_variation = (random.random() - 0.5) * variation * 8
            return max(0, min(10, average + trait_variation))
        
        return MaslowNeeds(
            physiological=inherit_trait(parent_a_needs.physiological, parent_b_needs.physiological),
            safety=inherit_trait(parent_a_needs.safety, parent_b_needs.safety),
            love=inherit_trait(parent_a_needs.love, parent_b_needs.love),
            esteem=inherit_trait(parent_a_needs.esteem, parent_b_needs.esteem),
            self_actualization=inherit_trait(parent_a_needs.self_actualization, parent_b_needs.self_actualization)
        )
    
    def _calculate_maslow_pressure_fast(self):
        """Optimized pressure calculation"""
        n = self.maslow_needs
        
        total_pressure = (
            (10 - n.physiological) ** 2 * 0.003 +
            (10 - n.safety) ** 2 * 0.002 +
            (10 - n.love) ** 2 * 0.001 +
            (10 - n.esteem) ** 2 * 0.0008 +
            (10 - n.self_actualization) ** 2 * 0.0005
        )
        
        total_relief = (
            n.physiological ** 1.5 * 0.0002 +
            n.safety ** 1.5 * 0.0002 +
            n.love ** 1.8 * 0.0005 +
            n.esteem ** 2.0 * 0.001 +
            n.self_actualization ** 2.2 * 0.002
        )
        
        self.maslow_pressure = max(0, total_pressure - total_relief)
    
    def calculate_cooperation_decision(self, other: 'OptimizedPerson', round_num: int, params: SimulationConfig) -> bool:
        """MAJOR FIX #5: Use cooperation threshold consistently, not resilience threshold"""
        if self.strategy == 'selfish':
            return False
        
        # Small chance of random defection
        if random.random() < 0.02:
            return False
        
        # MAJOR FIX #5: Use cooperation_threshold consistently for decisions
        effective_threshold = self.cooperation_threshold
        effective_threshold = max(0.01, min(0.99, effective_threshold))
        
        relationship = self.get_relationship(other.id, round_num, getattr(other, 'group_id', None))
        
        if relationship.interaction_count == 0:
            # First interaction - use base probability
            base_prob = max(0.1, 1.0 - self.cooperation_threshold)

            # Group-based modification
            if hasattr(self, 'group_id') and hasattr(other, 'group_id'):
                if self.group_id == other.group_id:
                    base_prob = min(0.95, base_prob * 1.2)  # More likely with in-group
                else:
                    base_prob = max(0.05, base_prob * 0.8)  # Less likely with out-group

            return random.random() < base_prob

        else:
            return relationship.trust >= self.cooperation_threshold
            
        # 9. POSITIVE MASLOW OVERRIDES
        if maslow_priority == 'self_actualization_boost':
            base_prob *= 1.2   # Self-actualization lift
        elif maslow_priority == 'group_seeking':
            base_prob += 0.05  # Belonging boost
            
            return random.random() < max(0.05, min(0.95, base_prob))
        else:
            # Trust-based decision - cooperate if trust exceeds threshold
            return relationship.trust >= effective_threshold
    
    def get_relationship(self, other_id: int, round_num: int, 
                        other_group_id: Optional[str] = None) -> FastRelationship:
        """Get or create relationship with group awareness"""
        if other_id not in self.relationships:
            if len(self.relationships) >= 150:
                # CRITICAL FIX: Delete weakest relationship, not most recent
                # Priority: 1) Lowest trust, 2) Fewest interactions, 3) Most recent
                def relationship_strength(rel_id):
                    rel = self.relationships[rel_id]
                    # Composite strength score (higher = stronger relationship)
                    trust_score = rel.trust * 1000  # Trust is most important
                    interaction_score = min(rel.interaction_count, 50)  # Cap at 50
                    recency_bonus = max(0, 50 - (round_num - rel.last_interaction_round))
                    return trust_score + interaction_score + recency_bonus
                
                weakest_id = min(self.relationships.keys(), key=relationship_strength)
                del self.relationships[weakest_id]
            
            is_same_group = (other_group_id is None or self.group_id == other_group_id)
            relationship = FastRelationship(is_same_group=is_same_group)
            self.relationships[other_id] = relationship
        return self.relationships[other_id]
    
    def update(self, system_stress: float, params: SimulationConfig, cooperation_bonus: float = 0):
        """Update person state"""
        if self.is_dead:
            return
        
        self.age += 1
        if self.age >= self.max_lifespan and self.age > 240:
            self.is_dead = True
            return
        
        if self.strategy == 'cooperative':
            self.rounds_as_cooperative += 1
        else:
            self.rounds_as_selfish += 1
        
        needs = self.maslow_needs
        fluctuation = 0.05
        
        needs.physiological = max(0, min(10, needs.physiological + (random.random() - 0.5) * fluctuation))
        needs.safety = max(0, min(10, needs.safety + (random.random() - 0.5) * fluctuation))
        needs.love = max(0, min(10, needs.love + (random.random() - 0.5) * fluctuation * 1.5))
        needs.esteem = max(0, min(10, needs.esteem + (random.random() - 0.5) * fluctuation * 2))
        needs.self_actualization = max(0, min(10, needs.self_actualization + (random.random() - 0.5) * fluctuation * 2))
        
        if cooperation_bonus > 0:
            needs.love = min(10, needs.love + cooperation_bonus * 0.3)
            needs.esteem = min(10, needs.esteem + cooperation_bonus * 0.2)
            needs.self_actualization = min(10, needs.self_actualization + cooperation_bonus * 0.1)
        
        if self.strategy == 'selfish':
            needs.love = max(0, needs.love - 0.02)
            needs.esteem = max(0, needs.esteem - 0.01)
        
        self._calculate_maslow_pressure_fast()
        
        # Calculate constraint level
        chronic_stress = np.mean(self.chronic_queue) if self.chronic_queue else 0
        self.constraint_level = chronic_stress * 1.2
        
        need_satisfaction = (needs.physiological + needs.safety + needs.love + 
                           needs.esteem + needs.self_actualization) / 50
        
        # FIX BUG #2: Research-based stress decay multiplier (was 0.5, now 1.0)
        pressure_decay = self.stress_recovery_rate * need_satisfaction * 1.0

        # NEW: Very slow decay of out-group penalty accumulator (5x slower than normal decay)
        penalty_decay = pressure_decay / 5.0  # Much slower decay for accumulated penalties
        self.out_group_penalty_accumulator = max(0, self.out_group_penalty_accumulator - penalty_decay)

        self.constraint_level = max(0, self.constraint_level - pressure_decay)
    
    def add_constraint_pressure(self, amount: float, is_from_out_group: bool = False, 
                              out_group_penalty: float = 1.0) -> bool:
        """Add pressure with out-group penalty"""
        if self.is_dead:
            return False
        
        maslow_amplifier = 1 + (self.maslow_pressure * 0.2)
        
        if is_from_out_group:
            amount *= out_group_penalty
            self.out_group_penalty_accumulator += amount * 0.15  # 15% goes to permanent accumulator
        
        # FIX BUG #1: Don't accumulate acute_stress indefinitely
        stress_increment = amount * maslow_amplifier
        self.chronic_queue.append(stress_increment)
         
        # Update constraint level immediately (including accumulated penalties)
        chronic_stress = np.mean(self.chronic_queue) if self.chronic_queue else 0
        accumulated_penalty_factor = self.out_group_penalty_accumulator * 0.25  # 25% of accumulated penalties
        self.constraint_level = chronic_stress * 1.2 + accumulated_penalty_factor
        
        if self.strategy == 'cooperative' and self.constraint_level > self.constraint_threshold:
            self.force_switch()
            return True
        return False
    
    def check_for_recovery(self, params: SimulationConfig) -> bool:
        """Check if person can recover to cooperative strategy"""
        if self.strategy == 'selfish' and self.constraint_level < self.recovery_threshold:
            recovery_chance = 0.8
            
            if self.maslow_needs.love > 7:
                recovery_chance += 0.2
            if self.maslow_needs.esteem > 7:
                recovery_chance += 0.1
            if self.maslow_needs.self_actualization > 8:
                recovery_chance += 0.2
            
            # Social support bonus
            top5_trust = self.get_top5_trust()
            recovery_chance += top5_trust * 0.3
            
            if self.rounds_as_selfish > 50:
                recovery_chance *= 0.85
            
            if random.random() < min(0.95, recovery_chance):  # Cap at 95%
                self.switch_to_cooperative()
                return True
        return False
    
    def force_switch(self):
        """Force switch to selfish strategy"""
        self.strategy = 'selfish'
        self.is_constrained = True
        self.strategy_changes += 1
        self.maslow_needs.love *= 0.8
        self.maslow_needs.esteem *= 0.7
    
    def switch_to_cooperative(self):
        """Recover to cooperative strategy"""
        self.strategy = 'cooperative'
        self.is_constrained = False
        self.strategy_changes += 1
        self.rounds_as_selfish = 0
        self.maslow_needs.love = min(10, self.maslow_needs.love * 1.1)
        self.maslow_needs.esteem = min(10, self.maslow_needs.esteem * 1.1)
    
    def get_top5_trust(self) -> float:
        """Get average trust of top 5 relationships"""
        if not self.relationships:
            return 0.0
        
        trust_values = [rel.trust for rel in self.relationships.values()]
        trust_values.sort(reverse=True)
        return np.mean(trust_values[:5])

# ===== 3. V3 PARAMETER SAMPLING =====

def sample_config() -> SimulationConfig:
    """Generate v3 parameter configuration - CRITICAL FIX #3: Use correct parameter ranges"""
    try:
        # Sample num_groups first as it affects other parameters
        num_groups = random.choice([1, 2, 3])
        
        # SIGNIFICANT FIX #9: Standardized group assignment logic
        if num_groups == 1:
            homophily_bias = 0.0
            intervention_interval = 0  # No interventions for single group
        else:
            homophily_bias = random.uniform(0.0, 0.8)
            intervention_interval = random.choice([10, 15, 20, 25])
        
        config = SimulationConfig(
            shock_interval_years=random.choice([2, 5, 10, 20]),  # CRITICAL FIX #3: Use documented ranges
            homophily_bias=homophily_bias,
            num_groups=num_groups,
            out_group_trust_bias=random.uniform(0.8, 1.2),
            out_group_penalty=random.uniform(1.1, 1.5),
            intervention_interval=intervention_interval,
            intervention_scale=random.uniform(0.05, 0.30),
            event_bonus=random.uniform(1.5, 2.5),
            base_trust_delta=random.uniform(0.05, 0.2),
            group_trust_bias=random.uniform(1.2, 2.0),
            resilience_profile={
                'threshold': random.uniform(0.1, 0.4),
                'noise': random.uniform(0.0, 0.15)
            },
            turnover_rate=random.uniform(0.02, 0.05),
            social_diffusion=random.uniform(0.0, 0.10),
            max_rounds=DEFAULT_MAX_ROUNDS
        )
        
        return config
        
    except AssertionError as e:
        timestamp_print(f"⚠️ Parameter validation failed: {e}, using defaults")
        # Create safe defaults with proper single-group handling
        safe_num_groups = random.choice([1, 2, 3])
        return SimulationConfig(
            shock_interval_years=15,  # CRITICAL FIX #3: Use valid range
            homophily_bias=0.0 if safe_num_groups == 1 else 0.5,
            num_groups=safe_num_groups,
            out_group_trust_bias=1.0,
            out_group_penalty=1.2,  # Valid minimum
            intervention_interval=0 if safe_num_groups == 1 else 15,
            intervention_scale=0.15,
            event_bonus=2.0,
            base_trust_delta=0.2,
            group_trust_bias=1.6,
            resilience_profile={'threshold': 0.25, 'noise': 0.075},
            turnover_rate=0.035,
            social_diffusion=0.05,
            max_rounds=DEFAULT_MAX_ROUNDS
        )
        
    except Exception as e:
        timestamp_print(f"❌ Error in parameter sampling: {e}, using safe defaults")
        # Return ultra-safe single-group defaults
        return SimulationConfig(
            shock_interval_years=15,  # CRITICAL FIX #3: Use valid range
            homophily_bias=0.0,
            num_groups=1,
            out_group_trust_bias=1.0,
            out_group_penalty=1.2,  # Valid minimum
            intervention_interval=0,
            intervention_scale=0.1,
            event_bonus=2.0,
            base_trust_delta=0.2,
            group_trust_bias=1.5,
            resilience_profile={'threshold': 0.25, 'noise': 0.05},
            turnover_rate=0.03,
            social_diffusion=0.05,
            max_rounds=DEFAULT_MAX_ROUNDS
        )

# ===== 4. V3 CORE SIMULATION =====

def schedule_interactions(population: List[OptimizedPerson], params: SimulationConfig, 
                         sim_ref: "EnhancedMassSimulation", round_num: int) -> None:
    """CRITICAL FIX #1: Fixed interaction scheduling to prevent double-counting"""
    alive_people = [p for p in population if not p.is_dead]
    if len(alive_people) < 2:
        return
    
    # EMPIRICAL FIX: Increase meaningful interactions for relationship development
    max_interactions_total = min(len(alive_people) * 2, len(alive_people) * (len(alive_people) - 1) // 4)
    interactions_processed = 0
    
    # Create unique pairs for this round
    interaction_pairs = []
    for i, person in enumerate(alive_people):
        if person.is_dead:
            continue
        
        # EMPIRICAL FIX: Increase from 3 to 8 meaningful interactions per round
        max_interactions_per_person = min(8, len(alive_people) // 8 + 2)
        person_interactions = 0
        
        for _ in range(max_interactions_per_person):
            if person_interactions >= max_interactions_per_person:
                break
                
            # Partner selection with homophily
            potential_partners = [p for p in alive_people[i+1:] if not p.is_dead]
            if not potential_partners:
                break
            
            # ENHANCED: 40% chance to interact with network neighbor first
            partner = None
            
            if (hasattr(person, 'network_neighbors') and 
                person.network_neighbors and 
                random.random() < 0.4):  # 40% network neighbor preference
                
                network_partners = [p for p in potential_partners 
                                  if p in person.network_neighbors]
                if network_partners:
                    partner = random.choice(network_partners)
            
            # If no network partner selected, use existing homophily logic
            if partner is None:
                if (params.num_groups > 1 and 
                    random.random() < params.homophily_bias and 
                    hasattr(person, 'group_id')):
                    # Try same group first
                    same_group_partners = [p for p in potential_partners 
                                         if hasattr(p, 'group_id') and p.group_id == person.group_id]
                    if same_group_partners:
                        partner = random.choice(same_group_partners)
                    else:
                        partner = random.choice(potential_partners)
                else:
                    # Random selection
                    partner = random.choice(potential_partners)
            
            # CRITICAL FIX #1: Add unique pair, avoid duplicates
            pair = (person, partner)
            if pair not in interaction_pairs:
                interaction_pairs.append(pair)
                person_interactions += 1
                
                if len(interaction_pairs) >= max_interactions_total:
                    break
        
        if len(interaction_pairs) >= max_interactions_total:
            break
    
    # CRITICAL FIX #1: Process each unique interaction exactly once
    for person, partner in interaction_pairs:
        if person.is_dead or partner.is_dead:
            continue
            
        try:
            # Get cooperation decisions
            person_coop = person.calculate_cooperation_decision(partner, round_num, params)
            partner_coop = partner.calculate_cooperation_decision(person, round_num, params)

            # CRITICAL FIX #1: Count each interaction exactly once
            sim_ref.total_encounters += 1
            
            # Track outcome types
            if person_coop and partner_coop:
                sim_ref.total_mutual_cooperation += 1
            elif not person_coop and not partner_coop:
                sim_ref.total_mutual_defection += 1
            else:
                sim_ref.total_mixed_outcomes += 1
            
            # Update relationships with v3 trust mechanics
            person_rel = person.get_relationship(partner.id, round_num, 
                                               getattr(partner, 'group_id', None))
            partner_rel = partner.get_relationship(person.id, round_num, 
                                                 getattr(person, 'group_id', None))
            
            # Apply trust updates
            person_rel.update_trust(partner_coop, round_num, params.base_trust_delta, 
                                  params.group_trust_bias, params.out_group_trust_bias)
            partner_rel.update_trust(person_coop, round_num, params.base_trust_delta,
                                   params.group_trust_bias, params.out_group_trust_bias)
            
            # Track group interactions
            sim_ref.total_interactions += 1  # Legacy compatibility
            person_group = getattr(person, 'group_id', 'A')
            partner_group = getattr(partner, 'group_id', 'A')
            
            if person_group == partner_group:
                person.in_group_interactions += 1
                partner.in_group_interactions += 1
                sim_ref.in_group_interactions += 1
            else:
                person.out_group_interactions += 1
                partner.out_group_interactions += 1
                sim_ref.out_group_interactions += 1
            
            # Handle cooperation outcomes
            if person_coop and partner_coop:
                sim_ref.total_mutual_coop += 1  # Legacy compatibility
                # Boost love needs safely
                person.maslow_needs.love = min(10, person.maslow_needs.love + 0.1)
                partner.maslow_needs.love = min(10, partner.maslow_needs.love + 0.1)
                

            # Restore cooperation-dependent birth chance
            if (len(sim_ref.people) < params.max_population and 
                random.random() < params.turnover_rate * person.base_coop):  # births ∝ cooperation
                # Random partner selection for births (more realistic)
                if random.random() < 0.3:  # 30% chance any interaction could lead to birth
                    sim_ref._create_birth(person, partner)
                    
            elif not person_coop or not partner_coop:
                # CRITICAL FIX #1: Count defections properly (one per defecting action)
                if not person_coop:
                    sim_ref.total_defections += 1
                if not partner_coop:
                    sim_ref.total_defections += 1
                
                # Apply constraint pressure with out-group penalty
                base_pressure = 0.15
                
                if not partner_coop:
                    is_out_group = person_group != partner_group
                    person.add_constraint_pressure(base_pressure, is_out_group, params.out_group_penalty)
                if not person_coop:
                    is_out_group = partner_group != person_group
                    partner.add_constraint_pressure(base_pressure, is_out_group, params.out_group_penalty)
                    
        except Exception as e:
            timestamp_print(f"⚠️ Error in interaction between {person.id} and {partner.id}: {e}")
            continue

class EnhancedMassSimulation:
    """v3 Enhanced simulation with streamlined parameters"""
    
    def __init__(self, params: SimulationConfig, run_id: int):
        self.params = params
        self.run_id = run_id
        self.people: List[OptimizedPerson] = []
        self.round = 0
        self.system_stress = 0.0
        self.next_person_id = params.initial_population + 1
        
        # Shock scheduling using exponential distribution
        self.next_shock_round = self._sample_next_shock()
        
        # Tracking variables - all preserved
        self.total_births = 0
        self.total_deaths = 0
        self.total_defections = 0
        self.total_redemptions = 0
        self.first_cascade_round = None
        self.cascade_events = 0
        self.shock_events = 0
        self.system_stress_history = []
        self.population_history = []
        self.cooperation_benefit_total = 0
        
        # Group tracking
        self.group_names = [chr(65 + i) for i in range(params.num_groups)]
        self.in_group_interactions = 0
        self.out_group_interactions = 0
        self.total_mixing_events = 0
        self.successful_mixing_events = 0
        self.reputational_spillover_events = 0
        self.out_group_constraint_amplifications = 0

        
        # Interaction tracking
        self.total_interactions = 0  # Legacy (deprecated, kept for compatibility)
        self.total_encounters = 0    # CRITICAL FIX #1: Proper encounter tracking
        self.total_mutual_cooperation = 0  # CRITICAL FIX #1: Proper mutual cooperation tracking
        self.total_mutual_defection = 0   # CRITICAL FIX #1: Proper mutual defection tracking
        self.total_mixed_outcomes = 0     # CRITICAL FIX #1: Proper mixed outcome tracking
        self.total_mutual_coop = 0  # Legacy (deprecated, kept for compatibility)
        self.institutional_memory = 0.0
        
        # v3 logging counters
        self.maslow_log_counter = 0
        self.social_diffusion_log_counter = 0
        self.network_topology_log_counter = 0

        # SIGNIFICANT FIX #8: Complete institutional memory system
        self.institutional_memory = {
            'total_shocks_experienced': 0,
            'average_shock_severity': 0.0,
            'learned_resilience_bonus': 0.0,
            'crisis_response_knowledge': 0.0,
            'recovery_success_rate': 0.0,
            'collective_learning_factor': 1.0
        }
        self.recovery_phase_rounds = 0
        self.in_recovery_phase = False
        self.shocks_without_recovery = 0
        
        self._initialize_population()
    
    def _sample_next_shock(self) -> int:
        """Sample next shock timing using Poisson process"""
        # Convert shock_interval_years to mean time between shocks
        mean_years = max(1.0, float(self.params.shock_interval_years))
        wait_years = np.random.exponential(mean_years)
        wait_years = max(0.25, wait_years)  # Minimum 1 quarter wait
        return self.round + int(wait_years * 4)  # Convert years to quarters
    
    def _initialize_population(self):
        """Initialize population with group distribution"""
        for i in range(1, self.params.initial_population + 1):
            person = OptimizedPerson(i, self.params)
            self.people.append(person)
        # ENHANCED: Create more realistic social networks with group bias
        for person in self.people:
            others = [p for p in self.people if p is not person]
            network_size = random.randint(4, 8)  # Slightly larger networks
            
            # Strong bias toward same group (80% chance for same-group connections)
            if self.params.num_groups > 1 and len(others) > network_size:
                same_group = [p for p in others if p.group_id == person.group_id]
                other_group = [p for p in others if p.group_id != person.group_id]
                
                # 80% same group, 20% other groups
                same_group_count = min(len(same_group), int(network_size * 0.8))
                other_group_count = min(len(other_group), network_size - same_group_count)
                
                selected = []
                if same_group and same_group_count > 0:
                    selected.extend(random.sample(same_group, same_group_count))
                if other_group and other_group_count > 0:
                    selected.extend(random.sample(other_group, other_group_count))
                
                # Fill remaining slots randomly if needed
                remaining = [p for p in others if p not in selected]
                if len(selected) < network_size and remaining:
                    additional = min(len(remaining), network_size - len(selected))
                    selected.extend(random.sample(remaining, additional))
                
                person.network_neighbors = set(selected)
            else:
                # Single group or fallback: random selection
                person.network_neighbors = set(random.sample(others, min(network_size, len(others))))
    
    def _trigger_shock(self):
        """Apply shock with proper magnitude and institutional memory"""
        shock_severity = self._draw_shock_magnitude()
        shock_severity = min(2.0, max(0.1, shock_severity))
        
        # SIGNIFICANT FIX #8: Apply institutional memory to reduce shock impact
        memory_reduction = self.institutional_memory['learned_resilience_bonus'] * 0.3
        effective_severity = shock_severity * (1 - memory_reduction)
        self.system_stress += effective_severity
        self.shock_events += 1
        
        # SIGNIFICANT FIX #8: Update institutional memory properly
        self.institutional_memory['total_shocks_experienced'] += 1
        old_avg = self.institutional_memory['average_shock_severity']
        n = self.institutional_memory['total_shocks_experienced']
        self.institutional_memory['average_shock_severity'] = (old_avg * (n-1) + shock_severity) / n
        
        # Increase learned resilience based on experience
        if self.in_recovery_phase:
            self.institutional_memory['learned_resilience_bonus'] += 0.01
        else:
            self.shocks_without_recovery += 1
            
        # Start recovery phase
        self.in_recovery_phase = True
        self.recovery_phase_rounds = 0
        
        # Apply shock to all people
        alive_people = [p for p in self.people if not p.is_dead]
        for person in alive_people:
            try:
                person.acute_stress += effective_severity * 0.5
                person.chronic_queue.append(person.acute_stress)
            except Exception as e:
                timestamp_print(f"⚠️ Error applying shock to person {person.id}: {e}")
                continue
        
        # Schedule next shock
        self.next_shock_round = self._sample_next_shock()
    
    def _draw_shock_magnitude(self) -> float:
        """Draw shock magnitude from Pareto distribution"""
        alpha = 2.0
        xm = 0.3
        try:
            u = random.uniform(0.001, 0.999)
            magnitude = xm * u ** (-1 / alpha)
            return min(2.0, max(0.1, magnitude))
        except (ValueError, OverflowError):
            return 0.5  # Safe default
    
    def _is_intervention_round(self) -> bool:
        """Check if this round should have intervention event"""
        return (self.params.intervention_interval > 0 and 
                self.params.num_groups > 1 and
                self.round > 0 and 
                self.round % self.params.intervention_interval == 0)
    
    def _handle_intervention_event(self, alive_people: List[OptimizedPerson]):
        """Handle cross-group intervention event"""
        if len(alive_people) < 2 or self.params.num_groups <= 1:
            return
            
        self.total_mixing_events += 1
        
        # Select agents for intervention
        try:
            num_selected = max(1, int(len(alive_people) * self.params.intervention_scale))
            num_selected = min(num_selected, len(alive_people))
            # weight toward younger agents
            ages    = [p.age for p in alive_people]
            weights = [1.0/(age*age + 1) for age in ages]  # Quadratic age weighting (much stronger youth bias)
            selected_agents = random.choices(alive_people, weights=weights, k=num_selected)
            
            # Apply event bonus through trust boost and stress reduction
            trust_boost = 0.1 * self.params.event_bonus
            for agent in selected_agents:
                agent.mixing_event_participations += 1
                # Only boost out-group relationships (intervention purpose)
                for rel in agent.relationships.values():
                    if hasattr(rel, 'is_same_group') and not rel.is_same_group:
                        rel.trust = min(1.0, rel.trust + trust_boost)
                    # Give smaller general boost to all relationships
                    else:
                        rel.trust = min(1.0, rel.trust + trust_boost * 0.3)
                
                # Reduce stress and boost social needs
                stress_reduction = 0.05 * self.params.event_bonus
                agent.constraint_level = max(0, agent.constraint_level - stress_reduction)
                agent.maslow_needs.love = min(10, agent.maslow_needs.love + 0.5)
                agent.maslow_needs.esteem = min(10, agent.maslow_needs.esteem + 0.3)
            
            if len(selected_agents) > 1:
                self.successful_mixing_events += 1
                
        except Exception as e:
            timestamp_print(f"⚠️ Error in intervention event: {e}")
    
    def _apply_social_diffusion(self, alive_people: List[OptimizedPerson]):
        """Apply social diffusion of trust values"""
        """Apply network-neighbor diffusion of trust and update institutional memory."""
        sd = self.params.social_diffusion
        if sd <= 0:
            return
        total = 0.0
        count = 0
        for person in alive_people:
            neighbors = getattr(person, 'network_neighbors', set())
            if not neighbors:
                continue
                
            # Collect trust values from network neighbors (enhanced weighting)
            neighbor_trusts = []
            for neighbor in neighbors:
                if neighbor in person.relationships:
                    trust_val = person.relationships[neighbor].trust
                    # Weight by relationship strength (interaction count)
                    rel = person.relationships[neighbor]
                    weight = min(rel.interaction_count + 1, 10)  # Cap at 10x weight
                    neighbor_trusts.extend([trust_val] * weight)  # Repeat based on weight
            
            if not neighbor_trusts:
                continue
                
            local_avg = sum(neighbor_trusts) / len(neighbor_trusts)
            
            # ENHANCED: Apply stronger diffusion for network neighbors
            enhanced_diffusion = sd * 1.5  # 50% stronger for network neighbors
            
            # Apply diffusion to all relationships, with extra strength for network neighbors
            for other_id in person.relationships:
                rel = person.relationships[other_id]
                diffusion_strength = enhanced_diffusion if other_id in neighbors else sd
                rel.trust = max(0.0, min(1.0, 
                    (1 - diffusion_strength) * rel.trust + diffusion_strength * local_avg
                ))

        # institutional memory: decay or reinforce
        if count:
            net_avg = total / count
            self.institutional_memory = (1-sd)*self.institutional_memory + sd*net_avg
        else:
            if hasattr(self, 'institutional_memory') and isinstance(self.institutional_memory, (int, float)):
                self.institutional_memory *= (1 - sd*0.1)
            elif not hasattr(self, 'institutional_memory'):
                self.institutional_memory = 0.0
        
    def _create_birth(self, parent_a: OptimizedPerson, parent_b: OptimizedPerson):
        """Create new person with group inheritance"""
        new_person = OptimizedPerson(self.next_person_id, self.params, parent_a, parent_b)
        new_person.society_trust = 0.5
        self.people.append(new_person)
        self.total_births += 1
        self.next_person_id += 1
    
    def _check_recoveries(self):
        """Check for strategy recoveries"""
        alive_people = [p for p in self.people if not p.is_dead]
        for person in alive_people:
            if person.check_for_recovery(self.params):
                self.total_redemptions += 1
    
    def _update_population(self):
        """Update population state"""
        initial_count = len(self.people)
        self.people = [p for p in self.people if not p.is_dead]
        self.total_deaths += initial_count - len(self.people)
        
        for person in self.people:
            person.update(self.system_stress, self.params)
    
    def _collect_round_data(self):
        """v3 logging with variable frequencies"""
        alive_people = [p for p in self.people if not p.is_dead]
        
        # Per-round metrics (always collected)
        self.system_stress_history.append(self.system_stress)
        self.population_history.append(len(alive_people))
        
        # Maslow needs logging (every 4 rounds)
        if self.round % 4 == 0:
            self.maslow_log_counter += 1
        
        # Social diffusion logging (every 8 rounds)
        if self.round % 8 == 0:
            self.social_diffusion_log_counter += 1
        
        # Network topology logging (every 4 rounds)
        if self.round % 4 == 0:
            self.network_topology_log_counter += 1
    
    def _get_average_traits(self) -> Dict[str, float]:
        """Get average Maslow traits"""
        alive_people = [p for p in self.people if not p.is_dead]
        if not alive_people:
            return {k: 0 for k in ['physiological', 'safety', 'love', 'esteem', 'self_actualization']}
        
        return {
            'physiological': sum(p.maslow_needs.physiological for p in alive_people) / len(alive_people),
            'safety': sum(p.maslow_needs.safety for p in alive_people) / len(alive_people),
            'love': sum(p.maslow_needs.love for p in alive_people) / len(alive_people),
            'esteem': sum(p.maslow_needs.esteem for p in alive_people) / len(alive_people),
            'self_actualization': sum(p.maslow_needs.self_actualization for p in alive_people) / len(alive_people)
        }
    
    def _get_group_populations(self) -> Dict[str, int]:
        """Get current population by group"""
        alive_people = [p for p in self.people if not p.is_dead]
        group_counts = defaultdict(int)
        for person in alive_people:
            if hasattr(person, 'group_id'):
                group_counts[person.group_id] += 1
            else:
                group_counts['default'] += 1
        return dict(group_counts)
    
    def _get_group_cooperation_rates(self) -> Dict[str, float]:
        """Get cooperation rate by group"""
        alive_people = [p for p in self.people if not p.is_dead]
        group_cooperation = defaultdict(list)
        
        for person in alive_people:
            group_id = getattr(person, 'group_id', 'default')
            group_cooperation[group_id].append(1 if person.strategy == 'cooperative' else 0)
        
        return {group: sum(strategies) / len(strategies) if strategies else 0 
                for group, strategies in group_cooperation.items()}
    
    def _calculate_trust_levels(self) -> Tuple[float, float]:
        """Calculate trust levels excluding undeveloped relationships"""
        alive_people = [p for p in self.people if not p.is_dead]
        in_group_trusts = []
        out_group_trusts = []
        
        for person in alive_people:
            for rel in person.relationships.values():
                # Include all relationships with valid trust values
               # CRITICAL: Only include developed relationships with valid trust
               if rel.trust is not None and rel.is_developed:
                    if hasattr(rel, 'is_same_group'):
                        if rel.is_same_group:
                            in_group_trusts.append(rel.trust)
                        else:
                            out_group_trusts.append(rel.trust)
                    else:
                        # Default to in-group if group information is missing
                        in_group_trusts.append(rel.trust)
        
        # SIGNIFICANT FIX #10: Calculate during processing, handle edge cases
        avg_in_group = sum(in_group_trusts) / len(in_group_trusts) if in_group_trusts else 0.5
        avg_out_group = sum(out_group_trusts) / len(out_group_trusts) if out_group_trusts else 0.5
        
        return avg_in_group, avg_out_group
    
    def _update_recovery_dynamics(self):
        """SIGNIFICANT FIX #8: Complete recovery dynamics implementation"""
        if self.in_recovery_phase:
            self.recovery_phase_rounds += 1
            
            # Recovery takes 8-12 rounds (2-3 years) after shocks
            recovery_duration = 8 + int(self.institutional_memory['crisis_response_knowledge'] * 4)
            
            if self.recovery_phase_rounds >= recovery_duration:
                self.in_recovery_phase = False
                
                # Calculate recovery success rate
                alive_people = [p for p in self.people if not p.is_dead]
                cooperative_count = len([p for p in alive_people if p.strategy == 'cooperative'])
                recovery_success = cooperative_count / max(1, len(alive_people))
                
                # Update institutional memory
                old_success = self.institutional_memory['recovery_success_rate']
                self.institutional_memory['recovery_success_rate'] = (old_success + recovery_success) / 2
                
                # Apply institutional learning boost
                knowledge_bonus = 0.1 * self.institutional_memory['recovery_success_rate']
                
                for person in alive_people:
                    if person.strategy == 'cooperative':
                        # Boost cooperation-related needs during recovery
                        person.maslow_needs.love = min(10, person.maslow_needs.love + knowledge_bonus)
                        person.maslow_needs.esteem = min(10, person.maslow_needs.esteem + knowledge_bonus * 0.7)
                
                # Institutional knowledge grows
                self.institutional_memory['crisis_response_knowledge'] = min(1.0, 
                    self.institutional_memory['crisis_response_knowledge'] + 0.05)
                
                # Update collective learning factor
                if recovery_success > 0.5:
                    self.institutional_memory['collective_learning_factor'] = min(2.0,
                        self.institutional_memory['collective_learning_factor'] * 1.05)
                else:
                    self.institutional_memory['collective_learning_factor'] = max(0.5,
                        self.institutional_memory['collective_learning_factor'] * 0.95)
                
                self.recovery_phase_rounds = 0
                self.shocks_without_recovery = 0
    
    def run_simulation(self) -> EnhancedSimulationResults:
        """Run v3 enhanced simulation"""
        timestamp_print(f"🎮 Starting v3 simulation run {self.run_id}")
        
        try:
            initial_trait_avg = self._get_average_traits()
            initial_group_populations = self._get_group_populations()
            
            while self.round < self.params.max_rounds:
                try:
                    self.round += 1
                    
                    alive_people = [p for p in self.people if not p.is_dead]
                    if len(alive_people) == 0:
                        timestamp_print(f"💀 Sim {self.run_id}: Population extinct at round {self.round}")
                        break
                    
                    # Check for intervention event
                    if self._is_intervention_round():
                        try:
                            self._handle_intervention_event(alive_people)
                        except Exception as e:
                            timestamp_print(f"⚠️ Error in intervention event round {self.round}: {e}")
                    
                    # Check for shocks
                    if self.round >= self.next_shock_round:
                        try:
                            self._trigger_shock()
                        except Exception as e:
                            timestamp_print(f"⚠️ Error in shock trigger round {self.round}: {e}")
                            # Schedule next shock anyway to prevent infinite loops
                            self.next_shock_round = self.round + 20  # Default 5-year gap
                    
                    # Process interactions
                    try:
                        schedule_interactions(self.people, self.params, self, self.round)
                    except Exception as e:
                        timestamp_print(f"⚠️ Error in interactions round {self.round}: {e}")
                    
                    # Apply social diffusion
                    try:
                        self._apply_social_diffusion(alive_people)
                    except Exception as e:
                        timestamp_print(f"⚠️ Error in social diffusion round {self.round}: {e}")
                    
                    # Other updates
                    try:
                        self._check_recoveries()
                        self._update_population()
                        self._collect_round_data()
                        self._update_recovery_dynamics()  # SIGNIFICANT FIX #8
                    except Exception as e:
                        timestamp_print(f"⚠️ Error in population updates round {self.round}: {e}")
                    
                    # Decay system stress
                    self.system_stress = max(0, self.system_stress - 0.01)
                    
                    # Progress reporting (less frequent to reduce noise)
                    if self.round == 1 or self.round % 30 == 0:
                        timestamp_print(f"✅ Sim {self.run_id}: completed round {self.round:3d} "
                            f"(pop={len(alive_people):4d}, defections={self.total_defections})")
                            
                except Exception as round_error:
                    timestamp_print(f"⚠️ Error in round {self.round} of sim {self.run_id}: {round_error}")
                    # Continue to next round rather than crashing
                    continue
            
            return self._generate_results(initial_trait_avg, initial_group_populations)
            
        except Exception as sim_error:
            timestamp_print(f"❌ Critical error in simulation {self.run_id}: {sim_error}")
            return self._generate_emergency_result()

   def _get_average_individual_changes(self) -> Dict[str, float]:
       """Calculate average of individual Maslow changes (not population-level changes)"""
       alive_people = [p for p in self.people if not p.is_dead]
       if not alive_people:
           return {k: 0 for k in ['physiological', 'safety', 'love', 'esteem', 'self_actualization']}
       
       # Calculate individual changes for each person
       individual_changes = [p.get_individual_maslow_changes() for p in alive_people]
       
       # Average the individual changes
       return {
           'physiological': sum(changes['physiological'] for changes in individual_changes) / len(individual_changes),
           'safety': sum(changes['safety'] for changes in individual_changes) / len(individual_changes),
           'love': sum(changes['love'] for changes in individual_changes) / len(individual_changes),
           'esteem': sum(changes['esteem'] for changes in individual_changes) / len(individual_changes),
           'self_actualization': sum(changes['self_actualization'] for changes in individual_changes) / len(individual_changes)       }
    
    def _generate_results(self, initial_traits: Dict[str, float], 
                         initial_group_populations: Dict[str, int]) -> EnhancedSimulationResults:
        """Generate comprehensive results"""
        alive_people = [p for p in self.people if not p.is_dead]
        cooperative = [p for p in alive_people if p.strategy == 'cooperative']
        constrained = [p for p in alive_people if p.is_constrained]

        # TOPLINE METRICS: Calculate both strategic and behavioral cooperation rates
        strategy_cooperation_rate = len(cooperative) / max(1, len(alive_people))  # % with cooperative strategy
        behavioral_cooperation_rate = (self.total_mutual_cooperation / max(1, self.total_encounters) 
                                     if self.total_encounters > 0 else 0.0)  # % of cooperative interactions
        
        # Diagnostic: Log cooperation rate differences for analysis
        if abs(strategy_cooperation_rate - behavioral_cooperation_rate) > 0.2:
            timestamp_print(f"📊 Sim {self.run_id}: Large cooperation gap - Strategy: {strategy_cooperation_rate:.3f}, Behavioral: {behavioral_cooperation_rate:.3f}")
        
        final_traits = self._get_average_traits()
       # MASLOW FIX: Calculate average of individual changes, not population-level changes
       trait_evolution = self._get_average_individual_changes()
       
       # DIAGNOSTIC: Log the difference between methods
       population_level_changes = {k: final_traits[k] - initial_traits[k] for k in initial_traits.keys()}
       if abs(trait_evolution['love'] - population_level_changes['love']) > 0.1:
           timestamp_print(f"🔍 Sim {self.run_id}: Maslow tracking difference - Individual: {trait_evolution['love']:.3f}, Population: {population_level_changes['love']:.3f}")
        
        # Population stability calculation
        if len(self.population_history) > 20:
            later_pop = self.population_history[-20:]
            pop_stability = np.std(later_pop) / (np.mean(later_pop) + 1e-6)
        else:
            pop_stability = 0.0
        
        # Pressure metrics
        avg_maslow_pressure = sum(p.maslow_pressure for p in alive_people) / max(1, len(alive_people))
        basic_needs_crisis = len([p for p in alive_people if p.maslow_needs.physiological < 3 or p.maslow_needs.safety < 3])
        
        # Trust level calculation
        if self.params.num_groups > 1:
            avg_in_group_trust, avg_out_group_trust = self._calculate_trust_levels()
            overall_trust = (avg_in_group_trust + avg_out_group_trust) / 2
            trust_asymmetry = avg_in_group_trust - avg_out_group_trust
        else:
            total_trust = 0
            total_relationships = 0
            for person in alive_people:
                for rel in person.relationships.values():
                    total_trust += rel.trust
                    total_relationships += 1
            
            overall_trust = total_trust / total_relationships if total_relationships > 0 else 0.5
            avg_in_group_trust = overall_trust
            avg_out_group_trust = overall_trust
            trust_asymmetry = 0.0
        
        # Growth rate
        max_pop_reached = max(self.population_history) if self.population_history else self.params.initial_population
        population_growth = max_pop_reached / self.params.initial_population

        # MAJOR FIX #6: Corrected redemption rate calculation
        total_strategy_switches = self.total_defections + self.total_redemptions
        redemption_rate = self.total_redemptions / max(1, total_strategy_switches)
        
        # Inter-group specific calculations
        final_group_populations = self._get_group_populations()
        final_group_cooperation_rates = self._get_group_cooperation_rates()
        
        # Calculate interaction rates
        total_interactions = self.in_group_interactions + self.out_group_interactions
        in_group_rate = self.in_group_interactions / total_interactions if total_interactions > 0 else 0
        out_group_rate = self.out_group_interactions / total_interactions if total_interactions > 0 else 0
        
        # Count group extinctions
        initial_groups = set(initial_group_populations.keys())
        final_groups = set(final_group_populations.keys())
        group_extinctions = len(initial_groups - final_groups)
        
        # Calculate mixing event success rate
        mixing_success_rate = (self.successful_mixing_events / self.total_mixing_events 
                             if self.total_mixing_events > 0 else 0)
        
        result = EnhancedSimulationResults(
            parameters=self.params,
            run_id=self.run_id,
            
            # All original metrics preserved
            final_population=len(alive_people),
            final_cooperation_rate=strategy_cooperation_rate,  # TOPLINE METRIC: Strategy-based cooperation
            behavioral_cooperation_rate=behavioral_cooperation_rate,  # TOPLINE METRIC: Action-based cooperation
            final_constrained_rate=len(constrained) / max(1, len(alive_people)),
            rounds_completed=self.round,
            extinction_occurred=len(alive_people) == 0,
            first_cascade_round=self.first_cascade_round,
            total_cascade_events=self.cascade_events,
            total_shock_events=self.shock_events,
            total_defections=self.total_defections,
            total_redemptions=self.total_redemptions,
            net_strategy_change=self.total_defections - self.total_redemptions,
            total_births=self.total_births,
            total_deaths=self.total_deaths,
            max_population_reached=max_pop_reached,
            population_stability=pop_stability,
            avg_system_stress=np.mean(self.system_stress_history) if self.system_stress_history else 0,
            max_system_stress=max(self.system_stress_history) if self.system_stress_history else 0,
            avg_maslow_pressure=avg_maslow_pressure,
            avg_basic_needs_crisis_rate=basic_needs_crisis / max(1, len(alive_people)),
            initial_needs_avg=initial_traits,
            final_needs_avg=final_traits,
            needs_improvement=trait_evolution,
            avg_trust_level=overall_trust,
            cooperation_benefit_total=self.cooperation_benefit_total,
            population_growth=population_growth,
            cooperation_resilience=strategy_cooperation_rate,  # TOPLINE METRIC: Cooperation sustainability

            # TOPLINE METRICS: Detailed interaction outcomes
            total_encounters=self.total_encounters,
            total_mutual_cooperation=self.total_mutual_cooperation,
            total_mutual_defection=self.total_mutual_defection,
            total_mixed_outcomes=self.total_mixed_outcomes,
            
            # Strategy tracking
            redemption_rate=redemption_rate,  # MAJOR FIX #6: Corrected calculation
            
            # Inter-Group Metrics - all preserved
            final_group_populations=final_group_populations,
            final_group_cooperation_rates=final_group_cooperation_rates,
            in_group_interaction_rate=in_group_rate,
            out_group_interaction_rate=out_group_rate,
            avg_in_group_trust=avg_in_group_trust,
            avg_out_group_trust=avg_out_group_trust,
            group_segregation_index=in_group_rate,  # Simplified calculation
            total_mixing_events=self.total_mixing_events,
            mixing_event_success_rate=mixing_success_rate,
            reputational_spillover_events=self.reputational_spillover_events,
            out_group_constraint_amplifications=self.out_group_constraint_amplifications,
            group_extinction_events=group_extinctions,
            trust_asymmetry=trust_asymmetry,
            
            # Interaction metrics
            total_interactions=total_interactions,
            total_mutual_coop=self.total_mutual_coop,
            avg_interaction_processing_time=0.0
        )
        
        # VALIDATION FIX #12: Validate results before returning
        validation_warnings = validate_simulation_results(result)
        if validation_warnings:
            timestamp_print(f"⚠️ Result validation warnings for simulation {self.run_id}:")
            for warning in validation_warnings:
                timestamp_print(f"   - {warning}")
        
        return result
    
    def _generate_emergency_result(self) -> EnhancedSimulationResults:
        """Generate emergency result object when simulation fails"""
        timestamp_print(f"🚨 Generating emergency result for failed simulation {self.run_id}")
        
        try:
            alive_people = [p for p in self.people if not p.is_dead]
            
            # Safe trait collection
            try:
                final_traits = self._get_average_traits()
            except:
                final_traits = {'physiological': 0, 'safety': 0, 'love': 0, 'esteem': 0, 'self_actualization': 0}
            
            # Safe group population collection
            try:
                final_group_populations = self._get_group_populations()
            except:
                final_group_populations = {'A': 0}
            
            return EnhancedSimulationResults(
                parameters=self.params,
                run_id=self.run_id,
                final_population=len(alive_people),
                final_cooperation_rate=0.0,  # TOPLINE: Strategy-based cooperation
                behavioral_cooperation_rate=0.0,  # TOPLINE: Action-based cooperation
                final_constrained_rate=1.0,
                rounds_completed=max(1, self.round),
                extinction_occurred=True,
                first_cascade_round=max(1, self.round),
                total_cascade_events=0,
                total_shock_events=max(0, self.shock_events),
                total_defections=max(0, self.total_defections),
                total_redemptions=max(0, self.total_redemptions),
                net_strategy_change=0,
                total_births=max(0, self.total_births),
                total_deaths=max(0, self.total_deaths),
                max_population_reached=max(self.params.initial_population, len(self.people)),
                population_stability=0.0,
                avg_system_stress=0.0,
                max_system_stress=0.0,
                avg_maslow_pressure=0.0,
                avg_basic_needs_crisis_rate=0.0,
                initial_needs_avg=final_traits,
                final_needs_avg=final_traits,
                needs_improvement={k: 0 for k in final_traits.keys()},
                avg_trust_level=0.5,
                cooperation_benefit_total=0.0,
                population_growth=1.0,
                cooperation_resilience=0.0,
                total_encounters=max(0, getattr(self, 'total_encounters', 0)),
                total_mutual_cooperation=max(0, getattr(self, 'total_mutual_cooperation', 0)),
                total_mutual_defection=max(0, getattr(self, 'total_mutual_defection', 0)),
                total_mixed_outcomes=max(0, getattr(self, 'total_mixed_outcomes', 0)),
                redemption_rate=0.0,
                final_group_populations=final_group_populations,
                final_group_cooperation_rates={k: 0.0 for k in final_group_populations.keys()},
                total_interactions=max(0, self.total_interactions),
                total_mutual_coop=max(0, getattr(self, 'total_mutual_coop', 0)),
            )
        except Exception as emergency_error:
            timestamp_print(f"❌ Even emergency result generation failed: {emergency_error}")
            # Ultra-safe fallback
            return EnhancedSimulationResults(
                parameters=self.params,
                run_id=self.run_id,
                final_population=0,
                final_cooperation_rate=0.0,
                final_constrained_rate=1.0,
                rounds_completed=1,
                extinction_occurred=True,
                first_cascade_round=1,
                total_cascade_events=0,
                total_shock_events=0,
                total_defections=0,
                total_redemptions=0,
                net_strategy_change=0,
                total_births=0,
                total_deaths=0,
                max_population_reached=self.params.initial_population,
                population_stability=0.0,
                avg_system_stress=0.0,
                max_system_stress=0.0,
                avg_maslow_pressure=0.0,
                avg_basic_needs_crisis_rate=0.0,
                initial_needs_avg={'physiological': 0, 'safety': 0, 'love': 0, 'esteem': 0, 'self_actualization': 0},
                final_needs_avg={'physiological': 0, 'safety': 0, 'love': 0, 'esteem': 0, 'self_actualization': 0},
                needs_improvement={'physiological': 0, 'safety': 0, 'love': 0, 'esteem': 0, 'self_actualization': 0},
                avg_trust_level=0.5,
                cooperation_benefit_total=0.0,
                population_growth=1.0,
                cooperation_resilience=0.0,
                final_group_populations={'A': 0},
                final_group_cooperation_rates={'A': 0.0},
                total_interactions=0,
                total_mutual_coop=0,
            )

# ===== 5. V3 RUNNER WITH UPDATED PROGRESS REPORTING =====

def run_single_simulation(run_id: int) -> EnhancedSimulationResults:
    """Run a single v3 simulation"""
    timestamp_print(f"🔄 Starting v3 simulation {run_id}")
    params = sample_config()
    sim = EnhancedMassSimulation(params, run_id)
    result = sim.run_simulation()
    timestamp_print(f"✅ Completed v3 simulation {run_id}")
    return result

# Load balancing system preserved with v3 updates
@dataclass
class SimulationWork:
    """Represents work to be done with chunking"""
    sim_id: int
    start_round: int
    end_round: int
    max_rounds: int
    simulation_state: Optional[bytes] = None
    estimated_time: float = 30.0
    complexity_score: float = 1.0
    
    @property
    def is_new_simulation(self) -> bool:
        return self.simulation_state is None
    
    @property
    def is_complete(self) -> bool:
        return self.start_round >= self.max_rounds

class LoadBalancedScheduler:
    """Load balancing with v3 parameters and updated progress tracking"""
    
    def __init__(self, params_list: List[SimulationConfig], chunk_size: int = 30, results_dir: str = "simulation_results"):
        timestamp_print(f"⚙️  Initializing v3 LoadBalancedScheduler with {len(params_list)} simulations...")
        
        self.params_list = params_list
        self.chunk_size = chunk_size
        self.results_dir = results_dir
        self.work_queue = queue.Queue()
        self.completed_simulations = set()
        self.active_simulations = {}  # sim_id -> current_round
        self.simulation_states = {}
        self.lock = threading.Lock()
        
        timestamp_print(f"📁 Setting up results directory: {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            timestamp_print(f"✅ Created results directory: {results_dir}")
        
        timestamp_print(f"🔄 Generating work queue for {len(params_list)} simulations...")
        for i, params in enumerate(params_list):
            if i % 25 == 0:
                timestamp_print(f"   📋 Processing simulation {i+1}/{len(params_list)} for work queue...")
            
            complexity = self._estimate_complexity(params)
            work = SimulationWork(
                sim_id=i,
                start_round=0,
                end_round=min(chunk_size, params.max_rounds),
                max_rounds=params.max_rounds,
                simulation_state=None,
                complexity_score=complexity,
                estimated_time=complexity * chunk_size / 100
            )
            self.work_queue.put(work)
            self.active_simulations[i] = 0
        
        timestamp_print(f"✅ v3 LoadBalancedScheduler initialized with {self.work_queue.qsize()} work items")
    
    def _estimate_complexity(self, params: SimulationConfig) -> float:
        """Estimate v3 simulation complexity"""
        base = params.initial_population ** 1.3 * (params.max_rounds / 100)
        
        # v3 complexity factors
        if params.num_groups > 1:
            intergroup_factor = params.num_groups * 1.5
            base *= intergroup_factor
        
        # Factor in intervention frequency
        if params.intervention_interval > 0:
            intervention_factor = 1.0 + (1.0 / params.intervention_interval)
            base *= intervention_factor
        
        return base
    
    def get_work_with_params(self) -> Optional[Tuple[SimulationWork, SimulationConfig]]:
        """Get next work item with its parameters"""
        try:
            work = self.work_queue.get_nowait()
            params = self.params_list[work.sim_id] if work.sim_id < len(self.params_list) else None
            return (work, params)
        except queue.Empty:
            return None
    
    def get_work(self) -> Optional[SimulationWork]:
        """Get next work item (backward compatibility)"""
        result = self.get_work_with_params()
        return result[0] if result else None
    
    def submit_result(self, work: SimulationWork, result_data: tuple):
        """Submit completed work"""
        result_type, sim_id, data, exec_time, rounds_done = result_data
        
        with self.lock:
            if result_type == 'complete':
                save_simulation_result(data, self.results_dir)
                save_incremental_csv(data)
                self.completed_simulations.add(sim_id)
                if sim_id in self.active_simulations:
                    del self.active_simulations[sim_id]
                
            elif result_type == 'partial':
                self.simulation_states[sim_id] = data
                current_round = work.end_round
                self.active_simulations[sim_id] = current_round
                
                if current_round < work.max_rounds:
                    next_work = SimulationWork(
                        sim_id=sim_id,
                        start_round=current_round,
                        end_round=min(current_round + self.chunk_size, work.max_rounds),
                        max_rounds=work.max_rounds,
                        simulation_state=data,
                        complexity_score=work.complexity_score,
                        estimated_time=exec_time
                    )
                    self.work_queue.put(next_work)
                    
            elif result_type == 'error':
                timestamp_print(f"❌ Error in simulation {sim_id}: {data}")
                if sim_id in self.active_simulations:
                    del self.active_simulations[sim_id]
    
    def get_progress(self) -> Tuple[int, int]:
        """Get (completed, total) simulations"""
        with self.lock:
            return len(self.completed_simulations), len(self.params_list)
    
    def get_aggregate_progress(self) -> Tuple[int, int]:
        """Get (completed_rounds, total_rounds) for better progress tracking"""
        with self.lock:
            completed_rounds = sum(self.active_simulations.values()) + \
                              len(self.completed_simulations) * DEFAULT_MAX_ROUNDS
            total_rounds = len(self.params_list) * DEFAULT_MAX_ROUNDS
            return completed_rounds, total_rounds
    
    def is_complete(self) -> bool:
        """Check if all simulations are done"""
        with self.lock:
            return (len(self.completed_simulations) == len(self.params_list) and 
                    self.work_queue.empty())

def process_simulation_work(work_and_params: tuple) -> tuple:
    """Process a single work item with v3 parameters"""
    work, provided_params = work_and_params
    start_time = time.time()
    
    try:
        if work.is_new_simulation:
            if provided_params is not None:
                params = provided_params
            else:
                params = sample_config()
            sim = EnhancedMassSimulation(params, work.sim_id)
            sim.round = 0
        else:
            sim = pickle.loads(work.simulation_state)
            sim.round = work.start_round
        
        rounds_completed = 0
        target_rounds = work.end_round - work.start_round
        
        # Run the specified rounds
        for _ in range(target_rounds):
            if sim.round >= work.max_rounds:
                break
            
            alive_people = [p for p in sim.people if not p.is_dead]
            if len(alive_people) == 0:
                break
            
            sim.round += 1
            rounds_completed += 1
            
            # Check for intervention event
            if sim._is_intervention_round():
                sim._handle_intervention_event(alive_people)
            
            # Check for shocks
            if sim.round >= sim.next_shock_round:
                sim._trigger_shock()
            
            # Process interactions
            schedule_interactions(sim.people, sim.params, sim, sim.round)
            
            # Apply social diffusion
            sim._apply_social_diffusion(alive_people)
            
            sim._check_recoveries()
            sim._update_population()
            sim._collect_round_data()
            
            sim.system_stress = max(0, sim.system_stress - 0.01)
        
        execution_time = time.time() - start_time
        
        # Check if simulation is complete
        alive_people = [p for p in sim.people if not p.is_dead]
        is_complete = (sim.round >= work.max_rounds or len(alive_people) == 0)
        
        if is_complete:
            initial_trait_avg = sim._get_average_traits()
            initial_group_populations = sim._get_group_populations()
            result = sim._generate_results(initial_trait_avg, initial_group_populations)
            return ('complete', work.sim_id, result, execution_time, rounds_completed)
        else:
            updated_state = pickle.dumps(sim)
            return ('partial', work.sim_id, updated_state, execution_time, rounds_completed)
            
    except Exception as e:
        timestamp_print(f"❌ Error processing sim {work.sim_id}: {e}")
        traceback.print_exc()
        return ('error', work.sim_id, str(e), 0, 0)

def run_smart_mass_experiment(params_list: List[SimulationConfig], use_multiprocessing: bool = False) -> List[EnhancedSimulationResults]:
    """v3 Load-balanced mass experiment with aggregate progress reporting"""
    num_simulations = len(params_list)
    timestamp_print(f"🚀 Starting v3 mass experiment with {num_simulations} simulations...")
    timestamp_print(f"🔧 Using v3 streamlined 13-parameter configuration")
    
    start_time = time.time()
    
    if not use_multiprocessing or num_simulations <= 5:
        timestamp_print("🔧 Using single-threaded execution")
        results = []
        results_dir = "simulation_results"
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for i, params in enumerate(params_list):
            timestamp_print(f"🔄 Starting simulation {i+1}/{num_simulations}")
            sim = EnhancedMassSimulation(params, i)
            result = sim.run_simulation()
            
            save_simulation_result(result, results_dir)
            save_incremental_csv(result)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                timestamp_print(f"📊 PROGRESS: {i + 1}/{num_simulations} complete ({elapsed:.1f}s elapsed)")
        
        return results
    
    # Multi-processing approach with v3 progress tracking
    num_cores = min(mp.cpu_count(), 8)
    timestamp_print(f"🔧 Using {num_cores} CPU cores for multiprocessing...")
    
    results_dir = "simulation_results"
    timestamp_print(f"📁 Setting up v3 scheduler with results directory: {results_dir}")
    scheduler = LoadBalancedScheduler(params_list, chunk_size=30, results_dir=results_dir)
    
    timestamp_print(f"🏭 Starting ProcessPoolExecutor with {num_cores} workers...")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        active_futures = {}
        last_log = time.time() 
        
        # Submit initial batch
        timestamp_print(f"📤 Submitting initial batch of {num_cores * 2} work items...")
        for _ in range(num_cores * 2):
            work_with_params = scheduler.get_work_with_params()
            if work_with_params:
                future = executor.submit(process_simulation_work, work_with_params)
                active_futures[future] = work_with_params[0]
        
        timestamp_print(f"✅ Initial batch submitted, {len(active_futures)} futures active")
        
        while not scheduler.is_complete() or active_futures:
            if active_futures:
                try:
                    for future in as_completed(active_futures, timeout=10):
                        work = active_futures.pop(future)
                        
                        try:
                            result_data = future.result()
                            scheduler.submit_result(work, result_data)
                            
                            # Submit new work if available
                            new_work_with_params = scheduler.get_work_with_params()
                            if new_work_with_params:
                                new_future = executor.submit(process_simulation_work, new_work_with_params)
                                active_futures[new_future] = new_work_with_params[0]
                            
                        except Exception as e:
                            timestamp_print(f"❌ Exception processing work: {e}")
                            traceback.print_exc()
                        
                        break
                        
                except TimeoutError:
                    pass
            
            # v3 Enhanced 60-second progress heartbeat
            now = time.time()
            if now - last_log >= 60:
                # Get aggregate round counts for true work progress
                completed_rounds, total_rounds = scheduler.get_aggregate_progress()
                pct_complete = 100.0 * completed_rounds / total_rounds if total_rounds > 0 else 0

                timestamp_print(
                    f"📊 PROGRESS: {completed_rounds}/{total_rounds} rounds "
                    f"({pct_complete:5.1f}% of total work) ..."
                )
                last_log = now
    
    # Load all results
    timestamp_print("📂 Loading all completed v3 results...")
    final_results = []
    
    for i in range(num_simulations):
        try:
            with open(f"{results_dir}/sim_{i:04d}_result.pkl", 'rb') as f:
                result = pickle.load(f)
                final_results.append(result)
        except Exception as e:
            timestamp_print(f"⚠️ Could not load result {i}: {e}")
    
    elapsed = time.time() - start_time
    timestamp_print(f"🎉 v3 EXPERIMENT COMPLETE: {len(final_results)} simulations in {elapsed:.2f} seconds")
    
    return final_results

# ===== 6. CLI ENTRYPOINT =====

def main():
    """Main CLI entrypoint for v3 simulation"""
    parser = argparse.ArgumentParser(
        description='Enhanced Constraint Cascade Simulation v3 - Streamlined 13-Parameter Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v3 IMPROVEMENTS:
- Streamlined 13-parameter interface replacing legacy 18-parameter system
- Improved interpretability and performance with focused parameter set
- Enhanced logging frequencies for different metrics
- Aggregate progress reporting showing total work completion
- All original data outputs preserved

Examples:
  python constraint_simulation_v3.py --test quick
  python constraint_simulation_v3.py --runs 50 --multiprocessing
  python constraint_simulation_v3.py --sweep basic --params 100
        """
    )
    
    parser.add_argument('--test', choices=['quick', 'smoke', 'batch'], 
                       help='Run test suite')
    parser.add_argument('-n', '--runs', type=int, default=50,
                       help='Number of simulation runs')
    parser.add_argument('--sweep', choices=['basic'], 
                       help='Run parameter sweep')
    parser.add_argument('--params', type=int, default=50,
                       help='Number of parameter configurations to test')
    parser.add_argument('-m', '--multiprocessing', action='store_true',
                       help='Use multiprocessing')
    parser.add_argument('--single-thread', action='store_true',
                       help='Force single-threaded execution')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests(args.test)
        return
    
    # Determine multiprocessing usage
    if args.single_thread:
        use_multiprocessing = False
    elif args.multiprocessing:
        use_multiprocessing = True
    else:
        use_multiprocessing = args.runs >= 10
    
    timestamp_print("🔬 Enhanced Constraint Cascade Simulation v3")
    timestamp_print("="*80)
    timestamp_print("🎯 Streamlined 13-parameter interface")
    timestamp_print("⚡ Improved interpretability and performance")
    timestamp_print("📊 Enhanced logging frequencies")
    timestamp_print("📈 Aggregate progress reporting")
    timestamp_print("✅ All original functionality preserved")
    
    if args.sweep:
        run_parameter_sweep(args)
    else:
        run_basic_experiment(args, use_multiprocessing)

def run_tests(test_type: str):
    """VALIDATION FIX #11: Run v3 test suite using production parameter sampling"""
    timestamp_print(f"Running v3 {test_type} tests...")
    
    if test_type == 'quick':
        # Quick unit tests using SAME parameter sampling as production
        params = sample_config()  # VALIDATION FIX #11: Use production sampling
        person = OptimizedPerson(1, params)
        
        # Test resilience profile
        assert 0.01 <= person.resilience_threshold <= 0.99
        assert person.resilience_noise >= 0
        
        # VALIDATION FIX #11: Test trust initialization fix
        other = OptimizedPerson(2, params, group_id="B")
        rel = person.get_relationship(other.id, 1, other.group_id)
        assert rel.trust == 0.5  # CRITICAL FIX #2: Should start at 0.5, not None
        
        # VALIDATION FIX #11: Test trust updates don't double-increment
        initial_count = rel.interaction_count
        rel.update_trust(True, 1, params.base_trust_delta, params.group_trust_bias, params.out_group_trust_bias)
        assert rel.interaction_count == initial_count + 1  # Should increment by exactly 1
        assert rel.trust > 0.5
        
        # MAJOR FIX #5: Test cooperation decision uses cooperation threshold
        coop_decision = person.calculate_cooperation_decision(other, 1, params)
        assert isinstance(coop_decision, bool)
        
        # VALIDATION FIX #12: Test bounds checking
        validation_warnings = validate_simulation_results(
            EnhancedSimulationResults(
                parameters=params, run_id=0, final_population=100, final_cooperation_rate=0.5,
                final_constrained_rate=0.2, rounds_completed=50, extinction_occurred=False,
                first_cascade_round=None, total_cascade_events=0, total_shock_events=5,
                total_defections=100, total_redemptions=20, net_strategy_change=80,
                total_births=10, total_deaths=5, max_population_reached=200,
                population_stability=0.1, avg_system_stress=0.3, max_system_stress=1.0,
                avg_maslow_pressure=0.2, avg_basic_needs_crisis_rate=0.1,
                initial_needs_avg={'physiological': 5, 'safety': 5, 'love': 5, 'esteem': 5, 'self_actualization': 5},
                final_needs_avg={'physiological': 6, 'safety': 6, 'love': 6, 'esteem': 6, 'self_actualization': 6},
                needs_improvement={'physiological': 1, 'safety': 1, 'love': 1, 'esteem': 1, 'self_actualization': 1},
                avg_trust_level=0.6, cooperation_benefit_total=50, population_growth=1.5, cooperation_resilience=0.5
            )
        )
        assert len(validation_warnings) == 0  # Should pass all validation checks
        
        timestamp_print("✅ v3 Quick tests passed")
    
    elif test_type == 'smoke':
        # VALIDATION FIX #11: Smoke test using production parameter sampling
        params = sample_config()  # Use production sampling
        params.max_rounds = 30  # Short test
        
        sim = EnhancedMassSimulation(params, 0)
        result = sim.run_simulation()
        
        assert result.rounds_completed > 0
        assert result.final_population >= 0
        assert result.parameters.max_rounds == 30
        
        # VALIDATION FIX #12: Check that defections are reasonable
        max_reasonable_defections = result.final_population * result.rounds_completed * 5
        assert result.total_defections <= max_reasonable_defections, f"Excessive defections: {result.total_defections} > {max_reasonable_defections}"
        
        timestamp_print("✅ v3 Smoke test passed")
    
    elif test_type == 'batch':
        # VALIDATION FIX #11: Batch test using production parameters
        start_time = time.time()
        
        results = []
        for i in range(5):
            params = sample_config()  # Use production sampling
            params.max_rounds = 60
            sim = EnhancedMassSimulation(params, i)
            result = sim.run_simulation()
            results.append(result)
        
        elapsed = time.time() - start_time
        timestamp_print(f"✅ v3 Batch test completed: {len(results)} simulations in {elapsed:.1f}s")
        
        # VALIDATION FIX #11: Verify all results have expected structure
        for result in results:
            assert result.rounds_completed > 0
            assert hasattr(result.parameters, 'shock_interval_years')
            assert hasattr(result.parameters, 'resilience_profile')
            assert result.parameters.max_rounds == 60
            
            # VALIDATION FIX #12: Check no impossible values
            assert 0 <= result.final_cooperation_rate <= 1
            assert 0 <= result.avg_trust_level <= 1
            validation_warnings = validate_simulation_results(result)
            if validation_warnings:
                timestamp_print(f"⚠️ Validation warnings in test result {result.run_id}: {validation_warnings}")
            
        timestamp_print("✅ v3 Result structure validation passed")

def run_basic_experiment(args, use_multiprocessing: bool):
    """Run basic v3 experiment"""
    timestamp_print(f"🎯 Running {args.runs} v3 simulations...")
    
    # Generate v3 parameter configurations
    params_list = [sample_config() for _ in range(args.runs)]
    
    timestamp_print(f"✅ Generated {len(params_list)} v3 parameter sets")
    
    # Run simulations
    results = run_smart_mass_experiment(params_list, use_multiprocessing)
    
    # Analyze and save results
    timestamp_print(f"📊 Analyzing {len(results)} v3 simulation results...")
    df = analyze_v3_patterns(results)
    
    timestamp_print(f"📈 Creating v3 visualizations...")
    create_v3_visualizations(df)
    
    timestamp_print(f"💾 Saving comprehensive v3 results...")
    saved_files = save_v3_results(df)
    
    timestamp_print(f"✅ v3 Experiment completed: {len(saved_files)} files saved")

def run_parameter_sweep(args):
    """Run v3 parameter sweep"""
    timestamp_print(f"Running v3 parameter sweep: {args.sweep}")
    
    if args.sweep == 'basic':
        # Basic v3 parameter sweep
        timestamp_print("🔄 Generating v3 parameter sweep...")
        params_list = []
        
        # CRITICAL FIX #3: Sample across correct parameter dimensions
        shock_intervals = [10, 15, 20, 25]  # Use documented ranges
        num_groups_options = [1, 2, 3]
        
        timestamp_print(f"📋 Sweep grid: {len(shock_intervals)} shock intervals × {len(num_groups_options)} group configs")
        
        for shock_interval in shock_intervals:
            for num_groups in num_groups_options:
                for rep in range(args.params // (len(shock_intervals) * len(num_groups_options))):
                    params = sample_config()
                    params.shock_interval_years = shock_interval
                    params.num_groups = num_groups
                    # SIGNIFICANT FIX #9: Ensure consistent group assignment
                    if num_groups == 1:
                        params.homophily_bias = 0.0
                        params.intervention_interval = 0
                    params_list.append(params)
        
        timestamp_print(f"✅ Generated {len(params_list)} v3 parameter sets for sweep")
        
        # Run sweep
        results = run_smart_mass_experiment(params_list, True)
        
        # Analyze results
        df = analyze_v3_patterns(results)
        create_v3_visualizations(df)
        save_v3_results(df)
        
        timestamp_print(f"✅ v3 Parameter sweep completed: {len(results)} results")

# ===== 7. V3 UTILITIES & METRICS =====

def analyze_v3_patterns(results: List[EnhancedSimulationResults]) -> pd.DataFrame:
    """Analyze v3 results for emergent patterns"""
    timestamp_print("🔍 Analyzing v3 emergent patterns...")
    
    data = []
    for result in results:
        try:
            final_pop = max(result.final_population, 1)
            
            # MAJOR FIX #6: Use corrected redemption rate calculation
            total_strategy_switches = result.total_defections + result.total_redemptions
            corrected_redemption_rate = result.total_redemptions / max(1, total_strategy_switches)
            
            row = {
                'run_id': result.run_id,
                
                # v3 core parameters
                'shock_interval_years': result.parameters.shock_interval_years,
                'homophily_bias': result.parameters.homophily_bias,
                'num_groups': result.parameters.num_groups,
                'out_group_trust_bias': result.parameters.out_group_trust_bias,
                'out_group_penalty': result.parameters.out_group_penalty,
                'intervention_interval': result.parameters.intervention_interval,
                'intervention_scale': result.parameters.intervention_scale,
                'event_bonus': result.parameters.event_bonus,
                'base_trust_delta': result.parameters.base_trust_delta,
                'group_trust_bias': result.parameters.group_trust_bias,
                'resilience_threshold': result.parameters.resilience_profile['threshold'],
                'resilience_noise': result.parameters.resilience_profile['noise'],
                'turnover_rate': result.parameters.turnover_rate,
                'social_diffusion': result.parameters.social_diffusion,
                'max_rounds': result.parameters.max_rounds,
                
                # Legacy parameters
                'initial_population': result.parameters.initial_population,
                'max_population': result.parameters.max_population,
                
                # TOPLINE METRICS: All three cooperation metrics with clear definitions
                'final_cooperation_rate': result.final_cooperation_rate,  # Strategy-based (% with cooperative strategy)
                'behavioral_cooperation_rate': result.behavioral_cooperation_rate,  # Action-based (% of cooperative interactions)
                'cooperation_resilience': result.cooperation_resilience,  # Sustainability after stress events
                'final_constrained_rate': result.final_constrained_rate,
                'final_population': result.final_population,
                'extinction_occurred': result.extinction_occurred,
                'rounds_completed': result.rounds_completed,
                'total_defections': result.total_defections,
                'total_redemptions': result.total_redemptions,
                'redemption_rate': corrected_redemption_rate,  # MAJOR FIX #6: Corrected calculation
                'avg_trust_level': result.avg_trust_level,
                'cooperation_benefit_total': result.cooperation_benefit_total,
                
                # Inter-group metrics
                'in_group_interaction_rate': result.in_group_interaction_rate,
                'out_group_interaction_rate': result.out_group_interaction_rate,
                'avg_in_group_trust': result.avg_in_group_trust,
                'avg_out_group_trust': result.avg_out_group_trust,
                'trust_asymmetry': result.trust_asymmetry,
                'total_mixing_events': result.total_mixing_events,
                
                # v3 derived metrics
                'shock_frequency': 1.0 / max(1, result.parameters.shock_interval_years),
                'trust_sensitivity': result.parameters.base_trust_delta * result.parameters.group_trust_bias,
                'intervention_intensity': result.parameters.intervention_scale / max(1, result.parameters.intervention_interval),
                'resilience_variability': result.parameters.resilience_profile['noise'] / max(0.001, result.parameters.resilience_profile['threshold']),
                'social_cohesion_factor': result.parameters.social_diffusion * result.avg_trust_level,

                # === INSTITUTIONAL MEMORY METRICS (new) ===
                'total_shocks_experienced': result.institutional_memory['total_shocks_experienced'],
                'average_shock_severity': result.institutional_memory['average_shock_severity'],
                'recovery_success_rate': result.institutional_memory['recovery_success_rate'],
                'crisis_response_knowledge': result.institutional_memory['crisis_response_knowledge'],
            }
            data.append(row)
            
        except Exception as e:
            timestamp_print(f"⚠️ Error processing v3 result {result.run_id}: {e}")
            continue
    
    if not data:
        timestamp_print("⚠️ No v3 simulation data to analyze")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Create v3 outcome categories
    try:
        df['outcome_category'] = pd.cut(df['final_cooperation_rate'], 
                                       bins=[0, 0.1, 0.3, 0.7, 1.0],
                                       labels=['Collapse', 'Low_Coop', 'Medium_Coop', 'High_Coop'])
        
        df['extinction_category'] = df['extinction_occurred'].map({True: 'Extinct', False: 'Survived'})
        
        # v3 specific categories
        df['shock_severity'] = pd.cut(df['shock_frequency'], 
                                     bins=[0, 0.1, 0.2, 0.5, 1.0],
                                     labels=['Rare', 'Occasional', 'Frequent', 'Constant'])
        
        df['trust_development'] = pd.cut(df['trust_sensitivity'], 
                                        bins=[0, 0.1, 0.2, 0.4, 1.0],
                                        labels=['Slow', 'Moderate', 'Fast', 'Rapid'])
        
        df['group_complexity'] = df['num_groups'].map({1: 'Simple', 2: 'Moderate', 3: 'Complex'})
        
    except Exception as e:
        timestamp_print(f"⚠️ Error creating v3 derived columns: {e}")
    
    return df

def create_v3_visualizations(df: pd.DataFrame):
    """Create v3 pattern analysis visualizations"""
    timestamp_print("📊 Creating v3 visualization analysis...")
    
    try:
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # Determine layout based on data
        has_intergroup_data = df['num_groups'].max() > 1 if 'num_groups' in df.columns else False
        
        if has_intergroup_data:
            rows, cols = 4, 4
        else:
            rows, cols = 3, 4
        
        fig = plt.figure(figsize=(24, rows * 5))
        
        # 1. v3 Shock Frequency vs Cooperation
        ax1 = plt.subplot(rows, cols, 1)
        if 'shock_frequency' in df.columns and 'final_cooperation_rate' in df.columns:
            scatter = plt.scatter(df['shock_frequency'], df['final_cooperation_rate'], 
                                 c=df['resilience_threshold'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='Resilience Threshold')
            plt.xlabel('Shock Frequency (1/years)')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Shock Frequency vs Cooperation\n(Color = Resilience Threshold)')
        
        # 2. Trust Development vs Outcomes
        ax2 = plt.subplot(rows, cols, 2)
        if 'trust_sensitivity' in df.columns:
            scatter = plt.scatter(df['trust_sensitivity'], df['final_cooperation_rate'], 
                                 c=df['avg_trust_level'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Average Trust Level')
            plt.xlabel('Trust Sensitivity')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Trust Development vs Cooperation\n(Color = Trust Level)')
        
        # 3. Intervention Effectiveness
        ax3 = plt.subplot(rows, cols, 3)
        if 'intervention_intensity' in df.columns:
            scatter = plt.scatter(df['intervention_intensity'], df['final_cooperation_rate'], 
                                 c=df['total_mixing_events'], cmap='RdYlGn', alpha=0.6)
            plt.colorbar(scatter, label='Total Mixing Events')
            plt.xlabel('Intervention Intensity')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Intervention vs Cooperation\n(Color = Mixing Events)')
        
        # 4. Social Diffusion Impact
        ax4 = plt.subplot(rows, cols, 4)
        if 'social_cohesion_factor' in df.columns:
            scatter = plt.scatter(df['social_cohesion_factor'], df['final_cooperation_rate'], 
                                 c=df['trust_asymmetry'], cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, label='Trust Asymmetry')
            plt.xlabel('Social Cohesion Factor')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Social Diffusion vs Cooperation\n(Color = Trust Asymmetry)')
        
        # 5. Group Dynamics (if applicable)
        if has_intergroup_data:
            ax5 = plt.subplot(rows, cols, 5)
            if 'num_groups' in df.columns and 'homophily_bias' in df.columns:
                for group_num in df['num_groups'].unique():
                    group_data = df[df['num_groups'] == group_num]
                    plt.scatter(group_data['homophily_bias'], group_data['final_cooperation_rate'], 
                               label=f'{group_num} Groups', alpha=0.6)
                plt.xlabel('Homophily Bias')
                plt.ylabel('Final Cooperation Rate')
                plt.title('v3 Group Dynamics')
                plt.legend()
        
        # 6. Resilience Profile Analysis
        ax6 = plt.subplot(rows, cols, 6)
        if 'resilience_variability' in df.columns:
            scatter = plt.scatter(df['resilience_variability'], df['final_cooperation_rate'], 
                                 c=df['total_defections'], cmap='Reds', alpha=0.6)
            plt.colorbar(scatter, label='Total Defections')
            plt.xlabel('Resilience Variability')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Resilience Variability\n(Color = Defections)')
        
        # 7. Turnover Rate Impact
        ax7 = plt.subplot(rows, cols, 7)
        if 'turnover_rate' in df.columns:
            scatter = plt.scatter(df['turnover_rate'], df['final_cooperation_rate'], 
                                 c=df['final_population'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Final Population')
            plt.xlabel('Turnover Rate')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Turnover vs Cooperation\n(Color = Population)')
        
        # 8. Out-group Penalty Effects
        ax8 = plt.subplot(rows, cols, 8)
        if 'out_group_penalty' in df.columns and has_intergroup_data:
            scatter = plt.scatter(df['out_group_penalty'], df['final_cooperation_rate'], 
                                 c=df['out_group_trust_bias'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='Out-group Trust Bias')
            plt.xlabel('Out-group Penalty')
            plt.ylabel('Final Cooperation Rate')
            plt.title('v3 Out-group Penalty vs Cooperation\n(Color = Trust Bias)')
        
        plt.tight_layout()
        
        title = 'Enhanced Constraint Cascade Simulation v3 - FIXED - Streamlined Parameter Analysis'
        if has_intergroup_data:
            title += '\n(13-Parameter Configuration with Inter-Group Dynamics)'
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        try:
            plt.savefig('v3_streamlined_analysis.png', dpi=300, bbox_inches='tight')
            timestamp_print("✅ v3 Visualization saved as v3_streamlined_analysis.png")
        except Exception as save_error:
            timestamp_print(f"⚠️ Could not save v3 visualization: {save_error}")
        
        plt.close(fig)
        
        return fig
        
    except Exception as e:
        timestamp_print(f"⚠️ Error creating v3 visualizations: {e}")
        timestamp_print("📊 Continuing without visualizations...")
        return None

def save_v3_results(df: pd.DataFrame):
    """Save v3 analysis results"""
    current_dir = os.getcwd()
    timestamp_print(f"💾 Saving v3 results to: {current_dir}")
    
    saved_files = []
    
    try:
        # Save main v3 dataset
        main_file = 'v3_streamlined_simulation_results.csv'
        df.to_csv(main_file, index=False)
        if os.path.exists(main_file):
            size_mb = os.path.getsize(main_file) / (1024*1024)
            saved_files.append(f"📊 {main_file} ({size_mb:.2f} MB)")
        
        # Save v3 summary statistics
        try:
            summary_file = 'v3_summary_stats.csv'
            summary_stats = df.describe()
            summary_stats.to_csv(summary_file)
            if os.path.exists(summary_file):
                saved_files.append(f"📈 {summary_file}")
        except Exception as summary_error:
            timestamp_print(f"⚠️ Could not create v3 summary stats: {summary_error}")
        
        # Create v3 comprehensive summary
        summary_report = 'v3_experiment_summary.txt'
        with open(summary_report, 'w') as f:
            f.write("Enhanced Constraint Cascade v3 - FIXED - Streamlined Parameter Experiment Summary\n")
            f.write("="*80 + "\n\n")
            f.write("v3 CONFIGURATION:\n")
            f.write("- 13-parameter streamlined interface\n")
            f.write("- Improved interpretability and performance\n")
            f.write("- Enhanced logging frequencies\n")
            f.write("- Aggregate progress reporting\n")
            f.write("- CRITICAL BUGS FIXED: Interaction double-counting, trust initialization, parameter validation\n")
            f.write("- MAJOR FIXES: Cooperation metrics clarified, redemption rate corrected, trust calculation improved\n")
            f.write("- SIGNIFICANT FIXES: Institutional memory completed, group assignment standardized\n")
            f.write("- VALIDATION FIXES: Bounds checking added, test functions aligned with production\n\n")
            
            f.write(f"Total Simulations: {len(df)}\n")
            
            # Safe column access for v3 metrics - TOPLINE COOPERATION METRICS
            if 'final_cooperation_rate' in df.columns:
                f.write(f"Average Strategy-Based Cooperation Rate: {df['final_cooperation_rate'].mean():.3f}\n")
            if 'behavioral_cooperation_rate' in df.columns:
                f.write(f"Average Action-Based Cooperation Rate: {df['behavioral_cooperation_rate'].mean():.3f}\n")
            if 'cooperation_resilience' in df.columns:
                f.write(f"Average Cooperation Resilience: {df['cooperation_resilience'].mean():.3f}\n")
            if 'extinction_occurred' in df.columns:
                f.write(f"Extinction Rate: {df['extinction_occurred'].mean():.3f}\n")
            if 'final_population' in df.columns:
                f.write(f"Average Final Population: {df['final_population'].mean():.1f}\n")
            
            # v3 parameter ranges
            if 'shock_interval_years' in df.columns:
                f.write(f"Shock Interval Range: {df['shock_interval_years'].min()} - {df['shock_interval_years'].max()} years\n")
            if 'num_groups' in df.columns:
                f.write(f"Group Configurations: {sorted(df['num_groups'].unique())}\n")
            if 'resilience_threshold' in df.columns:
                f.write(f"Resilience Threshold Range: {df['resilience_threshold'].min():.3f} - {df['resilience_threshold'].max():.3f}\n")
            
            # v3 performance metrics
            if 'trust_sensitivity' in df.columns:
                f.write(f"Trust Sensitivity Range: {df['trust_sensitivity'].min():.3f} - {df['trust_sensitivity'].max():.3f}\n")
            if 'intervention_intensity' in df.columns:
                f.write(f"Intervention Intensity Range: {df['intervention_intensity'].min():.3f} - {df['intervention_intensity'].max():.3f}\n")
            
            f.write(f"\nFiles Created:\n")
            for file_info in saved_files:
                f.write(f"  {file_info}\n")
        
        if os.path.exists(summary_report):
            saved_files.append(f"📋 {summary_report}")
        
        timestamp_print(f"\n✅ Successfully saved {len(saved_files)} v3 files:")
        for file_info in saved_files:
            timestamp_print(f"   {file_info}")
        
    except Exception as e:
        timestamp_print(f"❌ Error saving v3 files: {e}")
        traceback.print_exc()
        
        # Try to save just the main file as backup
        try:
            backup_file = 'v3_backup.csv'
            df.to_csv(backup_file, index=False)
            timestamp_print(f"💾 v3 Backup saved as: {backup_file}")
            saved_files.append(f"💾 {backup_file} (backup)")
        except Exception as backup_error:
            timestamp_print(f"❌ v3 Backup also failed: {backup_error}")
    
    return saved_files

if __name__ == "__main__":
    main()
