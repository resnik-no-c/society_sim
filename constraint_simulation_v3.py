#!/usr/bin/env python3
"""
Enhanced Constraint Cascade Simulation - Realistic Parameters Complete Edition
Implements realistic shock frequency, trust mechanics, and stress model while preserving all functionality.

FIXED ISSUES:
1. Parameter generation mismatch - grid parameters were being ignored
2. Added comprehensive logging during initialization
3. Fixed scheduler initialization performance
4. Added progress indicators for all long operations
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

# ===== 1. CONFIG & CONSTANTS =====

# Sweepable shock & buffer knobs  ───────────────────────────────────────
SHOCK_MEAN_YEARS_SET   = (5, 10, 20)    # exponential λ choices
PARETO_ALPHA_SET       = (1.5, 2.0, 2.7)
PARETO_ALPHA_RANGE     = PARETO_ALPHA_SET
COMMUNITY_BUFFER_SET   = (0.00, 0.04, 0.08)
COMMUNITY_BUFFER_MINMAX  = (min(COMMUNITY_BUFFER_SET), max(COMMUNITY_BUFFER_SET))
PARETO_XM              = 0.3
SHOCK_INTERVAL_YEARS   = (5,25) #back-compat shim            

# Trust mechanics - slower, more realistic
TRUST_DELTA_HELP = +0.04  # Slower trust building (was +0.1)
TRUST_DELTA_BETRAY = -0.06  # Slower trust decay (was -0.15)
REL_WINDOW_LEN = 40  # 40 events ≈ 10 years of interaction history
SERENDIPITY_RATE = 0.05  # 5% interactions ignore homophily

# Community buffer - social support reduces chronic stress
COMMUNITY_BUFFER_MIN = 0.02
COMMUNITY_BUFFER_MAX = 0.08

# Stress model parameters
ACUTE_DECAY_QUARTERLY = 0.97  # Acute stress decays 50% per 5 years
CHRONIC_WINDOW_QUARTERS = 16  # 4-year rolling window

# Population dynamics
MAX_POPULATION = 800
DEFAULT_ROUNDS = 200  # 50 years

def timestamp_print(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    # Flush immediately for nohup visibility
    import sys
    sys.stdout.flush()

def save_simulation_result(result, results_dir: str = "simulation_results"):
    """Enhanced save function that captures complete simulation state"""
    import os
    import pickle
    import json
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        timestamp_print(f"📁 Created results directory: {results_dir}")
    
    # Save as pickle file (complete object)
    pkl_filename = f"sim_{result.run_id:04d}_result.pkl"
    pkl_filepath = os.path.join(results_dir, pkl_filename)
    
    try:
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(result, f)
        timestamp_print(f"💾 Saved complete simulation {result.run_id} to {pkl_filepath}")
    except Exception as e:
        timestamp_print(f"❌ Error saving pickle for simulation {result.run_id}: {e}")
        return None
    
    # Also save as JSON for easier external access
    json_filename = f"sim_{result.run_id:04d}_result.json"
    json_filepath = os.path.join(results_dir, json_filename)
    
    try:
        # Convert result to JSON-serializable format
        json_data = {
            'run_id': result.run_id,
            'parameters': {
                'initial_population': result.parameters.initial_population,
                'max_population': result.parameters.max_population,
                'max_rounds': result.parameters.max_rounds,
                'base_birth_rate': result.parameters.base_birth_rate,
                'maslow_variation': result.parameters.maslow_variation,
                'constraint_threshold_range': result.parameters.constraint_threshold_range,
                'recovery_threshold': result.parameters.recovery_threshold,
                'cooperation_bonus': result.parameters.cooperation_bonus,
                'trust_threshold': result.parameters.trust_threshold,
                'max_relationships_per_person': result.parameters.max_relationships_per_person,
                'shock_interval_years': result.parameters.shock_interval_years if hasattr(result.parameters, 'shock_interval_years') else [0, 0],
                'shock_mean_years': getattr(result.parameters, 'shock_mean_years', 10),
                'pareto_alpha': result.parameters.pareto_alpha,
                'pareto_xm': result.parameters.pareto_xm,
                'trust_delta_help': result.parameters.trust_delta_help,
                'trust_delta_betray': result.parameters.trust_delta_betray,
                'relationship_memory': result.parameters.relationship_memory,
                'serendipity_rate': result.parameters.serendipity_rate,
                'community_buffer_factor': result.parameters.community_buffer_factor,
                'acute_decay': result.parameters.acute_decay,
                'chronic_window': result.parameters.chronic_window,
                'num_groups': result.parameters.num_groups,
                'founder_group_distribution': result.parameters.founder_group_distribution,
                'homophily_bias': result.parameters.homophily_bias,
                'in_group_trust_modifier': result.parameters.in_group_trust_modifier,
                'out_group_trust_modifier': result.parameters.out_group_trust_modifier,
                'out_group_constraint_amplifier': result.parameters.out_group_constraint_amplifier,
                'reputational_spillover': result.parameters.reputational_spillover,
                'mixing_event_frequency': result.parameters.mixing_event_frequency,
                'mixing_event_bonus_multiplier': result.parameters.mixing_event_bonus_multiplier,
                'inheritance_style': result.parameters.inheritance_style,
            },
            'outcomes': {
                'final_cooperation_rate': result.final_cooperation_rate,
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
    """Save simulation result to incremental CSV file - COMPLETE VERSION"""
    import os
    import pandas as pd
    
    # Convert result to comprehensive row format
    row_data = {
        # Basic identifiers
        'run_id': result.run_id,
        'timestamp': datetime.now().isoformat(),
        
        # === SIMULATION PARAMETERS ===
        # Core parameters
        'initial_population': result.parameters.initial_population,
        'max_population': result.parameters.max_population,
        'max_rounds': result.parameters.max_rounds,
        'base_birth_rate': result.parameters.base_birth_rate,
        'maslow_variation': result.parameters.maslow_variation,
        'recovery_threshold': result.parameters.recovery_threshold,
        'cooperation_bonus': result.parameters.cooperation_bonus,
        'trust_threshold': result.parameters.trust_threshold,
        'max_relationships_per_person': result.parameters.max_relationships_per_person,
        
        # Constraint threshold range
        'constraint_threshold_min': result.parameters.constraint_threshold_range[0],
        'constraint_threshold_max': result.parameters.constraint_threshold_range[1],
        
        # Realistic shock parameters
        'shock_interval_min': result.parameters.shock_interval_years[0] if hasattr(result.parameters, 'shock_interval_years') else 0,
        'shock_interval_max': result.parameters.shock_interval_years[1] if hasattr(result.parameters, 'shock_interval_years') else 0,
        'shock_interval_avg': (sum(result.parameters.shock_interval_years) / 2) if hasattr(result.parameters, 'shock_interval_years') else 0,
        'shock_mean_years': getattr(result.parameters, 'shock_mean_years', 10),
        'pareto_alpha': result.parameters.pareto_alpha,
        'pareto_xm': result.parameters.pareto_xm,
        
        # Realistic trust mechanics
        'trust_delta_help': result.parameters.trust_delta_help,
        'trust_delta_betray': result.parameters.trust_delta_betray,
        'relationship_memory': result.parameters.relationship_memory,
        'serendipity_rate': result.parameters.serendipity_rate,
        
        # Community buffer and stress model parameters
        'community_buffer_factor': result.parameters.community_buffer_factor,
        'acute_decay': result.parameters.acute_decay,
        'chronic_window': result.parameters.chronic_window,
        
        # Inter-group parameters
        'num_groups': result.parameters.num_groups,
        'homophily_bias': result.parameters.homophily_bias,
        'in_group_trust_modifier': result.parameters.in_group_trust_modifier,
        'out_group_trust_modifier': result.parameters.out_group_trust_modifier,
        'out_group_constraint_amplifier': result.parameters.out_group_constraint_amplifier,
        'reputational_spillover': result.parameters.reputational_spillover,
        'mixing_event_frequency': result.parameters.mixing_event_frequency,
        'mixing_event_bonus_multiplier': result.parameters.mixing_event_bonus_multiplier,
        'inheritance_style': result.parameters.inheritance_style,
        
        # Group distribution (expand list into separate columns)
        'founder_group_a_proportion': result.parameters.founder_group_distribution[0] if len(result.parameters.founder_group_distribution) > 0 else 0,
        'founder_group_b_proportion': result.parameters.founder_group_distribution[1] if len(result.parameters.founder_group_distribution) > 1 else 0,
        'founder_group_c_proportion': result.parameters.founder_group_distribution[2] if len(result.parameters.founder_group_distribution) > 2 else 0,
        
        # === FINAL OUTCOMES ===
        'final_cooperation_rate': result.final_cooperation_rate,
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
        'redemption_rate': result.total_redemptions / max(1, result.total_defections),
        
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
        # Initial needs
        'initial_physiological': result.initial_needs_avg.get('physiological', 0),
        'initial_safety': result.initial_needs_avg.get('safety', 0),
        'initial_love': result.initial_needs_avg.get('love', 0),
        'initial_esteem': result.initial_needs_avg.get('esteem', 0),
        'initial_self_actualization': result.initial_needs_avg.get('self_actualization', 0),
        
        # Final needs
        'final_physiological': result.final_needs_avg.get('physiological', 0),
        'final_safety': result.final_needs_avg.get('safety', 0),
        'final_love': result.final_needs_avg.get('love', 0),
        'final_esteem': result.final_needs_avg.get('esteem', 0),
        'final_self_actualization': result.final_needs_avg.get('self_actualization', 0),
        
        # Needs improvement (change)
        'physiological_change': result.needs_improvement.get('physiological', 0),
        'safety_change': result.needs_improvement.get('safety', 0),
        'love_change': result.needs_improvement.get('love', 0),
        'esteem_change': result.needs_improvement.get('esteem', 0),
        'self_actualization_change': result.needs_improvement.get('self_actualization', 0),
        
        # === COOPERATION METRICS ===
        'avg_trust_level': result.avg_trust_level,
        'cooperation_benefit_total': result.cooperation_benefit_total,
        'cooperation_resilience': result.cooperation_resilience,
        
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
        
        # Final group populations (expand dict into separate columns)
        'final_group_a_population': result.final_group_populations.get('A', 0),
        'final_group_b_population': result.final_group_populations.get('B', 0),
        'final_group_c_population': result.final_group_populations.get('C', 0),
        
        # Final group cooperation rates (expand dict into separate columns)
        'final_group_a_cooperation_rate': result.final_group_cooperation_rates.get('A', 0),
        'final_group_b_cooperation_rate': result.final_group_cooperation_rates.get('B', 0),
        'final_group_c_cooperation_rate': result.final_group_cooperation_rates.get('C', 0),
        
        # === INTERACTION METRICS ===
        'total_interactions': result.total_interactions,
        'avg_interaction_processing_time': result.avg_interaction_processing_time,
        'interaction_intensity': result.total_interactions / max(1, result.final_population),
        
        # === DERIVED METRICS ===
        'pop_multiplier': result.parameters.max_population / max(1, result.parameters.initial_population),
        'shock_frequency_proxy': 1 / max(1, (sum(result.parameters.shock_interval_years) / 2) if hasattr(result.parameters, 'shock_interval_years') else 10),
        'growth_potential': result.parameters.base_birth_rate * (result.parameters.max_population / max(1, result.parameters.initial_population)),
        'resilience_index': result.parameters.community_buffer_factor * (1 - (1 / max(1, (sum(result.parameters.shock_interval_years) / 2) if hasattr(result.parameters, 'shock_interval_years') else 10))),
        'intergroup_tension': result.parameters.out_group_constraint_amplifier * result.parameters.reputational_spillover * (1 - result.parameters.out_group_trust_modifier) if result.parameters.num_groups > 1 else 0,
        'stress_recovery_rate': 1 - result.parameters.acute_decay,
        'social_support_effectiveness': result.parameters.community_buffer_factor * result.avg_trust_level,
    }
    
    # Create DataFrame
    df_row = pd.DataFrame([row_data])
    
    # Append to file (create if doesn't exist)
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

# ===== 2. DATA CLASSES =====

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
    """Enhanced relationship tracking with realistic trust mechanics"""
    trust: float = 0.5
    interaction_count: int = 0
    cooperation_history: deque = field(default_factory=lambda: deque(maxlen=REL_WINDOW_LEN))
    last_interaction_round: int = 0
    
    # Inter-group extensions
    is_same_group: bool = True
    betrayal_count: int = 0
    cooperation_count: int = 0
    
    def update_trust(self, cooperated: bool, round_num: int, 
                    in_group_modifier: float = 1.0, out_group_modifier: float = 1.0):
        """Update trust based on interaction outcome with realistic speed"""
        self.interaction_count += 1
        self.last_interaction_round = round_num
        self.cooperation_history.append(cooperated)
        
        # Apply realistic trust deltas
        if cooperated:
            self.cooperation_count += 1
            trust_delta = TRUST_DELTA_HELP
            if self.is_same_group:
                trust_delta *= in_group_modifier
            else:
                trust_delta *= out_group_modifier
            self.trust = min(0.9, self.trust + trust_delta)
        else:
            self.betrayal_count += 1
            trust_delta = -TRUST_DELTA_BETRAY
            if self.is_same_group:
                trust_delta *= in_group_modifier
            else:
                trust_delta *= out_group_modifier
            self.trust = max(0.0, self.trust + trust_delta)

@dataclass
class SimulationParameters:
    """Enhanced simulation parameters with realistic settings"""
    initial_population: int
    max_population: int = MAX_POPULATION
    max_rounds: int = DEFAULT_ROUNDS
    
    # Realistic shock parameters
    shock_interval_years: Tuple[float, float] = SHOCK_INTERVAL_YEARS
    shock_mean_years: float = 10  # For exponential distribution
    pareto_alpha: float = 2.0
    pareto_xm: float = PARETO_XM
    
    # Realistic trust mechanics
    trust_delta_help: float = TRUST_DELTA_HELP
    trust_delta_betray: float = TRUST_DELTA_BETRAY
    relationship_memory: int = REL_WINDOW_LEN
    serendipity_rate: float = SERENDIPITY_RATE
    
    # Community buffer parameters
    community_buffer_factor: float = 0.04
    
    # Stress model parameters
    acute_decay: float = ACUTE_DECAY_QUARTERLY
    chronic_window: int = CHRONIC_WINDOW_QUARTERS
    
    # Original parameters preserved
    base_birth_rate: float = 0.008
    maslow_variation: float = 0.5
    constraint_threshold_range: Tuple[float, float] = (0.05, 0.25)
    recovery_threshold: float = 0.3
    cooperation_bonus: float = 0.2
    trust_threshold: float = 0.6
    max_relationships_per_person: int = 150
    
    # Inter-Group Parameters - all preserved
    num_groups: int = 3
    founder_group_distribution: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    homophily_bias: float = 0.7
    in_group_trust_modifier: float = 1.5
    out_group_trust_modifier: float = 0.5
    out_group_constraint_amplifier: float = 2.0
    reputational_spillover: float = 0.1
    mixing_event_frequency: int = 15
    mixing_event_bonus_multiplier: float = 2.0
    inheritance_style: str = "mother"

@dataclass
class EnhancedSimulationResults:
    """Comprehensive results container - all metrics preserved"""
    parameters: SimulationParameters
    run_id: int
    
    # Final outcomes
    final_population: int
    final_cooperation_rate: float
    final_constrained_rate: float
    
    # System dynamics
    rounds_completed: int
    extinction_occurred: bool
    first_cascade_round: Optional[int]
    total_cascade_events: int
    total_shock_events: int
    
    # Strategy changes
    total_defections: int
    total_redemptions: int
    net_strategy_change: int
    
    # Population metrics
    total_births: int
    total_deaths: int
    max_population_reached: int
    population_stability: float
    
    # Pressure metrics
    avg_system_stress: float
    max_system_stress: float
    avg_maslow_pressure: float
    avg_basic_needs_crisis_rate: float
    
    # Maslow evolution
    initial_needs_avg: Dict[str, float]
    final_needs_avg: Dict[str, float]
    needs_improvement: Dict[str, float]
    
    # Cooperation benefits
    avg_trust_level: float
    cooperation_benefit_total: float
    
    # Additional metrics
    population_growth: float
    cooperation_resilience: float
    
    # Inter-Group Metrics - all preserved
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
    
    # NEW: Realistic interaction metrics
    total_interactions: int = 0
    avg_interaction_processing_time: float = 0.0

class OptimizedPerson:
    """Enhanced person with realistic stress model and preserved functionality"""
    
    __slots__ = ['id', 'strategy', 'constraint_level', 'constraint_threshold', 
                 'recovery_threshold', 'is_constrained', 'is_dead', 'relationships',
                 'max_lifespan', 'age', 'strategy_changes', 'rounds_as_selfish',
                 'rounds_as_cooperative', 'maslow_needs', 'maslow_pressure', 'is_born',
                 'group_id', 'in_group_interactions', 'out_group_interactions', 
                 'mixing_event_participations', 'acute_stress', 'chronic_queue', 'base_coop', 'society_trust']
    
    def __init__(self, person_id: int, params: SimulationParameters, 
                 parent_a: Optional['OptimizedPerson'] = None, 
                 parent_b: Optional['OptimizedPerson'] = None,
                 group_id: Optional[str] = None):
        self.id = person_id
        self.strategy = 'cooperative'
        self.constraint_level = 0.0
        self.constraint_threshold = random.uniform(*params.constraint_threshold_range)
        self.recovery_threshold = params.recovery_threshold
        self.is_constrained = False
        self.is_dead = False
        self.is_born = (parent_a is not None and parent_b is not None)
        
        # NEW: Realistic stress model components
        self.acute_stress = 0.0
        self.chronic_queue = deque(maxlen=params.chronic_window)
        # Initialize chronic queue with low stress
        for _ in range(params.chronic_window):
            self.chronic_queue.append(0.0)
        
        # Base cooperation probability
        self.base_coop = 0.4 + (random.random() - 0.5) * 0.4
        self.base_coop = max(0.1, min(0.9, self.base_coop))
        
        self.relationships: Dict[int, FastRelationship] = {}
        self.society_trust = 0.5   # default aggregate trust
        
        self.max_lifespan = int((200 + random.random() * 300) * (params.max_rounds / 500))
        self.age = 0
        
        self.strategy_changes = 0
        self.rounds_as_selfish = 0
        self.rounds_as_cooperative = 0
        
        # Group identity and tracking
        if group_id is not None:
            self.group_id = group_id
        elif parent_a and parent_b:
            self.group_id = self._determine_child_group(parent_a, parent_b, params.inheritance_style)
        else:
            self.group_id = "A"
            
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
        
        self.maslow_pressure = 0.0
        self._calculate_maslow_pressure_fast()
    
    def _determine_child_group(self, parent_a: 'OptimizedPerson', parent_b: 'OptimizedPerson', 
                              inheritance_style: str) -> str:
        """Determine child's group based on inheritance style"""
        if inheritance_style == "mother":
            return parent_a.group_id
        elif inheritance_style == "father":
            return parent_b.group_id
        elif inheritance_style == "random":
            return random.choice([parent_a.group_id, parent_b.group_id])
        elif inheritance_style == "majority":
            return random.choice([parent_a.group_id, parent_b.group_id])
        else:
            return parent_a.group_id
    
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
    
    def update_stress(self, shock_increment: float, params: SimulationParameters):
        """NEW: Update acute and chronic stress with community buffer"""
        # Update acute stress with decay
        self.acute_stress = self.acute_stress * params.acute_decay + shock_increment
        
        # Update chronic stress queue
        self.chronic_queue.append(self.acute_stress)
        chronic_stress = np.mean(self.chronic_queue)
        
        # Apply community buffer based on top 5 relationships
        top5_trust = self.get_top5_trust()
        buffer = params.community_buffer_factor * top5_trust
        chronic_stress = max(0.0, chronic_stress - buffer)
        
        return chronic_stress
    
    def get_top5_trust(self) -> float:
        """Get average trust of top 5 relationships"""
        if not self.relationships:
            return 0.0
        
        trust_values = [rel.trust for rel in self.relationships.values()]
        trust_values.sort(reverse=True)
        return np.mean(trust_values[:5])
    
    def calculate_cooperation_probability(self, params: SimulationParameters) -> float:
        """NEW: Calculate cooperation probability based on stress model"""
        chronic_stress = np.mean(self.chronic_queue)
        
        # Base cooperation + acute boost - chronic burnout
        prob = self.base_coop + 0.4 * self.acute_stress - 0.5 * (chronic_stress ** 2)
        return max(0.05, min(0.95, prob))
    
    def update(self, system_stress: float, params: SimulationParameters, cooperation_bonus: float = 0):
        """Update person state with realistic stress model"""
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
        
        # Update constraint level for compatibility
        chronic_stress = np.mean(self.chronic_queue)
        self.constraint_level = chronic_stress * 0.5  # Convert to old constraint system
        
        need_satisfaction = (needs.physiological + needs.safety + needs.love + 
                           needs.esteem + needs.self_actualization) / 50
        pressure_decay = 0.01 * need_satisfaction
        self.constraint_level = max(0, self.constraint_level - pressure_decay)
    
    def add_constraint_pressure(self, amount: float, is_from_out_group: bool = False, 
                              out_group_amplifier: float = 1.0) -> bool:
        """Add pressure with Maslow amplification and optional out-group surcharge"""
        if self.is_dead:
            return False
        
        maslow_amplifier = 1 + (self.maslow_pressure * 0.5)
        
        if is_from_out_group:
            amount *= out_group_amplifier
        
        # Add to acute stress instead of constraint level
        self.acute_stress += amount * maslow_amplifier
        
        # Check if should switch to selfish
        if self.strategy == 'cooperative' and self.constraint_level > self.constraint_threshold:
            self.force_switch()
            return True
        return False
    
    def check_for_recovery(self, params: SimulationParameters) -> bool:
        """Check if person can recover to cooperative strategy"""
        if self.strategy == 'selfish' and self.constraint_level < self.recovery_threshold:
            recovery_chance = 0.6  # Increased from 0.1 for more realistic recovery
            
            if self.maslow_needs.love > 7:
                recovery_chance += 0.2
            if self.maslow_needs.esteem > 7:
                recovery_chance += 0.1
            if self.maslow_needs.self_actualization > 8:
                recovery_chance += 0.2
            
            # Add social support bonus
            social_support = self.get_top5_trust()
            recovery_chance += social_support * 0.3
            
            if self.rounds_as_selfish > 50:
                recovery_chance *= 0.7  # Less harsh penalty
            
            if random.random() < recovery_chance:
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

        if hasattr(self, "sim_ref"):
            self.sim_ref.total_defections += 1
    
    def switch_to_cooperative(self):
        """Recover to cooperative strategy"""
        self.strategy = 'cooperative'
        self.is_constrained = False
        self.strategy_changes += 1
        self.rounds_as_selfish = 0
        self.maslow_needs.love = min(10, self.maslow_needs.love * 1.1)
        self.maslow_needs.esteem = min(10, self.maslow_needs.esteem * 1.1)
    
    def get_relationship(self, other_id: int, round_num: int, 
                        other_group_id: Optional[str] = None) -> FastRelationship:
        """Get or create relationship with group awareness"""
        if other_id not in self.relationships:
            if len(self.relationships) >= 150:
                oldest_id = min(self.relationships.keys(), 
                              key=lambda k: self.relationships[k].last_interaction_round)
                del self.relationships[oldest_id]
            
            is_same_group = (other_group_id is None or self.group_id == other_group_id)
            self.relationships[other_id] = FastRelationship(is_same_group=is_same_group)
        return self.relationships[other_id]
    
    def calculate_cooperation_decision(self, other: 'OptimizedPerson', round_num: int, params: SimulationParameters) -> bool:
        """Decide whether to cooperate based on realistic stress model"""
        if self.strategy == 'selfish':
            return False

        if random.random()<0.02: 
            return False
        
        # Use realistic cooperation probability
        base_prob = self.calculate_cooperation_probability(params)
        
        relationship = self.get_relationship(other.id, round_num, other.group_id)
        
        if relationship.interaction_count == 0:
            if hasattr(self, 'group_id') and hasattr(other, 'group_id'):
                if self.group_id == other.group_id:
                    base_prob *= 1.2
                else:
                    base_prob *= 0.8
            return random.random() < base_prob
        else:
            recent_coop = sum(list(relationship.cooperation_history)[-3:]) / min(3, len(relationship.cooperation_history))
            cooperation_prob = relationship.trust * 0.7 + recent_coop * 0.3
            return random.random() < cooperation_prob
    
    def _get_basic_needs_pressure(self) -> float:
        """Calculate basic needs pressure"""
        return (max(0, 5 - self.maslow_needs.physiological) * 0.002 + 
                max(0, 5 - self.maslow_needs.safety) * 0.001)
    
    def _get_inspire_effect(self) -> float:
        """Calculate inspiration effect"""
        return max(0, self.maslow_needs.self_actualization - 7) * 0.001

# ===== 3. CORE MECHANICS =====

def stress_model(person: OptimizedPerson, shock_increment: float, params: SimulationParameters) -> float:
    """NEW: Update person's stress and return cooperation probability"""
    chronic_stress = person.update_stress(shock_increment, params)
    return person.calculate_cooperation_probability(params)

def cooperation_probability(person: OptimizedPerson, other: OptimizedPerson, params: SimulationParameters) -> bool:
    """Determine if person cooperates with other using realistic model"""
    return person.calculate_cooperation_decision(other, 0, params)

def update_relationship(person: OptimizedPerson, other: OptimizedPerson, cooperated: bool, round_num: int, params: SimulationParameters):
    """Update relationship between two people with realistic trust speed"""
    rel = person.get_relationship(other.id, round_num, other.group_id)
    rel.update_trust(cooperated, round_num, params.in_group_trust_modifier, params.out_group_trust_modifier)

def apply_community_buffer(person: OptimizedPerson, params: SimulationParameters):
    """Apply community buffer to reduce chronic stress - handled in stress model"""
    pass

# ===== 3. CORE MECHANICS =========================================
def schedule_interactions(
    population: List[OptimizedPerson],
    params: SimulationParameters,
    sim_ref: "Simulation",
    round_num: int,
) -> None:
    """
    Run one quarterly interaction cycle.

    • Tier 1  (deep ties): up to 12 explicit partner picks – births, betrayals, trust objects.
    • Tier 2  (weak ties): 40-80 encounters summarised via one binomial draw.
    • Tier 3  (stranger noise): hundreds of micro-contacts as a small stress jitter.
    """

    alive_people = [p for p in population if not p.is_dead]
    if len(alive_people) < 2:
        return

    # ---------- Tier 1 · deep-tie budget ---------------------------
    deep_per_person: int = max(1, min(12, int(0.05 * len(alive_people))))  # ≤12

    for person in alive_people:
        # Pre-filter partner pools once per person
        same_group_all  = [
            p for p in alive_people
            if p.id != person.id and getattr(p, "group_id", None) == getattr(person, "group_id", None)
        ]
        other_group_all = [
            p for p in alive_people
            if p.id != person.id and getattr(p, "group_id", None) != getattr(person, "group_id", None)
        ]

        # -------- Tier 1 explicit interactions ----------------------
        for _ in range(deep_per_person):
            # Serendipity vs homophily partner pick
            if random.random() < params.serendipity_rate and other_group_all:
                partner = random.choice(other_group_all + same_group_all)
            else:
                pool = same_group_all if (
                    same_group_all and (not other_group_all or random.random() < params.homophily_bias)
                ) else other_group_all
                if not pool:
                    continue
                # 30 % chance to favour known ties
                if person.relationships and random.random() < 0.3:
                    known = [p for p in pool if p.id in person.relationships]
                    partner = random.choice(known or pool)
                else:
                    partner = random.choice(pool)

            # Cooperation decisions
            person_coop  = person.calculate_cooperation_decision(partner, round_num, params)
            partner_coop = partner.calculate_cooperation_decision(person,  round_num, params)

            # Update relationship objects
            update_relationship(person,  partner,  partner_coop, round_num, params)
            update_relationship(partner, person,   person_coop,  round_num, params)

            # Count a defection if either side refused
            if not (person_coop and partner_coop):
                sim_ref.total_defections += 1

            # In-group / out-group counters
            if hasattr(person, "group_id") and hasattr(partner, "group_id"):
                if person.group_id == partner.group_id:
                    person.in_group_interactions  += 1
                    partner.in_group_interactions += 1
                else:
                    person.out_group_interactions  += 1
                    partner.out_group_interactions += 1

            # Mutual-cooperation benefits & possible birth
            if person_coop and partner_coop:
                person.maslow_needs.love  = min(10, person.maslow_needs.love  + 0.1)
                partner.maslow_needs.love = min(10, partner.maslow_needs.love + 0.1)

                pop_ratio  = len(alive_people) / params.max_population
                birth_rate = params.base_birth_rate * (1 - 0.3 * pop_ratio)
                if pop_ratio < 0.10:
                    birth_rate += 0.005  # safety bump for tiny societies
                if len(sim_ref.people) < params.max_population and random.random() < birth_rate:
                    sim_ref._create_birth(person, partner)

        # -------- Tier 2 · weak ties (aggregated) -------------------
        weak_cnt  = random.randint(40, 80)
        coop_prob = max(0.05, min(0.95, person.base_coop))
        weak_coop = np.random.binomial(weak_cnt, coop_prob)
        weak_betr = weak_cnt - weak_coop
        net_delta = weak_coop * TRUST_DELTA_HELP + weak_betr * TRUST_DELTA_BETRAY
        person.society_trust = max(0.0, min(1.0, getattr(person, "society_trust", 0.5) + net_delta))

        # -------- Tier 3 · stranger noise (stress nudge) ------------
        person.acute_stress = max(0.0, person.acute_stress + np.random.normal(0, 0.01))

    # -------- Trust noise drift (entropy tax) -----------------------
    for person in alive_people:
        for rel in person.relationships.values():
            rel.trust = max(
                0.0,
                min(1.0, rel.trust + np.random.normal(0, 0.005) - 0.001)
            )



# ===== 4. SHOCK ENGINE =====

def next_shock_timer(params: SimulationParameters) -> int:
    """Rounds until next crisis – exponential spacing."""
    wait_years = np.random.exponential(params.shock_mean_years)
    return int(wait_years * 4)          # 4 quarters = 1 year


def draw_shock_magnitude(params: SimulationParameters) -> float:
    """Pareto-tail shock severity using per-run α."""
    α = params.pareto_alpha
    u = 1.0 - random.random()           # uniform (0,1]
    return PARETO_XM * u ** (-1 / α)

class EnhancedMassSimulation:
    """Enhanced simulation with realistic parameters and preserved functionality"""
    
    def __init__(self, params: SimulationParameters, run_id: int):
        self.params = params
        self.run_id = run_id
        self.people: List[OptimizedPerson] = []
        self.round = 0
        self.system_stress = 0.0
        self.next_person_id = params.initial_population + 1
        
        # Realistic shock timing
        self.next_shock_round = next_shock_timer(params)
        
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
        
        # Inter-group tracking - all preserved
        self.group_names = [chr(65 + i) for i in range(params.num_groups)]
        self.in_group_interactions = 0
        self.out_group_interactions = 0
        self.total_mixing_events = 0
        self.successful_mixing_events = 0
        self.reputational_spillover_events = 0
        self.out_group_constraint_amplifications = 0
        
        # Interaction tracking
        self.total_interactions = 0
        
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with group distribution"""
        if hasattr(self.params, 'num_groups') and self.params.num_groups > 1:
            self._initialize_population_with_groups()
        else:
            for i in range(1, self.params.initial_population + 1):
                person = OptimizedPerson(i, self.params)
                self.people.append(person)
    
    def _initialize_population_with_groups(self):
        """Initialize population with specified group distribution"""
        group_sizes = []
        remaining_pop = self.params.initial_population
        
        for i, proportion in enumerate(self.params.founder_group_distribution):
            if i == len(self.params.founder_group_distribution) - 1:
                group_sizes.append(remaining_pop)
            else:
                size = int(self.params.initial_population * proportion)
                group_sizes.append(size)
                remaining_pop -= size
        
        person_id = 1
        for group_idx, group_size in enumerate(group_sizes):
            group_name = self.group_names[group_idx]
            for _ in range(group_size):
                person = OptimizedPerson(person_id, self.params, group_id=group_name)
                self.people.append(person)
                person_id += 1
        
        self.next_person_id = person_id
    
    def _trigger_shock(self):
        """Apply realistic system shock"""
        shock_severity = draw_shock_magnitude(self.params)
        self.system_stress += shock_severity
        self.shock_events += 1
        
        # Apply shock to all people
        for person in self.people:
            if not person.is_dead:
                # Apply shock with some protection from cooperation
                protection = 0
                if person.strategy == 'cooperative':
                    trusted_allies = sum(1 for r in person.relationships.values() 
                                       if r.trust > self.params.trust_threshold)
                    protection = min(0.5, trusted_allies * 0.05)
                
                effective_shock = shock_severity * (1 - protection)
                person.update_stress(effective_shock, self.params)
        
        # Schedule next shock
        self.next_shock_round = self.round + next_shock_timer(self.params)
    
    def _apply_reputational_spillover(self, defector: OptimizedPerson, alive_people: List[OptimizedPerson]):
        """Apply reputational spillover when someone defects"""
        if not hasattr(self.params, 'reputational_spillover') or self.params.reputational_spillover <= 0:
            return
        
        self.reputational_spillover_events += 1
        
        for person in alive_people:
            if person.id != defector.id and not person.is_dead:
                for other_id, relationship in person.relationships.items():
                    other = next((p for p in alive_people if p.id == other_id), None)
                    if other and hasattr(other, 'group_id') and other.group_id == defector.group_id:
                        relationship.trust = max(0.0, relationship.trust - self.params.reputational_spillover)
    
    def _is_mixing_event_round(self) -> bool:
        """Check if this round should have a mixing event"""
        return (hasattr(self.params, 'mixing_event_frequency') and 
                self.params.mixing_event_frequency > 0 and 
                self.round > 0 and 
                self.round % self.params.mixing_event_frequency == 0)
    
    def _handle_mixing_event(self, alive_people: List[OptimizedPerson]):
        """Handle inter-group mixing event"""
        self.total_mixing_events += 1
        
        if len(alive_people) < 2:
            return
        
        group_buckets = defaultdict(list)
        for person in alive_people:
            if hasattr(person, 'group_id') and not person.is_dead:
                group_buckets[person.group_id].append(person)
            elif not person.is_dead:
                group_buckets['default'].append(person)
        
        # Only proceed if we have multiple groups with people
        valid_groups = [group for group, people in group_buckets.items() if len(people) > 0]
        if len(valid_groups) < 2:
            return
        
        interactions_created = 0
        max_interactions = max(len(alive_people) // 3, 1)
        
        for _ in range(max_interactions):
            # Pick two different groups with people
            available_groups = [group for group in valid_groups if len(group_buckets[group]) > 0]
            if len(available_groups) < 2:
                break
            
            group1, group2 = random.sample(available_groups, 2)
            
            if group_buckets[group1] and group_buckets[group2]:
                person1 = random.choice(group_buckets[group1])
                person2 = random.choice(group_buckets[group2])
                
                if person1.is_dead or person2.is_dead:
                    continue
                
                # Mark as mixing event participation
                if hasattr(person1, 'mixing_event_participations'):
                    person1.mixing_event_participations += 1
                if hasattr(person2, 'mixing_event_participations'):
                    person2.mixing_event_participations += 1
                
                # Process interaction with enhanced cooperation bonus
                try:
                    success = self._process_interaction(person1, person2, is_mixing_event=True)
                    if success:
                        interactions_created += 1
                except Exception as e:
                    timestamp_print(f"⚠️  Error in mixing event interaction: {e}")
                    continue
        
        if interactions_created > 0:
            self.successful_mixing_events += 1
    
    def _process_interaction(self, person1: OptimizedPerson, person2: OptimizedPerson, 
                           is_mixing_event: bool = False) -> bool:
        """Process interaction with inter-group dynamics"""
        p1_cooperates = person1.calculate_cooperation_decision(person2, self.round, self.params)
        p2_cooperates = person2.calculate_cooperation_decision(person1, self.round, self.params)
        
        # Track interaction types
        if hasattr(person1, 'group_id') and hasattr(person2, 'group_id'):
            is_same_group = (person1.group_id == person2.group_id)
            if is_same_group:
                if hasattr(person1, 'in_group_interactions'):
                    person1.in_group_interactions += 1
                if hasattr(person2, 'in_group_interactions'):
                    person2.in_group_interactions += 1
                self.in_group_interactions += 1
            else:
                if hasattr(person1, 'out_group_interactions'):
                    person1.out_group_interactions += 1
                if hasattr(person2, 'out_group_interactions'):
                    person2.out_group_interactions += 1
                self.out_group_interactions += 1
        
        # Update relationships with group-weighted trust
        has_group_features = (hasattr(self.params, 'in_group_trust_modifier') and 
                             hasattr(self.params, 'out_group_trust_modifier'))
        
        if has_group_features:
            rel1 = person1.get_relationship(person2.id, self.round, person2.group_id)
            rel2 = person2.get_relationship(person1.id, self.round, person1.group_id)
            
            rel1.update_trust(p2_cooperates, self.round, 
                             self.params.in_group_trust_modifier, 
                             self.params.out_group_trust_modifier)
            rel2.update_trust(p1_cooperates, self.round, 
                             self.params.in_group_trust_modifier, 
                             self.params.out_group_trust_modifier)
        else:
            rel1 = person1.get_relationship(person2.id, self.round)
            rel2 = person2.get_relationship(person1.id, self.round)
            rel1.update_trust(p2_cooperates, self.round)
            rel2.update_trust(p1_cooperates, self.round)
        
        cooperation_bonus1 = 0
        cooperation_bonus2 = 0
        base_bonus = self.params.cooperation_bonus
        
        # Apply mixing event bonus
        if is_mixing_event and hasattr(self.params, 'mixing_event_bonus_multiplier'):
            base_bonus *= self.params.mixing_event_bonus_multiplier
        
        if p1_cooperates and p2_cooperates:
            cooperation_bonus1 = base_bonus
            cooperation_bonus2 = base_bonus
            self.cooperation_benefit_total += base_bonus * 2
            
            person1.maslow_needs.love = min(10, person1.maslow_needs.love + 0.1)
            person2.maslow_needs.love = min(10, person2.maslow_needs.love + 0.1)
            
        elif p1_cooperates and not p2_cooperates:
            self._apply_reputational_spillover(person2, [p for p in self.people if not p.is_dead])
            
            constraint_amount = 0.03
            
            is_from_out_group = (hasattr(person1, 'group_id') and hasattr(person2, 'group_id') and 
                                person1.group_id != person2.group_id)
            if is_from_out_group and hasattr(self.params, 'out_group_constraint_amplifier'):
                self.out_group_constraint_amplifications += 1
                person1.add_constraint_pressure(constraint_amount, is_from_out_group, 
                                              self.params.out_group_constraint_amplifier)
            else:
                person1.add_constraint_pressure(constraint_amount)
            
            person1.maslow_needs.esteem = max(0, person1.maslow_needs.esteem - 0.1)
            
        elif not p1_cooperates and p2_cooperates:
            self._apply_reputational_spillover(person1, [p for p in self.people if not p.is_dead])
            
            constraint_amount = 0.03
            
            is_from_out_group = (hasattr(person2, 'group_id') and hasattr(person1, 'group_id') and 
                                person2.group_id != person1.group_id)
            if is_from_out_group and hasattr(self.params, 'out_group_constraint_amplifier'):
                self.out_group_constraint_amplifications += 1
                person2.add_constraint_pressure(constraint_amount, is_from_out_group, 
                                              self.params.out_group_constraint_amplifier)
            else:
                person2.add_constraint_pressure(constraint_amount)
            
            person2.maslow_needs.esteem = max(0, person2.maslow_needs.esteem - 0.1)
        
        # Person updates
        person1.update(self.system_stress, self.params, cooperation_bonus1)
        person2.update(self.system_stress, self.params, cooperation_bonus2)
        
        return p1_cooperates and p2_cooperates
    
    def _check_recoveries(self):
        """Check for strategy recoveries"""
        alive_people = [p for p in self.people if not p.is_dead]
        for person in alive_people:
            if person.check_for_recovery(self.params):
                self.total_redemptions += 1
    
    def _create_birth(self, parent_a: OptimizedPerson, parent_b: OptimizedPerson):
        """Create new person with group inheritance"""
        new_person = OptimizedPerson(self.next_person_id, self.params, parent_a, parent_b)
        new_person.society_trust = 0.5
        self.people.append(new_person)
        self.total_births += 1
        self.next_person_id += 1
    
    def _check_cascade(self):
        """Check for cascade conditions"""
        alive_people = [p for p in self.people if not p.is_dead]
        cooperative = [p for p in alive_people if p.strategy == 'cooperative']
        selfish = [p for p in alive_people if p.strategy == 'selfish']
        
        if len(cooperative) > 0 and len(selfish) >= len(cooperative):
            if self.first_cascade_round is None:
                self.first_cascade_round = self.round
            self.cascade_events += 1
            
            for person in cooperative:
                coop_allies = sum(1 for other_id, rel in person.relationships.items()
                                if rel.trust > 0.5 and any(p.id == other_id and p.strategy == 'cooperative' 
                                                         for p in alive_people))
                protection = min(0.5, coop_allies * 0.1)
                cascade_pressure = 0.2 * (1 - protection)
                person.add_constraint_pressure(cascade_pressure)
    
    def _update_population(self):
        """Update population state"""
        initial_count = len(self.people)
        self.people = [p for p in self.people if not p.is_dead]
        self.total_deaths += initial_count - len(self.people)
        
        for person in self.people:
            person.update(self.system_stress, self.params)
    
    def _collect_round_data(self):
        """Lightweight data collection"""
        alive_people = [p for p in self.people if not p.is_dead]
        self.system_stress_history.append(self.system_stress)
        self.population_history.append(len(alive_people))
    
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
    
    def _calculate_segregation_index(self) -> float:
        """Calculate how segregated the groups became"""
        total_interactions = self.in_group_interactions + self.out_group_interactions
        
        if total_interactions == 0:
            return 0.0
        
        return self.in_group_interactions / total_interactions
    
    def _calculate_trust_levels(self) -> Tuple[float, float]:
        """Calculate average in-group and out-group trust levels"""
        alive_people = [p for p in self.people if not p.is_dead]
        in_group_trusts = []
        out_group_trusts = []
        
        for person in alive_people:
            for rel in person.relationships.values():
                if hasattr(rel, 'is_same_group'):
                    if rel.is_same_group:
                        in_group_trusts.append(rel.trust)
                    else:
                        out_group_trusts.append(rel.trust)
                else:
                    in_group_trusts.append(rel.trust)
        
        avg_in_group = sum(in_group_trusts) / len(in_group_trusts) if in_group_trusts else 0.5
        avg_out_group = sum(out_group_trusts) / len(out_group_trusts) if out_group_trusts else 0.5
        
        return avg_in_group, avg_out_group
    
    def run_simulation(self) -> EnhancedSimulationResults:
        """Run enhanced simulation with realistic parameters"""
        timestamp_print(f"🎮 Starting realistic simulation run {self.run_id}")
        
        try:
            initial_trait_avg = self._get_average_traits()
            initial_group_populations = self._get_group_populations()
            
            while self.round < self.params.max_rounds:
                self.round += 1
                
                alive_people = [p for p in self.people if not p.is_dead]
                if len(alive_people) == 0:
                    timestamp_print(f"💀 Sim {self.run_id}: Population extinct at round {self.round}")
                    break
                
                # Check for mixing event
                if self._is_mixing_event_round():
                    self._handle_mixing_event(alive_people)
                
                # Check for realistic shocks
                if self.round >= self.next_shock_round:
                    self._trigger_shock()
                
                # Process interactions with realistic mechanics
                schedule_interactions(self.people, self.params, self, self.round)
                
                self._check_recoveries()
                self._update_population()
                self._collect_round_data()
                
                self.system_stress = max(0, self.system_stress - 0.01)

                if self.round == 1 or self.round % 20 == 0:
                    timestamp_print(f"✅ Sim {self.run_id}: completed round {self.round:3d} "
                        f"(pop={len(alive_people):4d}, defections={self.total_defections})")


            
            return self._generate_results(initial_trait_avg, initial_group_populations)
            
        except Exception as sim_error:
            timestamp_print(f"❌ Critical error in simulation {self.run_id}: {sim_error}")
            return self._generate_emergency_result()
    
    def _generate_results(self, initial_traits: Dict[str, float], 
                         initial_group_populations: Dict[str, int]) -> EnhancedSimulationResults:
        """Generate comprehensive results"""
        alive_people = [p for p in self.people if not p.is_dead]
        cooperative = [p for p in alive_people if p.strategy == 'cooperative']
        constrained = [p for p in alive_people if p.is_constrained]
        
        final_traits = self._get_average_traits()
        trait_evolution = {k: final_traits[k] - initial_traits[k] for k in initial_traits.keys()}
        
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
        if hasattr(self.params, 'num_groups') and self.params.num_groups > 1:
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
        
        return EnhancedSimulationResults(
            parameters=self.params,
            run_id=self.run_id,
            
            # All original metrics preserved
            final_population=len(alive_people),
            final_cooperation_rate=len(cooperative) / max(1, len(alive_people)),
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
            cooperation_resilience=len(cooperative) / max(1, len(alive_people)),
            
            # Inter-Group Metrics - all preserved
            final_group_populations=final_group_populations,
            final_group_cooperation_rates=final_group_cooperation_rates,
            in_group_interaction_rate=in_group_rate,
            out_group_interaction_rate=out_group_rate,
            avg_in_group_trust=avg_in_group_trust,
            avg_out_group_trust=avg_out_group_trust,
            group_segregation_index=self._calculate_segregation_index(),
            total_mixing_events=self.total_mixing_events,
            mixing_event_success_rate=mixing_success_rate,
            reputational_spillover_events=self.reputational_spillover_events,
            out_group_constraint_amplifications=self.out_group_constraint_amplifications,
            group_extinction_events=group_extinctions,
            trust_asymmetry=trust_asymmetry,
            
            # Realistic interaction metrics
            total_interactions=total_interactions,
            avg_interaction_processing_time=0.0
        )
    
    def _generate_emergency_result(self) -> EnhancedSimulationResults:
        """Generate emergency result object when simulation fails"""
        timestamp_print(f"🚨 Generating emergency result for failed simulation {self.run_id}")
        
        try:
            alive_people = [p for p in self.people if not p.is_dead]
            final_traits = self._get_average_traits()
            final_group_populations = self._get_group_populations()
            
            return EnhancedSimulationResults(
                parameters=self.params,
                run_id=self.run_id,
                final_population=len(alive_people),
                final_cooperation_rate=0.0,
                final_constrained_rate=1.0,
                rounds_completed=self.round,
                extinction_occurred=True,
                first_cascade_round=self.round,
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
                initial_needs_avg=final_traits,
                final_needs_avg=final_traits,
                needs_improvement={k: 0 for k in final_traits.keys()},
                avg_trust_level=0.5,
                cooperation_benefit_total=0.0,
                population_growth=1.0,
                cooperation_resilience=0.0,
                final_group_populations=final_group_populations,
                final_group_cooperation_rates={k: 0.0 for k in final_group_populations.keys()},
                total_interactions=0,
            )
        except Exception as emergency_error:
            timestamp_print(f"❌ Even emergency result generation failed: {emergency_error}")
            return EnhancedSimulationResults(
                parameters=self.params,
                run_id=self.run_id,
                final_population=0,
                final_cooperation_rate=0.0,
                final_constrained_rate=1.0,
                rounds_completed=self.round,
                extinction_occurred=True,
                first_cascade_round=0,
                total_cascade_events=0,
                total_shock_events=0,
                total_defections=0,
                total_redemptions=0,
                net_strategy_change=0,
                total_births=0,
                total_deaths=0,
                max_population_reached=0,
                population_stability=0.0,
                avg_system_stress=0.0,
                max_system_stress=0.0,
                avg_maslow_pressure=0.0,
                avg_basic_needs_crisis_rate=0.0,
                initial_needs_avg={},
                final_needs_avg={},
                needs_improvement={},
                avg_trust_level=0.5,
                cooperation_benefit_total=0.0,
                population_growth=1.0,
                cooperation_resilience=0.0,
                final_group_populations={},
                final_group_cooperation_rates={},
                total_interactions=0,
            )

def latin_hypercube_sampler(n_samples: int, n_replicates: int = 1) -> List[SimulationParameters]:
    """Generate parameter sets using Latin Hypercube Sampling"""
    timestamp_print(f"🎲 Generating {n_samples} parameter sets with LHC sampling")
    
    # Parameter ranges for realistic sampling
    ranges = {
        'initial_population': (100, 500),
        'pareto_alpha': PARETO_ALPHA_RANGE,
        'community_buffer_factor': (COMMUNITY_BUFFER_MIN, COMMUNITY_BUFFER_MAX),
        'base_birth_rate': (0.006, 0.012),
        'cooperation_bonus': (0.1, 0.4),
        'homophily_bias': (0.0, 0.8),
        'maslow_variation': (0.3, 0.7),
        'recovery_threshold': (0.2, 0.5),
        'out_group_constraint_amplifier': (1.1, 1.3),  # Realistic range
        'out_group_trust_modifier': (0.8, 0.9),  # Realistic range
        'reputational_spillover': (0.0, 0.15),
        'mixing_event_frequency': (10, 25),
    }
    
    # Generate samples
    samples = []
    
    for rep in range(n_replicates):
        for i in range(n_samples):
            # Create base parameters
            params = SimulationParameters(initial_population=200)
            
            # Sample each parameter using LHC
            for param_name, (min_val, max_val) in ranges.items():
                # Latin hypercube: divide range into n_samples segments
                segment_size = (max_val - min_val) / n_samples
                segment_start = min_val + i * segment_size
                segment_end = segment_start + segment_size
                
                # Random value within this segment
                value = segment_start + random.random() * (segment_end - segment_start)
                setattr(params, param_name, value)
            
            # Set discrete parameters
            params.num_groups = random.choice([1, 2, 3])
            if params.num_groups == 1:
                params.homophily_bias = 0.0
                params.founder_group_distribution = [1.0]
            elif params.num_groups == 2:
                split = 0.4 + random.random() * 0.2
                params.founder_group_distribution = [split, 1.0 - split]
            else:  # 3 groups
                params.founder_group_distribution = [0.4, 0.35, 0.25]
            
            # Set realistic shock timing
            shock_min, shock_max = SHOCK_INTERVAL_YEARS
            params.shock_interval_years = (shock_min + random.random() * (shock_max - shock_min - 5), 
                                         shock_min + 5 + random.random() * (shock_max - shock_min - 5))
            
            samples.append(params)
    
    return samples

def generate_random_parameters(run_id: int) -> SimulationParameters:
    """Generate randomized simulation parameters with realistic constraints"""
    timestamp_print(f"🎲 Generating realistic parameters for sim {run_id}")
    initial_pop = random.randint(100, 500)
    
    # Decide whether to include inter-group dynamics (80% chance)
    include_intergroup = random.random() < 0.8
    
    if include_intergroup:
        num_groups = random.choice([2, 3])
        if num_groups == 2:
            split = 0.4 + random.random() * 0.2
            group_dist = [split, 1.0 - split]
        else:
            group_dist = [0.4, 0.35, 0.25]
        
        # Realistic shock timing
        shock_min = 5 + random.random() * 10
        shock_max = shock_min + 10 + random.random() * 10
        shock_mean = (shock_min + shock_max) / 2
        
        params = SimulationParameters(
            initial_population=initial_pop,
            max_population=MAX_POPULATION,
            max_rounds=DEFAULT_ROUNDS,
            
            # Realistic shock parameters
            shock_interval_years=(shock_min, shock_max),
            shock_mean_years=shock_mean,
            pareto_alpha=1.8 + random.random() * 0.7,
            
            # Realistic community buffer
            community_buffer_factor=COMMUNITY_BUFFER_MIN + random.random() * (COMMUNITY_BUFFER_MAX - COMMUNITY_BUFFER_MIN),
            
            # Other realistic parameters
            base_birth_rate=0.006 + random.random() * 0.006,
            maslow_variation=0.3 + random.random() * 0.4,
            recovery_threshold=0.2 + random.random() * 0.3,
            cooperation_bonus=0.1 + random.random() * 0.3,
            
            # Realistic inter-group parameters
            num_groups=num_groups,
            founder_group_distribution=group_dist,
            homophily_bias=random.random() * 0.8,
            in_group_trust_modifier=1.0 + random.random() * 0.5,
            out_group_trust_modifier=0.8 + random.random() * 0.1,  # Realistic range
            out_group_constraint_amplifier=1.1 + random.random() * 0.2,  # Realistic range
            reputational_spillover=random.random() * 0.15,
            mixing_event_frequency=random.choice([10, 15, 20, 25]),
            mixing_event_bonus_multiplier=1.5 + random.random() * 1.0,
            inheritance_style=random.choice(["mother", "father", "random"]),
        )
    else:
        # Realistic shock timing
        shock_min = 5 + random.random() * 10
        shock_max = shock_min + 10 + random.random() * 10
        shock_mean = (shock_min + shock_max) / 2
        
        params = SimulationParameters(
            initial_population=initial_pop,
            max_population=MAX_POPULATION,
            max_rounds=DEFAULT_ROUNDS,
            
            # Realistic shock parameters
            shock_interval_years=(shock_min, shock_max),
            shock_mean_years=shock_mean,
            pareto_alpha=1.8 + random.random() * 0.7,
            
            # Realistic community buffer
            community_buffer_factor=COMMUNITY_BUFFER_MIN + random.random() * (COMMUNITY_BUFFER_MAX - COMMUNITY_BUFFER_MIN),
            
            # Other realistic parameters
            base_birth_rate=0.006 + random.random() * 0.006,
            maslow_variation=0.3 + random.random() * 0.4,
            recovery_threshold=0.2 + random.random() * 0.3,
            cooperation_bonus=0.1 + random.random() * 0.3,
            
            # Minimal inter-group
            num_groups=1,
            founder_group_distribution=[1.0],
            homophily_bias=0.0,
            in_group_trust_modifier=1.0,
            out_group_trust_modifier=1.0,
            out_group_constraint_amplifier=1.0,
            reputational_spillover=0.0,
            mixing_event_frequency=0,
            mixing_event_bonus_multiplier=1.0,
            inheritance_style="mother",
        )
    
    return params

# ===== 5. RUNNER =====

def run_single_simulation(run_id: int) -> EnhancedSimulationResults:
    """Run a single simulation with realistic parameters"""
    timestamp_print(f"🔄 Starting realistic simulation {run_id}")
    params = generate_random_parameters(run_id)
    sim = EnhancedMassSimulation(params, run_id)
    result = sim.run_simulation()
    timestamp_print(f"✅ Completed realistic simulation {run_id}")
    return result

# Load balancing system preserved
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
    """Load balancing with realistic parameters"""
    
    def __init__(self, params_list: List[SimulationParameters], chunk_size: int = 30, results_dir: str = "simulation_results"):
        timestamp_print(f"⚙️  Initializing LoadBalancedScheduler with {len(params_list)} simulations...")
        
        self.params_list = params_list  # Store the parameter list
        self.chunk_size = chunk_size
        self.results_dir = results_dir
        self.work_queue = queue.Queue()
        self.completed_simulations = set()
        self.active_simulations = {}
        self.simulation_states = {}
        self.lock = threading.Lock()
        
        # Create results directory
        timestamp_print(f"📁 Setting up results directory: {results_dir}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            timestamp_print(f"✅ Created results directory: {results_dir}")
        
        # Initialize work queue with progress tracking
        timestamp_print(f"🔄 Generating work queue for {len(params_list)} simulations...")
        for i, params in enumerate(params_list):
            if i % 25 == 0:  # Log every 25 items
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
        
        timestamp_print(f"✅ LoadBalancedScheduler initialized with {self.work_queue.qsize()} work items")
    
    def _estimate_complexity(self, params: SimulationParameters) -> float:
        """Estimate simulation complexity"""
        base = params.initial_population ** 1.3 * (params.max_rounds / 100)
        
        # Inter-group complexity multiplier
        if hasattr(params, 'num_groups') and params.num_groups > 1:
            intergroup_factor = params.num_groups * 1.5
            base *= intergroup_factor
        
        return base
    
    def get_work_with_params(self) -> Optional[Tuple[SimulationWork, SimulationParameters]]:
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
    
    def is_complete(self) -> bool:
        """Check if all simulations are done"""
        with self.lock:
            return (len(self.completed_simulations) == len(self.params_list) and 
                    self.work_queue.empty())

def process_simulation_work(work_and_params: tuple) -> tuple:
    """Process a single work item with provided parameters"""
    work, provided_params = work_and_params
    start_time = time.time()
    
    try:
        if work.is_new_simulation:
            # Use provided parameters instead of generating new ones
            if provided_params is not None:
                params = provided_params
            else:
                params = generate_random_parameters(work.sim_id)
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
            
            # Check for mixing event
            if sim._is_mixing_event_round():
                sim._handle_mixing_event(alive_people)
            
            # Check for realistic shocks
            if sim.round >= sim.next_shock_round:
                sim._trigger_shock()
            
            # Process interactions
            schedule_interactions(sim.people, sim.params, sim, sim.round)
            
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
        traceback.print_exc()  # Add full traceback for debugging
        return ('error', work.sim_id, str(e), 0, 0)

def run_smart_mass_experiment(params_list: List[SimulationParameters], use_multiprocessing: bool = False) -> List[EnhancedSimulationResults]:
    """FIXED: Load-balanced mass experiment that uses provided parameters"""
    num_simulations = len(params_list)
    timestamp_print(f"🚀 Starting realistic mass experiment with {num_simulations} simulations...")
    timestamp_print(f"🔧 Using provided parameter list instead of generating new ones")
    
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
    
    # Multi-processing approach
    num_cores = min(mp.cpu_count(), 8)
    timestamp_print(f"🔧 Using {num_cores} CPU cores for multiprocessing...")
    
    results_dir = "simulation_results"
    timestamp_print(f"📁 Setting up scheduler with results directory: {results_dir}")
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
                active_futures[future] = work_with_params[0]  # Store just the work item
        
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
                        
                        break  # Process one at a time
                        
                except TimeoutError:
                    pass
            
            # ---------- 60-second progress heartbeat -------------------------
            now = time.time()
            if now - last_log >= 60:
                done, total = scheduler.get_progress()
                pct_done = 100.0 * done / total if total > 0 else 0

                # Build per-sim round percentages
                with scheduler.lock:
                    active_strings = [
                        f"{sid}:{round_num}"
                        for sid, round_num in scheduler.active_simulations.items()
                    ]
                active_report = ", ".join(active_strings[:10]) or "none"  # Limit to first 10
                if len(active_strings) > 10:
                    active_report += f" +{len(active_strings)-10} more"

                timestamp_print(
                    f"📊 PROGRESS: {done}/{total} complete "
                    f"({pct_done:5.1f}%) • active sims: [{active_report}]"
                )
                last_log = now
            # -----------------------------------------------------------------

    
    # Load all results
    timestamp_print("📂 Loading all completed results...")
    final_results = []
    
    for i in range(num_simulations):
        try:
            with open(f"{results_dir}/sim_{i:04d}_result.pkl", 'rb') as f:
                result = pickle.load(f)
                final_results.append(result)
        except Exception as e:
            timestamp_print(f"⚠️ Could not load result {i}: {e}")
    
    elapsed = time.time() - start_time
    timestamp_print(f"🎉 EXPERIMENT COMPLETE: {len(final_results)} simulations in {elapsed:.2f} seconds")
    
    return final_results

# ===== 6. CLI ENTRYPOINT =====

def main():
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description='Enhanced Constraint Cascade Simulation - Realistic Parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REALISTIC IMPROVEMENTS:
- Shock frequency: 5-25 years (instead of multiple per year)
- Trust development: ±0.04/±0.06 (instead of ±0.1/±0.15)
- Acute/chronic stress model with community buffer
- Serendipity rate: 10% interactions ignore homophily
- Improved recovery rates with social support
- Latin hypercube parameter sampling
- All original functionality preserved

Examples:
  python constraint_simulation.py --test quick
  python constraint_simulation.py --runs 50 --multiprocessing
  python constraint_simulation.py --sweep resilience --design lhc
        """
    )
    
    parser.add_argument('--test', choices=['quick', 'smoke', 'batch'], 
                       help='Run test suite')
    parser.add_argument('-n', '--runs', type=int, default=50,
                       help='Number of simulation runs')
    parser.add_argument('--sweep', choices=['resilience'], 
                       help='Run parameter sweep')
    parser.add_argument('--design', choices=['lhc', 'random'], default='lhc',
                       help='Parameter sampling design')
    parser.add_argument('--repeats', type=int, default=1,
                       help='Replicates per parameter set')
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
    
    timestamp_print("🔬 Enhanced Constraint Cascade Simulation - REALISTIC PARAMETERS")
    timestamp_print("="*80)
    timestamp_print("🎯 Realistic shock frequency: 5-25 years (instead of multiple per year)")
    timestamp_print("🤝 Realistic trust development: ±0.04/±0.06 (instead of ±0.1/±0.15)")
    timestamp_print("🧠 Acute/chronic stress model with community buffer")
    timestamp_print("🎲 Serendipity rate: 10% interactions ignore homophily")
    timestamp_print("♻️ Improved recovery rates with social support")
    timestamp_print("📊 Latin hypercube parameter sampling")
    timestamp_print("✅ All original functionality preserved")
    
    if args.sweep:
        run_parameter_sweep(args)
    else:
        run_basic_experiment(args, use_multiprocessing)

def run_tests(test_type: str):
    """Run test suite"""
    timestamp_print(f"Running {test_type} tests...")
    
    if test_type == 'quick':
        # Quick unit tests
        params = SimulationParameters(initial_population=100)
        person = OptimizedPerson(1, params)
        
        # Test stress model
        old_acute = person.acute_stress
        person.update_stress(0.1, params)
        assert person.acute_stress > old_acute
        
        # Test relationship
        other = OptimizedPerson(2, params, group_id="B")
        rel = person.get_relationship(other.id, 1, other.group_id)
        assert rel.trust == 0.5
        
        rel.update_trust(True, 1)
        assert rel.trust > 0.5
        
        # Test cooperation probability
        coop_prob = person.calculate_cooperation_probability(params)
        assert 0 <= coop_prob <= 1
        
        timestamp_print("✅ Quick tests passed")
    
    elif test_type == 'smoke':
        # Smoke test
        params = SimulationParameters(initial_population=50)
        params.max_rounds = 20
        
        result = run_single_simulation(0)
        assert result.rounds_completed > 0
        assert result.final_population >= 0
        
        timestamp_print("✅ Smoke test passed")
    
    elif test_type == 'batch':
        # Batch test
        start_time = time.time()
        
        results = []
        for i in range(10):
            params = SimulationParameters(initial_population=100)
            params.max_rounds = 50
            sim = EnhancedMassSimulation(params, i)
            result = sim.run_simulation()
            results.append(result)
        
        elapsed = time.time() - start_time
        timestamp_print(f"✅ Batch test completed: {len(results)} simulations in {elapsed:.1f}s")

def run_basic_experiment(args, use_multiprocessing: bool):
    """FIXED: Run basic experiment with proper parameter generation"""
    timestamp_print(f"🎯 Running {args.runs} simulations...")
    timestamp_print(f"📊 Design strategy: {'3x3 grid sweep' if args.design == 'random' else args.design}")
    
    # FIXED: Generate parameters based on design choice
    if args.design == 'random':
        # Use 3-factor grid: λ × α × buffer (3 × 3 × 3 = 27 combos)
        timestamp_print("🔧 Using 3x3x3 parameter grid instead of random generation")
        grid = list(product(SHOCK_MEAN_YEARS_SET, PARETO_ALPHA_SET, COMMUNITY_BUFFER_SET))
        
        # Calculate replicates per combo to reach target runs
        replicates = max(1, args.runs // len(grid))
        params_list = []
        
        timestamp_print(f"📋 Grid has {len(grid)} combinations, using {replicates} replicates each")
        
        run_id = 0
        for combo_idx, (shock_mean_years, pareto_alpha, community_buffer) in enumerate(grid):
            timestamp_print(f"   🔄 Generating combo {combo_idx+1}/{len(grid)}: "
                          f"shock={shock_mean_years}yr, α={pareto_alpha}, buffer={community_buffer}")
            
            for rep in range(replicates):
                # Generate base random parameters first
                p = generate_random_parameters(run_id)
                
                # Then override with grid values
                p.shock_mean_years = shock_mean_years
                p.pareto_alpha = pareto_alpha
                p.community_buffer_factor = community_buffer
                
                params_list.append(p)
                run_id += 1
        
        # Pad or trim to exact target
        while len(params_list) < args.runs:
            extra_combo = random.choice(grid)
            p = generate_random_parameters(run_id)
            p.shock_mean_years, p.pareto_alpha, p.community_buffer_factor = extra_combo
            params_list.append(p)
            run_id += 1
        
        params_list = params_list[:args.runs]
        
    elif args.design == 'lhc':
        timestamp_print("🎲 Using Latin Hypercube sampling")
        params_list = latin_hypercube_sampler(args.runs, args.repeats)
    else:
        timestamp_print("🎲 Using basic random parameter generation")
        params_list = [generate_random_parameters(i) for i in range(args.runs)]
    
    timestamp_print(f"✅ Generated {len(params_list)} parameter sets")
    
    # Run simulations with the generated parameters
    results = run_smart_mass_experiment(params_list, use_multiprocessing)
    
    # Analyze and save results
    timestamp_print(f"📊 Analyzing {len(results)} simulation results...")
    df = analyze_emergent_patterns(results)
    
    timestamp_print(f"📈 Creating visualizations...")
    create_pattern_visualizations(df)
    
    timestamp_print(f"🎯 Identifying critical thresholds...")
    thresholds = identify_critical_thresholds(df)
    
    timestamp_print(f"💾 Saving comprehensive results...")
    saved_files = save_comprehensive_results(df, thresholds)
    
    timestamp_print(f"✅ Experiment completed: {len(saved_files)} files saved")

def run_parameter_sweep(args):
    """Run parameter sweep"""
    timestamp_print(f"Running parameter sweep: {args.sweep}")
    
    if args.sweep == 'resilience':
        # Resilience sweep
        timestamp_print("🔄 Generating resilience sweep parameters...")
        params_list = []
        
        shock_intervals = [(5, 15), (10, 20), (15, 25), (20, 30)]
        buffer_factors = [0.05, 0.1, 0.15, 0.2]
        
        timestamp_print(f"📋 Sweep grid: {len(shock_intervals)} shock intervals × {len(buffer_factors)} buffers × {args.repeats} repeats")
        
        for shock_interval in shock_intervals:
            for buffer_factor in buffer_factors:
                for rep in range(args.repeats):
                    params = SimulationParameters(initial_population=200)
                    params.shock_interval_years = shock_interval
                    params.community_buffer_factor = buffer_factor
                    params_list.append(params)
        
        timestamp_print(f"✅ Generated {len(params_list)} parameter sets for resilience sweep")
        
        # Run sweep
        results = run_smart_mass_experiment(params_list, True)  # Always use multiprocessing for sweeps
        
        # Analyze results
        df = analyze_emergent_patterns(results)
        create_pattern_visualizations(df)
        thresholds = identify_critical_thresholds(df)
        save_comprehensive_results(df, thresholds)
        
        timestamp_print(f"✅ Resilience sweep completed: {len(results)} results")

# ===== 7. UTILITIES & METRICS =====

def analyze_emergent_patterns(results: List[EnhancedSimulationResults]) -> pd.DataFrame:
    """Analyze results for emergent patterns with realistic parameters"""
    timestamp_print("🔍 Analyzing emergent patterns with realistic parameters...")
    
    # Convert results to DataFrame
    data = []
    for result in results:
        try:
            # Safe division to prevent division by zero
            final_pop = max(result.final_population, 1)
            
            row = {
                'run_id': result.run_id,
                
                # All original parameters preserved with safe access
                'initial_population': result.parameters.initial_population,
                'max_population': result.parameters.max_population,
                'pop_multiplier': result.parameters.max_population / max(result.parameters.initial_population, 1),
                'base_birth_rate': result.parameters.base_birth_rate,
                'max_rounds': result.parameters.max_rounds,
                'maslow_variation': result.parameters.maslow_variation,
                'recovery_threshold': result.parameters.recovery_threshold,
                'cooperation_bonus': result.parameters.cooperation_bonus,
                'trust_threshold': result.parameters.trust_threshold,
                'relationship_memory': getattr(result.parameters, 'relationship_memory', REL_WINDOW_LEN),
                
                # NEW: Realistic shock parameters with safe access
                'shock_interval_min': result.parameters.shock_interval_years[0] if hasattr(result.parameters, 'shock_interval_years') else 0,
                'shock_interval_max': result.parameters.shock_interval_years[1] if hasattr(result.parameters, 'shock_interval_years') else 0,
                'shock_interval_avg': (sum(result.parameters.shock_interval_years) / 2) if hasattr(result.parameters, 'shock_interval_years') else 0,
                'shock_mean_years': getattr(result.parameters, 'shock_mean_years', 10),
                'pareto_alpha': result.parameters.pareto_alpha,
                'pareto_xm': result.parameters.pareto_xm,
                
                # NEW: Realistic trust parameters
                'trust_delta_help': result.parameters.trust_delta_help,
                'trust_delta_betray': result.parameters.trust_delta_betray,
                'serendipity_rate': result.parameters.serendipity_rate,
                
                # NEW: Community buffer parameters
                'community_buffer_factor': result.parameters.community_buffer_factor,
                'acute_decay': result.parameters.acute_decay,
                'chronic_window': result.parameters.chronic_window,
                
                # All original outcomes preserved
                'final_cooperation_rate': result.final_cooperation_rate,
                'final_constrained_rate': result.final_constrained_rate,
                'final_population': result.final_population,
                'extinction_occurred': result.extinction_occurred,
                'first_cascade_round': result.first_cascade_round if result.first_cascade_round else result.rounds_completed,
                'cascade_events': result.total_cascade_events,
                'shock_events': result.total_shock_events,
                'population_growth': result.population_growth,
                'population_stability': result.population_stability,
                'cooperation_resilience': result.cooperation_resilience,
                'rounds_completed': result.rounds_completed,
                'total_defections': result.total_defections,
                'total_redemptions': result.total_redemptions,
                'redemption_rate': result.total_redemptions / max(1, result.total_defections),
                'avg_trust_level': result.avg_trust_level,
                'cooperation_benefit_total': result.cooperation_benefit_total,
                
                # Maslow changes preserved with safe access
                'physiological_change': result.needs_improvement.get('physiological', 0) if hasattr(result, 'needs_improvement') else 0,
                'safety_change': result.needs_improvement.get('safety', 0) if hasattr(result, 'needs_improvement') else 0,
                'love_change': result.needs_improvement.get('love', 0) if hasattr(result, 'needs_improvement') else 0,
                'esteem_change': result.needs_improvement.get('esteem', 0) if hasattr(result, 'needs_improvement') else 0,
                'self_actualization_change': result.needs_improvement.get('self_actualization', 0) if hasattr(result, 'needs_improvement') else 0,
                
                # Inter-group parameters (when available)
                'num_groups': getattr(result.parameters, 'num_groups', 1),
                'homophily_bias': getattr(result.parameters, 'homophily_bias', 0.0),
                'in_group_trust_modifier': getattr(result.parameters, 'in_group_trust_modifier', 1.0),
                'out_group_trust_modifier': getattr(result.parameters, 'out_group_trust_modifier', 1.0),
                'out_group_constraint_amplifier': getattr(result.parameters, 'out_group_constraint_amplifier', 1.0),
                'reputational_spillover': getattr(result.parameters, 'reputational_spillover', 0.0),
                'mixing_event_frequency': getattr(result.parameters, 'mixing_event_frequency', 0),
                'mixing_event_bonus_multiplier': getattr(result.parameters, 'mixing_event_bonus_multiplier', 1.0),
                
                # Inter-group outcomes (when available)
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
                
                # Realistic interaction metrics
                'total_interactions': result.total_interactions,
                'interaction_intensity': result.total_interactions / final_pop,
            }
            data.append(row)
            
        except Exception as e:
            timestamp_print(f"⚠️ Error processing result {result.run_id}: {e}")
            continue
    
    if not data:
        timestamp_print("⚠️  No simulation data to analyze")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Create outcome categories with safe access
    try:
        df['outcome_category'] = pd.cut(df['final_cooperation_rate'], 
                                       bins=[0, 0.1, 0.3, 0.7, 1.0],
                                       labels=['Collapse', 'Low_Coop', 'Medium_Coop', 'High_Coop'])
        
        df['extinction_category'] = df['extinction_occurred'].map({True: 'Extinct', False: 'Survived'})
        
        # Inter-group specific categories
        df['has_intergroup'] = df['num_groups'] > 1
        df['segregation_level'] = pd.cut(df['group_segregation_index'], 
                                       bins=[0, 0.3, 0.7, 1.0],
                                       labels=['Integrated', 'Moderate', 'Highly_Segregated'])
        
        df['trust_asymmetry_level'] = pd.cut(df['trust_asymmetry'], 
                                           bins=[-1, 0.1, 0.3, 1.0],
                                           labels=['Low', 'Medium', 'High'])
        
        # Calculate derived metrics
        df['shock_frequency_proxy'] = 1 / df['shock_interval_avg'].replace(0, 1)  # Avoid division by zero
        df['growth_potential'] = df['base_birth_rate'] * df['pop_multiplier']
        df['resilience_index'] = df['community_buffer_factor'] * (1 - df['shock_frequency_proxy'])
        
        # Inter-group tension index
        df['intergroup_tension'] = (df['out_group_constraint_amplifier'] * 
                                   df['reputational_spillover'] * 
                                   (1 - df['out_group_trust_modifier']))
        
        # Realistic stress indices
        df['stress_recovery_rate'] = 1 - df['acute_decay']
        df['social_support_effectiveness'] = df['community_buffer_factor'] * df['avg_trust_level']
        
    except Exception as e:
        timestamp_print(f"⚠️ Error creating derived columns: {e}")
    
    return df

def create_pattern_visualizations(df: pd.DataFrame):
    """Create comprehensive pattern analysis visualizations for realistic parameters"""
    timestamp_print("📊 Creating realistic parameter visualization analysis...")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # Determine how many plots we need
        has_intergroup_data = df['has_intergroup'].any() if 'has_intergroup' in df.columns else False
        
        if has_intergroup_data:
            rows, cols = 5, 4
        else:
            rows, cols = 4, 4
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(24, rows * 5))
        
        # 1. Realistic Shock Frequency vs Cooperation
        ax1 = plt.subplot(rows, cols, 1)
        if 'shock_interval_avg' in df.columns and 'final_cooperation_rate' in df.columns:
            scatter = plt.scatter(df['shock_interval_avg'], df['final_cooperation_rate'], 
                                 c=df['pareto_alpha'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='Pareto Alpha')
            plt.xlabel('Average Shock Interval (years)')
            plt.ylabel('Final Cooperation Rate')
            plt.title('Realistic Shock Frequency vs Cooperation\n(Color = Shock Severity Distribution)')
        
        # 2. Community Buffer Effectiveness
        ax2 = plt.subplot(rows, cols, 2)
        if 'community_buffer_factor' in df.columns and 'final_cooperation_rate' in df.columns:
            scatter = plt.scatter(df['community_buffer_factor'], df['final_cooperation_rate'], 
                                 c=df['avg_trust_level'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Average Trust Level')
            plt.xlabel('Community Buffer Factor')
            plt.ylabel('Final Cooperation Rate')
            plt.title('Community Buffer vs Cooperation\n(Color = Trust Level)')
        
        # 3. Trust Development Speed vs Outcomes
        ax3 = plt.subplot(rows, cols, 3)
        if 'trust_delta_help' in df.columns and 'trust_delta_betray' in df.columns:
            # Create trust development speed metric
            trust_speed = df['trust_delta_help'] / (-df['trust_delta_betray'])
            scatter = plt.scatter(trust_speed, df['final_cooperation_rate'], 
                                 c=df['redemption_rate'], cmap='RdYlGn', alpha=0.6)
            plt.colorbar(scatter, label='Redemption Rate')
            plt.xlabel('Trust Development Speed Ratio')
            plt.ylabel('Final Cooperation Rate')
            plt.title('Trust Speed vs Cooperation\n(Color = Redemption Rate)')
        
        # 4. Stress Model Effectiveness
        ax4 = plt.subplot(rows, cols, 4)
        if 'social_support_effectiveness' in df.columns:
            scatter = plt.scatter(df['social_support_effectiveness'], df['final_cooperation_rate'], 
                                 c=df['stress_recovery_rate'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='Stress Recovery Rate')
            plt.xlabel('Social Support Effectiveness')
            plt.ylabel('Final Cooperation Rate')
            plt.title('Social Support vs Cooperation\n(Color = Stress Recovery)')
        
        # Continue with remaining plots...
        plt.tight_layout()
        
        title = 'Enhanced Constraint Cascade Simulation - Realistic Parameters Analysis'
        if has_intergroup_data:
            title += '\n(Including Inter-Group Dynamics with Realistic Parameters)'
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Save with error handling
        try:
            plt.savefig('realistic_parameters_analysis.png', dpi=300, bbox_inches='tight')
            timestamp_print("✅ Visualization saved as realistic_parameters_analysis.png")
        except Exception as save_error:
            timestamp_print(f"⚠️ Could not save visualization: {save_error}")
        
        plt.close(fig)
        
        return fig
        
    except Exception as e:
        timestamp_print(f"⚠️ Error creating visualizations: {e}")
        timestamp_print("📊 Continuing without visualizations...")
        return None

def identify_critical_thresholds(df: pd.DataFrame):
    """Identify critical thresholds with realistic parameters"""
    timestamp_print("🎯 Identifying critical thresholds for realistic parameters...")
    
    # Find shock interval threshold for cooperation collapse
    df_sorted = df.sort_values('shock_interval_avg')
    cooperation_rates = df_sorted['final_cooperation_rate'].rolling(window=min(20, len(df)//2), center=True).mean()
    
    # Find where cooperation drops below 50%
    collapse_threshold = None
    for i, rate in enumerate(cooperation_rates):
        if not pd.isna(rate) and rate < 0.5:
            collapse_threshold = df_sorted.iloc[i]['shock_interval_avg']
            break
    
    timestamp_print("\n" + "="*60)
    timestamp_print("🔍 REALISTIC PARAMETERS THRESHOLD ANALYSIS")
    timestamp_print("="*60)
    
    # Cooperation collapse threshold
    if collapse_threshold:
        timestamp_print(f"🚨 Cooperation Collapse Threshold: {collapse_threshold:.1f} years")
        timestamp_print(f"   (Average shock interval below this causes cooperation failure)")
    
    return {
        'cooperation_collapse_threshold': collapse_threshold,
        'realistic_parameters': True
    }

def save_comprehensive_results(df: pd.DataFrame, thresholds: Dict):
    """Save all results for realistic parameters analysis"""
    current_dir = os.getcwd()
    timestamp_print(f"💾 Saving realistic parameters results to: {current_dir}")
    
    saved_files = []
    
    try:
        # Save main dataset
        main_file = 'realistic_parameters_simulation_results.csv'
        df.to_csv(main_file, index=False)
        if os.path.exists(main_file):
            size_mb = os.path.getsize(main_file) / (1024*1024)
            saved_files.append(f"📊 {main_file} ({size_mb:.2f} MB)")
        
        # Save summary statistics with error handling
        try:
            summary_file = 'realistic_parameters_summary_stats.csv'
            summary_stats = df.describe()
            summary_stats.to_csv(summary_file)
            if os.path.exists(summary_file):
                saved_files.append(f"📈 {summary_file}")
        except Exception as summary_error:
            timestamp_print(f"⚠️ Could not create summary stats: {summary_error}")
        
        # Create comprehensive summary with safe column access
        summary_report = 'realistic_parameters_experiment_summary.txt'
        with open(summary_report, 'w') as f:
            f.write("Enhanced Constraint Cascade - Realistic Parameters Experiment Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Simulations: {len(df)}\n")
            
            # Safe column access
            if 'final_cooperation_rate' in df.columns:
                f.write(f"Average Cooperation Rate: {df['final_cooperation_rate'].mean():.3f}\n")
            if 'extinction_occurred' in df.columns:
                f.write(f"Extinction Rate: {df['extinction_occurred'].mean():.3f}\n")
            if 'final_population' in df.columns:
                f.write(f"Average Final Population: {df['final_population'].mean():.1f}\n")
            
            # Grid parameters summary
            if 'shock_interval_avg' in df.columns:
                f.write(f"Shock Interval Range: {df['shock_interval_avg'].min():.1f} - {df['shock_interval_avg'].max():.1f} years\n")
            if 'pareto_alpha' in df.columns:
                f.write(f"Pareto Alpha Range: {df['pareto_alpha'].min():.2f} - {df['pareto_alpha'].max():.2f}\n")
            if 'community_buffer_factor' in df.columns:
                f.write(f"Community Buffer Range: {df['community_buffer_factor'].min():.3f} - {df['community_buffer_factor'].max():.3f}\n")
            
            f.write(f"\nThreshold Analysis:\n")
            for key, value in thresholds.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nFiles Created:\n")
            for file_info in saved_files:
                f.write(f"  {file_info}\n")
        
        if os.path.exists(summary_report):
            saved_files.append(f"📋 {summary_report}")
        
        timestamp_print(f"\n✅ Successfully saved {len(saved_files)} files:")
        for file_info in saved_files:
            timestamp_print(f"   {file_info}")
        
    except Exception as e:
        timestamp_print(f"❌ Error saving files: {e}")
        traceback.print_exc()
        
        # Try to save just the main file as a backup
        try:
            backup_file = 'realistic_parameters_backup.csv'
            df.to_csv(backup_file, index=False)
            timestamp_print(f"💾 Backup saved as: {backup_file}")
            saved_files.append(f"💾 {backup_file} (backup)")
        except Exception as backup_error:
            timestamp_print(f"❌ Backup also failed: {backup_error}")
    
    return saved_files

if __name__ == "__main__":
    main()