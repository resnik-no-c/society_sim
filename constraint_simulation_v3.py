#!/usr/bin/env python3
"""
Enhanced Constraint Cascade Simulation with Weighted Sampling and Inter-Group Dynamics
Implements Option 1: Weighted Sampling with differentiated interaction types
- Transformational events: 2 per person per quarter (ALL modeled)
- Significant interactions: 12 per person per quarter (ALL modeled) 
- Maintenance interactions: 10 per person per quarter (sampled 1 in 5)
"""

import random
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, NamedTuple
from collections import defaultdict, deque
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import traceback
import argparse
import threading
import queue
import pickle
import heapq

def timestamp_print(message: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


# Try to import seaborn for better visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
    timestamp_print("✅ Seaborn loaded successfully")
except ImportError:
    HAS_SEABORN = False
    timestamp_print("⚠️  Seaborn not available - using matplotlib only")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    timestamp_print("⚠️  SciPy not available - some statistical analysis will be limited")

def save_simulation_result(result: EnhancedSimulationResults, results_dir: str = "simulation_results"):
    """Save individual simulation result immediately upon completion"""
    import os
    import pickle
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        timestamp_print(f"📁 Created results directory: {results_dir}")
    
    # Save as pickle file
    filename = f"sim_{result.run_id:04d}_result.pkl"
    filepath = os.path.join(results_dir, filename)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        timestamp_print(f"💾 Saved simulation {result.run_id} result to {filepath}")
        return filepath
    except Exception as e:
        timestamp_print(f"❌ Error saving simulation {result.run_id}: {e}")
        return None

def load_all_simulation_results(results_dir: str = "simulation_results") -> List[EnhancedSimulationResults]:
    """Load all saved simulation results from directory"""
    import os
    import pickle
    import glob
    
    if not os.path.exists(results_dir):
        timestamp_print(f"⚠️  Results directory {results_dir} does not exist")
        return []
    
    # Find all result files
    pattern = os.path.join(results_dir, "sim_*_result.pkl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        timestamp_print(f"⚠️  No result files found in {results_dir}")
        return []
    
    timestamp_print(f"📂 Found {len(result_files)} result files in {results_dir}")
    
    # Load all results
    results = []
    failed_loads = 0
    
    for filepath in sorted(result_files):
        try:
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
                results.append(result)
        except Exception as e:
            timestamp_print(f"❌ Error loading {filepath}: {e}")
            failed_loads += 1
    
    timestamp_print(f"✅ Successfully loaded {len(results)} simulation results")
    if failed_loads > 0:
        timestamp_print(f"⚠️  Failed to load {failed_loads} result files")
    
    return results

def save_incremental_csv(result: EnhancedSimulationResults, csv_file: str = "simulation_results_incremental.csv"):
    """Save simulation result to incremental CSV file"""
    import os
    import pandas as pd
    
    # Convert result to row format
    row_data = {
        'run_id': result.run_id,
        'timestamp': datetime.now().isoformat(),
        'final_cooperation_rate': result.final_cooperation_rate,
        'final_population': result.final_population,
        'extinction_occurred': result.extinction_occurred,
        'rounds_completed': result.rounds_completed,
        'total_defections': result.total_defections,
        'total_redemptions': result.total_redemptions,
        'avg_trust_level': result.avg_trust_level,
        'initial_population': result.parameters.initial_population,
        'max_population': result.parameters.max_population,
        'shock_frequency': result.parameters.shock_frequency,
        'pressure_multiplier': result.parameters.pressure_multiplier,
        'homophily_bias': getattr(result.parameters, 'homophily_bias', 0.0),
        'num_groups': getattr(result.parameters, 'num_groups', 1),
        'relationship_memory': getattr(result.parameters, 'relationship_memory', 10),
        'total_transformational_events': result.total_transformational_events,
        'total_significant_interactions': result.total_significant_interactions,
        'total_maintenance_interactions': result.total_maintenance_interactions,
        'trust_asymmetry': result.trust_asymmetry,
        'group_segregation_index': result.group_segregation_index,
    }
    
    # Create DataFrame
    df_row = pd.DataFrame([row_data])
    
    # Append to file (create if doesn't exist)
    try:
        if os.path.exists(csv_file):
            df_row.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_file, mode='w', header=True, index=False)
            timestamp_print(f"📊 Created incremental CSV: {csv_file}")
        
        return True
    except Exception as e:
        timestamp_print(f"❌ Error saving to incremental CSV: {e}")
        return False

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
    """Enhanced relationship tracking with increased memory (15-20 interactions)"""
    trust: float = 0.5
    interaction_count: int = 0
    cooperation_history: deque = field(default_factory=lambda: deque(maxlen=18))  # Increased from 10 to 18
    last_interaction_round: int = 0
    
    # Inter-group extensions
    is_same_group: bool = True
    betrayal_count: int = 0
    cooperation_count: int = 0
    
    def update_trust(self, cooperated: bool, round_num: int, 
                    in_group_modifier: float = 1.0, out_group_modifier: float = 1.0):
        """Update trust based on interaction outcome with optional group-based modifiers"""
        self.interaction_count += 1
        self.last_interaction_round = round_num
        self.cooperation_history.append(cooperated)
        
        # Apply group-based trust modifiers
        if cooperated:
            self.cooperation_count += 1
            trust_delta = 0.1
            if self.is_same_group:
                trust_delta *= in_group_modifier
            else:
                trust_delta *= out_group_modifier
            self.trust = min(1.0, self.trust + trust_delta)
        else:
            self.betrayal_count += 1
            trust_delta = 0.15
            if self.is_same_group:
                trust_delta *= in_group_modifier
            else:
                trust_delta *= out_group_modifier
            self.trust = max(0.0, self.trust - trust_delta)

@dataclass
class SimulationParameters:
    """Enhanced simulation parameters with weighted sampling and 800 max population"""
    initial_population: int
    max_population: int = 800  # Fixed at 800 as requested
    shock_frequency: float = 0.1
    pressure_multiplier: float = 0.5
    base_birth_rate: float = 0.05
    max_rounds: int = 200  # Fixed at 200 (50 years) as requested
    maslow_variation: float = 0.5
    constraint_threshold_range: Tuple[float, float] = (0.3, 0.8)
    recovery_threshold: float = 0.3
    cooperation_bonus: float = 0.2
    trust_threshold: float = 0.6
    relationship_memory: int = 18  # Increased from 10 to 18 (15-20 range)
    max_relationships_per_person: int = 150  # Keep at 150 as requested
    interaction_batch_size: int = 50
    
    # Inter-Group Parameters
    num_groups: int = 3
    founder_group_distribution: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    homophily_bias: float = 0.7  # Keep as-is as requested
    in_group_trust_modifier: float = 1.5
    out_group_trust_modifier: float = 0.5
    out_group_constraint_amplifier: float = 2.0
    reputational_spillover: float = 0.1
    mixing_event_frequency: int = 15  # More frequent mixing events (was 25)
    mixing_event_bonus_multiplier: float = 2.0
    inheritance_style: str = "mother"
    
    # NEW: Weighted Sampling Parameters
    transformational_events_per_quarter: int = 2
    significant_interactions_per_quarter: int = 12
    maintenance_interactions_per_quarter: int = 10  # Represents 50 real interactions (1 in 5 sampling)

@dataclass
class InteractionEvent:
    """Represents a weighted interaction event"""
    person1_id: int
    person2_id: int
    interaction_type: str  # 'transformational', 'significant', 'maintenance'
    round_num: int
    complexity_weight: float
    
    def __lt__(self, other):
        """For priority queue sorting"""
        return self.complexity_weight < other.complexity_weight

@dataclass
class EnhancedSimulationResults:
    """Comprehensive results container"""
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
    
    # Inter-Group Metrics
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
    
    # NEW: Weighted Sampling Metrics
    total_transformational_events: int = 0
    total_significant_interactions: int = 0
    total_maintenance_interactions: int = 0
    avg_transformational_processing_time: float = 0.0
    avg_significant_processing_time: float = 0.0
    avg_maintenance_processing_time: float = 0.0

class OptimizedPerson:
    """Enhanced person with group identity"""
    
    __slots__ = ['id', 'strategy', 'constraint_level', 'constraint_threshold', 
                 'recovery_threshold', 'is_constrained', 'is_dead', 'relationships',
                 'max_lifespan', 'age', 'strategy_changes', 'rounds_as_selfish',
                 'rounds_as_cooperative', 'maslow_needs', 'maslow_pressure', 'is_born',
                 'group_id', 'in_group_interactions', 'out_group_interactions', 
                 'mixing_event_participations']
    
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
        
        self.relationships: Dict[int, FastRelationship] = {}
        
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
            self.group_id = "A"  # Default fallback
            
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
        self._calculate_maslow_pressure_fast(params.pressure_multiplier)
    
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
    
    def _calculate_maslow_pressure_fast(self, pressure_multiplier: float):
        """Optimized pressure calculation"""
        n = self.maslow_needs
        
        total_pressure = (
            (10 - n.physiological) ** 2 * 0.003 +
            (10 - n.safety) ** 2 * 0.002 +
            (10 - n.love) ** 2 * 0.001 +
            (10 - n.esteem) ** 2 * 0.0008 +
            (10 - n.self_actualization) ** 2 * 0.0005
        ) * pressure_multiplier
        
        total_relief = (
            n.physiological ** 1.5 * 0.0002 +
            n.safety ** 1.5 * 0.0002 +
            n.love ** 1.8 * 0.0005 +
            n.esteem ** 2.0 * 0.001 +
            n.self_actualization ** 2.2 * 0.002
        )
        
        self.maslow_pressure = max(0, total_pressure - total_relief)
        self.constraint_level += self.maslow_pressure * 0.02
    
    def update(self, system_stress: float, pressure_multiplier: float, cooperation_bonus: float = 0):
        """Update person state with full fidelity"""
        if self.is_dead:
            return
        
        self.age += 1
        if self.age >= self.max_lifespan:
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
        
        self._calculate_maslow_pressure_fast(pressure_multiplier)
        
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
        
        self.constraint_level += amount * maslow_amplifier
        
        if self.strategy == 'cooperative' and self.constraint_level > self.constraint_threshold:
            self.force_switch()
            return True
        return False
    
    def check_for_recovery(self) -> bool:
        """Check if person can recover to cooperative strategy"""
        if self.strategy == 'selfish' and self.constraint_level < self.recovery_threshold:
            recovery_chance = 0.1
            
            if self.maslow_needs.love > 7:
                recovery_chance += 0.2
            if self.maslow_needs.esteem > 7:
                recovery_chance += 0.1
            if self.maslow_needs.self_actualization > 8:
                recovery_chance += 0.2
            
            if self.rounds_as_selfish > 50:
                recovery_chance *= 0.5
            
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
    
    def calculate_cooperation_decision(self, other: 'OptimizedPerson', round_num: int) -> bool:
        """Decide whether to cooperate based on relationship, strategy, and group"""
        if self.strategy == 'selfish':
            return False
        
        relationship = self.get_relationship(other.id, round_num, other.group_id)
        
        if relationship.interaction_count == 0:
            base_coop_prob = self.maslow_needs.love / 10
            if hasattr(self, 'group_id') and hasattr(other, 'group_id'):
                if self.group_id == other.group_id:
                    base_coop_prob *= 1.2
                else:
                    base_coop_prob *= 0.8
            return random.random() < base_coop_prob
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

class EnhancedMassSimulation:
    """Enhanced simulation with weighted sampling"""
    
    def __init__(self, params: SimulationParameters, run_id: int):
        self.params = params
        self.run_id = run_id
        self.people: List[OptimizedPerson] = []
        self.round = 0
        self.system_stress = 0.0
        self.next_person_id = params.initial_population + 1
        
        # Tracking variables
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
        
        # Inter-group tracking
        self.group_names = [chr(65 + i) for i in range(params.num_groups)]
        self.in_group_interactions = 0
        self.out_group_interactions = 0
        self.total_mixing_events = 0
        self.successful_mixing_events = 0
        self.reputational_spillover_events = 0
        self.out_group_constraint_amplifications = 0
        
        # NEW: Weighted sampling tracking
        self.total_transformational_events = 0
        self.total_significant_interactions = 0
        self.total_maintenance_interactions = 0
        
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
    
    def _generate_weighted_interactions(self, alive_people: List[OptimizedPerson]) -> List[InteractionEvent]:
        """Generate weighted interactions according to Option 1 sampling strategy"""
        interaction_queue = []
        
        for person in alive_people:
            # Transformational events: 2 per person per quarter (ALL modeled)
            for _ in range(self.params.transformational_events_per_quarter):
                partner = self._select_interaction_partner(person, alive_people)
                if partner:
                    event = InteractionEvent(
                        person1_id=person.id,
                        person2_id=partner.id,
                        interaction_type='transformational',
                        round_num=self.round,
                        complexity_weight=0.005  # High complexity
                    )
                    interaction_queue.append(event)
            
            # Significant interactions: 12 per person per quarter (ALL modeled)
            for _ in range(self.params.significant_interactions_per_quarter):
                partner = self._select_interaction_partner(person, alive_people)
                if partner:
                    event = InteractionEvent(
                        person1_id=person.id,
                        person2_id=partner.id,
                        interaction_type='significant',
                        round_num=self.round,
                        complexity_weight=0.003  # Medium complexity
                    )
                    interaction_queue.append(event)
            
            # Maintenance interactions: 10 per person per quarter (sample 1 in 5)
            for _ in range(self.params.maintenance_interactions_per_quarter):
                partner = self._select_interaction_partner(person, alive_people)
                if partner:
                    event = InteractionEvent(
                        person1_id=person.id,
                        person2_id=partner.id,
                        interaction_type='maintenance',
                        round_num=self.round,
                        complexity_weight=0.001  # Low complexity
                    )
                    interaction_queue.append(event)
        
        return interaction_queue
    
    def _select_interaction_partner(self, person: OptimizedPerson, 
                                  alive_people: List[OptimizedPerson]) -> Optional[OptimizedPerson]:
        """Select interaction partner with homophily bias and safety checks"""
        # BUGFIX: Ensure we have potential partners
        available_partners = [p for p in alive_people if p.id != person.id and not p.is_dead]
        if not available_partners:
            return None
        
        if not hasattr(self.params, 'homophily_bias'):
            # Original selection logic with safety check
            if person.relationships and random.random() < 0.3:
                known_alive = [p for p in available_partners 
                             if p.id in person.relationships]
                if known_alive:
                    return random.choice(known_alive)
            return random.choice(available_partners)
        
        # Apply homophily bias with safety checks
        if random.random() < self.params.homophily_bias:
            # Try to find same-group partner
            same_group_partners = [p for p in available_partners 
                                 if hasattr(p, 'group_id') and hasattr(person, 'group_id') 
                                 and p.group_id == person.group_id]
            if same_group_partners:
                # Prefer known relationships with some probability
                if person.relationships and random.random() < 0.3:
                    known_same_group = [p for p in same_group_partners if p.id in person.relationships]
                    if known_same_group:
                        return random.choice(known_same_group)
                return random.choice(same_group_partners)
        
        # Fall back to random selection (including out-group)
        if person.relationships and random.random() < 0.2:
            known_partners = [p for p in available_partners if p.id in person.relationships]
            if known_partners:
                return random.choice(known_partners)
        
        return random.choice(available_partners)
    
    def _process_weighted_interaction(self, event: InteractionEvent, alive_people: List[OptimizedPerson]) -> bool:
        """Process a weighted interaction event with appropriate complexity"""
        # BUGFIX: Validate that people still exist and are alive
        person1 = None
        person2 = None
        
        for person in alive_people:
            if person.id == event.person1_id and not person.is_dead:
                person1 = person
            if person.id == event.person2_id and not person.is_dead:
                person2 = person
            if person1 and person2:
                break
        
        if not person1 or not person2:
            # One or both people died/missing since event was created
            return False
        
        # Track interaction type
        if event.interaction_type == 'transformational':
            self.total_transformational_events += 1
        elif event.interaction_type == 'significant':
            self.total_significant_interactions += 1
        elif event.interaction_type == 'maintenance':
            self.total_maintenance_interactions += 1
        
        # Process interaction with type-specific complexity
        return self._process_interaction(person1, person2, interaction_type=event.interaction_type)
    
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
        """Check if this round should have a mixing event (more frequent now)"""
        return (hasattr(self.params, 'mixing_event_frequency') and 
                self.params.mixing_event_frequency > 0 and 
                self.round > 0 and 
                self.round % self.params.mixing_event_frequency == 0)
    
    def _handle_mixing_event(self, alive_people: List[OptimizedPerson]):
        """Handle inter-group mixing event with safety checks"""
        self.total_mixing_events += 1
        
        # BUGFIX: Ensure we have enough people for mixing
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
        max_interactions = max(len(alive_people) // 3, 1)  # BUGFIX: Ensure at least 1 interaction possible
        
        for _ in range(max_interactions):
            # Pick two different groups with people
            available_groups = [group for group in valid_groups if len(group_buckets[group]) > 0]
            if len(available_groups) < 2:
                break
            
            group1, group2 = random.sample(available_groups, 2)
            
            if group_buckets[group1] and group_buckets[group2]:
                person1 = random.choice(group_buckets[group1])
                person2 = random.choice(group_buckets[group2])
                
                # BUGFIX: Verify people are still alive and valid
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
    
    def run_simulation(self) -> EnhancedSimulationResults:
        """Run enhanced simulation with weighted sampling and error handling"""
        timestamp_print(f"🎮 Starting weighted sampling simulation run {self.run_id}")
        
        try:
            initial_trait_avg = self._get_average_traits()
            initial_group_populations = self._get_group_populations()
            timestamp_print(f"📊 Initial setup complete for sim {self.run_id}")
            
            while self.round < self.params.max_rounds:
                self.round += 1
                
                if self.round % 50 == 0:
                    timestamp_print(f"🔄 Sim {self.run_id}: Round {self.round}/{self.params.max_rounds}")
                
                alive_people = [p for p in self.people if not p.is_dead]
                if len(alive_people) == 0:
                    timestamp_print(f"💀 Sim {self.run_id}: Population extinct at round {self.round}")
                    break
                
                try:
                    # Check for mixing event
                    if self._is_mixing_event_round():
                        self._handle_mixing_event(alive_people)
                    
                    if random.random() < self.params.shock_frequency:
                        self._trigger_shock()
                    
                    # NEW: Weighted interaction handling with error recovery
                    self._handle_weighted_interactions(alive_people)
                    
                    self._check_recoveries()
                    self._update_population()
                    self._collect_round_data()
                    
                    self.system_stress = max(0, self.system_stress - 0.01)
                    
                except Exception as round_error:
                    timestamp_print(f"⚠️  Error in round {self.round} of sim {self.run_id}: {round_error}")
                    # Continue with next round rather than crashing entire simulation
                    continue
            
            timestamp_print(f"🏁 Sim {self.run_id}: Completed {self.round} rounds, generating results...")
            return self._generate_results(initial_trait_avg, initial_group_populations)
            
        except Exception as sim_error:
            timestamp_print(f"❌ Critical error in simulation {self.run_id}: {sim_error}")
            traceback.print_exc()
            
            # Return emergency result to prevent total failure
            emergency_result = self._generate_emergency_result()
            return emergency_result
    
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
                final_cooperation_rate=0.0,  # Default safe values
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
                total_transformational_events=0,
                total_significant_interactions=0,
                total_maintenance_interactions=0,
            )
        except Exception as emergency_error:
            timestamp_print(f"❌ Even emergency result generation failed: {emergency_error}")
            # Absolute fallback - minimal result
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
            )
    
    def _handle_weighted_interactions(self, alive_people: List[OptimizedPerson]):
        """Handle interactions using weighted sampling approach"""
        if len(alive_people) < 2:
            return
        
        # Generate weighted interaction events
        interaction_events = self._generate_weighted_interactions(alive_people)
        
        # Process all interaction events
        for event in interaction_events:
            self._process_weighted_interaction(event, alive_people)
    
    def _trigger_shock(self):
        """Apply system shock"""
        shock_amount = 0.15 + random.random() * 0.25
        self.system_stress += shock_amount
        self.shock_events += 1
        
        for person in self.people:
            if not person.is_dead:
                protection = 0
                if person.strategy == 'cooperative':
                    trusted_allies = sum(1 for r in person.relationships.values() 
                                       if r.trust > self.params.trust_threshold)
                    protection = min(0.5, trusted_allies * 0.05)
                
                effective_shock = shock_amount * (1 - protection) * 0.1 * self.params.pressure_multiplier
                person.add_constraint_pressure(effective_shock)
    
    def _process_interaction(self, person1: OptimizedPerson, person2: OptimizedPerson, 
                           is_mixing_event: bool = False, interaction_type: str = 'significant') -> bool:
        """Process interaction with inter-group dynamics and type-specific effects"""
        p1_cooperates = person1.calculate_cooperation_decision(person2, self.round)
        p2_cooperates = person2.calculate_cooperation_decision(person1, self.round)
        
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
        
        # Apply interaction type multiplier
        if interaction_type == 'transformational':
            base_bonus *= 1.5  # Stronger effects for transformational events
        elif interaction_type == 'maintenance':
            base_bonus *= 0.5  # Weaker effects for maintenance interactions
        
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
            if interaction_type == 'transformational':
                constraint_amount *= 2.0  # Stronger constraint from transformational betrayals
            
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
            if interaction_type == 'transformational':
                constraint_amount *= 2.0
            
            is_from_out_group = (hasattr(person2, 'group_id') and hasattr(person1, 'group_id') and 
                                person2.group_id != person1.group_id)
            if is_from_out_group and hasattr(self.params, 'out_group_constraint_amplifier'):
                self.out_group_constraint_amplifications += 1
                person2.add_constraint_pressure(constraint_amount, is_from_out_group, 
                                              self.params.out_group_constraint_amplifier)
            else:
                person2.add_constraint_pressure(constraint_amount)
            
            person2.maslow_needs.esteem = max(0, person2.maslow_needs.esteem - 0.1)
        
        # Base pressure calculations
        base_pressure = 0.005 + self.system_stress * 0.02
        
        pressure1 = base_pressure * self.params.pressure_multiplier
        pressure2 = base_pressure * self.params.pressure_multiplier
        
        if person2.strategy == 'selfish':
            pressure1 += 0.02
        if person1.strategy == 'selfish':
            pressure2 += 0.02
        
        pressure1 += person1._get_basic_needs_pressure() * self.params.pressure_multiplier - person1._get_inspire_effect()
        pressure2 += person2._get_basic_needs_pressure() * self.params.pressure_multiplier - person2._get_inspire_effect()
        
        switched1 = person1.add_constraint_pressure(pressure1)
        switched2 = person2.add_constraint_pressure(pressure2)
        
        if switched1:
            self.total_defections += 1
        if switched2:
            self.total_defections += 1
        
        if switched1 or switched2:
            self._check_cascade()
        
        # Birth mechanics with 800 max population
        population_ratio = len(self.people) / 800  # Use fixed 800 max
        adjusted_birth_rate = self.params.base_birth_rate * (1 - population_ratio * 0.8)
        
        if random.random() < adjusted_birth_rate and len(self.people) < 800:
            self._create_birth(person1, person2)
        
        # Person updates
        person1.update(self.system_stress, self.params.pressure_multiplier, cooperation_bonus1)
        person2.update(self.system_stress, self.params.pressure_multiplier, cooperation_bonus2)
        
        return p1_cooperates and p2_cooperates
    
    def _check_recoveries(self):
        """Check for strategy recoveries"""
        alive_people = [p for p in self.people if not p.is_dead]
        for person in alive_people:
            if person.check_for_recovery():
                self.total_redemptions += 1
    
    def _create_birth(self, parent_a: OptimizedPerson, parent_b: OptimizedPerson):
        """Create new person with group inheritance"""
        new_person = OptimizedPerson(self.next_person_id, self.params, parent_a, parent_b)
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
            person.update(self.system_stress, self.params.pressure_multiplier)
    
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
    
    def _generate_results(self, initial_traits: Dict[str, float], 
                         initial_group_populations: Dict[str, int]) -> EnhancedSimulationResults:
        """Generate comprehensive results with weighted sampling metrics"""
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
            
            # Inter-Group Metrics
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
            
            # NEW: Weighted Sampling Metrics
            total_transformational_events=self.total_transformational_events,
            total_significant_interactions=self.total_significant_interactions,
            total_maintenance_interactions=self.total_maintenance_interactions,
            avg_transformational_processing_time=0.005,  # Complexity weights as proxy
            avg_significant_processing_time=0.003,
            avg_maintenance_processing_time=0.001
        )

def generate_random_parameters(run_id: int) -> SimulationParameters:
    """Generate randomized simulation parameters with weighted sampling"""
    timestamp_print(f"🎲 Generating weighted sampling parameters for sim {run_id}")
    initial_pop = random.randint(100, 500)
    
    # Decide whether to include inter-group dynamics (80% chance)
    include_intergroup = random.random() < 0.8
    timestamp_print(f"🏷️  Sim {run_id}: include_intergroup = {include_intergroup}")
    
    if include_intergroup:
        num_groups = random.choice([2, 3, 4])
        group_dist = [random.random() for _ in range(num_groups)]
        total = sum(group_dist)
        group_dist = [x/total for x in group_dist]
        
        params = SimulationParameters(
            initial_population=initial_pop,
            max_population=800,  # Fixed at 800
            shock_frequency=0.01 + random.random() * 0.19,
            pressure_multiplier=0.1 + random.random() * 0.9,
            base_birth_rate=0.01 + random.random() * 0.09,
            max_rounds=200,  # Fixed at 200 (50 years)
            maslow_variation=0.3 + random.random() * 0.7,
            constraint_threshold_range=(
                0.2 + random.random() * 0.3,
                0.6 + random.random() * 0.3
            ),
            recovery_threshold=0.2 + random.random() * 0.3,
            cooperation_bonus=0.1 + random.random() * 0.3,
            trust_threshold=0.4 + random.random() * 0.4,
            relationship_memory=random.randint(15, 20),  # 15-20 range
            
            # Inter-group parameters
            num_groups=num_groups,
            founder_group_distribution=group_dist,
            homophily_bias=random.random(),  # Keep as-is
            in_group_trust_modifier=1.0 + random.random() * 1.0,
            out_group_trust_modifier=0.1 + random.random() * 0.9,
            out_group_constraint_amplifier=1.0 + random.random() * 2.0,
            reputational_spillover=random.random() * 0.3,
            mixing_event_frequency=random.choice([10, 15, 20, 25]),  # More frequent mixing
            mixing_event_bonus_multiplier=1.5 + random.random() * 1.5,
            inheritance_style=random.choice(["mother", "father", "random"]),
            
            # Weighted sampling parameters (fixed as per analysis)
            transformational_events_per_quarter=2,
            significant_interactions_per_quarter=12,
            maintenance_interactions_per_quarter=10
        )
    else:
        params = SimulationParameters(
            initial_population=initial_pop,
            max_population=800,  # Fixed at 800
            shock_frequency=0.01 + random.random() * 0.19,
            pressure_multiplier=0.1 + random.random() * 0.9,
            base_birth_rate=0.01 + random.random() * 0.09,
            max_rounds=200,  # Fixed at 200
            maslow_variation=0.3 + random.random() * 0.7,
            constraint_threshold_range=(
                0.2 + random.random() * 0.3,
                0.6 + random.random() * 0.3
            ),
            recovery_threshold=0.2 + random.random() * 0.3,
            cooperation_bonus=0.1 + random.random() * 0.3,
            trust_threshold=0.4 + random.random() * 0.4,
            relationship_memory=random.randint(15, 20),  # 15-20 range
            
            # Set inter-group to minimal/disabled
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
            
            # Weighted sampling parameters
            transformational_events_per_quarter=2,
            significant_interactions_per_quarter=12,
            maintenance_interactions_per_quarter=10
        )
    
    timestamp_print(f"📋 Sim {run_id}: 200 rounds, 800 max pop, weighted sampling enabled")
    return params

def run_single_simulation(run_id: int) -> EnhancedSimulationResults:
    """Run a single simulation with weighted sampling"""
    timestamp_print(f"🔄 Starting weighted sampling simulation {run_id}")
    params = generate_random_parameters(run_id)
    timestamp_print(f"🎛️  Generated weighted sampling parameters for sim {run_id}")
    sim = EnhancedMassSimulation(params, run_id)
    timestamp_print(f"🏗️  Created weighted sampling simulation object for sim {run_id}")
    result = sim.run_simulation()
    timestamp_print(f"✅ Completed weighted sampling simulation {run_id}")
    return result

# ============================================================================
# LOAD-BALANCED SIMULATION SYSTEM WITH CHUNKING AT 30
# ============================================================================

@dataclass
class SimulationWork:
    """Represents work to be done with chunking at 30"""
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
    """Load balancing with chunk size 30 and immediate result saving"""
    
    def __init__(self, simulations: List[SimulationParameters], chunk_size: int = 30, results_dir: str = "simulation_results"):  # Changed to 30
        self.simulations = simulations
        self.chunk_size = chunk_size
        self.results_dir = results_dir
        self.work_queue = queue.Queue()
        self.completed_simulations = set()  # Changed to set for tracking IDs only
        self.active_simulations = {}
        self.simulation_states = {}
        self.lock = threading.Lock()
        
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            timestamp_print(f"📁 Created results directory: {results_dir}")
        
        # Clear any existing result files
        import glob
        existing_files = glob.glob(os.path.join(results_dir, "sim_*_result.pkl"))
        if existing_files:
            timestamp_print(f"🧹 Clearing {len(existing_files)} existing result files...")
            for file_path in existing_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    timestamp_print(f"⚠️  Could not remove {file_path}: {e}")
        
        # Initialize work queue
        for i, params in enumerate(simulations):
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
    
    def _estimate_complexity(self, params: SimulationParameters) -> float:
        """Estimate simulation complexity with weighted sampling overhead"""
        # Base complexity from population and interactions
        base = params.initial_population ** 1.3 * (params.max_rounds / 100)
        
        # Weighted sampling adds complexity (more interactions)
        weighted_factor = (
            params.transformational_events_per_quarter * 0.005 +  # High complexity
            params.significant_interactions_per_quarter * 0.003 +  # Medium complexity  
            params.maintenance_interactions_per_quarter * 0.001    # Low complexity
        ) * 1000  # Scale up
        
        base *= (1 + weighted_factor)
        
        # Inter-group complexity multiplier
        if hasattr(params, 'num_groups') and params.num_groups > 1:
            intergroup_factor = (
                params.num_groups * 1.5 +
                params.homophily_bias * 1.2 +
                (2.0 - params.out_group_trust_modifier) * 1.5
            )
            base *= intergroup_factor
        
        return base
    
    def get_work(self) -> Optional[SimulationWork]:
        """Get next work item"""
        try:
            return self.work_queue.get_nowait()
        except queue.Empty:
            return None
    
    def submit_result(self, work: SimulationWork, result_data: tuple):
        """Submit completed work and save immediately if complete"""
        result_type, sim_id, data, exec_time, rounds_done = result_data
        
        with self.lock:
            if result_type == 'complete':
                # Save result immediately
                save_simulation_result(data, self.results_dir)
                save_incremental_csv(data)
                
                # Track completion
                self.completed_simulations.add(sim_id)
                if sim_id in self.active_simulations:
                    del self.active_simulations[sim_id]
                if sim_id in self.simulation_states:
                    del self.simulation_states[sim_id]
                timestamp_print(f"🎉 Weighted sampling simulation {sim_id} completed and saved!")
                
            elif result_type == 'partial':
                self.simulation_states[sim_id] = data
                current_round = work.end_round
                self.active_simulations[sim_id] = current_round
                
                if current_round < work.max_rounds:
                    # Adaptive chunk sizing based on execution time (but centered around 30)
                    if exec_time > 60:
                        new_chunk_size = max(20, int(self.chunk_size * 0.8))  # Reduce but not below 20
                    elif exec_time < 20:
                        new_chunk_size = min(40, int(self.chunk_size * 1.2))  # Increase but not above 40
                    else:
                        new_chunk_size = self.chunk_size  # Keep at 30
                    
                    next_work = SimulationWork(
                        sim_id=sim_id,
                        start_round=current_round,
                        end_round=min(current_round + new_chunk_size, work.max_rounds),
                        max_rounds=work.max_rounds,
                        simulation_state=data,
                        complexity_score=work.complexity_score,
                        estimated_time=exec_time * (new_chunk_size / rounds_done) if rounds_done > 0 else work.estimated_time
                    )
                    self.work_queue.put(next_work)
                    
            elif result_type == 'error':
                timestamp_print(f"❌ Error in weighted sampling simulation {sim_id}: {data}")
                if sim_id in self.active_simulations:
                    del self.active_simulations[sim_id]
    
    def get_progress(self) -> Tuple[int, int]:
        """Get (completed, total) simulations"""
        with self.lock:
            return len(self.completed_simulations), len(self.simulations)
    
    def is_complete(self) -> bool:
        """Check if all simulations are done"""
        with self.lock:
            return (len(self.completed_simulations) == len(self.simulations) and 
                    self.work_queue.empty())
    
    def get_completed_results(self) -> List[EnhancedSimulationResults]:
        """Load all completed results from disk"""
        return load_all_simulation_results(self.results_dir)

def process_simulation_work(work: SimulationWork) -> tuple:
    """Process a single work item with weighted sampling"""
    start_time = time.time()
    
    try:
        if work.is_new_simulation:
            params = generate_random_parameters(work.sim_id)
            sim = EnhancedMassSimulation(params, work.sim_id)
            sim.round = 0
        else:
            sim = pickle.loads(work.simulation_state)
            sim.round = work.start_round
        
        rounds_completed = 0
        target_rounds = work.end_round - work.start_round
        
        if work.is_new_simulation:
            timestamp_print(f"🚀 Starting weighted sampling simulation {work.sim_id} (200 rounds)")
        
        # Run the specified rounds with weighted sampling
        for _ in range(target_rounds):
            if sim.round >= work.max_rounds:
                break
            if sim.round >= work.end_round:
                break
            
            alive_people = [p for p in sim.people if not p.is_dead]
            if len(alive_people) == 0:
                timestamp_print(f"💀 Weighted sampling simulation {sim.run_id} population extinct at round {sim.round}")
                break
            
            sim.round += 1
            rounds_completed += 1
            
            # Standard round logic with weighted interactions
            if sim._is_mixing_event_round():
                sim._handle_mixing_event(alive_people)
            
            if random.random() < sim.params.shock_frequency:
                sim._trigger_shock()
            
            sim._handle_weighted_interactions(alive_people)  # NEW: Weighted sampling
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
        timestamp_print(f"❌ Error processing weighted sampling sim {work.sim_id}: {e}")
        import traceback
        traceback.print_exc()
        return ('error', work.sim_id, str(e), 0, 0)

def run_smart_mass_experiment(num_simulations: int = 100, use_multiprocessing: bool = False) -> List[EnhancedSimulationResults]:
    """Load-balanced mass experiment with weighted sampling, chunk size 30, and save-on-completion"""
    timestamp_print(f"🚀 Starting WEIGHTED SAMPLING mass experiment with {num_simulations} simulations...")
    timestamp_print("✨ Using weighted sampling with chunk size 30 and save-on-completion")
    
    start_time = time.time()
    
    # Generate simulation parameters
    timestamp_print("🎲 Generating weighted sampling simulation parameters...")
    simulations = [generate_random_parameters(i) for i in range(num_simulations)]
    
    if not use_multiprocessing or num_simulations <= 5:
        timestamp_print("🔧 Using single-threaded execution with immediate saving")
        results = []
        results_dir = "simulation_results"
        
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for i, params in enumerate(simulations):
            timestamp_print(f"🔄 Starting weighted sampling simulation {i}")
            sim = EnhancedMassSimulation(params, i)
            result = sim.run_simulation()
            
            # Save immediately
            save_simulation_result(result, results_dir)
            save_incremental_csv(result)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_simulations - (i + 1)) / rate
                timestamp_print(f"📊 PROGRESS: {i + 1}/{num_simulations} complete ({(i+1)/num_simulations*100:.1f}%) - All saved to disk")
        
        return results
    
    # Load-balanced multi-processing approach with save-on-completion
    num_cores = min(mp.cpu_count(), 8)
    timestamp_print(f"🔧 Using {num_cores} CPU cores with chunk size 30 and immediate result saving...")
    
    # Create scheduler with chunk size 30 and results directory
    results_dir = "simulation_results"
    scheduler = LoadBalancedScheduler(simulations, chunk_size=30, results_dir=results_dir)
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        active_futures = {}
        
        # Submit initial batch of work
        for _ in range(num_cores * 2):
            work = scheduler.get_work()
            if work:
                future = executor.submit(process_simulation_work, work)
                active_futures[future] = work
        
        last_progress_time = time.time()
        
        while not scheduler.is_complete() or active_futures:
            if not active_futures and not scheduler.is_complete():
                timestamp_print("⚠️  Warning: No active work but simulations not complete")
                break
            
            if active_futures:
                try:
                    completed_futures = []
                    for future in as_completed(active_futures, timeout=10):
                        completed_futures.append(future)
                        break
                    
                    for future in completed_futures:
                        work = active_futures.pop(future)
                        
                        try:
                            result_data = future.result()
                            scheduler.submit_result(work, result_data)
                            
                            if result_data[0] == 'complete':
                                sim_id = result_data[1]
                                timestamp_print(f"✅ Weighted sampling simulation {sim_id} completed and saved!")
                            
                            # Submit new work if available
                            new_work = scheduler.get_work()
                            if new_work:
                                new_future = executor.submit(process_simulation_work, new_work)
                                active_futures[new_future] = new_work
                            
                        except Exception as e:
                            timestamp_print(f"❌ Exception processing weighted sampling work: {e}")
                            continue
                
                except TimeoutError:
                    pass
                
                # Progress reporting every 60 seconds
                current_time = time.time()
                if current_time - last_progress_time > 60:
                    _print_streamlined_progress(scheduler, active_futures, simulations)
                    last_progress_time = current_time
    
    # Load all results from disk
    timestamp_print("📂 Loading all completed results from disk...")
    final_results = scheduler.get_completed_results()
    
    # Verify we got all results
    if len(final_results) != num_simulations:
        timestamp_print(f"⚠️  Warning: Expected {num_simulations} results, got {len(final_results)}")
        missing_ids = set(range(num_simulations)) - {r.run_id for r in final_results}
        if missing_ids:
            timestamp_print(f"⚠️  Missing simulation IDs: {sorted(missing_ids)}")
    
    elapsed = time.time() - start_time
    timestamp_print(f"🎉 WEIGHTED SAMPLING EXPERIMENT COMPLETE: {len(final_results)} simulations in {elapsed:.2f} seconds")
    timestamp_print(f"⚡ Average: {elapsed/len(final_results):.1f} seconds per simulation")
    timestamp_print(f"💾 All results saved to {results_dir}/ and incremental CSV")
    
    return final_results

def _print_streamlined_progress(scheduler, active_futures, simulations):
    """Print clean, consolidated progress update"""
    completed, total = scheduler.get_progress()
    
    active_sims = []
    for future, work in active_futures.items():
        if work.sim_id in scheduler.active_simulations:
            current_round = scheduler.active_simulations[work.sim_id]
            max_rounds = simulations[work.sim_id].max_rounds
            progress_pct = int(100 * current_round / max_rounds)
            active_sims.append(f"Sim {work.sim_id} ({current_round}/{max_rounds} {progress_pct}%)")
    
    active_sims.sort(key=lambda x: int(x.split()[1]))
    
    if active_sims:
        active_str = " | Active: " + ", ".join(active_sims[:4])
        if len(active_sims) > 4:
            active_str += f" +{len(active_sims)-4} more"
    else:
        active_str = " | No active simulations"
    
    timestamp_print(f"📊 WEIGHTED SAMPLING PROGRESS: {completed}/{total} complete{active_str}")

# Keep remaining analysis functions (analyze_emergent_patterns, create_pattern_visualizations, etc.)
# but add weighted sampling metrics to the analysis...

def analyze_emergent_patterns(results: List[EnhancedSimulationResults]) -> pd.DataFrame:
    """Analyze results for emergent patterns including weighted sampling metrics"""
    timestamp_print("🔍 Analyzing emergent patterns with weighted sampling...")
    
    # Convert results to DataFrame
    data = []
    for result in results:
        # BUGFIX: Safe division to prevent division by zero
        final_pop = max(result.final_population, 1)  # Prevent division by zero
        total_weighted = (result.total_transformational_events + 
                         result.total_significant_interactions + 
                         result.total_maintenance_interactions)
        
        row = {
            'run_id': result.run_id,
            
            # All original parameters preserved
            'initial_population': result.parameters.initial_population,
            'max_population': result.parameters.max_population,
            'pop_multiplier': result.parameters.max_population / max(result.parameters.initial_population, 1),  # BUGFIX: Safe division
            'shock_frequency': result.parameters.shock_frequency,
            'pressure_multiplier': result.parameters.pressure_multiplier,
            'birth_rate': result.parameters.base_birth_rate,
            'max_rounds': result.parameters.max_rounds,
            'maslow_variation': result.parameters.maslow_variation,
            'threshold_range': result.parameters.constraint_threshold_range[1] - result.parameters.constraint_threshold_range[0],
            'threshold_min': result.parameters.constraint_threshold_range[0],
            'threshold_max': result.parameters.constraint_threshold_range[1],
            'recovery_threshold': result.parameters.recovery_threshold,
            'cooperation_bonus': result.parameters.cooperation_bonus,
            'trust_threshold': result.parameters.trust_threshold,
            'relationship_memory': getattr(result.parameters, 'relationship_memory', 10),  # BUGFIX: Safe attribute access
            
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
            'redemption_rate': result.total_redemptions / max(1, result.total_defections),  # BUGFIX: Safe division
            'avg_trust_level': result.avg_trust_level,
            'cooperation_benefit_total': result.cooperation_benefit_total,
            
            # Maslow changes preserved
            'physiological_change': result.needs_improvement.get('physiological', 0),  # BUGFIX: Safe dict access
            'safety_change': result.needs_improvement.get('safety', 0),
            'love_change': result.needs_improvement.get('love', 0),
            'esteem_change': result.needs_improvement.get('esteem', 0),
            'self_actualization_change': result.needs_improvement.get('self_actualization', 0),
            
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
            
            # NEW: Weighted Sampling Metrics
            'total_transformational_events': result.total_transformational_events,
            'total_significant_interactions': result.total_significant_interactions,
            'total_maintenance_interactions': result.total_maintenance_interactions,
            'transformational_events_per_quarter': getattr(result.parameters, 'transformational_events_per_quarter', 2),
            'significant_interactions_per_quarter': getattr(result.parameters, 'significant_interactions_per_quarter', 12),
            'maintenance_interactions_per_quarter': getattr(result.parameters, 'maintenance_interactions_per_quarter', 10),
            
            # Calculate interaction intensities with safe division
            'total_weighted_interactions': total_weighted,
            'transformational_ratio': result.total_transformational_events / max(1, total_weighted),  # BUGFIX: Safe division
            'significant_ratio': result.total_significant_interactions / max(1, total_weighted),
            'maintenance_ratio': result.total_maintenance_interactions / max(1, total_weighted),
            'interaction_intensity': total_weighted / final_pop,  # BUGFIX: Safe division
        }
        data.append(row)
    
    if not data:  # BUGFIX: Handle empty results
        timestamp_print("⚠️  No simulation data to analyze")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Create outcome categories
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
    df['pressure_index'] = df['shock_frequency'] * df['pressure_multiplier']
    df['growth_potential'] = df['birth_rate'] * df['pop_multiplier']
    df['resilience_index'] = df['threshold_range'] * (1 - df['pressure_index'])
    
    # Inter-group tension index
    df['intergroup_tension'] = (df['out_group_constraint_amplifier'] * 
                               df['reputational_spillover'] * 
                               (1 - df['out_group_trust_modifier']))
    
    # NEW: Weighted sampling indices
    df['interaction_complexity_index'] = (
        df['transformational_ratio'] * 0.005 +
        df['significant_ratio'] * 0.003 +
        df['maintenance_ratio'] * 0.001
    )
    
    df['interaction_intensity'] = df['total_weighted_interactions'] / df['final_population']
    
    return df

def main():
    """Run the complete enhanced mass experiment with weighted sampling and save-on-completion"""
    
    parser = argparse.ArgumentParser(
        description='Enhanced Constraint Cascade Simulation - Weighted Sampling v2 with Save-on-Completion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NEW FEATURES:
- Weighted Sampling: 2 transformational + 12 significant + 10 maintenance per quarter
- Max population: 800 (fixed)
- Simulation length: 200 rounds (50 years)
- Relationship memory: 15-20 interactions
- More frequent mixing events (every 15 rounds)
- Chunking at 30 rounds for load balancing
- Save-on-completion: Each simulation saved immediately when done
- Incremental CSV: Real-time progress tracking
- Crash recovery: Resume from saved results

Examples:
  python constraint_simulation_v2.py --num-runs 200 --multiprocessing
  python constraint_simulation_v2.py -n 50 --single-thread
  python constraint_simulation_v2.py -n 1000 -m --resume
        """
    )
    
    parser.add_argument('-n', '--num-runs', type=int, default=100,
                        help='Number of simulation runs to execute (default: 100)')
    parser.add_argument('-m', '--multiprocessing', action='store_true',
                        help='Enable multiprocessing with smart load balancing')
    parser.add_argument('--single-thread', action='store_true',
                        help='Force single-threaded execution (overrides --multiprocessing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing saved results (skip completed simulations)')
    
    args = parser.parse_args()
    
    # Configuration from arguments
    num_simulations = args.num_runs
    
    # Check for existing results if resume mode
    if args.resume:
        existing_results = load_all_simulation_results()
        if existing_results:
            timestamp_print(f"🔄 RESUME MODE: Found {len(existing_results)} existing results")
            if len(existing_results) >= num_simulations:
                timestamp_print(f"✅ All simulations already complete! Proceeding to analysis...")
                # Skip to analysis phase
                df = analyze_emergent_patterns(existing_results[:num_simulations])
                timestamp_print(f"📊 Loaded {len(df)} simulation records from saved results")
                
                # Create visualizations
                timestamp_print(f"📈 Creating weighted sampling visualizations...")
                create_pattern_visualizations(df)
                timestamp_print(f"✅ Weighted sampling pattern analysis completed")
                
                # Identify thresholds  
                timestamp_print(f"🎯 Identifying critical thresholds...")
                thresholds = identify_critical_thresholds(df)
                
                # Save comprehensive results
                timestamp_print(f"💾 Saving comprehensive analysis...")
                saved_files = save_comprehensive_results(df, thresholds)
                timestamp_print(f"✅ Analysis complete: {len(saved_files)} files saved")
                return
            else:
                timestamp_print(f"⚠️  Only {len(existing_results)} of {num_simulations} simulations complete")
                timestamp_print(f"🚀 Continuing with remaining {num_simulations - len(existing_results)} simulations...")
        else:
            timestamp_print(f"⚠️  No existing results found, starting fresh...")
    
    # Determine multiprocessing usage
    if args.single_thread:
        use_multiprocessing = False
    elif args.multiprocessing:
        use_multiprocessing = True
    else:
        use_multiprocessing = num_simulations >= 10
    
    timestamp_print("🔬 Enhanced Constraint Cascade Simulation - WEIGHTED SAMPLING v2")
    timestamp_print("="*80)
    timestamp_print("🎯 Weighted Sampling: 72x more realistic social interaction frequency")
    timestamp_print("⚡ Transformational: 2 per quarter (complex cascade, strategy switches)")
    timestamp_print("🤝 Significant: 12 per quarter (standard trust updates, cooperation)")
    timestamp_print("🔄 Maintenance: 10 per quarter (basic trust, routine cooperation)")
    timestamp_print("🏠 Max population: 800 (realistic community size)")
    timestamp_print("📅 Simulation length: 200 rounds (50 years, 1 round = 1 quarter)")
    timestamp_print("🧠 Relationship memory: 15-20 interactions (increased from 10)")
    timestamp_print("🎭 Mixing events: Every 15 rounds (more frequent cross-group contact)")
    timestamp_print("⚡ Load balancing: Chunking at 30 rounds for optimal performance")
    timestamp_print("💾 Save-on-completion: Each simulation saved immediately when done")
    timestamp_print("📊 Incremental tracking: Real-time CSV updates + crash recovery")
    timestamp_print(f"📂 Working directory: {os.getcwd()}")
    
    timestamp_print(f"\n⚙️  Experiment Configuration:")
    timestamp_print(f"   🔢 Number of simulations: {num_simulations}")
    timestamp_print(f"   👥 Population range: 100-500 (max 800)")
    timestamp_print(f"   🎛️  Parameters: Fully randomized with weighted sampling")
    timestamp_print(f"   🆕 Inter-group features: 80% of simulations include group dynamics")
    timestamp_print(f"   🏷️  Groups: 2-4 groups with randomized distributions")
    timestamp_print(f"   🔗 Homophily: 0-100% same-group preference (kept as-is)")
    timestamp_print(f"   ⚖️  Trust asymmetry: In-group vs out-group modifiers")
    timestamp_print(f"   ⚡ Out-group surcharge: 1.0x-3.0x constraint amplification")
    timestamp_print(f"   📢 Reputational spillover: 0-30% collective blame")
    timestamp_print(f"   🎭 Mixing events: More frequent cross-group institutions")
    timestamp_print(f"   🖥️  Multiprocessing: {'✅ ENABLED' if use_multiprocessing else '❌ DISABLED'} (Chunk size: 30)")
    timestamp_print(f"   💾 Save mode: {'🔄 RESUME' if args.resume else '🆕 FRESH START'}")
    
    try:
        # Run mass experiment with weighted sampling and save-on-completion
        timestamp_print(f"\n🚀 PHASE 1: Running {num_simulations} weighted sampling simulations with save-on-completion...")
        
        results = run_smart_mass_experiment(num_simulations, use_multiprocessing)
        
        timestamp_print(f"✅ Phase 1 complete: {len(results)} weighted sampling simulations finished and saved")
        
        # Analyze patterns
        timestamp_print(f"\n📊 PHASE 2: Analyzing emergent patterns with weighted sampling metrics...")
        df = analyze_emergent_patterns(results)
        timestamp_print(f"✅ Phase 2 complete: {len(df)} weighted sampling simulation records analyzed")
        
        # Create visualizations
        timestamp_print(f"\n📈 PHASE 3: Creating weighted sampling visualizations...")
        create_pattern_visualizations(df)
        timestamp_print(f"✅ Phase 3 complete: Weighted sampling pattern analysis completed")
        
        # Identify thresholds
        timestamp_print(f"\n🎯 PHASE 4: Identifying critical thresholds...")
        thresholds = identify_critical_thresholds(df)
        timestamp_print(f"✅ Phase 4 complete: Weighted sampling thresholds identified")
        
        # Save results
        timestamp_print(f"\n💾 PHASE 5: Saving comprehensive weighted sampling results...")
        saved_files = save_comprehensive_results(df, thresholds)
        timestamp_print(f"✅ Phase 5 complete: {len(saved_files)} files saved")
        
        timestamp_print(f"\n🎉 WEIGHTED SAMPLING EXPERIMENT COMPLETE!")
        timestamp_print(f"📊 Analyzed {len(results)} simulations with weighted sampling + inter-group dynamics")
        timestamp_print(f"🔍 Discovered patterns across {len(df.columns)} measured variables")
        timestamp_print(f"💾 Saved detailed results for further research")
        timestamp_print(f"🛡️  All results preserved with save-on-completion (crash-safe)")
        
        # Summary statistics
        has_intergroup = df['has_intergroup'].any()
        standard_sims = len(df[~df['has_intergroup']]) if has_intergroup else len(df)
        intergroup_sims = len(df[df['has_intergroup']]) if has_intergroup else 0
        
        timestamp_print(f"\n📋 Key Findings:")
        timestamp_print(f"   🔄 Standard simulations: {standard_sims}")
        if has_intergroup:
            timestamp_print(f"   🆕 Inter-group simulations: {intergroup_sims}")
        timestamp_print(f"   🤝 Average cooperation: {df['final_cooperation_rate'].mean():.3f}")
        timestamp_print(f"   ♻️  Average redemption rate: {df['redemption_rate'].mean():.3f}")
        timestamp_print(f"   📈 Scenarios with redemptions: {(df['total_redemptions'] > 0).sum()}")
        timestamp_print(f"   🤝 Average trust level: {df['avg_trust_level'].mean():.3f}")
        
        # NEW: Weighted sampling specific metrics
        timestamp_print(f"   🆕 Weighted Sampling Metrics:")
        timestamp_print(f"      ⚡ Avg transformational events: {df['total_transformational_events'].mean():.0f}")
        timestamp_print(f"      🤝 Avg significant interactions: {df['total_significant_interactions'].mean():.0f}")
        timestamp_print(f"      🔄 Avg maintenance interactions: {df['total_maintenance_interactions'].mean():.0f}")
        timestamp_print(f"      📊 Avg total weighted interactions: {df['total_weighted_interactions'].mean():.0f}")
        timestamp_print(f"      🎯 Avg interaction intensity: {df['interaction_intensity'].mean():.1f} per person")
        
        if has_intergroup:
            intergroup_df = df[df['has_intergroup']]
            timestamp_print(f"   🆕 Inter-group specific:")
            timestamp_print(f"      🔗 Average trust asymmetry: {intergroup_df['trust_asymmetry'].mean():.3f}")
            timestamp_print(f"      🏘️  Average segregation index: {intergroup_df['group_segregation_index'].mean():.3f}")
            timestamp_print(f"      🎭 Scenarios with mixing events: {(intergroup_df['total_mixing_events'] > 0).sum()}")
            timestamp_print(f"      💀 Group extinction events: {intergroup_df['group_extinction_events'].sum()}")
        
    except Exception as e:
        timestamp_print(f"\n❌ ERROR in weighted sampling experiment: {e}")
        timestamp_print(f"🔧 Please check the error message above for debugging")
        timestamp_print(f"💾 Note: Any completed simulations are saved and can be resumed with --resume")
        traceback.print_exc()

def create_pattern_visualizations(df: pd.DataFrame):
    """Create comprehensive pattern analysis visualizations for weighted sampling"""
    timestamp_print("📊 Creating weighted sampling pattern analysis visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # Determine how many plots we need
    has_intergroup_data = df['has_intergroup'].any()
    has_weighted_data = 'total_weighted_interactions' in df.columns
    
    total_plots = 20 if has_intergroup_data else 16
    if has_weighted_data:
        total_plots += 4  # Add weighted sampling specific plots
    
    # Calculate grid size
    if total_plots <= 16:
        rows, cols = 4, 4
    elif total_plots <= 20:
        rows, cols = 5, 4
    else:
        rows, cols = 6, 4
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, rows * 5))
    
    # 1. Parameter Space Overview (with weighted sampling)
    ax1 = plt.subplot(rows, cols, 1)
    scatter = plt.scatter(df['shock_frequency'], df['pressure_multiplier'], 
                         c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6,
                         s=df['total_weighted_interactions']/100)  # Size by interaction count
    plt.colorbar(scatter)
    plt.xlabel('Shock Frequency')
    plt.ylabel('Pressure Multiplier')
    plt.title('Parameter Space: Cooperation Outcomes\n(Size = Weighted Interactions)')
    
    # 2. Weighted Sampling Analysis
    ax2 = plt.subplot(rows, cols, 2)
    if has_weighted_data:
        plt.scatter(df['transformational_ratio'], df['final_cooperation_rate'], 
                   c=df['significant_ratio'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Significant Ratio')
        plt.xlabel('Transformational Event Ratio')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Interaction Type vs Cooperation')
    
    # 3. Interaction Intensity Analysis
    ax3 = plt.subplot(rows, cols, 3)
    if has_weighted_data:
        plt.scatter(df['interaction_intensity'], df['final_cooperation_rate'], alpha=0.6)
        plt.xlabel('Interaction Intensity (per person)')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Interaction Intensity vs Cooperation')
    
    # 4. Trust and Cooperation (with relationship memory)
    ax4 = plt.subplot(rows, cols, 4)
    scatter = plt.scatter(df['avg_trust_level'], df['final_cooperation_rate'], 
                         c=df['relationship_memory'], cmap='plasma', alpha=0.6)
    plt.colorbar(scatter, label='Relationship Memory')
    plt.xlabel('Average Trust Level')
    plt.ylabel('Final Cooperation Rate')
    plt.title('Trust-Cooperation Relationship\n(Color = Memory Length)')
    
    # 5-8. Interaction Type Breakdowns
    interaction_types = ['transformational', 'significant', 'maintenance']
    for i, interaction_type in enumerate(interaction_types):
        if i < 3:  # Only plot first 3
            ax = plt.subplot(rows, cols, 5 + i)
            col_name = f'total_{interaction_type}_events' if interaction_type == 'transformational' else f'total_{interaction_type}_interactions'
            if col_name in df.columns:
                plt.scatter(df[col_name], df['final_cooperation_rate'], alpha=0.6)
                plt.xlabel(f'Total {interaction_type.title()} Events')
                plt.ylabel('Final Cooperation Rate')
                plt.title(f'{interaction_type.title()} Events Impact')
    
    # 8. Weighted Sampling Effectiveness
    ax8 = plt.subplot(rows, cols, 8)
    if has_weighted_data:
        plt.scatter(df['interaction_complexity_index'], df['cooperation_benefit_total'], 
                   c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
        plt.colorbar(label='Final Cooperation')
        plt.xlabel('Interaction Complexity Index')
        plt.ylabel('Total Cooperation Benefits')
        plt.title('Complexity vs Benefits')
    
    # 9-12. Population and System Dynamics
    for i in range(4):
        ax = plt.subplot(rows, cols, 9 + i)
        if i == 0:  # Population growth vs interaction intensity
            if has_weighted_data:
                plt.scatter(df['population_growth'], df['interaction_intensity'], 
                           c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
                plt.colorbar(label='Cooperation')
                plt.xlabel('Population Growth')
                plt.ylabel('Interaction Intensity')
                plt.title('Growth vs Interaction Intensity')
        elif i == 1:  # Redemption analysis with memory
            plt.scatter(df['recovery_threshold'], df['redemption_rate'], 
                       c=df['relationship_memory'], cmap='plasma', alpha=0.6)
            plt.colorbar(label='Memory Length')
            plt.xlabel('Recovery Threshold')
            plt.ylabel('Redemption Rate')
            plt.title('Recovery Dynamics\n(Color = Memory)')
        elif i == 2:  # System stress distribution
            df['pressure_index'].hist(bins=30, alpha=0.7)
            plt.xlabel('Pressure Index')
            plt.ylabel('Frequency')
            plt.title('Pressure Distribution')
        elif i == 3:  # Cascade timing
            non_extinct = df[~df['extinction_occurred']]
            if len(non_extinct) > 0:
                plt.scatter(non_extinct['pressure_index'], non_extinct['first_cascade_round'], alpha=0.6)
                plt.xlabel('Pressure Index')
                plt.ylabel('First Cascade Round')
                plt.title('Cascade Timing vs Pressure')
    
    # 13-16. Inter-group Analysis (if available)
    if has_intergroup_data:
        intergroup_df = df[df['has_intergroup']]
        
        # 13. Homophily vs Cooperation with weighted interactions
        ax13 = plt.subplot(rows, cols, 13)
        scatter = plt.scatter(intergroup_df['homophily_bias'], intergroup_df['final_cooperation_rate'], 
                             c=intergroup_df['total_weighted_interactions'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Total Interactions')
        plt.xlabel('Homophily Bias')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Homophily vs Cooperation\n(Color = Interaction Count)')
        
        # 14. Trust Asymmetry with interaction types
        ax14 = plt.subplot(rows, cols, 14)
        scatter = plt.scatter(intergroup_df['avg_in_group_trust'], intergroup_df['avg_out_group_trust'], 
                             c=intergroup_df['transformational_ratio'], cmap='plasma', alpha=0.6)
        plt.colorbar(scatter, label='Transformational Ratio')
        plt.xlabel('Average In-Group Trust')
        plt.ylabel('Average Out-Group Trust')
        plt.title('Trust Asymmetry\n(Color = Transformational %)')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal Trust')
        plt.legend()
        
        # 15. Mixing Events vs Interaction Intensity
        ax15 = plt.subplot(rows, cols, 15)
        mixing_data = intergroup_df[intergroup_df['mixing_event_frequency'] > 0]
        if len(mixing_data) > 0:
            scatter = plt.scatter(mixing_data['mixing_event_frequency'], mixing_data['mixing_event_success_rate'], 
                                 c=mixing_data['interaction_intensity'], cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, label='Interaction Intensity')
        plt.xlabel('Mixing Event Frequency')
        plt.ylabel('Mixing Event Success Rate')
        plt.title('Mixing Events vs Intensity')
        
        # 16. Group Segregation vs Weighted Interactions
        ax16 = plt.subplot(rows, cols, 16)
        plt.scatter(intergroup_df['group_segregation_index'], intergroup_df['total_weighted_interactions'], 
                   c=intergroup_df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
        plt.colorbar(label='Cooperation')
        plt.xlabel('Group Segregation Index')
        plt.ylabel('Total Weighted Interactions')
        plt.title('Segregation vs Interactions')
    
    # Additional weighted sampling plots if space allows
    if total_plots > 16:
        # 17. Memory vs Trust Development
        ax17 = plt.subplot(rows, cols, 17)
        plt.scatter(df['relationship_memory'], df['avg_trust_level'], 
                   c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
        plt.colorbar(label='Cooperation')
        plt.xlabel('Relationship Memory Length')
        plt.ylabel('Average Trust Level')
        plt.title('Memory vs Trust Development')
        
        # 18. Interaction Balance Analysis
        ax18 = plt.subplot(rows, cols, 18)
        if has_weighted_data:
            # Create a stacked bar chart of interaction types
            bottom_sig = df['transformational_ratio'].values
            bottom_main = bottom_sig + df['significant_ratio'].values
            
            plt.bar(range(len(df)), df['transformational_ratio'], label='Transformational', alpha=0.7)
            plt.bar(range(len(df)), df['significant_ratio'], bottom=bottom_sig, label='Significant', alpha=0.7)
            plt.bar(range(len(df)), df['maintenance_ratio'], bottom=bottom_main, label='Maintenance', alpha=0.7)
            
            plt.xlabel('Simulation ID')
            plt.ylabel('Interaction Type Ratio')
            plt.title('Interaction Type Distribution')
            plt.legend()
            
        # 19. Weighted Sampling Efficiency
        ax19 = plt.subplot(rows, cols, 19)
        if has_weighted_data:
            efficiency = df['cooperation_benefit_total'] / df['total_weighted_interactions']
            plt.scatter(df['total_weighted_interactions'], efficiency, 
                       c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
            plt.colorbar(label='Cooperation')
            plt.xlabel('Total Weighted Interactions')
            plt.ylabel('Benefit per Interaction')
            plt.title('Interaction Efficiency')
        
        # 20. Memory and Complexity Combined
        ax20 = plt.subplot(rows, cols, 20)
        if has_weighted_data:
            plt.scatter(df['relationship_memory'], df['interaction_complexity_index'], 
                       c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
            plt.colorbar(label='Cooperation')
            plt.xlabel('Relationship Memory')
            plt.ylabel('Interaction Complexity Index')
            plt.title('Memory vs Complexity')
    
    plt.tight_layout()
    
    title = 'Enhanced Constraint Cascade Simulation - Weighted Sampling Analysis'
    if has_intergroup_data:
        title += '\n(Including Inter-Group Dynamics + Weighted Interactions)'
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('weighted_sampling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def identify_critical_thresholds(df: pd.DataFrame):
    """Identify critical thresholds with weighted sampling considerations"""
    timestamp_print("🎯 Identifying critical thresholds for weighted sampling...")
    
    # Find pressure threshold for cooperation collapse
    df_sorted = df.sort_values('pressure_index')
    cooperation_rates = df_sorted['final_cooperation_rate'].rolling(window=50, center=True).mean()
    
    # Find where cooperation drops below 50%
    collapse_threshold = None
    for i, rate in enumerate(cooperation_rates):
        if not pd.isna(rate) and rate < 0.5:
            collapse_threshold = df_sorted.iloc[i]['pressure_index']
            break
    
    timestamp_print("\n" + "="*60)
    timestamp_print("🔍 WEIGHTED SAMPLING THRESHOLD ANALYSIS")
    timestamp_print("="*60)
    
    # Cooperation collapse threshold
    if collapse_threshold:
        timestamp_print(f"🚨 Cooperation Collapse Threshold: {collapse_threshold:.3f}")
        timestamp_print(f"   (Shock Frequency × Pressure Multiplier)")
    
    # Population size effects with 800 max
    small_pop_coop = df[df['initial_population'] < 200]['final_cooperation_rate'].mean()
    large_pop_coop = df[df['initial_population'] > 400]['final_cooperation_rate'].mean()
    
    timestamp_print(f"👥 Population Size Effect (Max 800):")
    timestamp_print(f"   Small populations (<200): {small_pop_coop:.1%} cooperation")
    timestamp_print(f"   Large populations (>400): {large_pop_coop:.1%} cooperation")
    timestamp_print(f"   Size penalty: {(small_pop_coop - large_pop_coop):.1%}")
    
    # Weighted sampling specific thresholds
    has_weighted_data = 'total_weighted_interactions' in df.columns
    if has_weighted_data:
        timestamp_print(f"\n🆕 WEIGHTED SAMPLING THRESHOLDS:")
        timestamp_print("="*40)
        
        # Interaction intensity threshold
        high_intensity = df[df['interaction_intensity'] > df['interaction_intensity'].quantile(0.8)]
        low_intensity = df[df['interaction_intensity'] < df['interaction_intensity'].quantile(0.2)]
        
        timestamp_print(f"⚡ Interaction Intensity Effects:")
        timestamp_print(f"   High intensity (top 20%): {high_intensity['final_cooperation_rate'].mean():.1%} cooperation")
        timestamp_print(f"   Low intensity (bottom 20%): {low_intensity['final_cooperation_rate'].mean():.1%} cooperation")
        timestamp_print(f"   Intensity benefit: {(high_intensity['final_cooperation_rate'].mean() - low_intensity['final_cooperation_rate'].mean()):.1%}")
        
        # Transformational event threshold
        high_transformational = df[df['transformational_ratio'] > 0.1]
        low_transformational = df[df['transformational_ratio'] < 0.05]
        
        if len(high_transformational) > 0 and len(low_transformational) > 0:
            timestamp_print(f"⚡ Transformational Event Effects:")
            timestamp_print(f"   High transformational (>10%): {high_transformational['final_cooperation_rate'].mean():.1%} cooperation")
            timestamp_print(f"   Low transformational (<5%): {low_transformational['final_cooperation_rate'].mean():.1%} cooperation")
            timestamp_print(f"   Transformational benefit: {(high_transformational['final_cooperation_rate'].mean() - low_transformational['final_cooperation_rate'].mean()):.1%}")
        
        # Memory length effects
        high_memory = df[df['relationship_memory'] >= 18]
        low_memory = df[df['relationship_memory'] <= 15]
        
        if len(high_memory) > 0 and len(low_memory) > 0:
            timestamp_print(f"🧠 Relationship Memory Effects:")
            timestamp_print(f"   High memory (≥18): {high_memory['final_cooperation_rate'].mean():.1%} cooperation")
            timestamp_print(f"   Low memory (≤15): {low_memory['final_cooperation_rate'].mean():.1%} cooperation")
            timestamp_print(f"   Memory benefit: {(high_memory['final_cooperation_rate'].mean() - low_memory['final_cooperation_rate'].mean()):.1%}")
        
        # Interaction complexity threshold
        high_complexity = df[df['interaction_complexity_index'] > df['interaction_complexity_index'].quantile(0.8)]
        low_complexity = df[df['interaction_complexity_index'] < df['interaction_complexity_index'].quantile(0.2)]
        
        timestamp_print(f"🔄 Interaction Complexity Effects:")
        timestamp_print(f"   High complexity (top 20%): {high_complexity['final_cooperation_rate'].mean():.1%} cooperation")
        timestamp_print(f"   Low complexity (bottom 20%): {low_complexity['final_cooperation_rate'].mean():.1%} cooperation")
        timestamp_print(f"   Complexity benefit: {(high_complexity['final_cooperation_rate'].mean() - low_complexity['final_cooperation_rate'].mean()):.1%}")
    
    # Inter-group analysis (if available)
    has_intergroup_data = df['has_intergroup'].any()
    if has_intergroup_data:
        intergroup_df = df[df['has_intergroup']]
        
        timestamp_print(f"\n🆕 INTER-GROUP + WEIGHTED SAMPLING:")
        timestamp_print("="*40)
        
        # Homophily with weighted interactions
        if len(intergroup_df) > 0:
            high_homophily = intergroup_df[intergroup_df['homophily_bias'] > 0.7]
            low_homophily = intergroup_df[intergroup_df['homophily_bias'] < 0.3]
            
            if len(high_homophily) > 0 and len(low_homophily) > 0:
                timestamp_print(f"🏘️  Homophily + Weighted Interactions:")
                timestamp_print(f"   High homophily (>0.7): {high_homophily['final_cooperation_rate'].mean():.1%} cooperation")
                timestamp_print(f"   Low homophily (<0.3): {low_homophily['final_cooperation_rate'].mean():.1%} cooperation")
                timestamp_print(f"   Homophily penalty: {(low_homophily['final_cooperation_rate'].mean() - high_homophily['final_cooperation_rate'].mean()):.1%}")
                
                if has_weighted_data:
                    timestamp_print(f"   High homophily avg interactions: {high_homophily['total_weighted_interactions'].mean():.0f}")
                    timestamp_print(f"   Low homophily avg interactions: {low_homophily['total_weighted_interactions'].mean():.0f}")
    
    return {
        'cooperation_collapse_threshold': collapse_threshold,
        'population_size_effect': large_pop_coop - small_pop_coop if not pd.isna(large_pop_coop) and not pd.isna(small_pop_coop) else None,
        'has_weighted_data': has_weighted_data,
        'intensity_benefit': (high_intensity['final_cooperation_rate'].mean() - low_intensity['final_cooperation_rate'].mean()) if has_weighted_data and len(high_intensity) > 0 and len(low_intensity) > 0 else None,
        'memory_benefit': (high_memory['final_cooperation_rate'].mean() - low_memory['final_cooperation_rate'].mean()) if has_weighted_data and len(high_memory) > 0 and len(low_memory) > 0 else None,
        'complexity_benefit': (high_complexity['final_cooperation_rate'].mean() - low_complexity['final_cooperation_rate'].mean()) if has_weighted_data and len(high_complexity) > 0 and len(low_complexity) > 0 else None,
        'has_intergroup_data': has_intergroup_data,
    }

def save_comprehensive_results(df: pd.DataFrame, thresholds: Dict):
    """Save all results for weighted sampling analysis"""
    current_dir = os.getcwd()
    timestamp_print(f"💾 Saving weighted sampling results to: {current_dir}")
    
    saved_files = []
    
    try:
        # Save main dataset
        main_file = 'weighted_sampling_simulation_results.csv'
        df.to_csv(main_file, index=False)
        if os.path.exists(main_file):
            size_mb = os.path.getsize(main_file) / (1024*1024)
            saved_files.append(f"📊 {main_file} ({size_mb:.2f} MB)")
        
        # Save summary statistics
        summary_file = 'weighted_sampling_summary_stats.csv'
        summary_stats = df.describe()
        summary_stats.to_csv(summary_file)
        if os.path.exists(summary_file):
            saved_files.append(f"📈 {summary_file}")
        
        # Save threshold analysis
        threshold_file = 'weighted_sampling_thresholds.txt'
        with open(threshold_file, 'w') as f:
            f.write("Weighted Sampling Critical Threshold Analysis\n")
            f.write("="*50 + "\n\n")
            for key, value in thresholds.items():
                f.write(f"{key}: {value}\n")
        if os.path.exists(threshold_file):
            saved_files.append(f"🎯 {threshold_file}")
        
        # Save high cooperation scenarios
        high_coop = df[df['final_cooperation_rate'] > 0.8]
        if len(high_coop) > 0:
            high_file = 'weighted_sampling_high_cooperation.csv'
            high_coop.to_csv(high_file, index=False)
            if os.path.exists(high_file):
                saved_files.append(f"✅ {high_file} ({len(high_coop)} scenarios)")
        
        # Save low cooperation scenarios (RESTORED from previous version)
        low_coop = df[df['final_cooperation_rate'] < 0.2]
        if len(low_coop) > 0:
            low_file = 'weighted_sampling_low_cooperation.csv'
            low_coop.to_csv(low_file, index=False)
            if os.path.exists(low_file):
                saved_files.append(f"❌ {low_file} ({len(low_coop)} scenarios)")
        
        # Save high redemption scenarios (RESTORED from previous version)
        high_redemption = df[df['redemption_rate'] > 0.5]
        if len(high_redemption) > 0:
            redemption_file = 'high_redemption_scenarios.csv'
            high_redemption.to_csv(redemption_file, index=False)
            if os.path.exists(redemption_file):
                saved_files.append(f"♻️  {redemption_file} ({len(high_redemption)} scenarios)")
        
        # Save high interaction intensity scenarios (NEW for weighted sampling)
        has_weighted_data = 'total_weighted_interactions' in df.columns
        if has_weighted_data:
            high_intensity = df[df['interaction_intensity'] > df['interaction_intensity'].quantile(0.9)]
            if len(high_intensity) > 0:
                intensity_file = 'high_interaction_intensity_scenarios.csv'
                high_intensity.to_csv(intensity_file, index=False)
                if os.path.exists(intensity_file):
                    saved_files.append(f"⚡ {intensity_file} ({len(high_intensity)} scenarios)")
        
        # Create comprehensive summary
        summary_report = 'weighted_sampling_experiment_summary.txt'
        with open(summary_report, 'w') as f:
            f.write("Enhanced Constraint Cascade - Weighted Sampling Experiment Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Simulations: {len(df)}\n")
            f.write(f"Average Cooperation Rate: {df['final_cooperation_rate'].mean():.3f}\n")
            f.write(f"Extinction Rate: {df['extinction_occurred'].mean():.3f}\n")
            f.write(f"Average Final Population: {df['final_population'].mean():.1f}\n")
            f.write(f"Max Population Limit: 800 (fixed)\n")
            f.write(f"Simulation Length: 200 rounds (50 years)\n")
            f.write(f"Relationship Memory: {df['relationship_memory'].min()}-{df['relationship_memory'].max()} interactions\n")
            f.write(f"High Cooperation Scenarios: {len(high_coop)}\n")
            f.write(f"Low Cooperation Scenarios: {len(low_coop)}\n")
            
            if has_weighted_data:
                f.write(f"\nWeighted Sampling Metrics:\n")
                f.write(f"Average Transformational Events: {df['total_transformational_events'].mean():.0f}\n")
                f.write(f"Average Significant Interactions: {df['total_significant_interactions'].mean():.0f}\n")
                f.write(f"Average Maintenance Interactions: {df['total_maintenance_interactions'].mean():.0f}\n")
                f.write(f"Average Total Weighted Interactions: {df['total_weighted_interactions'].mean():.0f}\n")
                f.write(f"Average Interaction Intensity: {df['interaction_intensity'].mean():.1f} per person\n")
                f.write(f"Average Interaction Complexity Index: {df['interaction_complexity_index'].mean():.4f}\n")
            
            # Add enhanced metrics for compatibility with previous version
            f.write(f"\nEnhanced Metrics:\n")
            f.write(f"Average Redemption Rate: {df['redemption_rate'].mean():.3f}\n")
            f.write(f"Average Trust Level: {df['avg_trust_level'].mean():.3f}\n")
            f.write(f"Total Defections: {df['total_defections'].sum()}\n")
            f.write(f"Total Redemptions: {df['total_redemptions'].sum()}\n")
            f.write(f"High Redemption Scenarios: {len(high_redemption)}\n")
            
            has_intergroup = df['has_intergroup'].any()
            if has_intergroup:
                intergroup_df = df[df['has_intergroup']]
                f.write(f"\nInter-Group Dynamics:\n")
                f.write(f"Simulations with Inter-Group Features: {len(intergroup_df)}\n")
                f.write(f"Average Trust Asymmetry: {intergroup_df['trust_asymmetry'].mean():.3f}\n")
                f.write(f"Average Segregation Index: {intergroup_df['group_segregation_index'].mean():.3f}\n")
                f.write(f"Total Mixing Events: {intergroup_df['total_mixing_events'].sum()}\n")
                f.write(f"Average In-Group Trust: {intergroup_df['avg_in_group_trust'].mean():.3f}\n")
                f.write(f"Average Out-Group Trust: {intergroup_df['avg_out_group_trust'].mean():.3f}\n")
            
            f.write(f"\nFiles Created:\n")
            for file_info in saved_files:
                f.write(f"  {file_info}\n")
        
        if os.path.exists(summary_report):
            saved_files.append(f"📋 {summary_report}")
        
        timestamp_print(f"\n✅ Successfully saved {len(saved_files)} files:")
        for file_info in saved_files:
            timestamp_print(f"   {file_info}")
        
        timestamp_print(f"\n📂 Full path: {current_dir}")
        
    except Exception as e:
        timestamp_print(f"❌ Error saving files: {e}")
        traceback.print_exc()
        
        # Try to save just the main file as a backup
        try:
            backup_file = 'weighted_sampling_backup.csv'
            df.to_csv(backup_file, index=False)
            timestamp_print(f"💾 Backup saved as: {backup_file}")
        except Exception as backup_error:
            timestamp_print(f"❌ Backup also failed: {backup_error}")
    
    return saved_files

if __name__ == "__main__":
    main()