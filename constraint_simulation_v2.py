#!/usr/bin/env python3
"""
Complete Enhanced Constraint Cascade Simulation with Inter-Group Dynamics
Full fidelity simulation with comprehensive analysis and reporting
EXTENDS the original with: Group Tags, Homophily, Group-Weighted Trust, 
Out-group Surcharge, Reputational Spillover, and Inter-group Institutions
"""

import random
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Try to import seaborn for better visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
    print("✅ Seaborn loaded successfully")
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Seaborn not available - using matplotlib only")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  SciPy not available - some statistical analysis will be limited")

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
    """Enhanced relationship tracking with group awareness (extends original)"""
    trust: float = 0.5
    interaction_count: int = 0
    cooperation_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_interaction_round: int = 0
    
    # NEW: Inter-group extensions
    is_same_group: bool = True
    betrayal_count: int = 0
    cooperation_count: int = 0
    
    def update_trust(self, cooperated: bool, round_num: int, 
                    in_group_modifier: float = 1.0, out_group_modifier: float = 1.0):
        """Update trust based on interaction outcome with optional group-based modifiers"""
        self.interaction_count += 1
        self.last_interaction_round = round_num
        self.cooperation_history.append(cooperated)
        
        # Apply group-based trust modifiers (defaults to original behavior if not specified)
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
    """Enhanced simulation parameters (extends original with inter-group dynamics)"""
    initial_population: int
    max_population: int
    shock_frequency: float
    pressure_multiplier: float
    base_birth_rate: float
    max_rounds: int
    maslow_variation: float
    constraint_threshold_range: Tuple[float, float]
    recovery_threshold: float = 0.3
    cooperation_bonus: float = 0.2
    trust_threshold: float = 0.6
    relationship_memory: int = 10
    max_relationships_per_person: int = 150
    interaction_batch_size: int = 50
    
    # NEW: Inter-Group Parameters
    num_groups: int = 3  # Number of distinct groups (A, B, C, etc.)
    founder_group_distribution: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])  # Initial group proportions
    homophily_bias: float = 0.7  # h ∈ [0,1] - probability of choosing same-group partner
    in_group_trust_modifier: float = 1.5  # Amplifier for in-group trust changes
    out_group_trust_modifier: float = 0.5  # Dampener for out-group trust changes
    out_group_constraint_amplifier: float = 2.0  # λ > 1 - amplify constraint from out-group betrayals
    reputational_spillover: float = 0.1  # σ - trust reduction to entire group when one member defects
    mixing_event_frequency: int = 25  # M - rounds between mixing events
    mixing_event_bonus_multiplier: float = 2.0  # Double cooperation bonus at mixing events
    inheritance_style: str = "mother"  # "mother", "father", "random", or "majority"

@dataclass
class EnhancedSimulationResults:
    """Comprehensive results container (extends original with inter-group metrics)"""
    parameters: SimulationParameters
    run_id: int
    
    # ORIGINAL: Final outcomes
    final_population: int
    final_cooperation_rate: float
    final_constrained_rate: float
    
    # ORIGINAL: System dynamics
    rounds_completed: int
    extinction_occurred: bool
    first_cascade_round: Optional[int]
    total_cascade_events: int
    total_shock_events: int
    
    # ORIGINAL: Strategy changes
    total_defections: int
    total_redemptions: int
    net_strategy_change: int
    
    # ORIGINAL: Population metrics
    total_births: int
    total_deaths: int
    max_population_reached: int
    population_stability: float
    
    # ORIGINAL: Pressure metrics
    avg_system_stress: float
    max_system_stress: float
    avg_maslow_pressure: float
    avg_basic_needs_crisis_rate: float
    
    # ORIGINAL: Maslow evolution
    initial_needs_avg: Dict[str, float]
    final_needs_avg: Dict[str, float]
    needs_improvement: Dict[str, float]
    
    # ORIGINAL: Cooperation benefits
    avg_trust_level: float
    cooperation_benefit_total: float
    
    # ORIGINAL: Additional metrics for compatibility
    population_growth: float
    cooperation_resilience: float
    
    # NEW: Inter-Group Metrics (extensions)
    final_group_populations: Dict[str, int] = field(default_factory=dict)
    final_group_cooperation_rates: Dict[str, float] = field(default_factory=dict)
    in_group_interaction_rate: float = 0.0
    out_group_interaction_rate: float = 0.0
    avg_in_group_trust: float = 0.5
    avg_out_group_trust: float = 0.5
    group_segregation_index: float = 0.0  # How segregated the groups became
    total_mixing_events: int = 0
    mixing_event_success_rate: float = 0.0
    reputational_spillover_events: int = 0
    out_group_constraint_amplifications: int = 0
    group_extinction_events: int = 0  # How many groups went extinct
    trust_asymmetry: float = 0.0  # In-group trust - out-group trust

class OptimizedPerson:
    """Enhanced person with optional group identity (extends original)"""
    
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
        
        # NEW: Group identity and tracking
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
        """NEW: Determine child's group based on inheritance style"""
        if inheritance_style == "mother":
            return parent_a.group_id  # Assume parent_a is mother
        elif inheritance_style == "father":
            return parent_b.group_id  # Assume parent_b is father
        elif inheritance_style == "random":
            return random.choice([parent_a.group_id, parent_b.group_id])
        elif inheritance_style == "majority":
            # This would require group population counts - simplified to random for now
            return random.choice([parent_a.group_id, parent_b.group_id])
        else:
            return parent_a.group_id  # Default to mother
    
    def _inherit_traits(self, parent_a_needs: MaslowNeeds, parent_b_needs: MaslowNeeds, 
                       variation: float, parent_a: Optional['OptimizedPerson'] = None, 
                       parent_b: Optional['OptimizedPerson'] = None) -> MaslowNeeds:
        """ORIGINAL: Inherit traits with variation"""
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
        """ORIGINAL: Optimized pressure calculation"""
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
        """ORIGINAL: Update person state with full fidelity"""
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
        """EXTENDED: Add pressure with Maslow amplification and optional out-group surcharge"""
        if self.is_dead:
            return False
        
        maslow_amplifier = 1 + (self.maslow_pressure * 0.5)
        
        # NEW: Apply out-group surcharge if applicable
        if is_from_out_group:
            amount *= out_group_amplifier
        
        self.constraint_level += amount * maslow_amplifier
        
        if self.strategy == 'cooperative' and self.constraint_level > self.constraint_threshold:
            self.force_switch()
            return True
        return False
    
    def check_for_recovery(self) -> bool:
        """ORIGINAL: Check if person can recover to cooperative strategy"""
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
        """ORIGINAL: Force switch to selfish strategy"""
        self.strategy = 'selfish'
        self.is_constrained = True
        self.strategy_changes += 1
        self.maslow_needs.love *= 0.8
        self.maslow_needs.esteem *= 0.7
    
    def switch_to_cooperative(self):
        """ORIGINAL: Recover to cooperative strategy"""
        self.strategy = 'cooperative'
        self.is_constrained = False
        self.strategy_changes += 1
        self.rounds_as_selfish = 0
        self.maslow_needs.love = min(10, self.maslow_needs.love * 1.1)
        self.maslow_needs.esteem = min(10, self.maslow_needs.esteem * 1.1)
    
    def get_relationship(self, other_id: int, round_num: int, 
                        other_group_id: Optional[str] = None) -> FastRelationship:
        """EXTENDED: Get or create relationship with optional group awareness"""
        if other_id not in self.relationships:
            if len(self.relationships) >= 150:
                oldest_id = min(self.relationships.keys(), 
                              key=lambda k: self.relationships[k].last_interaction_round)
                del self.relationships[oldest_id]
            
            # NEW: Set group relationship status
            is_same_group = (other_group_id is None or self.group_id == other_group_id)
            self.relationships[other_id] = FastRelationship(is_same_group=is_same_group)
        return self.relationships[other_id]
    
    def calculate_cooperation_decision(self, other: 'OptimizedPerson', round_num: int) -> bool:
        """EXTENDED: Decide whether to cooperate based on relationship, strategy, and group"""
        if self.strategy == 'selfish':
            return False
        
        relationship = self.get_relationship(other.id, round_num, other.group_id)
        
        if relationship.interaction_count == 0:
            # NEW: First interaction - bias based on group membership
            base_coop_prob = self.maslow_needs.love / 10
            if hasattr(self, 'group_id') and hasattr(other, 'group_id'):
                if self.group_id == other.group_id:
                    base_coop_prob *= 1.2  # 20% bonus for in-group
                else:
                    base_coop_prob *= 0.8  # 20% penalty for out-group
            return random.random() < base_coop_prob
        else:
            recent_coop = sum(list(relationship.cooperation_history)[-3:]) / min(3, len(relationship.cooperation_history))
            cooperation_prob = relationship.trust * 0.7 + recent_coop * 0.3
            return random.random() < cooperation_prob
    
    def _get_basic_needs_pressure(self) -> float:
        """ORIGINAL: Calculate basic needs pressure"""
        return (max(0, 5 - self.maslow_needs.physiological) * 0.002 + 
                max(0, 5 - self.maslow_needs.safety) * 0.001)
    
    def _get_inspire_effect(self) -> float:
        """ORIGINAL: Calculate inspiration effect"""
        return max(0, self.maslow_needs.self_actualization - 7) * 0.001

class EnhancedMassSimulation:
    """Enhanced simulation with comprehensive tracking (extends original with inter-group dynamics)"""
    
    def __init__(self, params: SimulationParameters, run_id: int):
        self.params = params
        self.run_id = run_id
        self.people: List[OptimizedPerson] = []
        self.round = 0
        self.system_stress = 0.0
        self.next_person_id = params.initial_population + 1
        
        # ORIGINAL: Tracking variables
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
        
        # NEW: Inter-group tracking
        self.group_names = [chr(65 + i) for i in range(params.num_groups)]  # A, B, C, ...
        self.in_group_interactions = 0
        self.out_group_interactions = 0
        self.total_mixing_events = 0
        self.successful_mixing_events = 0
        self.reputational_spillover_events = 0
        self.out_group_constraint_amplifications = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """EXTENDED: Initialize population with optional group distribution"""
        if hasattr(self.params, 'num_groups') and self.params.num_groups > 1:
            self._initialize_population_with_groups()
        else:
            # ORIGINAL: Standard initialization
            for i in range(1, self.params.initial_population + 1):
                person = OptimizedPerson(i, self.params)
                self.people.append(person)
    
    def _initialize_population_with_groups(self):
        """NEW: Initialize population with specified group distribution"""
        group_sizes = []
        remaining_pop = self.params.initial_population
        
        # Calculate group sizes based on distribution
        for i, proportion in enumerate(self.params.founder_group_distribution):
            if i == len(self.params.founder_group_distribution) - 1:
                # Last group gets remaining population
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
    
    def _select_interaction_partner(self, person: OptimizedPerson, 
                                  alive_people: List[OptimizedPerson]) -> Optional[OptimizedPerson]:
        """NEW: Select interaction partner with homophily bias"""
        if not hasattr(self.params, 'homophily_bias'):
            # ORIGINAL: No homophily - use original selection logic
            available_partners = [p for p in alive_people if p.id != person.id]
            if not available_partners:
                return None
            
            if person.relationships and random.random() < 0.3:
                known_alive = [p for p in alive_people 
                             if p.id in person.relationships and p.id != person.id]
                if known_alive:
                    return random.choice(known_alive)
            return random.choice(available_partners)
        
        # NEW: Apply homophily bias
        available_partners = [p for p in alive_people if p.id != person.id]
        if not available_partners:
            return None
        
        # Apply homophily bias
        if random.random() < self.params.homophily_bias:
            # Try to find same-group partner
            same_group_partners = [p for p in available_partners if p.group_id == person.group_id]
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
    
    def _apply_reputational_spillover(self, defector: OptimizedPerson, alive_people: List[OptimizedPerson]):
        """NEW: Apply reputational spillover when someone defects"""
        if not hasattr(self.params, 'reputational_spillover') or self.params.reputational_spillover <= 0:
            return
        
        self.reputational_spillover_events += 1
        
        # Reduce trust toward all members of defector's group
        for person in alive_people:
            if person.id != defector.id and not person.is_dead:
                for other_id, relationship in person.relationships.items():
                    # Find the other person
                    other = next((p for p in alive_people if p.id == other_id), None)
                    if other and hasattr(other, 'group_id') and other.group_id == defector.group_id:
                        # Reduce trust toward this group member
                        relationship.trust = max(0.0, relationship.trust - self.params.reputational_spillover)
    
    def _is_mixing_event_round(self) -> bool:
        """NEW: Check if this round should have a mixing event"""
        return (hasattr(self.params, 'mixing_event_frequency') and 
                self.params.mixing_event_frequency > 0 and 
                self.round > 0 and 
                self.round % self.params.mixing_event_frequency == 0)
    
    def _handle_mixing_event(self, alive_people: List[OptimizedPerson]):
        """NEW: Handle inter-group mixing event"""
        self.total_mixing_events += 1
        
        # Create cross-group pairs
        group_buckets = defaultdict(list)
        for person in alive_people:
            if hasattr(person, 'group_id'):
                group_buckets[person.group_id].append(person)
            else:
                group_buckets['default'].append(person)
        
        # Only proceed if we have multiple groups
        if len(group_buckets) < 2:
            return
        
        # Create cross-group interactions
        interactions_created = 0
        max_interactions = len(alive_people) // 3  # Limit mixing event size
        
        for _ in range(max_interactions):
            # Pick two different groups
            groups = list(group_buckets.keys())
            if len(groups) < 2:
                break
            
            group1, group2 = random.sample(groups, 2)
            
            if group_buckets[group1] and group_buckets[group2]:
                person1 = random.choice(group_buckets[group1])
                person2 = random.choice(group_buckets[group2])
                
                # Mark as mixing event participation
                if hasattr(person1, 'mixing_event_participations'):
                    person1.mixing_event_participations += 1
                if hasattr(person2, 'mixing_event_participations'):
                    person2.mixing_event_participations += 1
                
                # Process interaction with enhanced cooperation bonus
                success = self._process_interaction(person1, person2, is_mixing_event=True)
                if success:
                    interactions_created += 1
        
        if interactions_created > 0:
            self.successful_mixing_events += 1
    
    def run_simulation(self) -> EnhancedSimulationResults:
        """ORIGINAL: Run enhanced simulation (with inter-group extensions)"""
        print(f"🎮 Starting simulation run {self.run_id}")
        initial_trait_avg = self._get_average_traits()
        initial_group_populations = self._get_group_populations()
        print(f"📊 Initial setup complete for sim {self.run_id}")
        
        while self.round < self.params.max_rounds:
            self.round += 1
            
            # Debug output every 50 rounds
            if self.round % 50 == 0:
                print(f"🔄 Sim {self.run_id}: Round {self.round}/{self.params.max_rounds}")
            
            alive_people = [p for p in self.people if not p.is_dead]
            if len(alive_people) == 0:
                print(f"💀 Sim {self.run_id}: Population extinct at round {self.round}")
                break
            
            # NEW: Check for mixing event
            if self._is_mixing_event_round():
                self._handle_mixing_event(alive_people)
            
            if random.random() < self.params.shock_frequency:
                self._trigger_shock()
            
            self._handle_interactions()
            self._check_recoveries()
            self._update_population()
            self._collect_round_data()
            
            self.system_stress = max(0, self.system_stress - 0.01)
        
        print(f"🏁 Sim {self.run_id}: Completed {self.round} rounds, generating results...")
        return self._generate_results(initial_trait_avg, initial_group_populations)
    
    def _trigger_shock(self):
        """ORIGINAL: Apply system shock"""
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
    
    def _handle_interactions(self):
        """EXTENDED: Optimized interaction handling with optional homophily"""
        alive_people = [p for p in self.people if not p.is_dead]
        if len(alive_people) < 2:
            return
        
        num_interactions = max(len(alive_people) // 4, 10)
        
        for _ in range(num_interactions):
            if len(alive_people) >= 2:
                person1 = random.choice(alive_people)
                person2 = self._select_interaction_partner(person1, alive_people)
                
                if person2:
                    self._process_interaction(person1, person2)
    
    def _process_interaction(self, person1: OptimizedPerson, person2: OptimizedPerson, 
                           is_mixing_event: bool = False) -> bool:
        """EXTENDED: Process interaction with inter-group dynamics"""
        p1_cooperates = person1.calculate_cooperation_decision(person2, self.round)
        p2_cooperates = person2.calculate_cooperation_decision(person1, self.round)
        
        # NEW: Track interaction types
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
        
        # EXTENDED: Update relationships with group-weighted trust
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
            # ORIGINAL: Standard trust updates
            rel1 = person1.get_relationship(person2.id, self.round)
            rel2 = person2.get_relationship(person1.id, self.round)
            rel1.update_trust(p2_cooperates, self.round)
            rel2.update_trust(p1_cooperates, self.round)
        
        cooperation_bonus1 = 0
        cooperation_bonus2 = 0
        base_bonus = self.params.cooperation_bonus
        
        # NEW: Apply mixing event bonus
        if is_mixing_event and hasattr(self.params, 'mixing_event_bonus_multiplier'):
            base_bonus *= self.params.mixing_event_bonus_multiplier
        
        if p1_cooperates and p2_cooperates:
            cooperation_bonus1 = base_bonus
            cooperation_bonus2 = base_bonus
            self.cooperation_benefit_total += base_bonus * 2
            
            person1.maslow_needs.love = min(10, person1.maslow_needs.love + 0.1)
            person2.maslow_needs.love = min(10, person2.maslow_needs.love + 0.1)
            
        elif p1_cooperates and not p2_cooperates:
            # NEW: Person2 defected - apply reputational spillover and out-group surcharge
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
            # NEW: Person1 defected
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
        
        # ORIGINAL: Base pressure calculations
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
        
        # ORIGINAL: Birth mechanics
        population_ratio = len(self.people) / self.params.max_population
        adjusted_birth_rate = self.params.base_birth_rate * (1 - population_ratio * 0.8)
        
        if random.random() < adjusted_birth_rate and len(self.people) < self.params.max_population:
            self._create_birth(person1, person2)
        
        # ORIGINAL: Person updates
        person1.update(self.system_stress, self.params.pressure_multiplier, cooperation_bonus1)
        person2.update(self.system_stress, self.params.pressure_multiplier, cooperation_bonus2)
        
        return p1_cooperates and p2_cooperates  # Return success for mixing events
    
    def _check_recoveries(self):
        """ORIGINAL: Check for strategy recoveries"""
        alive_people = [p for p in self.people if not p.is_dead]
        for person in alive_people:
            if person.check_for_recovery():
                self.total_redemptions += 1
    
    def _create_birth(self, parent_a: OptimizedPerson, parent_b: OptimizedPerson):
        """ORIGINAL: Create new person (with group inheritance)"""
        new_person = OptimizedPerson(self.next_person_id, self.params, parent_a, parent_b)
        self.people.append(new_person)
        self.total_births += 1
        self.next_person_id += 1
    
    def _check_cascade(self):
        """ORIGINAL: Check for cascade conditions"""
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
        """ORIGINAL: Update population state"""
        initial_count = len(self.people)
        self.people = [p for p in self.people if not p.is_dead]
        self.total_deaths += initial_count - len(self.people)
        
        for person in self.people:
            person.update(self.system_stress, self.params.pressure_multiplier)
    
    def _collect_round_data(self):
        """ORIGINAL: Lightweight data collection"""
        alive_people = [p for p in self.people if not p.is_dead]
        self.system_stress_history.append(self.system_stress)
        self.population_history.append(len(alive_people))
    
    def _get_average_traits(self) -> Dict[str, float]:
        """ORIGINAL: Get average Maslow traits"""
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
        """NEW: Get current population by group"""
        alive_people = [p for p in self.people if not p.is_dead]
        group_counts = defaultdict(int)
        for person in alive_people:
            if hasattr(person, 'group_id'):
                group_counts[person.group_id] += 1
            else:
                group_counts['default'] += 1
        return dict(group_counts)
    
    def _get_group_cooperation_rates(self) -> Dict[str, float]:
        """NEW: Get cooperation rate by group"""
        alive_people = [p for p in self.people if not p.is_dead]
        group_cooperation = defaultdict(list)
        
        for person in alive_people:
            group_id = getattr(person, 'group_id', 'default')
            group_cooperation[group_id].append(1 if person.strategy == 'cooperative' else 0)
        
        return {group: sum(strategies) / len(strategies) if strategies else 0 
                for group, strategies in group_cooperation.items()}
    
    def _calculate_segregation_index(self) -> float:
        """NEW: Calculate how segregated the groups became"""
        total_interactions = self.in_group_interactions + self.out_group_interactions
        
        if total_interactions == 0:
            return 0.0
        
        return self.in_group_interactions / total_interactions
    
    def _calculate_trust_levels(self) -> Tuple[float, float]:
        """NEW: Calculate average in-group and out-group trust levels"""
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
                    # Fallback for relationships without group info
                    in_group_trusts.append(rel.trust)
        
        avg_in_group = sum(in_group_trusts) / len(in_group_trusts) if in_group_trusts else 0.5
        avg_out_group = sum(out_group_trusts) / len(out_group_trusts) if out_group_trusts else 0.5
        
        return avg_in_group, avg_out_group
    
    def _generate_results(self, initial_traits: Dict[str, float], 
                         initial_group_populations: Dict[str, int]) -> EnhancedSimulationResults:
        """EXTENDED: Generate comprehensive results with inter-group metrics"""
        alive_people = [p for p in self.people if not p.is_dead]
        cooperative = [p for p in alive_people if p.strategy == 'cooperative']
        constrained = [p for p in alive_people if p.is_constrained]
        
        final_traits = self._get_average_traits()
        trait_evolution = {k: final_traits[k] - initial_traits[k] for k in initial_traits.keys()}
        
        # ORIGINAL: Population stability calculation
        if len(self.population_history) > 20:
            later_pop = self.population_history[-20:]
            pop_stability = np.std(later_pop) / (np.mean(later_pop) + 1e-6)
        else:
            pop_stability = 0.0
        
        # ORIGINAL: Pressure metrics
        avg_maslow_pressure = sum(p.maslow_pressure for p in alive_people) / max(1, len(alive_people))
        basic_needs_crisis = len([p for p in alive_people if p.maslow_needs.physiological < 3 or p.maslow_needs.safety < 3])
        
        # ORIGINAL: Trust level calculation
        if hasattr(self.params, 'num_groups') and self.params.num_groups > 1:
            avg_in_group_trust, avg_out_group_trust = self._calculate_trust_levels()
            overall_trust = (avg_in_group_trust + avg_out_group_trust) / 2
            trust_asymmetry = avg_in_group_trust - avg_out_group_trust
        else:
            # Standard trust calculation for non-group simulations
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
        
        # ORIGINAL: Growth rate
        max_pop_reached = max(self.population_history) if self.population_history else self.params.initial_population
        population_growth = max_pop_reached / self.params.initial_population
        
        # NEW: Inter-group specific calculations
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
            
            # ORIGINAL: All original metrics preserved
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
            
            # NEW: Inter-Group Metrics (extensions)
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
            trust_asymmetry=trust_asymmetry
        )

def generate_random_parameters(run_id: int) -> SimulationParameters:
    """EXTENDED: Generate randomized simulation parameters (with optional inter-group settings)"""
    print(f"🎲 Generating parameters for sim {run_id}")
    initial_pop = random.randint(100, 500)
    max_pop_multiplier = 5 + random.random() * 5
    max_pop = int(initial_pop * max_pop_multiplier)
    
    # Decide whether to include inter-group dynamics (80% chance)
    include_intergroup = random.random() < 0.8
    print(f"🏷️  Sim {run_id}: include_intergroup = {include_intergroup}")
    
    if include_intergroup:
        # Random group distribution
        num_groups = random.choice([2, 3, 4])
        group_dist = [random.random() for _ in range(num_groups)]
        total = sum(group_dist)
        group_dist = [x/total for x in group_dist]
        
        params = SimulationParameters(
            initial_population=initial_pop,
            max_population=max_pop,
            shock_frequency=0.01 + random.random() * 0.19,
            pressure_multiplier=0.1 + random.random() * 0.9,
            base_birth_rate=0.01 + random.random() * 0.09,
            max_rounds=300 + random.randint(0, 500),
            maslow_variation=0.3 + random.random() * 0.7,
            constraint_threshold_range=(
                0.2 + random.random() * 0.3,
                0.6 + random.random() * 0.3
            ),
            recovery_threshold=0.2 + random.random() * 0.3,
            cooperation_bonus=0.1 + random.random() * 0.3,
            trust_threshold=0.4 + random.random() * 0.4,
            
            # Inter-group parameters
            num_groups=num_groups,
            founder_group_distribution=group_dist,
            homophily_bias=random.random(),  # 0 to 1
            in_group_trust_modifier=1.0 + random.random() * 1.0,  # 1.0 to 2.0
            out_group_trust_modifier=0.1 + random.random() * 0.9,  # 0.1 to 1.0
            out_group_constraint_amplifier=1.0 + random.random() * 2.0,  # 1.0 to 3.0
            reputational_spillover=random.random() * 0.3,  # 0 to 0.3
            mixing_event_frequency=random.choice([0, 10, 20, 30, 50]),  # 0 = no mixing events
            mixing_event_bonus_multiplier=1.5 + random.random() * 1.5,  # 1.5 to 3.0
            inheritance_style=random.choice(["mother", "father", "random"])
        )
    else:
        # ORIGINAL: Standard parameters without inter-group features
        params = SimulationParameters(
            initial_population=initial_pop,
            max_population=max_pop,
            shock_frequency=0.01 + random.random() * 0.19,
            pressure_multiplier=0.1 + random.random() * 0.9,
            base_birth_rate=0.01 + random.random() * 0.09,
            max_rounds=300 + random.randint(0, 500),
            maslow_variation=0.3 + random.random() * 0.7,
            constraint_threshold_range=(
                0.2 + random.random() * 0.3,
                0.6 + random.random() * 0.3
            ),
            recovery_threshold=0.2 + random.random() * 0.3,
            cooperation_bonus=0.1 + random.random() * 0.3,
            trust_threshold=0.4 + random.random() * 0.4,
            
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
            inheritance_style="mother"
        )
    
    print(f"📋 Sim {run_id}: max_rounds={params.max_rounds}, initial_pop={params.initial_population}")
    return params

def run_single_simulation(run_id: int) -> EnhancedSimulationResults:
    """ORIGINAL: Run a single simulation with random parameters"""
    print(f"🔄 Starting simulation {run_id}")
    params = generate_random_parameters(run_id)
    print(f"🎛️  Generated parameters for sim {run_id}")
    sim = EnhancedMassSimulation(params, run_id)
    print(f"🏗️  Created simulation object for sim {run_id}")
    result = sim.run_simulation()
    print(f"✅ Completed simulation {run_id}")
    return result

def run_mass_experiment(num_simulations: int = 100, use_multiprocessing: bool = False) -> List[EnhancedSimulationResults]:
    """ORIGINAL: Run mass parameter exploration experiment (with inter-group extensions)"""
    print(f"🚀 Starting enhanced mass experiment with {num_simulations} simulations...")
    print("✨ Including: Strategy reversals, trust dynamics, cooperation benefits")
    print("🆕 NEW: Group Tags, Homophily, Trust Asymmetry, Out-group Surcharge, Spillover, Mixing Events")
    
    # DEBUG: Explicit confirmation of threading mode
    print(f"🔧 DEBUG: use_multiprocessing = {use_multiprocessing}")
    print(f"🔧 DEBUG: num_simulations = {num_simulations}")
    
    start_time = time.time()
    
    if use_multiprocessing and num_simulations > 20:
        print(f"🔧 TAKING MULTIPROCESSING PATH")
        num_cores = min(mp.cpu_count(), 8)
        print(f"🔧 Using {num_cores} CPU cores for parallel processing...")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(run_single_simulation, i) for i in range(num_simulations)]
            
            results = []
            for completed_future in as_completed(futures):
                result = completed_future.result()
                results.append(result)
                
                if len(results) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(results) / elapsed
                    eta = (num_simulations - len(results)) / rate
                    print(f"⏳ Progress: {len(results)}/{num_simulations} ({len(results)/num_simulations*100:.1f}%) | "
                          f"Rate: {rate:.1f} sim/sec | ETA: {eta:.1f}s")
    else:
        print(f"🔧 TAKING SINGLE-THREADED PATH")
        results = []
        for i in range(num_simulations):
            if i % 10 == 0:
                elapsed = time.time() - start_time if i > 0 else 0.1
                rate = i / elapsed if i > 0 else 0
                eta = (num_simulations - i) / rate if rate > 0 else 0
                print(f"⏳ Progress: {i}/{num_simulations} ({i/num_simulations*100:.1f}%) | "
                      f"Rate: {rate:.1f} sim/sec | ETA: {eta:.1f}s")
            results.append(run_single_simulation(i))
    
    elapsed = time.time() - start_time
    print(f"✅ Completed {num_simulations} simulations in {elapsed:.2f} seconds")
    print(f"⚡ Average: {elapsed/num_simulations:.3f} seconds per simulation")
    print(f"🏁 Final rate: {num_simulations/elapsed:.1f} simulations per second")
    
    return results

def analyze_emergent_patterns(results: List[EnhancedSimulationResults]) -> pd.DataFrame:
    """EXTENDED: Analyze results for emergent patterns (includes inter-group analysis)"""
    print("🔍 Analyzing emergent patterns...")
    
    # Convert results to DataFrame
    data = []
    for result in results:
        row = {
            'run_id': result.run_id,
            
            # ORIGINAL: All original parameters preserved
            'initial_population': result.parameters.initial_population,
            'max_population': result.parameters.max_population,
            'pop_multiplier': result.parameters.max_population / result.parameters.initial_population,
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
            
            # ORIGINAL: All original outcomes preserved
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
            
            # ORIGINAL: Maslow changes preserved
            'physiological_change': result.needs_improvement['physiological'],
            'safety_change': result.needs_improvement['safety'],
            'love_change': result.needs_improvement['love'],
            'esteem_change': result.needs_improvement['esteem'],
            'self_actualization_change': result.needs_improvement['self_actualization'],
            
            # NEW: Inter-group parameters (when available)
            'num_groups': getattr(result.parameters, 'num_groups', 1),
            'homophily_bias': getattr(result.parameters, 'homophily_bias', 0.0),
            'in_group_trust_modifier': getattr(result.parameters, 'in_group_trust_modifier', 1.0),
            'out_group_trust_modifier': getattr(result.parameters, 'out_group_trust_modifier', 1.0),
            'out_group_constraint_amplifier': getattr(result.parameters, 'out_group_constraint_amplifier', 1.0),
            'reputational_spillover': getattr(result.parameters, 'reputational_spillover', 0.0),
            'mixing_event_frequency': getattr(result.parameters, 'mixing_event_frequency', 0),
            'mixing_event_bonus_multiplier': getattr(result.parameters, 'mixing_event_bonus_multiplier', 1.0),
            
            # NEW: Inter-group outcomes (when available)
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
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # ORIGINAL: Create outcome categories
    df['outcome_category'] = pd.cut(df['final_cooperation_rate'], 
                                   bins=[0, 0.1, 0.3, 0.7, 1.0],
                                   labels=['Collapse', 'Low_Coop', 'Medium_Coop', 'High_Coop'])
    
    df['extinction_category'] = df['extinction_occurred'].map({True: 'Extinct', False: 'Survived'})
    
    # NEW: Inter-group specific categories
    df['has_intergroup'] = df['num_groups'] > 1
    df['segregation_level'] = pd.cut(df['group_segregation_index'], 
                                   bins=[0, 0.3, 0.7, 1.0],
                                   labels=['Integrated', 'Moderate', 'Highly_Segregated'])
    
    df['trust_asymmetry_level'] = pd.cut(df['trust_asymmetry'], 
                                       bins=[-1, 0.1, 0.3, 1.0],
                                       labels=['Low', 'Medium', 'High'])
    
    # ORIGINAL: Calculate derived metrics
    df['pressure_index'] = df['shock_frequency'] * df['pressure_multiplier']
    df['growth_potential'] = df['birth_rate'] * df['pop_multiplier']
    df['resilience_index'] = df['threshold_range'] * (1 - df['pressure_index'])
    
    # NEW: Inter-group tension index
    df['intergroup_tension'] = (df['out_group_constraint_amplifier'] * 
                               df['reputational_spillover'] * 
                               (1 - df['out_group_trust_modifier']))
    
    return df

def create_pattern_visualizations(df: pd.DataFrame):
    """EXTENDED: Create comprehensive pattern analysis visualizations (preserves all original + adds inter-group)"""
    print("📊 Creating comprehensive pattern analysis visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # Determine how many plots we need (original 16 + new inter-group plots)
    has_intergroup_data = df['has_intergroup'].any()
    total_plots = 16 + (8 if has_intergroup_data else 0)
    
    # Calculate grid size
    if total_plots <= 16:
        rows, cols = 4, 4
    else:
        rows, cols = 6, 4
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, rows * 5))
    
    # ORIGINAL VISUALIZATIONS (1-16) - PRESERVED EXACTLY
    
    # 1. Parameter Space Overview
    ax1 = plt.subplot(rows, cols, 1)
    scatter = plt.scatter(df['shock_frequency'], df['pressure_multiplier'], 
                         c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Shock Frequency')
    plt.ylabel('Pressure Multiplier')
    plt.title('Parameter Space: Cooperation Outcomes')
    
    # 2. Redemption Analysis
    ax2 = plt.subplot(rows, cols, 2)
    plt.scatter(df['recovery_threshold'], df['redemption_rate'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(label='Final Cooperation')
    plt.xlabel('Recovery Threshold')
    plt.ylabel('Redemption Rate')
    plt.title('Recovery Dynamics')
    
    # 3. Pressure Index Distribution
    ax3 = plt.subplot(rows, cols, 3)
    df['pressure_index'].hist(bins=30, alpha=0.7)
    plt.xlabel('Pressure Index (Shock Freq × Pressure Mult)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pressure Conditions')
    
    # 4. Trust and Cooperation
    ax4 = plt.subplot(rows, cols, 4)
    plt.scatter(df['avg_trust_level'], df['final_cooperation_rate'], alpha=0.6)
    plt.xlabel('Average Trust Level')
    plt.ylabel('Final Cooperation Rate')
    plt.title('Trust-Cooperation Relationship')
    
    # 5. Cascade Timing Analysis
    ax5 = plt.subplot(rows, cols, 5)
    non_extinct = df[~df['extinction_occurred']]
    if len(non_extinct) > 0:
        plt.scatter(non_extinct['pressure_index'], non_extinct['first_cascade_round'], alpha=0.6)
        plt.xlabel('Pressure Index')
        plt.ylabel('First Cascade Round')
        plt.title('Cascade Timing vs System Pressure')
    
    # 6. Population Growth Patterns
    ax6 = plt.subplot(rows, cols, 6)
    plt.scatter(df['birth_rate'], df['population_growth'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.xlabel('Birth Rate')
    plt.ylabel('Population Growth Ratio')
    plt.title('Population Growth vs Birth Rate')
    
    # 7. Maslow Need Changes
    ax7 = plt.subplot(rows, cols, 7)
    need_changes = df[['physiological_change', 'safety_change', 'love_change', 
                      'esteem_change', 'self_actualization_change']].mean()
    need_changes.plot(kind='bar')
    plt.xlabel('Need Type')
    plt.ylabel('Average Change')
    plt.title('Need Level Changes During Simulation')
    plt.xticks(rotation=45)
    
    # 8. Correlation Heatmap
    ax8 = plt.subplot(rows, cols, 8)
    key_vars = ['shock_frequency', 'pressure_multiplier', 'recovery_threshold', 
                'cooperation_bonus', 'final_cooperation_rate', 'redemption_rate', 
                'avg_trust_level']
    corr_matrix = df[key_vars].corr()
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    else:
        im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(key_vars)), key_vars, rotation=45)
        plt.yticks(range(len(key_vars)), key_vars)
        for i in range(len(key_vars)):
            for j in range(len(key_vars)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    plt.title('Parameter Correlations')
    
    # 9. Phase Transition Analysis
    ax9 = plt.subplot(rows, cols, 9)
    pressure_bins = pd.cut(df['pressure_index'], bins=15)
    coop_by_pressure = df.groupby(pressure_bins)['final_cooperation_rate'].mean()
    coop_by_pressure.plot(kind='line', marker='o')
    plt.xlabel('Pressure Index Bins')
    plt.ylabel('Average Cooperation Rate')
    plt.title('Phase Transition: Pressure vs Cooperation')
    plt.xticks(rotation=45)
    
    # 10. Redemption vs Defection
    ax10 = plt.subplot(rows, cols, 10)
    plt.scatter(df['total_defections'], df['total_redemptions'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(label='Final Cooperation')
    plt.xlabel('Total Defections')
    plt.ylabel('Total Redemptions')
    plt.title('Strategy Change Dynamics')
    
    # 11. Cooperation Benefits Impact
    ax11 = plt.subplot(rows, cols, 11)
    plt.scatter(df['cooperation_bonus'], df['cooperation_benefit_total'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.xlabel('Cooperation Bonus Parameter')
    plt.ylabel('Total Cooperation Benefits')
    plt.title('Cooperation Incentive Impact')
    
    # 12. Resilience Analysis
    ax12 = plt.subplot(rows, cols, 12)
    if HAS_SEABORN:
        sns.boxplot(data=df, x='outcome_category', y='resilience_index')
    else:
        categories = df['outcome_category'].unique()
        box_data = [df[df['outcome_category'] == cat]['resilience_index'].values for cat in categories]
        plt.boxplot(box_data, labels=categories)
    plt.title('Resilience Index by Outcome')
    plt.xticks(rotation=45)
    
    # 13-16. Outcome Distribution Analysis
    for i, category in enumerate(['Collapse', 'Low_Coop', 'Medium_Coop', 'High_Coop']):
        ax = plt.subplot(rows, cols, 13 + i)
        subset = df[df['outcome_category'] == category]
        if len(subset) > 0:
            plt.scatter(subset['shock_frequency'], subset['pressure_multiplier'], alpha=0.7)
            plt.xlabel('Shock Frequency')
            plt.ylabel('Pressure Multiplier')
            plt.title(f'{category} Conditions\n(n={len(subset)})')
    
    # NEW INTER-GROUP VISUALIZATIONS (17-24) - ONLY IF DATA EXISTS
    if has_intergroup_data:
        intergroup_df = df[df['has_intergroup']]
        
        # 17. Homophily vs Cooperation
        ax17 = plt.subplot(rows, cols, 17)
        scatter = plt.scatter(intergroup_df['homophily_bias'], intergroup_df['final_cooperation_rate'], 
                             c=intergroup_df['group_segregation_index'], cmap='RdYlBu_r', alpha=0.6)
        plt.colorbar(scatter, label='Segregation Index')
        plt.xlabel('Homophily Bias')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Homophily vs Cooperation\n(Color = Segregation)')
        
        # 18. Trust Asymmetry Analysis
        ax18 = plt.subplot(rows, cols, 18)
        plt.scatter(intergroup_df['avg_in_group_trust'], intergroup_df['avg_out_group_trust'], 
                   c=intergroup_df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
        plt.colorbar(label='Final Cooperation')
        plt.xlabel('Average In-Group Trust')
        plt.ylabel('Average Out-Group Trust')
        plt.title('Trust Asymmetry Patterns')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal Trust')
        plt.legend()
        
        # 19. Out-Group Constraint Amplifier Impact
        ax19 = plt.subplot(rows, cols, 19)
        plt.scatter(intergroup_df['out_group_constraint_amplifier'], intergroup_df['final_cooperation_rate'], 
                   c=intergroup_df['out_group_constraint_amplifications'], cmap='plasma', alpha=0.6)
        plt.colorbar(label='Amplification Events')
        plt.xlabel('Out-Group Constraint Amplifier')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Out-Group Surcharge Impact')
        
        # 20. Mixing Events Analysis
        ax20 = plt.subplot(rows, cols, 20)
        mixing_data = intergroup_df[intergroup_df['mixing_event_frequency'] > 0]
        if len(mixing_data) > 0:
            plt.scatter(mixing_data['mixing_event_frequency'], mixing_data['mixing_event_success_rate'], 
                       c=mixing_data['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
            plt.colorbar(label='Final Cooperation')
        plt.xlabel('Mixing Event Frequency')
        plt.ylabel('Mixing Event Success Rate')
        plt.title('Mixing Events Effectiveness')
        
        # 21. Group Diversity vs Outcomes
        ax21 = plt.subplot(rows, cols, 21)
        if HAS_SEABORN:
            sns.boxplot(data=intergroup_df, x='num_groups', y='final_cooperation_rate')
        else:
            groups = intergroup_df['num_groups'].unique()
            box_data = [intergroup_df[intergroup_df['num_groups'] == g]['final_cooperation_rate'].values for g in sorted(groups)]
            plt.boxplot(box_data, labels=sorted(groups))
        plt.xlabel('Number of Groups')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Group Diversity Impact')
        
        # 22. Segregation Outcomes
        ax22 = plt.subplot(rows, cols, 22)
        if HAS_SEABORN:
            sns.boxplot(data=intergroup_df, x='segregation_level', y='final_cooperation_rate')
        else:
            seg_levels = intergroup_df['segregation_level'].dropna().unique()
            box_data = [intergroup_df[intergroup_df['segregation_level'] == level]['final_cooperation_rate'].values 
                       for level in seg_levels]
            if box_data:
                plt.boxplot(box_data, labels=[str(level) for level in seg_levels])
        plt.xlabel('Segregation Level')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Segregation vs Cooperation')
        plt.xticks(rotation=45)
        
        # 23. Inter-Group Tension Index
        ax23 = plt.subplot(rows, cols, 23)
        plt.scatter(intergroup_df['intergroup_tension'], intergroup_df['final_cooperation_rate'], alpha=0.6)
        plt.xlabel('Inter-Group Tension Index')
        plt.ylabel('Final Cooperation Rate')
        plt.title('Tension vs Cooperation')
        
        # 24. Comparison: Standard vs Inter-Group
        ax24 = plt.subplot(rows, cols, 24)
        standard_df = df[~df['has_intergroup']]
        if len(standard_df) > 0 and len(intergroup_df) > 0:
            plt.hist([standard_df['final_cooperation_rate'], intergroup_df['final_cooperation_rate']], 
                    bins=20, alpha=0.7, label=['Standard', 'Inter-Group'])
            plt.xlabel('Final Cooperation Rate')
            plt.ylabel('Frequency')
            plt.title('Standard vs Inter-Group\nSimulation Outcomes')
            plt.legend()
    
    plt.tight_layout()
    
    title = 'Enhanced Constraint Cascade Simulation - Comprehensive Analysis'
    if has_intergroup_data:
        title += '\n(Including Inter-Group Dynamics)'
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('enhanced_sim_visuals.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def identify_critical_thresholds(df: pd.DataFrame):
    """EXTENDED: Identify critical thresholds and phase transitions (preserves original + adds inter-group)"""
    print("🎯 Identifying critical thresholds...")
    
    # ORIGINAL ANALYSIS - PRESERVED
    
    # Find pressure threshold for cooperation collapse
    df_sorted = df.sort_values('pressure_index')
    cooperation_rates = df_sorted['final_cooperation_rate'].rolling(window=50, center=True).mean()
    
    # Find where cooperation drops below 50%
    collapse_threshold = None
    for i, rate in enumerate(cooperation_rates):
        if not pd.isna(rate) and rate < 0.5:
            collapse_threshold = df_sorted.iloc[i]['pressure_index']
            break
    
    # Population size thresholds
    size_effect = df.groupby(pd.cut(df['initial_population'], bins=10))['final_cooperation_rate'].mean()
    
    # Extinction thresholds
    extinction_by_pressure = df.groupby(pd.cut(df['pressure_index'], bins=20))['extinction_occurred'].mean()
    
    # Redemption analysis
    redemption_by_recovery = df.groupby(pd.cut(df['recovery_threshold'], bins=10))['redemption_rate'].mean()
    
    print("\n" + "="*60)
    print("🔍 CRITICAL THRESHOLD ANALYSIS")
    print("="*60)
    
    # ORIGINAL THRESHOLDS
    if collapse_threshold:
        print(f"🚨 Cooperation Collapse Threshold: {collapse_threshold:.3f}")
        print(f"   (Shock Frequency × Pressure Multiplier)")
    
    # Find extinction threshold
    extinction_threshold = None
    for pressure_bin, extinction_rate in extinction_by_pressure.items():
        if extinction_rate > 0.5:
            extinction_threshold = pressure_bin.mid
            break
    
    if extinction_threshold is not None:
        print(f"💀 Population Extinction Threshold: {extinction_threshold:.3f}")
    
    # Population size effects
    small_pop_coop = df[df['initial_population'] < 200]['final_cooperation_rate'].mean()
    large_pop_coop = df[df['initial_population'] > 400]['final_cooperation_rate'].mean()
    
    print(f"👥 Population Size Effect:")
    print(f"   Small populations (<200): {small_pop_coop:.1%} cooperation")
    print(f"   Large populations (>400): {large_pop_coop:.1%} cooperation")
    print(f"   Size penalty: {(small_pop_coop - large_pop_coop):.1%}")
    
    # Resilience factors
    high_resilience = df[df['resilience_index'] > df['resilience_index'].quantile(0.8)]
    low_resilience = df[df['resilience_index'] < df['resilience_index'].quantile(0.2)]
    
    print(f"🛡️  Resilience Factors:")
    print(f"   High resilience systems: {high_resilience['final_cooperation_rate'].mean():.1%} cooperation")
    print(f"   Low resilience systems: {low_resilience['final_cooperation_rate'].mean():.1%} cooperation")
    
    # Redemption thresholds
    if redemption_by_recovery.dropna().empty:
        optimal_recovery = None
    else:
        optimal_recovery = redemption_by_recovery.idxmax()
    print(f"♻️  Redemption Dynamics:")
    if optimal_recovery is not None:
        print(f"   Optimal recovery threshold: {optimal_recovery.mid:.3f}")
    else:
        print("   Optimal recovery threshold: N/A (no redemptions recorded)")
    print(f"   Average redemption rate: {df['redemption_rate'].mean():.1%}")
    print(f"   Max redemption rate: {df['redemption_rate'].max():.1%}")
    
    # NEW: INTER-GROUP ANALYSIS - ONLY IF DATA EXISTS
    has_intergroup_data = df['has_intergroup'].any()
    if has_intergroup_data:
        intergroup_df = df[df['has_intergroup']]
        
        print(f"\n🆕 INTER-GROUP THRESHOLDS:")
        print("="*40)
        
        # Homophily threshold for segregation
        if len(intergroup_df) > 0:
            df_sorted = intergroup_df.sort_values('homophily_bias')
            segregation_by_homophily = df_sorted.groupby(pd.cut(df_sorted['homophily_bias'], bins=20))['group_segregation_index'].mean()
            
            segregation_threshold = None
            for interval, segregation in segregation_by_homophily.items():
                if segregation > 0.7:  # High segregation threshold
                    segregation_threshold = interval.mid
                    break
            
            if segregation_threshold:
                print(f"🏘️  Segregation Emergence Threshold: {segregation_threshold:.3f}")
                print(f"   (Homophily bias above this leads to high segregation)")
            
            # Trust asymmetry effects
            high_trust_asymmetry = intergroup_df[intergroup_df['trust_asymmetry'] > 0.3]
            low_trust_asymmetry = intergroup_df[intergroup_df['trust_asymmetry'] < 0.1]
            
            print(f"🤝 Trust Asymmetry Effects:")
            if len(high_trust_asymmetry) > 0 and len(low_trust_asymmetry) > 0:
                print(f"   High asymmetry (>0.3): {high_trust_asymmetry['final_cooperation_rate'].mean():.1%} cooperation")
                print(f"   Low asymmetry (<0.1): {low_trust_asymmetry['final_cooperation_rate'].mean():.1%} cooperation")
                print(f"   Asymmetry penalty: {(low_trust_asymmetry['final_cooperation_rate'].mean() - high_trust_asymmetry['final_cooperation_rate'].mean()):.1%}")
            
            # Out-group constraint amplifier
            high_amplifier = intergroup_df[intergroup_df['out_group_constraint_amplifier'] > 2.0]
            low_amplifier = intergroup_df[intergroup_df['out_group_constraint_amplifier'] < 1.5]
            
            print(f"⚡ Out-Group Constraint Amplifier:")
            if len(high_amplifier) > 0 and len(low_amplifier) > 0:
                print(f"   High amplifier (>2.0): {high_amplifier['final_cooperation_rate'].mean():.1%} cooperation")
                print(f"   Low amplifier (<1.5): {low_amplifier['final_cooperation_rate'].mean():.1%} cooperation")
                print(f"   Amplification penalty: {(low_amplifier['final_cooperation_rate'].mean() - high_amplifier['final_cooperation_rate'].mean()):.1%}")
            
            # Mixing events
            effective_mixing = intergroup_df[(intergroup_df['total_mixing_events'] > 0) & (intergroup_df['mixing_event_success_rate'] > 0.5)]
            no_mixing = intergroup_df[intergroup_df['total_mixing_events'] == 0]
            
            print(f"🎭 Mixing Events Impact:")
            if len(effective_mixing) > 0 and len(no_mixing) > 0:
                print(f"   Effective mixing events: {effective_mixing['final_cooperation_rate'].mean():.1%} cooperation")
                print(f"   No mixing events: {no_mixing['final_cooperation_rate'].mean():.1%} cooperation")
                print(f"   Mixing benefit: {(effective_mixing['final_cooperation_rate'].mean() - no_mixing['final_cooperation_rate'].mean()):.1%}")
            
            # Group number effects
            group_effects = intergroup_df.groupby('num_groups')['final_cooperation_rate'].mean()
            print(f"👥 Group Number Effects:")
            for num_groups, coop_rate in group_effects.items():
                print(f"   {num_groups} groups: {coop_rate:.1%} cooperation")
    
    return {
        # ORIGINAL thresholds preserved
        'cooperation_collapse_threshold': collapse_threshold,
        'extinction_threshold': extinction_threshold.mid if extinction_threshold is not None else None,
        'population_size_effect': large_pop_coop - small_pop_coop,
        'resilience_benefit': high_resilience['final_cooperation_rate'].mean() - low_resilience['final_cooperation_rate'].mean(),
        'optimal_recovery_threshold': optimal_recovery.mid if optimal_recovery is not None else None,
        'average_redemption_rate': df['redemption_rate'].mean(),
        
        # NEW inter-group thresholds
        'has_intergroup_data': has_intergroup_data,
        'segregation_threshold': segregation_threshold if has_intergroup_data else None,
        'trust_asymmetry_penalty': (low_trust_asymmetry['final_cooperation_rate'].mean() - 
                                   high_trust_asymmetry['final_cooperation_rate'].mean()) if has_intergroup_data and len(high_trust_asymmetry) > 0 and len(low_trust_asymmetry) > 0 else None,
        'amplification_penalty': (low_amplifier['final_cooperation_rate'].mean() - 
                                high_amplifier['final_cooperation_rate'].mean()) if has_intergroup_data and len(high_amplifier) > 0 and len(low_amplifier) > 0 else None,
        'mixing_benefit': (effective_mixing['final_cooperation_rate'].mean() - 
                         no_mixing['final_cooperation_rate'].mean()) if has_intergroup_data and len(effective_mixing) > 0 and len(no_mixing) > 0 else None,
        'optimal_group_number': group_effects.idxmax() if has_intergroup_data else None,
    }

def save_comprehensive_results(df: pd.DataFrame, thresholds: Dict):
    """EXTENDED: Save all results for further analysis (preserves original files + adds inter-group)"""
    current_dir = os.getcwd()
    print(f"💾 Saving comprehensive results to: {current_dir}")
    
    saved_files = []
    
    try:
        # ORIGINAL FILES - PRESERVED EXACTLY
        
        # Save main dataset
        main_file = 'enhanced_mass_simulation_results.csv'
        df.to_csv(main_file, index=False)
        if os.path.exists(main_file):
            size_mb = os.path.getsize(main_file) / (1024*1024)
            saved_files.append(f"📊 {main_file} ({size_mb:.2f} MB)")
        
        # Save summary statistics
        summary_file = 'enhanced_simulation_summary_stats.csv'
        summary_stats = df.describe()
        summary_stats.to_csv(summary_file)
        if os.path.exists(summary_file):
            saved_files.append(f"📈 {summary_file}")
        
        # Save threshold analysis
        threshold_file = 'enhanced_critical_thresholds.txt'
        with open(threshold_file, 'w') as f:
            f.write("Enhanced Critical Threshold Analysis\n")
            f.write("="*50 + "\n\n")
            for key, value in thresholds.items():
                f.write(f"{key}: {value}\n")
        if os.path.exists(threshold_file):
            saved_files.append(f"🎯 {threshold_file}")
        
        # Save parameter combinations for extreme outcomes
        high_coop = df[df['final_cooperation_rate'] > 0.8]
        low_coop = df[df['final_cooperation_rate'] < 0.2]
        
        if len(high_coop) > 0:
            high_file = 'enhanced_high_cooperation_parameters.csv'
            high_coop.to_csv(high_file, index=False)
            if os.path.exists(high_file):
                saved_files.append(f"✅ {high_file} ({len(high_coop)} scenarios)")
        
        if len(low_coop) > 0:
            low_file = 'enhanced_low_cooperation_parameters.csv'
            low_coop.to_csv(low_file, index=False)
            if os.path.exists(low_file):
                saved_files.append(f"❌ {low_file} ({len(low_coop)} scenarios)")
        
        # Save redemption success scenarios
        high_redemption = df[df['redemption_rate'] > 0.5]
        if len(high_redemption) > 0:
            redemption_file = 'high_redemption_scenarios.csv'
            high_redemption.to_csv(redemption_file, index=False)
            if os.path.exists(redemption_file):
                saved_files.append(f"♻️  {redemption_file} ({len(high_redemption)} scenarios)")
        
        # NEW: INTER-GROUP SPECIFIC FILES (only if inter-group data exists)
        has_intergroup_data = df['has_intergroup'].any()
        if has_intergroup_data:
            intergroup_df = df[df['has_intergroup']]
            
            # Save high cooperation + low segregation scenarios
            integrated_coop = intergroup_df[(intergroup_df['group_segregation_index'] < 0.3) & 
                                          (intergroup_df['final_cooperation_rate'] > 0.7)]
            if len(integrated_coop) > 0:
                integrated_file = 'integrated_high_cooperation_scenarios.csv'
                integrated_coop.to_csv(integrated_file, index=False)
                if os.path.exists(integrated_file):
                    saved_files.append(f"🤝 {integrated_file} ({len(integrated_coop)} scenarios)")
            
            # Save effective mixing event scenarios
            effective_mixing = intergroup_df[(intergroup_df['total_mixing_events'] > 0) & 
                                           (intergroup_df['mixing_event_success_rate'] > 0.5)]
            if len(effective_mixing) > 0:
                mixing_file = 'effective_mixing_event_scenarios.csv'
                effective_mixing.to_csv(mixing_file, index=False)
                if os.path.exists(mixing_file):
                    saved_files.append(f"🎭 {mixing_file} ({len(effective_mixing)} scenarios)")
            
            # Save inter-group analysis subset
            intergroup_file = 'intergroup_dynamics_subset.csv'
            intergroup_df.to_csv(intergroup_file, index=False)
            if os.path.exists(intergroup_file):
                size_mb = os.path.getsize(intergroup_file) / (1024*1024)
                saved_files.append(f"🏷️  {intergroup_file} ({size_mb:.2f} MB)")
        
        # Create a comprehensive data summary file
        summary_report = 'enhanced_experiment_summary.txt'
        with open(summary_report, 'w') as f:
            f.write("Enhanced Constraint Cascade Mass Experiment Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Simulations: {len(df)}\n")
            f.write(f"Average Cooperation Rate: {df['final_cooperation_rate'].mean():.3f}\n")
            f.write(f"Extinction Rate: {df['extinction_occurred'].mean():.3f}\n")
            f.write(f"Average Population: {df['final_population'].mean():.1f}\n")
            f.write(f"High Cooperation Scenarios: {len(high_coop)}\n")
            f.write(f"Low Cooperation Scenarios: {len(low_coop)}\n")
            f.write(f"\nOriginal Enhanced Metrics:\n")
            f.write(f"Average Redemption Rate: {df['redemption_rate'].mean():.3f}\n")
            f.write(f"Average Trust Level: {df['avg_trust_level'].mean():.3f}\n")
            f.write(f"Total Defections: {df['total_defections'].sum()}\n")
            f.write(f"Total Redemptions: {df['total_redemptions'].sum()}\n")
            
            if has_intergroup_data:
                f.write(f"\nInter-Group Dynamics:\n")
                f.write(f"Simulations with Inter-Group Features: {len(intergroup_df)}\n")
                f.write(f"Average Trust Asymmetry: {intergroup_df['trust_asymmetry'].mean():.3f}\n")
                f.write(f"Average Segregation Index: {intergroup_df['group_segregation_index'].mean():.3f}\n")
                f.write(f"Effective Mixing Events: {len(effective_mixing) if 'effective_mixing' in locals() else 0}\n")
                f.write(f"High Cooperation + Low Segregation: {len(integrated_coop) if 'integrated_coop' in locals() else 0}\n")
                f.write(f"Group Extinction Events: {intergroup_df['group_extinction_events'].sum()}\n")
                f.write(f"Average In-Group Trust: {intergroup_df['avg_in_group_trust'].mean():.3f}\n")
                f.write(f"Average Out-Group Trust: {intergroup_df['avg_out_group_trust'].mean():.3f}\n")
                f.write(f"Total Out-Group Amplifications: {intergroup_df['out_group_constraint_amplifications'].sum()}\n")
                f.write(f"Total Spillover Events: {intergroup_df['reputational_spillover_events'].sum()}\n")
            
            f.write(f"\nFiles Created:\n")
            for file_info in saved_files:
                f.write(f"  {file_info}\n")
        if os.path.exists(summary_report):
            saved_files.append(f"📋 {summary_report}")
        
        print(f"\n✅ Successfully saved {len(saved_files)} files:")
        for file_info in saved_files:
            print(f"   {file_info}")
        
        print(f"\n📂 Full path: {current_dir}")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        traceback.print_exc()
        
        # Try to save just the main file as a backup
        try:
            backup_file = 'enhanced_simulation_backup.csv'
            df.to_csv(backup_file, index=False)
            print(f"💾 Backup saved as: {backup_file}")
        except Exception as backup_error:
            print(f"❌ Backup also failed: {backup_error}")
    
    return saved_files

def main():
    """Run the complete enhanced mass experiment with inter-group dynamics"""
    print("🔬 Enhanced Constraint Cascade Simulation - Mass Parameter Exploration")
    print("="*80)
    print("🎯 Discovering emergent patterns with enhanced dynamics AND inter-group features")
    print("✨ Original features: Strategy reversals, trust dynamics, cooperation benefits, Maslow tracking")
    print("🆕 NEW features: Group Tags, Homophily, Trust Asymmetry, Out-group Surcharge,")
    print("🆕              Reputational Spillover, Inter-group Institutions (Mixing Events)")
    print("📊 Full analysis and reporting capabilities (preserves ALL original functionality)")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Configuration
    num_simulations = 100  # Reduced for testing
    use_multiprocessing = False  # Single-threaded
    
    print(f"\n⚙️  Experiment Configuration:")
    print(f"   🔢 Number of simulations: {num_simulations}")
    print(f"   👥 Population range: 100-500 (5x-10x max)")
    print(f"   🎛️  Parameters: Fully randomized with recovery dynamics")
    print(f"   🆕 Inter-group features: 80% of simulations include group dynamics")
    print(f"   🏷️  Groups: 2-4 groups with randomized distributions")
    print(f"   🔗 Homophily: 0-100% same-group preference")
    print(f"   ⚖️  Trust asymmetry: In-group vs out-group modifiers")
    print(f"   ⚡ Out-group surcharge: 1.0x-3.0x constraint amplification")
    print(f"   📢 Reputational spillover: 0-30% collective blame")
    print(f"   🎭 Mixing events: Periodic cross-group institutions")
    print(f"   🖥️  Multiprocessing: {use_multiprocessing}")
    
    try:
        # Run mass experiment
        print(f"\n🚀 PHASE 1: Running {num_simulations} enhanced simulations...")
        results = run_mass_experiment(num_simulations, use_multiprocessing)
        print(f"✅ Phase 1 complete: {len(results)} simulations finished")
        
        # Analyze patterns
        print(f"\n📊 PHASE 2: Analyzing emergent patterns...")
        df = analyze_emergent_patterns(results)
        print(f"✅ Phase 2 complete: {len(df)} simulation records analyzed")
        
        # Create visualizations
        print(f"\n📈 PHASE 3: Creating comprehensive visualizations...")
        create_pattern_visualizations(df)
        print(f"✅ Phase 3 complete: Pattern analysis charts generated")
        
        # Identify critical thresholds
        print(f"\n🎯 PHASE 4: Identifying critical thresholds...")
        thresholds = identify_critical_thresholds(df)
        print(f"✅ Phase 4 complete: Critical thresholds identified")
        
        # Save results
        print(f"\n💾 PHASE 5: Saving comprehensive results...")
        saved_files = save_comprehensive_results(df, thresholds)
        print(f"✅ Phase 5 complete: {len(saved_files)} files saved")
        
        print(f"\n🎉 ENHANCED EXPERIMENT COMPLETE!")
        print(f"📊 Analyzed {len(results)} simulations with full original + inter-group dynamics")
        print(f"🔍 Discovered patterns across {len(df.columns)} measured variables")
        print(f"📈 Generated comprehensive analysis charts (original + inter-group)")
        print(f"💾 Saved detailed results for further research")
        
        # Summary statistics
        has_intergroup = df['has_intergroup'].any()
        standard_sims = len(df[~df['has_intergroup']]) if has_intergroup else len(df)
        intergroup_sims = len(df[df['has_intergroup']]) if has_intergroup else 0
        
        print(f"\n📋 Key Findings:")
        print(f"   🔄 Standard simulations: {standard_sims}")
        if has_intergroup:
            print(f"   🆕 Inter-group simulations: {intergroup_sims}")
        print(f"   🤝 Average cooperation: {df['final_cooperation_rate'].mean():.3f}")
        print(f"   ♻️  Average redemption rate: {df['redemption_rate'].mean():.3f}")
        print(f"   📈 Scenarios with redemptions: {(df['total_redemptions'] > 0).sum()}")
        print(f"   🤝 Average trust level: {df['avg_trust_level'].mean():.3f}")
        
        if has_intergroup:
            intergroup_df = df[df['has_intergroup']]
            print(f"   🆕 Inter-group specific:")
            print(f"      🔗 Average trust asymmetry: {intergroup_df['trust_asymmetry'].mean():.3f}")
            print(f"      🏘️  Average segregation index: {intergroup_df['group_segregation_index'].mean():.3f}")
            print(f"      🎭 Scenarios with mixing events: {(intergroup_df['total_mixing_events'] > 0).sum()}")
            print(f"      🤝 High cooperation + low segregation: {len(intergroup_df[(intergroup_df['final_cooperation_rate'] > 0.7) & (intergroup_df['group_segregation_index'] < 0.3)])}")
            print(f"      💀 Group extinction events: {intergroup_df['group_extinction_events'].sum()}")
        
    except Exception as e:
        print(f"\n❌ ERROR in main experiment: {e}")
        print(f"🔧 Please check the error message above for debugging")
        traceback.print_exc()

if __name__ == "__main__":
    main()
