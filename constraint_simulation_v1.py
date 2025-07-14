#!/usr/bin/env python3
"""
Complete Enhanced Constraint Cascade Simulation
Full fidelity simulation with comprehensive analysis and reporting
Combines all enhancements with original reporting capabilities
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
from concurrent.futures import ProcessPoolExecutor
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
    """Optimized relationship tracking that maintains full history"""
    trust: float = 0.5
    interaction_count: int = 0
    cooperation_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_interaction_round: int = 0
    
    def update_trust(self, cooperated: bool, round_num: int):
        """Update trust based on interaction outcome"""
        self.interaction_count += 1
        self.last_interaction_round = round_num
        self.cooperation_history.append(cooperated)
        
        if cooperated:
            self.trust = min(1.0, self.trust + 0.1)
        else:
            self.trust = max(0.0, self.trust - 0.15)

@dataclass
class SimulationParameters:
    """Enhanced simulation parameters"""
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
    
    # Additional metrics for compatibility
    population_growth: float
    cooperation_resilience: float

class OptimizedPerson:
    """Optimized person maintaining full fidelity"""
    
    __slots__ = ['id', 'strategy', 'constraint_level', 'constraint_threshold', 
                 'recovery_threshold', 'is_constrained', 'is_dead', 'relationships',
                 'max_lifespan', 'age', 'strategy_changes', 'rounds_as_selfish',
                 'rounds_as_cooperative', 'maslow_needs', 'maslow_pressure', 'is_born']
    
    def __init__(self, person_id: int, params: SimulationParameters, 
                 parent_a: Optional['OptimizedPerson'] = None, 
                 parent_b: Optional['OptimizedPerson'] = None):
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
    
    def add_constraint_pressure(self, amount: float) -> bool:
        """Add pressure with Maslow amplification"""
        if self.is_dead:
            return False
        
        maslow_amplifier = 1 + (self.maslow_pressure * 0.5)
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
    
    def get_relationship(self, other_id: int, round_num: int) -> FastRelationship:
        """Get or create relationship"""
        if other_id not in self.relationships:
            if len(self.relationships) >= 150:
                oldest_id = min(self.relationships.keys(), 
                              key=lambda k: self.relationships[k].last_interaction_round)
                del self.relationships[oldest_id]
            
            self.relationships[other_id] = FastRelationship()
        return self.relationships[other_id]
    
    def calculate_cooperation_decision(self, other: 'OptimizedPerson', round_num: int) -> bool:
        """Decide whether to cooperate based on relationship and strategy"""
        if self.strategy == 'selfish':
            return False
        
        relationship = self.get_relationship(other.id, round_num)
        
        if relationship.interaction_count == 0:
            return random.random() < (self.maslow_needs.love / 10)
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
    """Enhanced simulation with comprehensive tracking"""
    
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
        
        # Initialize population
        for i in range(1, params.initial_population + 1):
            person = OptimizedPerson(i, params)
            self.people.append(person)
    
    def run_simulation(self) -> EnhancedSimulationResults:
        """Run enhanced simulation"""
        initial_trait_avg = self._get_average_traits()
        
        while self.round < self.params.max_rounds:
            self.round += 1
            
            if random.random() < self.params.shock_frequency:
                self._trigger_shock()
            
            self._handle_interactions()
            self._check_recoveries()
            self._update_population()
            self._collect_round_data()
            
            alive_people = [p for p in self.people if not p.is_dead]
            if len(alive_people) == 0:
                break
            
            self.system_stress = max(0, self.system_stress - 0.01)
        
        return self._generate_results(initial_trait_avg)
    
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
    
    def _handle_interactions(self):
        """Optimized interaction handling"""
        alive_people = [p for p in self.people if not p.is_dead]
        if len(alive_people) < 2:
            return
        
        num_interactions = max(len(alive_people) // 4, 10)
        
        for _ in range(num_interactions):
            if len(alive_people) >= 2:
                person1 = random.choice(alive_people)
                
                if person1.relationships and random.random() < 0.3:
                    known_alive = [p for p in alive_people 
                                 if p.id in person1.relationships and p.id != person1.id]
                    if known_alive:
                        person2 = random.choice(known_alive)
                    else:
                        person2 = random.choice([p for p in alive_people if p.id != person1.id])
                else:
                    person2 = random.choice([p for p in alive_people if p.id != person1.id])
                
                self._process_interaction(person1, person2)
    
    def _process_interaction(self, person1: OptimizedPerson, person2: OptimizedPerson):
        """Process interaction with cooperation dynamics"""
        p1_cooperates = person1.calculate_cooperation_decision(person2, self.round)
        p2_cooperates = person2.calculate_cooperation_decision(person1, self.round)
        
        person1.get_relationship(person2.id, self.round).update_trust(p2_cooperates, self.round)
        person2.get_relationship(person1.id, self.round).update_trust(p1_cooperates, self.round)
        
        cooperation_bonus1 = 0
        cooperation_bonus2 = 0
        
        if p1_cooperates and p2_cooperates:
            cooperation_bonus1 = self.params.cooperation_bonus
            cooperation_bonus2 = self.params.cooperation_bonus
            self.cooperation_benefit_total += self.params.cooperation_bonus * 2
            
            person1.maslow_needs.love = min(10, person1.maslow_needs.love + 0.1)
            person2.maslow_needs.love = min(10, person2.maslow_needs.love + 0.1)
            
        elif p1_cooperates and not p2_cooperates:
            person1.add_constraint_pressure(0.03)
            person1.maslow_needs.esteem = max(0, person1.maslow_needs.esteem - 0.1)
            
        elif not p1_cooperates and p2_cooperates:
            person2.add_constraint_pressure(0.03)
            person2.maslow_needs.esteem = max(0, person2.maslow_needs.esteem - 0.1)
        
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
        
        population_ratio = len(self.people) / self.params.max_population
        adjusted_birth_rate = self.params.base_birth_rate * (1 - population_ratio * 0.8)
        
        if random.random() < adjusted_birth_rate and len(self.people) < self.params.max_population:
            self._create_birth(person1, person2)
        
        person1.update(self.system_stress, self.params.pressure_multiplier, cooperation_bonus1)
        person2.update(self.system_stress, self.params.pressure_multiplier, cooperation_bonus2)
    
    def _check_recoveries(self):
        """Check for strategy recoveries"""
        alive_people = [p for p in self.people if not p.is_dead]
        for person in alive_people:
            if person.check_for_recovery():
                self.total_redemptions += 1
    
    def _create_birth(self, parent_a: OptimizedPerson, parent_b: OptimizedPerson):
        """Create new person"""
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
    
    def _generate_results(self, initial_traits: Dict[str, float]) -> EnhancedSimulationResults:
        """Generate comprehensive results"""
        alive_people = [p for p in self.people if not p.is_dead]
        cooperative = [p for p in alive_people if p.strategy == 'cooperative']
        constrained = [p for p in alive_people if p.is_constrained]
        
        final_traits = self._get_average_traits()
        trait_evolution = {k: final_traits[k] - initial_traits[k] for k in initial_traits.keys()}
        
        if len(self.population_history) > 20:
            later_pop = self.population_history[-20:]
            pop_stability = np.std(later_pop) / (np.mean(later_pop) + 1e-6)
        else:
            pop_stability = 0.0
        
        avg_maslow_pressure = sum(p.maslow_pressure for p in alive_people) / max(1, len(alive_people))
        basic_needs_crisis = len([p for p in alive_people if p.maslow_needs.physiological < 3 or p.maslow_needs.safety < 3])
        
        # Calculate trust level
        total_trust = 0
        total_relationships = 0
        for person in alive_people:
            for rel in person.relationships.values():
                total_trust += rel.trust
                total_relationships += 1
        
        avg_trust = total_trust / total_relationships if total_relationships > 0 else 0.5
        
        # Calculate growth rate
        max_pop_reached = max(self.population_history) if self.population_history else self.params.initial_population
        population_growth = max_pop_reached / self.params.initial_population
        
        return EnhancedSimulationResults(
            parameters=self.params,
            run_id=self.run_id,
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
            avg_trust_level=avg_trust,
            cooperation_benefit_total=self.cooperation_benefit_total,
            population_growth=population_growth,
            cooperation_resilience=len(cooperative) / max(1, len(alive_people))
        )

def generate_random_parameters(run_id: int) -> SimulationParameters:
    """Generate randomized simulation parameters"""
    initial_pop = random.randint(100, 500)
    max_pop_multiplier = 5 + random.random() * 5
    max_pop = int(initial_pop * max_pop_multiplier)
    
    return SimulationParameters(
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
        trust_threshold=0.4 + random.random() * 0.4
    )

def run_single_simulation(run_id: int) -> EnhancedSimulationResults:
    """Run a single simulation with random parameters"""
    params = generate_random_parameters(run_id)
    sim = EnhancedMassSimulation(params, run_id)
    return sim.run_simulation()

def run_mass_experiment(num_simulations: int = 1000, use_multiprocessing: bool = True) -> List[EnhancedSimulationResults]:
    """Run mass parameter exploration experiment"""
    print(f"🚀 Starting enhanced mass experiment with {num_simulations} simulations...")
    print("✨ Including: Strategy reversals, trust dynamics, cooperation benefits")
    start_time = time.time()
    
    if use_multiprocessing and num_simulations > 20:
        num_cores = min(mp.cpu_count(), 8)
        print(f"🔧 Using {num_cores} CPU cores for parallel processing...")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(run_single_simulation, i) for i in range(num_simulations)]
            
            results = []
            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (num_simulations - (i + 1)) / rate
                    print(f"⏳ Progress: {i + 1}/{num_simulations} ({(i+1)/num_simulations*100:.1f}%) | "
                          f"Rate: {rate:.1f} sim/sec | ETA: {eta:.1f}s")
    else:
        results = []
        for i in range(num_simulations):
            if i % 100 == 0:
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
    """Analyze results for emergent patterns"""
    print("🔍 Analyzing emergent patterns...")
    
    # Convert results to DataFrame
    data = []
    for result in results:
        row = {
            'run_id': result.run_id,
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
            
            # Enhanced parameters
            'recovery_threshold': result.parameters.recovery_threshold,
            'cooperation_bonus': result.parameters.cooperation_bonus,
            'trust_threshold': result.parameters.trust_threshold,
            
            # Outcomes
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
            
            # New metrics
            'total_defections': result.total_defections,
            'total_redemptions': result.total_redemptions,
            'redemption_rate': result.total_redemptions / max(1, result.total_defections),
            'avg_trust_level': result.avg_trust_level,
            'cooperation_benefit_total': result.cooperation_benefit_total,
            
            # Maslow changes
            'physiological_change': result.needs_improvement['physiological'],
            'safety_change': result.needs_improvement['safety'],
            'love_change': result.needs_improvement['love'],
            'esteem_change': result.needs_improvement['esteem'],
            'self_actualization_change': result.needs_improvement['self_actualization'],
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create outcome categories
    df['outcome_category'] = pd.cut(df['final_cooperation_rate'], 
                                   bins=[0, 0.1, 0.3, 0.7, 1.0],
                                   labels=['Collapse', 'Low_Coop', 'Medium_Coop', 'High_Coop'])
    
    df['extinction_category'] = df['extinction_occurred'].map({True: 'Extinct', False: 'Survived'})
    
    # Calculate derived metrics
    df['pressure_index'] = df['shock_frequency'] * df['pressure_multiplier']
    df['growth_potential'] = df['birth_rate'] * df['pop_multiplier']
    df['resilience_index'] = df['threshold_range'] * (1 - df['pressure_index'])
    
    return df

def create_pattern_visualizations(df: pd.DataFrame):
    """Create comprehensive pattern analysis visualizations"""
    print("📊 Creating pattern analysis visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Parameter Space Overview
    ax1 = plt.subplot(4, 4, 1)
    scatter = plt.scatter(df['shock_frequency'], df['pressure_multiplier'], 
                         c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Shock Frequency')
    plt.ylabel('Pressure Multiplier')
    plt.title('Parameter Space: Cooperation Outcomes')
    
    # 2. Redemption Analysis
    ax2 = plt.subplot(4, 4, 2)
    plt.scatter(df['recovery_threshold'], df['redemption_rate'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(label='Final Cooperation')
    plt.xlabel('Recovery Threshold')
    plt.ylabel('Redemption Rate')
    plt.title('Recovery Dynamics')
    
    # 3. Pressure Index Distribution
    ax3 = plt.subplot(4, 4, 3)
    df['pressure_index'].hist(bins=30, alpha=0.7)
    plt.xlabel('Pressure Index (Shock Freq × Pressure Mult)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pressure Conditions')
    
    # 4. Trust and Cooperation
    ax4 = plt.subplot(4, 4, 4)
    plt.scatter(df['avg_trust_level'], df['final_cooperation_rate'], alpha=0.6)
    plt.xlabel('Average Trust Level')
    plt.ylabel('Final Cooperation Rate')
    plt.title('Trust-Cooperation Relationship')
    
    # 5. Cascade Timing Analysis
    ax5 = plt.subplot(4, 4, 5)
    non_extinct = df[~df['extinction_occurred']]
    if len(non_extinct) > 0:
        plt.scatter(non_extinct['pressure_index'], non_extinct['first_cascade_round'], alpha=0.6)
        plt.xlabel('Pressure Index')
        plt.ylabel('First Cascade Round')
        plt.title('Cascade Timing vs System Pressure')
    
    # 6. Population Growth Patterns
    ax6 = plt.subplot(4, 4, 6)
    plt.scatter(df['birth_rate'], df['population_growth'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.xlabel('Birth Rate')
    plt.ylabel('Population Growth Ratio')
    plt.title('Population Growth vs Birth Rate')
    
    # 7. Maslow Need Changes
    ax7 = plt.subplot(4, 4, 7)
    need_changes = df[['physiological_change', 'safety_change', 'love_change', 
                      'esteem_change', 'self_actualization_change']].mean()
    need_changes.plot(kind='bar')
    plt.xlabel('Need Type')
    plt.ylabel('Average Change')
    plt.title('Need Level Changes During Simulation')
    plt.xticks(rotation=45)
    
    # 8. Correlation Heatmap
    ax8 = plt.subplot(4, 4, 8)
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
    ax9 = plt.subplot(4, 4, 9)
    pressure_bins = pd.cut(df['pressure_index'], bins=15)
    coop_by_pressure = df.groupby(pressure_bins)['final_cooperation_rate'].mean()
    coop_by_pressure.plot(kind='line', marker='o')
    plt.xlabel('Pressure Index Bins')
    plt.ylabel('Average Cooperation Rate')
    plt.title('Phase Transition: Pressure vs Cooperation')
    plt.xticks(rotation=45)
    
    # 10. Redemption vs Defection
    ax10 = plt.subplot(4, 4, 10)
    plt.scatter(df['total_defections'], df['total_redemptions'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.colorbar(label='Final Cooperation')
    plt.xlabel('Total Defections')
    plt.ylabel('Total Redemptions')
    plt.title('Strategy Change Dynamics')
    
    # 11. Cooperation Benefits Impact
    ax11 = plt.subplot(4, 4, 11)
    plt.scatter(df['cooperation_bonus'], df['cooperation_benefit_total'], 
               c=df['final_cooperation_rate'], cmap='RdYlGn', alpha=0.6)
    plt.xlabel('Cooperation Bonus Parameter')
    plt.ylabel('Total Cooperation Benefits')
    plt.title('Cooperation Incentive Impact')
    
    # 12. Resilience Analysis
    ax12 = plt.subplot(4, 4, 12)
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
        ax = plt.subplot(4, 4, 13 + i)
        subset = df[df['outcome_category'] == category]
        if len(subset) > 0:
            plt.scatter(subset['shock_frequency'], subset['pressure_multiplier'], alpha=0.7)
            plt.xlabel('Shock Frequency')
            plt.ylabel('Pressure Multiplier')
            plt.title(f'{category} Conditions\n(n={len(subset)})')
    
    plt.tight_layout()
    plt.suptitle('Enhanced Constraint Cascade Simulation - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('enhanced_sim_visuals.png', dpi=300)
    plt.close(fig)                # frees memory and avoids blocking
    
    return fig

def identify_critical_thresholds(df: pd.DataFrame):
    """Identify critical thresholds and phase transitions"""
    print("🎯 Identifying critical thresholds...")
    
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
    
    # New: Redemption thresholds
    if redemption_by_recovery.dropna().empty:
        optimal_recovery = None            # nothing to analyse
    else:
        optimal_recovery = redemption_by_recovery.idxmax()
    print(f"♻️  Redemption Dynamics:")
    if optimal_recovery is not None:
        print(f"   Optimal recovery threshold: {optimal_recovery.mid:.3f}")
    else:
        print("   Optimal recovery threshold: N/A (no redemptions recorded)")
    print(f"   Average redemption rate: {df['redemption_rate'].mean():.1%}")
    print(f"   Max redemption rate: {df['redemption_rate'].max():.1%}")
    
    return {
        'cooperation_collapse_threshold': collapse_threshold,
        'extinction_threshold': extinction_threshold.mid if extinction_threshold is not None else None,
        'population_size_effect': large_pop_coop - small_pop_coop,
        'resilience_benefit': high_resilience['final_cooperation_rate'].mean() - low_resilience['final_cooperation_rate'].mean(),
        'optimal_recovery_threshold': optimal_recovery.mid if optimal_recovery is not None else None,
        'average_redemption_rate': df['redemption_rate'].mean()
    }

def save_comprehensive_results(df: pd.DataFrame, thresholds: Dict):
    """Save all results for further analysis"""
    current_dir = os.getcwd()
    print(f"💾 Saving comprehensive results to: {current_dir}")
    
    saved_files = []
    
    try:
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
        
        # New: Save redemption success scenarios
        high_redemption = df[df['redemption_rate'] > 0.5]
        if len(high_redemption) > 0:
            redemption_file = 'high_redemption_scenarios.csv'
            high_redemption.to_csv(redemption_file, index=False)
            if os.path.exists(redemption_file):
                saved_files.append(f"♻️  {redemption_file} ({len(high_redemption)} scenarios)")
        
        # Create a data summary file
        summary_report = 'enhanced_experiment_summary.txt'
        with open(summary_report, 'w') as f:
            f.write("Enhanced Constraint Cascade Mass Experiment Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Simulations: {len(df)}\n")
            f.write(f"Average Cooperation Rate: {df['final_cooperation_rate'].mean():.3f}\n")
            f.write(f"Extinction Rate: {df['extinction_occurred'].mean():.3f}\n")
            f.write(f"Average Population: {df['final_population'].mean():.1f}\n")
            f.write(f"High Cooperation Scenarios: {len(high_coop)}\n")
            f.write(f"Low Cooperation Scenarios: {len(low_coop)}\n")
            f.write(f"\nEnhanced Metrics:\n")
            f.write(f"Average Redemption Rate: {df['redemption_rate'].mean():.3f}\n")
            f.write(f"Average Trust Level: {df['avg_trust_level'].mean():.3f}\n")
            f.write(f"Total Defections: {df['total_defections'].sum()}\n")
            f.write(f"Total Redemptions: {df['total_redemptions'].sum()}\n")
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
    """Run the complete enhanced mass experiment"""
    print("🔬 Enhanced Constraint Cascade Simulation - Mass Parameter Exploration")
    print("="*70)
    print("🎯 Discovering emergent patterns with enhanced dynamics")
    print("✨ New features: Redemption, Trust, Cooperation Benefits, Maslow Tracking")
    print("📊 Full analysis and reporting capabilities")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Configuration
    num_simulations = 1500  # Full scale!
    use_multiprocessing = True
    
    print(f"\n⚙️  Experiment Configuration:")
    print(f"   🔢 Number of simulations: {num_simulations}")
    print(f"   👥 Population range: 100-500 (5x-10x max)")
    print(f"   🎛️  Parameters: Fully randomized with recovery dynamics")
    print(f"   🖥️  Multiprocessing: {use_multiprocessing}")
    
    try:
        # Run mass experiment
        print(f"\n🚀 PHASE 1: Running {num_simulations} simulations...")
        results = run_mass_experiment(num_simulations, use_multiprocessing)
        print(f"✅ Phase 1 complete: {len(results)} simulations finished")
        
        # Analyze patterns
        print(f"\n📊 PHASE 2: Analyzing emergent patterns...")
        df = analyze_emergent_patterns(results)
        print(f"✅ Phase 2 complete: {len(df)} simulation records analyzed")
        
        # Create visualizations
        print(f"\n📈 PHASE 3: Creating visualizations...")
        create_pattern_visualizations(df)
        print(f"✅ Phase 3 complete: Pattern analysis charts generated")
        
        # Identify critical thresholds
        print(f"\n🎯 PHASE 4: Identifying critical thresholds...")
        thresholds = identify_critical_thresholds(df)
        print(f"✅ Phase 4 complete: Critical thresholds identified")
        
        # Save results
        print(f"\n💾 PHASE 5: Saving results...")
        saved_files = save_comprehensive_results(df, thresholds)
        print(f"✅ Phase 5 complete: {len(saved_files)} files saved")
        
        print(f"\n🎉 ENHANCED EXPERIMENT COMPLETE!")
        print(f"📊 Analyzed {len(results)} simulations")
        print(f"🔍 Discovered patterns across {len(df.columns)} measured variables")
        print(f"📈 Generated comprehensive analysis charts")
        print(f"💾 Saved detailed results for further research")
        
        # Summary statistics
        print(f"\n📋 Key Findings:")
        print(f"   Average cooperation: {df['final_cooperation_rate'].mean():.3f}")
        print(f"   Average redemption rate: {df['redemption_rate'].mean():.3f}")
        print(f"   Scenarios with redemptions: {(df['total_redemptions'] > 0).sum()}")
        print(f"   Average trust level: {df['avg_trust_level'].mean():.3f}")
        
    except Exception as e:
        print(f"\n❌ ERROR in main experiment: {e}")
        print(f"🔧 Please check the error message above for debugging")
        traceback.print_exc()

if __name__ == "__main__":
    main()
