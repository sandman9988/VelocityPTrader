#!/usr/bin/env python3
"""
Intelligent Experience Replay System
Advanced experience replay with prioritization and trade journey analysis

Features:
- Priority-based experience selection
- Trade journey reconstruction
- Multi-criteria replay optimization
- Performance-based sample weighting
"""

import random
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import math
# import numpy as np  # Disabled for GraalPy compatibility

@dataclass
class TradeJourney:
    """Complete trade journey for experience analysis"""
    journey_id: str
    symbol: str
    agent_type: str
    start_time: datetime
    end_time: datetime
    
    # Journey metrics
    total_pnl: float
    max_drawdown: float
    max_profit: float
    duration_hours: float
    trade_count: int
    
    # Learning metrics
    initial_confidence: float
    final_confidence: float
    exploration_ratio: float
    regime_changes: int
    
    # Experience value
    learning_value: float = 0.0
    replay_priority: float = 0.0
    replay_count: int = 0

@dataclass
class PrioritizedExperience:
    """Experience with priority weighting"""
    experience_id: str
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    
    # Priority factors
    temporal_priority: float = 1.0  # Recent experiences get higher priority
    performance_priority: float = 1.0  # High reward experiences
    diversity_priority: float = 1.0  # Diverse state-action pairs
    error_priority: float = 1.0  # High TD-error experiences
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    agent_type: str = ""
    symbol: str = ""
    regime: str = ""
    
    @property
    def total_priority(self) -> float:
        """Calculate total priority score"""
        return (self.temporal_priority * 
                self.performance_priority * 
                self.diversity_priority * 
                self.error_priority)

class IntelligentExperienceReplay:
    """Advanced experience replay system with intelligent prioritization"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        
        # Storage
        self.experiences: Dict[str, PrioritizedExperience] = {}
        self.priority_heap: List[Tuple[float, str]] = []
        self.trade_journeys: Dict[str, TradeJourney] = {}
        
        # Replay statistics
        self.total_replays = 0
        self.diversity_tracker: Dict[str, int] = {}
        self.performance_history = deque(maxlen=1000)
        
        print("🧠 Intelligent Experience Replay initialized")
        print(f"   📚 Capacity: {capacity}")
        print(f"   🎯 Priority alpha: {alpha}, Beta: {beta}")
    
    def add_experience(self, experience_data: Dict[str, Any]) -> str:
        """Add new experience with intelligent prioritization"""
        
        experience_id = f"{experience_data.get('agent', 'unknown')}_{experience_data.get('symbol', 'unknown')}_{int(datetime.now().timestamp())}"
        
        # Create prioritized experience
        experience = PrioritizedExperience(
            experience_id=experience_id,
            state=experience_data.get('state', []),
            action=experience_data.get('action', 0),
            reward=experience_data.get('reward', 0.0),
            next_state=experience_data.get('next_state', []),
            done=experience_data.get('done', False),
            agent_type=experience_data.get('agent', ''),
            symbol=experience_data.get('symbol', ''),
            regime=experience_data.get('regime', '')
        )
        
        # Calculate priority factors
        experience.temporal_priority = 1.0  # Most recent = highest
        experience.performance_priority = self._calculate_performance_priority(experience.reward)
        experience.diversity_priority = self._calculate_diversity_priority(experience)
        experience.error_priority = self._estimate_td_error_priority(experience)
        
        # Add to storage
        self.experiences[experience_id] = experience
        
        # Update priority heap
        priority_score = -experience.total_priority  # Negative for max heap
        heapq.heappush(self.priority_heap, (priority_score, experience_id))
        
        # Manage capacity
        if len(self.experiences) > self.capacity:
            self._remove_lowest_priority()
        
        # Update diversity tracker
        state_key = self._create_state_key(experience.state)
        self.diversity_tracker[state_key] = self.diversity_tracker.get(state_key, 0) + 1
        
        return experience_id
    
    def sample_batch(self, batch_size: int, focus_type: str = 'balanced') -> List[PrioritizedExperience]:
        """Sample batch of experiences with intelligent selection"""
        
        if len(self.experiences) < batch_size:
            return list(self.experiences.values())
        
        if focus_type == 'priority':
            return self._sample_priority_batch(batch_size)
        elif focus_type == 'diverse':
            return self._sample_diverse_batch(batch_size)
        elif focus_type == 'recent':
            return self._sample_recent_batch(batch_size)
        elif focus_type == 'performance':
            return self._sample_performance_batch(batch_size)
        else:  # balanced
            return self._sample_balanced_batch(batch_size)
    
    def _sample_priority_batch(self, batch_size: int) -> List[PrioritizedExperience]:
        """Sample based on total priority scores"""
        
        # Get all experiences with priorities
        experience_priorities = [
            (exp.total_priority ** self.alpha, exp_id, exp) 
            for exp_id, exp in self.experiences.items()
        ]
        
        # Calculate probabilities
        total_priority = sum(priority for priority, _, _ in experience_priorities)
        probabilities = [priority / total_priority for priority, _, _ in experience_priorities]
        
        # Sample based on probabilities (using standard library)
        import random
        sample_size = min(batch_size, len(experience_priorities))
        indices = random.choices(
            range(len(experience_priorities)),
            weights=[priority for priority, _, _ in experience_priorities],
            k=sample_size
        )
        
        sampled = [experience_priorities[i][2] for i in indices]
        
        # Update replay counts
        for exp in sampled:
            exp.replay_count += 1
        
        self.total_replays += 1
        return sampled
    
    def _sample_diverse_batch(self, batch_size: int) -> List[PrioritizedExperience]:
        """Sample for maximum diversity"""
        
        # Group experiences by state-action diversity
        diversity_groups = {}
        for exp_id, exp in self.experiences.items():
            key = self._create_diversity_key(exp)
            if key not in diversity_groups:
                diversity_groups[key] = []
            diversity_groups[key].append(exp)
        
        # Sample from different groups
        sampled = []
        group_keys = list(diversity_groups.keys())
        random.shuffle(group_keys)
        
        for i in range(batch_size):
            if i < len(group_keys):
                group = diversity_groups[group_keys[i]]
                sampled.append(random.choice(group))
            else:
                # Fill remaining with random samples
                all_experiences = list(self.experiences.values())
                sampled.append(random.choice(all_experiences))
        
        return sampled[:batch_size]
    
    def _sample_recent_batch(self, batch_size: int) -> List[PrioritizedExperience]:
        """Sample most recent experiences"""
        
        sorted_experiences = sorted(
            self.experiences.values(), 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        
        return sorted_experiences[:batch_size]
    
    def _sample_performance_batch(self, batch_size: int) -> List[PrioritizedExperience]:
        """Sample highest performing experiences"""
        
        sorted_experiences = sorted(
            self.experiences.values(),
            key=lambda x: abs(x.reward),  # High absolute reward
            reverse=True
        )
        
        return sorted_experiences[:batch_size]
    
    def _sample_balanced_batch(self, batch_size: int) -> List[PrioritizedExperience]:
        """Sample balanced mix of different criteria"""
        
        quarter_size = batch_size // 4
        
        # 25% priority-based
        priority_samples = self._sample_priority_batch(quarter_size)
        
        # 25% recent
        recent_samples = self._sample_recent_batch(quarter_size)
        
        # 25% diverse
        diverse_samples = self._sample_diverse_batch(quarter_size)
        
        # 25% performance
        performance_samples = self._sample_performance_batch(quarter_size)
        
        # Combine and fill to exact batch size
        all_samples = priority_samples + recent_samples + diverse_samples + performance_samples
        
        # Remove duplicates and fill remaining
        seen_ids = set()
        unique_samples = []
        for exp in all_samples:
            if exp.experience_id not in seen_ids:
                unique_samples.append(exp)
                seen_ids.add(exp.experience_id)
        
        # Fill remaining slots randomly
        remaining_needed = batch_size - len(unique_samples)
        if remaining_needed > 0:
            remaining_experiences = [
                exp for exp in self.experiences.values() 
                if exp.experience_id not in seen_ids
            ]
            if remaining_experiences:
                additional = random.sample(
                    remaining_experiences, 
                    min(remaining_needed, len(remaining_experiences))
                )
                unique_samples.extend(additional)
        
        return unique_samples[:batch_size]
    
    def add_trade_journey(self, journey_data: Dict[str, Any]) -> str:
        """Add complete trade journey for analysis"""
        
        journey = TradeJourney(
            journey_id=journey_data['journey_id'],
            symbol=journey_data['symbol'],
            agent_type=journey_data['agent_type'],
            start_time=datetime.fromisoformat(journey_data['start_time']),
            end_time=datetime.fromisoformat(journey_data['end_time']),
            total_pnl=journey_data['total_pnl'],
            max_drawdown=journey_data['max_drawdown'],
            max_profit=journey_data['max_profit'],
            duration_hours=journey_data['duration_hours'],
            trade_count=journey_data['trade_count'],
            initial_confidence=journey_data['initial_confidence'],
            final_confidence=journey_data['final_confidence'],
            exploration_ratio=journey_data['exploration_ratio'],
            regime_changes=journey_data['regime_changes']
        )
        
        # Calculate learning value
        journey.learning_value = self._calculate_journey_learning_value(journey)
        journey.replay_priority = self._calculate_journey_priority(journey)
        
        self.trade_journeys[journey.journey_id] = journey
        
        return journey.journey_id
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive replay statistics"""
        
        return {
            'total_experiences': len(self.experiences),
            'total_journeys': len(self.trade_journeys),
            'total_replays': self.total_replays,
            'capacity_utilization': len(self.experiences) / self.capacity,
            'diversity_score': len(self.diversity_tracker) / max(len(self.experiences), 1),
            'avg_replay_count': sum(exp.replay_count for exp in self.experiences.values()) / max(len(self.experiences), 1),
            'top_priority_experiences': self._get_top_priority_experiences(5),
            'journey_summary': self._get_journey_summary()
        }
    
    def _calculate_performance_priority(self, reward: float) -> float:
        """Calculate priority based on reward performance"""
        
        # Higher absolute rewards get higher priority
        # Normalize to 0.1 - 2.0 range
        abs_reward = abs(reward)
        return min(2.0, max(0.1, 0.5 + abs_reward))
    
    def _calculate_diversity_priority(self, experience: PrioritizedExperience) -> float:
        """Calculate priority based on state-action diversity"""
        
        state_key = self._create_state_key(experience.state)
        frequency = self.diversity_tracker.get(state_key, 0)
        
        # Less frequent states get higher priority
        if frequency == 0:
            return 2.0
        else:
            return min(2.0, max(0.1, 2.0 / (1 + frequency * 0.1)))
    
    def _estimate_td_error_priority(self, experience: PrioritizedExperience) -> float:
        """Estimate TD-error priority (simplified)"""
        
        # Simple heuristic: higher rewards likely have higher TD-error
        # In real implementation, this would use actual TD-error
        reward_factor = min(2.0, abs(experience.reward) + 0.5)
        
        # Add some randomness to prevent overfitting
        random_factor = 0.8 + random.random() * 0.4
        
        return reward_factor * random_factor
    
    def _create_state_key(self, state: List[float]) -> str:
        """Create hashable key for state"""
        
        # Discretize continuous states for diversity tracking
        discretized = [round(s, 2) for s in state[:4]]  # Use first 4 elements
        return str(discretized)
    
    def _create_diversity_key(self, experience: PrioritizedExperience) -> str:
        """Create diversity key for experience"""
        
        state_key = self._create_state_key(experience.state)
        return f"{experience.agent_type}_{experience.symbol}_{experience.action}_{state_key}"
    
    def _remove_lowest_priority(self):
        """Remove experience with lowest priority"""
        
        if not self.priority_heap:
            return
        
        # Find and remove lowest priority experience
        lowest_priority, exp_id = heapq.heappop(self.priority_heap)
        
        if exp_id in self.experiences:
            # Update diversity tracker
            exp = self.experiences[exp_id]
            state_key = self._create_state_key(exp.state)
            if state_key in self.diversity_tracker:
                self.diversity_tracker[state_key] -= 1
                if self.diversity_tracker[state_key] <= 0:
                    del self.diversity_tracker[state_key]
            
            # Remove experience
            del self.experiences[exp_id]
    
    def _calculate_journey_learning_value(self, journey: TradeJourney) -> float:
        """Calculate learning value of trade journey"""
        
        # Factors that increase learning value
        pnl_factor = min(2.0, abs(journey.total_pnl) / 100.0)
        confidence_change = abs(journey.final_confidence - journey.initial_confidence)
        exploration_factor = journey.exploration_ratio
        complexity_factor = min(2.0, journey.regime_changes / 3.0)
        
        learning_value = (pnl_factor + confidence_change + exploration_factor + complexity_factor) / 4.0
        
        return min(1.0, learning_value)
    
    def _calculate_journey_priority(self, journey: TradeJourney) -> float:
        """Calculate replay priority for journey"""
        
        return journey.learning_value * (1.0 + journey.total_pnl / 1000.0)
    
    def _get_top_priority_experiences(self, count: int) -> List[Dict[str, Any]]:
        """Get top priority experiences summary"""
        
        sorted_experiences = sorted(
            self.experiences.values(),
            key=lambda x: x.total_priority,
            reverse=True
        )
        
        return [
            {
                'experience_id': exp.experience_id[:20] + '...',
                'agent': exp.agent_type,
                'symbol': exp.symbol,
                'reward': exp.reward,
                'priority': exp.total_priority,
                'replay_count': exp.replay_count
            }
            for exp in sorted_experiences[:count]
        ]
    
    def _get_journey_summary(self) -> Dict[str, Any]:
        """Get trade journey summary"""
        
        if not self.trade_journeys:
            return {}
        
        journeys = list(self.trade_journeys.values())
        
        return {
            'total_journeys': len(journeys),
            'avg_learning_value': sum(j.learning_value for j in journeys) / len(journeys),
            'avg_duration_hours': sum(j.duration_hours for j in journeys) / len(journeys),
            'total_pnl': sum(j.total_pnl for j in journeys),
            'best_journey': max(journeys, key=lambda j: j.total_pnl).journey_id if journeys else None
        }

def demonstrate_intelligent_replay():
    """Demonstrate intelligent experience replay"""
    
    print("🧠 INTELLIGENT EXPERIENCE REPLAY DEMONSTRATION")
    print("=" * 60)
    
    replay_system = IntelligentExperienceReplay(capacity=100, alpha=0.6, beta=0.4)
    
    # Add various experiences
    for i in range(50):
        experience_data = {
            'agent': 'BERSERKER' if i % 2 == 0 else 'SNIPER',
            'symbol': ['EURUSD+', 'GBPUSD+', 'BTCUSD+'][i % 3],
            'state': [random.random() for _ in range(6)],
            'action': random.randint(0, 2),
            'reward': random.uniform(-10, 20),
            'next_state': [random.random() for _ in range(6)],
            'done': random.random() > 0.7,
            'regime': ['CHAOTIC', 'UNDERDAMPED', 'CRITICALLY_DAMPED'][i % 3]
        }
        
        exp_id = replay_system.add_experience(experience_data)
        if i % 10 == 0:
            print(f"📝 Added experience {i+1}: {exp_id[:30]}...")
    
    # Test different sampling strategies
    strategies = ['priority', 'diverse', 'recent', 'performance', 'balanced']
    
    print(f"\n🎯 Testing sampling strategies:")
    for strategy in strategies:
        batch = replay_system.sample_batch(8, focus_type=strategy)
        avg_reward = sum(exp.reward for exp in batch) / len(batch)
        agent_distribution = {}
        for exp in batch:
            agent_distribution[exp.agent_type] = agent_distribution.get(exp.agent_type, 0) + 1
        
        print(f"   {strategy.upper()}: {len(batch)} experiences, avg reward: {avg_reward:+.2f}")
        print(f"     Agents: {agent_distribution}")
    
    # Add some trade journeys
    for i in range(5):
        journey_data = {
            'journey_id': f"journey_{i}",
            'symbol': ['EURUSD+', 'GBPUSD+'][i % 2],
            'agent_type': 'BERSERKER' if i % 2 == 0 else 'SNIPER',
            'start_time': (datetime.now() - timedelta(hours=i+1)).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_pnl': random.uniform(-100, 200),
            'max_drawdown': random.uniform(0, 50),
            'max_profit': random.uniform(0, 150),
            'duration_hours': random.uniform(0.5, 4.0),
            'trade_count': random.randint(3, 15),
            'initial_confidence': random.uniform(0.3, 0.8),
            'final_confidence': random.uniform(0.4, 0.9),
            'exploration_ratio': random.uniform(0.1, 0.4),
            'regime_changes': random.randint(0, 3)
        }
        
        journey_id = replay_system.add_trade_journey(journey_data)
        print(f"🗺️ Added journey {i+1}: {journey_id}")
    
    # Get statistics
    stats = replay_system.get_replay_statistics()
    print(f"\n📊 REPLAY STATISTICS:")
    print(f"   Total experiences: {stats['total_experiences']}")
    print(f"   Capacity utilization: {stats['capacity_utilization']:.1%}")
    print(f"   Diversity score: {stats['diversity_score']:.3f}")
    print(f"   Avg replay count: {stats['avg_replay_count']:.1f}")
    
    if stats['top_priority_experiences']:
        print(f"   Top priority experiences:")
        for i, exp in enumerate(stats['top_priority_experiences'][:3]):
            print(f"     {i+1}. {exp['agent']} {exp['symbol']}: reward {exp['reward']:+.1f}, priority {exp['priority']:.3f}")
    
    journey_summary = stats.get('journey_summary', {})
    if journey_summary:
        print(f"   Journey summary:")
        print(f"     Total journeys: {journey_summary['total_journeys']}")
        print(f"     Avg learning value: {journey_summary['avg_learning_value']:.3f}")
        print(f"     Total P&L: {journey_summary['total_pnl']:+.1f}")
    
    print("\n✅ Intelligent Experience Replay demonstration completed!")

if __name__ == "__main__":
    demonstrate_intelligent_replay()