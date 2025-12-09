"""
Base LLM Agent
Abstract base class for all LLM trading agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from src.config.config import Config
from src.util.logging import get_logger


class BaseLLMAgent(ABC):
    """
    Abstract base class for LLM trading agents.
    All LLM agents (GPT, Claude, etc.) inherit from this.
    """
    
    def __init__(self, name: str, config: Optional[Config] = None):
        """
        Initialize base LLM agent.
        
        Args:
            name: Agent name
            config: Optional config object
        """
        self.name = name
        self.config = config or Config()
        self.log = get_logger(name=f"Agent.{name}", enable_console=False)
        
        # Trading state
        self.portfolio = {
            'cash': 100000.0,  # Starting cash
            'positions': {},   # {symbol: shares}
            'history': []      # Trade history
        }
        
        # Performance tracking
        self.decisions = []
        self.total_trades = 0
        
        self.log.info(f"{name} agent initialized")
    
    @abstractmethod
    def analyze_market(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market data and return trading decision.
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            market_context: Optional market context (news, sentiment, etc.)
        
        Returns:
            dict with decision and reasoning
        """
        pass
    
    @abstractmethod
    def make_decision(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make trading decision based on analysis.
        
        Args:
            analysis: Market analysis results
        
        Returns:
            dict with action, quantity, confidence
        """
        pass
    
    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float
    ) -> Dict[str, Any]:
        """
        Execute a trade.
        
        Args:
            symbol: Stock symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Price per share
        
        Returns:
            dict with trade result
        """
        if action == 'buy':
            cost = quantity * price
            if cost > self.portfolio['cash']:
                self.log.warning(f"Insufficient funds for {symbol}: need ${cost:.2f}, have ${self.portfolio['cash']:.2f}")
                return {'success': False, 'reason': 'insufficient_funds'}
            
            self.portfolio['cash'] -= cost
            self.portfolio['positions'][symbol] = self.portfolio['positions'].get(symbol, 0) + quantity
            
        elif action == 'sell':
            if symbol not in self.portfolio['positions'] or self.portfolio['positions'][symbol] < quantity:
                self.log.warning(f"Insufficient shares of {symbol}")
                return {'success': False, 'reason': 'insufficient_shares'}
            
            proceeds = quantity * price
            self.portfolio['cash'] += proceeds
            self.portfolio['positions'][symbol] -= quantity
            
            if self.portfolio['positions'][symbol] == 0:
                del self.portfolio['positions'][symbol]
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'total': quantity * price
        }
        self.portfolio['history'].append(trade)
        self.total_trades += 1
        
        self.log.info(f"Executed {action} {quantity} {symbol} @ ${price:.2f}")
        
        return {'success': True, 'trade': trade}
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dict of {symbol: current_price}
        
        Returns:
            Total portfolio value
        """
        value = self.portfolio['cash']
        
        for symbol, shares in self.portfolio['positions'].items():
            if symbol in current_prices:
                value += shares * current_prices[symbol]
        
        return value
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'agent_name': self.name,
            'total_trades': self.total_trades,
            'current_cash': self.portfolio['cash'],
            'positions': len(self.portfolio['positions']),
            'decisions_made': len(self.decisions)
        }
    
    def reset(self):
        """Reset agent state."""
        self.portfolio = {
            'cash': 100000.0,
            'positions': {},
            'history': []
        }
        self.decisions = []
        self.total_trades = 0
        self.log.info(f"{self.name} agent reset")