"""
Claude Trading Agent
LLM-based trading agent using Anthropic Claude
"""

from typing import Dict, Any, Optional
import pandas as pd
import json

from src.agents.agentsconfig.agentconfig import BaseLLMAgent
from src.agents.llm_agents.shared_llm_config import SharedLLMConfig


class ClaudeAgent(BaseLLMAgent):
    """Trading agent powered by Anthropic Claude."""
    
    def __init__(self, config=None):
        """Initialize Claude agent."""
        super().__init__(name="Claude", config=config)
        
        # Initialize LLM
        self.llm = SharedLLMConfig(config)
        
        if not self.llm.is_available('anthropic'):
            raise ValueError("Claude API not available. Check config and API key.")
        
        # Claude-specific settings
        self.model_config = self.llm.get_model_config('anthropic')
        
        # System prompt for trading
        self.system_prompt = """You are an expert AI trading agent with deep knowledge of:
- Technical analysis and chart patterns
- Fundamental analysis of companies
- Market psychology and sentiment
- Risk management and position sizing
- Portfolio optimization

Your goal is to make profitable trading decisions while managing risk.
You must provide clear, actionable recommendations with detailed reasoning.

Always respond in JSON format with this structure:
{
    "analysis": "Your detailed market analysis",
    "sentiment": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "recommendation": "buy/sell/hold",
    "reasoning": "Why you made this decision",
    "risk_factors": ["list", "of", "risks"],
    "entry_price": 123.45,
    "exit_price": 130.00,
    "stop_loss": 120.00
}"""
        
        self.log.info("Claude agent initialized")
    
    def analyze_market(
        self,
        symbol: str,
        data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market using Claude.
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            market_context: Optional market context
        
        Returns:
            Analysis results
        """
        self.log.info(f"Analyzing {symbol} with Claude...")
        
        # Prepare market data summary
        latest = data.iloc[-1]
        prev_5d = data.iloc[-6]
        prev_20d = data.iloc[-21] if len(data) >= 21 else data.iloc[0]
        
        # Calculate key metrics
        current_price = latest['close']
        change_5d = ((current_price - prev_5d['close']) / prev_5d['close']) * 100
        change_20d = ((current_price - prev_20d['close']) / prev_20d['close']) * 100
        avg_volume = data['volume'].tail(20).mean()
        current_volume = latest['volume']
        
        # Prepare prompt
        prompt = f"""Analyze {symbol} and provide a trading recommendation.

CURRENT DATA:
- Symbol: {symbol}
- Current Price: ${current_price:.2f}
- 5-day change: {change_5d:+.2f}%
- 20-day change: {change_20d:+.2f}%
- Current Volume: {current_volume:,.0f}
- Average Volume (20d): {avg_volume:,.0f}
- Volume Ratio: {current_volume/avg_volume:.2f}x

RECENT PRICE ACTION:
- High (5d): ${data['high'].tail(5).max():.2f}
- Low (5d): ${data['low'].tail(5).min():.2f}
- High (20d): ${data['high'].tail(20).max():.2f}
- Low (20d): ${data['low'].tail(20).min():.2f}

TECHNICAL INDICATORS:
- 5-day MA: ${data['close'].tail(5).mean():.2f}
- 20-day MA: ${data['close'].tail(20).mean():.2f}
- 50-day MA: ${data['close'].tail(50).mean():.2f} (if available)
"""
        
        # Add market context if available
        if market_context:
            prompt += f"\n\nMARKET CONTEXT:\n{json.dumps(market_context, indent=2)}"
        
        prompt += "\n\nProvide your analysis and recommendation in JSON format."
        
        # Get Claude's analysis
        result = self.llm.generate(
            prompt=prompt,
            provider='anthropic',
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        if 'error' in result:
            self.log.error(f"Error from Claude: {result['error']}")
            return {'error': result['error']}
        
        # Parse JSON response
        try:
            response_text = result['response']
            # Claude might wrap JSON in markdown
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            analysis = json.loads(response_text.strip())
            
            # Add metadata
            analysis['symbol'] = symbol
            analysis['model'] = 'claude'
            analysis['cost'] = result['cost']
            analysis['tokens'] = result['usage']['total_tokens']
            
            self.log.info(f"Claude analysis complete: {analysis['recommendation']} with {analysis['confidence']:.2f} confidence")
            
            return analysis
            
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to parse Claude response: {e}")
            self.log.error(f"Response: {result['response']}")
            return {
                'error': 'Failed to parse response',
                'raw_response': result['response']
            }
    
    def make_decision(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make trading decision based on Claude's analysis.
        
        Args:
            analysis: Analysis from analyze_market()
        
        Returns:
            Trading decision
        """
        if 'error' in analysis:
            return {
                'action': 'hold',
                'quantity': 0,
                'confidence': 0.0,
                'reason': analysis['error']
            }
        
        recommendation = analysis.get('recommendation', 'hold').lower()
        confidence = analysis.get('confidence', 0.0)
        
        # Position sizing based on confidence
        # Higher confidence = larger position
        if confidence > 0.8:
            position_size = 100  # Full position
        elif confidence > 0.6:
            position_size = 50   # Half position
        elif confidence > 0.5:
            position_size = 25   # Quarter position
        else:
            position_size = 0    # No trade
        
        # Determine action
        if recommendation == 'buy' and confidence >= 0.5:
            action = 'buy'
            quantity = position_size
        elif recommendation == 'sell' and confidence >= 0.5:
            action = 'sell'
            quantity = position_size
        else:
            action = 'hold'
            quantity = 0
        
        decision = {
            'action': action,
            'quantity': quantity,
            'confidence': confidence,
            'reasoning': analysis.get('reasoning', ''),
            'entry_price': analysis.get('entry_price'),
            'exit_price': analysis.get('exit_price'),
            'stop_loss': analysis.get('stop_loss'),
            'risk_factors': analysis.get('risk_factors', [])
        }
        
        # Record decision
        self.decisions.append({
            'timestamp': pd.Timestamp.now(),
            'symbol': analysis.get('symbol'),
            'decision': decision,
            'analysis': analysis
        })
        
        self.log.info(f"Decision: {action} {quantity} shares (confidence: {confidence:.2f})")
        
        return decision
