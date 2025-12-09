from typing import Dict, Any, Optional
import pandas as pd
import json

from ..agentsconfig.agentconfig import BaseLLMAgent
from .shared_llm_config import SharedLLMConfig


class GPTAgent(BaseLLMAgent):
    """Trading agent powered by OpenAI GPT."""
    
    def __init__(self, config=None):
        """Initialize GPT agent."""
        super().__init__(name="GPT", config=config)
        
        # Initialize LLM
        self.llm = SharedLLMConfig(config)
        
        if not self.llm.is_available('openai'):
            raise ValueError("OpenAI API not available. Check config and API key.")
        
        # GPT-specific settings
        self.model_config = self.llm.get_model_config('openai')
        
        # System prompt for trading
        self.system_prompt = """You are a professional AI trading agent specializing in:
- Quantitative analysis and statistical modeling
- Pattern recognition in price charts
- Market microstructure and order flow
- Machine learning-based predictions
- Risk-adjusted return optimization

Provide data-driven, quantitative trading recommendations.

Respond in JSON format:
{
    "analysis": "Quantitative analysis summary",
    "signal": "strong_buy/buy/neutral/sell/strong_sell",
    "confidence": 0.0-1.0,
    "action": "buy/sell/hold",
    "reasoning": "Statistical reasoning",
    "metrics": {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
        "risk_score": 0.3
    },
    "entry_price": 123.45,
    "exit_price": 130.00,
    "stop_loss": 120.00
}"""
        
        self.log.info("GPT agent initialized")
    
    def analyze_market(
        self,
        symbol: str,
        data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market using GPT.
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            market_context: Optional market context
        
        Returns:
            Analysis results
        """
        self.log.info(f"Analyzing {symbol} with GPT...")
        
        # Calculate advanced metrics
        returns = data['close'].pct_change()
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        
        # Moving averages
        ma5 = data['close'].rolling(5).mean().iloc[-1]
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        ma50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Volume analysis
        volume_ma = data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = data['volume'].iloc[-1] / volume_ma
        
        # Prepare prompt
        current_price = data['close'].iloc[-1]
        
        ma50_str = f"${ma50:.2f}" if ma50 is not None else "N/A"

        prompt = f"""Analyze {symbol} using quantitative methods.

PRICE DATA:
- Current: ${current_price:.2f}
- MA(5): ${ma5:.2f}
- MA(20): ${ma20:.2f}
- MA(50): {ma50_str}
- Price vs MA(20): {((current_price - ma20) / ma20 * 100):+.2f}%

MOMENTUM & VOLATILITY:
- RSI(14): {current_rsi:.2f}
- Annualized Volatility: {volatility:.2%}
- 5-day return: {returns.tail(5).sum():.2%}
- 20-day return: {returns.tail(20).sum():.2%}

VOLUME:
- Current: {data['volume'].iloc[-1]:,.0f}
- 20-day avg: {volume_ma:,.0f}
- Volume ratio: {volume_ratio:.2f}x

STATISTICAL MEASURES:
- Daily avg return: {returns.mean():.4%}
- Sharpe (approx): {(returns.mean() / returns.std() * (252**0.5)):.2f}
- Max drawdown (20d): {((data['close'].tail(20) / data['close'].tail(20).cummax() - 1).min()):.2%}
"""
        
        if market_context:
            prompt += f"\n\nCONTEXT:\n{json.dumps(market_context, indent=2)}"
        
        prompt += "\n\nProvide quantitative analysis and recommendation in JSON."
        
        # Get GPT's analysis
        result = self.llm.generate(
            prompt=prompt,
            provider='openai',
            system_prompt=self.system_prompt,
            temperature=0.5,  # Lower temp for more consistent analysis
            max_tokens=2000
        )
        
        if 'error' in result:
            self.log.error(f"Error from GPT: {result['error']}")
            return {'error': result['error']}
        
        # Parse JSON response
        try:
            response_text = result['response']
            # Clean markdown
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            analysis = json.loads(response_text.strip())
            
            # Add metadata
            analysis['symbol'] = symbol
            analysis['model'] = 'gpt'
            analysis['cost'] = result['cost']
            analysis['tokens'] = result['usage']['total_tokens']
            
            self.log.info(f"GPT analysis complete: {analysis['action']} with {analysis['confidence']:.2f} confidence")
            
            return analysis
            
        except json.JSONDecodeError as e:
            self.log.error(f"Failed to parse GPT response: {e}")
            return {
                'error': 'Failed to parse response',
                'raw_response': result['response']
            }
    
    def make_decision(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make trading decision based on GPT's analysis.
        
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
        
        action = analysis.get('action', 'hold').lower()
        confidence = analysis.get('confidence', 0.0)
        signal = analysis.get('signal', 'neutral')
        
        # Position sizing based on signal strength and confidence
        if signal == 'strong_buy' and confidence > 0.8:
            position_size = 100
        elif signal in ['strong_buy', 'buy'] and confidence > 0.7:
            position_size = 75
        elif signal == 'buy' and confidence > 0.6:
            position_size = 50
        elif signal in ['strong_sell', 'sell'] and confidence > 0.6:
            position_size = 50  # For selling
        else:
            position_size = 0
        
        # Final decision
        if action == 'buy' and position_size > 0:
            final_action = 'buy'
            quantity = position_size
        elif action == 'sell' and position_size > 0:
            final_action = 'sell'
            quantity = position_size
        else:
            final_action = 'hold'
            quantity = 0
        
        decision = {
            'action': final_action,
            'quantity': quantity,
            'confidence': confidence,
            'signal': signal,
            'reasoning': analysis.get('reasoning', ''),
            'entry_price': analysis.get('entry_price'),
            'exit_price': analysis.get('exit_price'),
            'stop_loss': analysis.get('stop_loss'),
            'metrics': analysis.get('metrics', {})
        }
        
        # Record decision
        self.decisions.append({
            'timestamp': pd.Timestamp.now(),
            'symbol': analysis.get('symbol'),
            'decision': decision,
            'analysis': analysis
        })
        
        self.log.info(f"Decision: {final_action} {quantity} shares (signal: {signal}, confidence: {confidence:.2f})")
        return decision

