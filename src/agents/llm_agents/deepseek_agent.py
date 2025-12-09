"""
DeepSeek Trading Agent
LLM-based trading agent using DeepSeek AI
"""

from typing import Dict, Any, Optional
import pandas as pd
import json

from src.agents.agentsconfig.agentconfig import BaseLLMAgent
from src.agents.llm_agents.shared_llm_config import SharedLLMConfig


class DeepSeekAgent(BaseLLMAgent):
    """Trading agent powered by DeepSeek AI."""

    def __init__(self, config=None):
        """Initialize DeepSeek agent."""
        super().__init__(name="DeepSeek", config=config)

        # Initialize LLM
        self.llm = SharedLLMConfig(config)

        if not self.llm.is_available('deepseek'):
            raise ValueError("DeepSeek API not available. Check config and API key.")

        # DeepSeek-specific settings
        self.model_config = self.llm.get_model_config('deepseek')
        self.is_reasoner = "reasoner" in self.model_config.get("model", "").lower()

        # System prompt for trading
        self.system_prompt = """You are an advanced AI trading agent specializing in:
- Deep learning-based market analysis
- Multi-timeframe technical analysis
- Quantitative momentum and trend detection
- Risk-adjusted portfolio strategies
- Advanced statistical modeling

You excel at identifying complex patterns and providing data-driven insights.
Focus on precision, accuracy, and robust risk management.
Think step-by-step through the analysis, considering multiple perspectives.

Respond in JSON format:
{
    "analysis": "Detailed technical and quantitative analysis",
    "signal": "strong_buy/buy/neutral/sell/strong_sell",
    "confidence": 0.0-1.0,
    "action": "buy/sell/hold",
    "reasoning": "Clear reasoning with data support",
    "technical_score": 0.0-10.0,
    "risk_assessment": {
        "level": "low/medium/high",
        "factors": ["list", "of", "risk", "factors"]
    },
    "entry_price": 123.45,
    "exit_price": 130.00,
    "stop_loss": 120.00,
    "timeframe": "short/medium/long"
}"""

        self.log.info("DeepSeek agent initialized")

    def analyze_market(
        self,
        symbol: str,
        data: pd.DataFrame,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze market using DeepSeek.

        Args:
            symbol: Stock symbol
            data: Historical price data
            market_context: Optional market context

        Returns:
            Analysis results
        """
        self.log.info(f"Analyzing {symbol} with DeepSeek...")

        # Calculate comprehensive technical indicators
        returns = data['close'].pct_change()
        volatility = returns.std() * (252 ** 0.5)  # Annualized

        # Moving averages (multiple timeframes)
        ma5 = data['close'].rolling(5).mean().iloc[-1]
        ma10 = data['close'].rolling(10).mean().iloc[-1]
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        ma50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
        ma100 = data['close'].rolling(100).mean().iloc[-1] if len(data) >= 100 else None

        # Exponential moving averages
        ema12 = data['close'].ewm(span=12).mean().iloc[-1]
        ema26 = data['close'].ewm(span=26).mean().iloc[-1]

        # MACD
        macd_line = ema12 - ema26
        signal_line = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
        signal_line = signal_line.ewm(span=9).mean().iloc[-1]
        macd_histogram = macd_line - signal_line

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Bollinger Bands
        bb_ma = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        bb_upper = bb_ma + (2 * bb_std)
        bb_lower = bb_ma - (2 * bb_std)
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]

        # Volume analysis
        volume_ma = data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = data['volume'].iloc[-1] / volume_ma

        # On-Balance Volume
        obv = (data['volume'] * ((data['close'].diff() > 0).astype(int) * 2 - 1)).cumsum()
        obv_trend = obv.diff(5).iloc[-1]

        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        # Momentum indicators
        momentum_5d = returns.tail(5).sum()
        momentum_20d = returns.tail(20).sum()

        # Prepare comprehensive prompt
        current_price = data['close'].iloc[-1]

        # Format MA strings properly
        ma50_str = f"${ma50:.2f}" if ma50 is not None else "N/A"
        ma100_str = f"${ma100:.2f}" if ma100 is not None else "N/A"

        prompt = f"""Perform deep technical analysis on {symbol}.

PRICE DATA:
- Current: ${current_price:.2f}
- MA(5): ${ma5:.2f} | MA(10): ${ma10:.2f} | MA(20): ${ma20:.2f}
- MA(50): {ma50_str} | MA(100): {ma100_str}
- EMA(12): ${ema12:.2f} | EMA(26): ${ema26:.2f}
- Price position: MA5: {((current_price - ma5) / ma5 * 100):+.2f}%, MA20: {((current_price - ma20) / ma20 * 100):+.2f}%

MOMENTUM INDICATORS:
- RSI(14): {current_rsi:.2f} ({('Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral')})
- MACD: {macd_line:.4f} | Signal: {signal_line:.4f} | Histogram: {macd_histogram:.4f}
- 5-day momentum: {momentum_5d:.2%}
- 20-day momentum: {momentum_20d:.2%}

VOLATILITY & RISK:
- ATR(14): ${atr:.2f}
- Annualized Volatility: {volatility:.2%}
- Bollinger Bands: Upper ${current_bb_upper:.2f} | Lower ${current_bb_lower:.2f}
- BB Width: {((current_bb_upper - current_bb_lower) / bb_ma.iloc[-1] * 100):.2f}%
- Price in BB: {((current_price - current_bb_lower) / (current_bb_upper - current_bb_lower) * 100):.1f}%

VOLUME ANALYSIS:
- Current: {data['volume'].iloc[-1]:,.0f}
- 20-day avg: {volume_ma:,.0f}
- Volume ratio: {volume_ratio:.2f}x
- OBV trend (5d): {('Bullish' if obv_trend > 0 else 'Bearish' if obv_trend < 0 else 'Neutral')}

STATISTICAL MEASURES:
- Daily avg return: {returns.mean():.4%}
- Sharpe ratio (approx): {(returns.mean() / returns.std() * (252**0.5)):.2f}
- Max drawdown (20d): {((data['close'].tail(20) / data['close'].tail(20).cummax() - 1).min()):.2%}
- Win rate (20d): {(returns.tail(20) > 0).sum() / 20:.2%}

PRICE ACTION:
- 5-day range: ${data['low'].tail(5).min():.2f} - ${data['high'].tail(5).max():.2f}
- 20-day range: ${data['low'].tail(20).min():.2f} - ${data['high'].tail(20).max():.2f}
"""

        if market_context:
            prompt += f"\n\nMARKET CONTEXT:\n{json.dumps(market_context, indent=2)}"

        prompt += "\n\nProvide comprehensive technical analysis and recommendation in JSON."

        # Get DeepSeek's analysis
        # Use higher max_tokens for reasoner model to allow for reasoning process
        max_tokens = 8000 if self.is_reasoner else 2500
        temperature = 1.0 if self.is_reasoner else 0.4

        result = self.llm.generate(
            prompt=prompt,
            provider='deepseek',
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if 'error' in result:
            self.log.error(f"Error from DeepSeek: {result['error']}")
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
            analysis['model'] = 'deepseek-reasoner' if self.is_reasoner else 'deepseek'
            analysis['cost'] = result['cost']
            analysis['tokens'] = result['usage']['total_tokens']

            # Add reasoning process if available (for reasoner model)
            if 'reasoning_process' in result:
                analysis['reasoning_process'] = result['reasoning_process']
                analysis['reasoning_tokens'] = result['usage'].get('reasoning_tokens', 0)

            # Add calculated indicators for reference
            analysis['indicators'] = {
                'rsi': current_rsi,
                'macd': macd_line,
                'atr': atr,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            }

            log_msg = f"DeepSeek analysis complete: {analysis['action']} with {analysis['confidence']:.2f} confidence"
            if self.is_reasoner and 'reasoning_tokens' in analysis:
                log_msg += f" (reasoning tokens: {analysis['reasoning_tokens']})"
            self.log.info(log_msg)

            return analysis

        except json.JSONDecodeError as e:
            self.log.error(f"Failed to parse DeepSeek response: {e}")
            return {
                'error': 'Failed to parse response',
                'raw_response': result['response']
            }

    def make_decision(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make trading decision based on DeepSeek's analysis.

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
        technical_score = analysis.get('technical_score', 5.0)

        # Advanced position sizing based on multiple factors
        # Combines signal strength, confidence, and technical score
        if signal == 'strong_buy' and confidence > 0.85 and technical_score >= 8.0:
            position_size = 100  # Full position
        elif signal == 'strong_buy' and confidence > 0.75 and technical_score >= 7.0:
            position_size = 80
        elif signal in ['strong_buy', 'buy'] and confidence > 0.70:
            position_size = 60
        elif signal == 'buy' and confidence > 0.65:
            position_size = 40
        elif signal == 'buy' and confidence > 0.55:
            position_size = 25
        elif signal in ['strong_sell', 'sell'] and confidence > 0.65:
            position_size = 50  # For selling
        elif signal == 'sell' and confidence > 0.55:
            position_size = 30
        else:
            position_size = 0

        # Risk-adjusted final decision
        risk_level = analysis.get('risk_assessment', {}).get('level', 'medium')
        if risk_level == 'high':
            position_size = int(position_size * 0.5)  # Reduce position by 50% for high risk

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
            'technical_score': technical_score,
            'reasoning': analysis.get('reasoning', ''),
            'entry_price': analysis.get('entry_price'),
            'exit_price': analysis.get('exit_price'),
            'stop_loss': analysis.get('stop_loss'),
            'timeframe': analysis.get('timeframe', 'medium'),
            'risk_assessment': analysis.get('risk_assessment', {}),
            'indicators': analysis.get('indicators', {})
        }

        # Record decision
        self.decisions.append({
            'timestamp': pd.Timestamp.now(),
            'symbol': analysis.get('symbol'),
            'decision': decision,
            'analysis': analysis
        })

        self.log.info(
            f"Decision: {final_action} {quantity} shares "
            f"(signal: {signal}, confidence: {confidence:.2f}, score: {technical_score:.1f})"
        )

        return decision
