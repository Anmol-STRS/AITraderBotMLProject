import React from 'react';
import { TrendingUp, TrendingDown, Minus, ShoppingCart, DollarSign, Hand } from 'lucide-react';
import { fetchLivePredictions } from '../services/api';
import type { LivePredictionsProps, LivePrediction } from '../types';

const TradingSignalBadge: React.FC<{ action?: string; signal?: string; confidence?: number }> = ({
  action,
  signal,
  confidence
}) => {
  if (!action) return null;

  const actionConfig = {
    buy: {
      icon: <ShoppingCart className="w-4 h-4" />,
      label: 'BUY',
      bgColor: 'bg-green-500 dark:bg-green-600',
      textColor: 'text-white',
    },
    sell: {
      icon: <DollarSign className="w-4 h-4" />,
      label: 'SELL',
      bgColor: 'bg-red-500 dark:bg-red-600',
      textColor: 'text-white',
    },
    hold: {
      icon: <Hand className="w-4 h-4" />,
      label: 'HOLD',
      bgColor: 'bg-gray-500 dark:bg-gray-600',
      textColor: 'text-white',
    },
  };

  const config = actionConfig[action.toLowerCase() as keyof typeof actionConfig];
  if (!config) return null;

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${config.bgColor} ${config.textColor} shadow-sm`}>
      {config.icon}
      <div className="flex flex-col">
        <span className="text-sm font-bold leading-tight">{config.label}</span>
        {confidence !== undefined && (
          <span className="text-xs opacity-90 leading-tight">
            {(confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {signal && (
        <span className="text-xs opacity-75 ml-1 capitalize">
          {signal.replace('_', ' ')}
        </span>
      )}
    </div>
  );
};

export const LivePredictions: React.FC<LivePredictionsProps> = ({ symbol }) => {
  const [predictions, setPredictions] = React.useState<LivePrediction[]>([]);
  const [loading, setLoading] = React.useState<boolean>(true);

  React.useEffect(() => {
    let isMounted = true;

    const load = async (): Promise<void> => {
      setLoading(true);
      try {
        const data = await fetchLivePredictions(symbol);
        if (isMounted) {
          setPredictions(data);
        }
      } catch (error) {
        console.error('Error loading live predictions:', error);
        if (isMounted) {
          setPredictions([]);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    load();

    return () => {
      isMounted = false;
    };
  }, [symbol]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!predictions.length) {
    return (
      <div className="text-center text-sm text-gray-500 dark:text-gray-400">
        No recent predictions available for {symbol}.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {predictions.map((pred, i) => (
        <div
          key={i}
          className="flex flex-col gap-3 p-5 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600"
        >
          {/* Top Row: Date/Agent and Trading Signal Badge */}
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="font-medium text-gray-900 dark:text-white">
                {pred.agent ? pred.agent.toUpperCase() : 'AI Agent'}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                {new Date(pred.date || pred.timestamp || Date.now()).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric'
                })}
              </p>
            </div>

            <TradingSignalBadge
              action={pred.action}
              signal={pred.signal}
              confidence={pred.confidence}
            />
          </div>

          {/* Bottom Row: Price Data */}
          <div className="flex items-center gap-6">
            {(pred.actual || pred.current_price) && (
              <div className="flex-1">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Current Price</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  ${(pred.actual || pred.current_price || 0).toFixed(2)}
                </p>
              </div>
            )}

            {typeof (pred.predicted || pred.predicted_price) === 'number' && (
              <div className="flex-1">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Predicted Price</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  ${(pred.predicted || pred.predicted_price || 0).toFixed(2)}
                </p>
              </div>
            )}

            {typeof (pred.change || pred.change_pct) === 'number' && (
              <div className="flex-1">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Change</p>
                <div className={`inline-flex items-center px-3 py-1.5 rounded-full ${
                  (pred.change || pred.change_pct || 0) > 0
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                    : (pred.change || pred.change_pct || 0) < 0
                    ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                    : 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400'
                }`}>
                  {(pred.change || pred.change_pct || 0) > 0 ? (
                    <TrendingUp className="w-4 h-4 mr-1.5" />
                  ) : (pred.change || pred.change_pct || 0) < 0 ? (
                    <TrendingDown className="w-4 h-4 mr-1.5" />
                  ) : (
                    <Minus className="w-4 h-4 mr-1.5" />
                  )}
                  <span className="text-sm font-semibold">{(pred.change || pred.change_pct || 0).toFixed(2)}%</span>
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default LivePredictions;
