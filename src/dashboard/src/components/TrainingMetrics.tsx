import React, { useState, useEffect, useCallback } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import { fetchTrainingMetrics } from '../services/api';
import type { TrainingMetricsProps, TrainingMetrics as TMetrics } from '../types';

const safeNumber = (value?: number | null): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const formatNumber = (value?: number | null, digits: number = 2, suffix = ''): string => {
  const safe = safeNumber(value);
  return safe !== null ? `${safe.toFixed(digits)}${suffix}` : 'N/A';
};

const formatCurrency = (value?: number | null): string => {
  const safe = safeNumber(value);
  return safe !== null ? `$${safe.toFixed(2)}` : 'N/A';
};

const formatPercent = (value?: number | null, digits: number = 1): string => {
  const safe = safeNumber(value);
  return safe !== null ? `${safe.toFixed(digits)}%` : 'N/A';
};

const TrainingMetrics: React.FC<TrainingMetricsProps> = ({ symbol }) => {
  const [metrics, setMetrics] = useState<TMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const loadMetrics = useCallback(async (): Promise<void> => {
    setLoading(true);
    try {
      const data = await fetchTrainingMetrics(symbol);
      setMetrics(data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    loadMetrics();
  }, [loadMetrics]);

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  const r2 = safeNumber(metrics.test_r2);
  const direction = safeNumber(metrics.test_direction_accuracy);
  const rmse = safeNumber(metrics.test_rmse);
  const mae = safeNumber(metrics.test_mae);
  const mape = safeNumber(metrics.test_mape);

  const radarData = [
    {
      metric: 'R^2 Score',
      value: r2 !== null ? Math.max(0, r2 * 100) : 0,
      fullMark: 100
    },
    {
      metric: 'Direction',
      value: direction !== null ? direction : 0,
      fullMark: 100
    },
    {
      metric: 'Low RMSE',
      value: rmse !== null && rmse !== 0 ? Math.max(0, 100 - rmse) : 0,
      fullMark: 100
    },
    {
      metric: 'MAPE',
      value: mape !== null ? Math.max(0, 100 - mape) : 0,
      fullMark: 100
    }
  ];

  return (
    <div>
      <ResponsiveContainer width="100%" height={250}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#374151" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#9CA3AF', fontSize: 10 }} />
          <Radar
            name={symbol}
            dataKey="value"
            stroke="#3B82F6"
            fill="#3B82F6"
            fillOpacity={0.6}
          />
        </RadarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-2 gap-4 mt-6">
        <MetricCard label="R^2 Score" value={formatNumber(r2, 4)} isGood={!!r2 && r2 > 0.8} />
        <MetricCard label="RMSE" value={formatCurrency(rmse)} isGood={!!rmse && rmse < 5} />
        <MetricCard label="MAE" value={formatCurrency(mae)} isGood={!!mae && mae < 3} />
        <MetricCard label="MAPE" value={formatPercent(mape, 2)} isGood={!!mape && mape < 5} />
        <MetricCard label="Direction Acc" value={formatPercent(direction)} isGood={!!direction && direction > 55} />
        <MetricCard label="Samples" value={metrics.test_samples?.toLocaleString?.() || 'N/A'} isGood={true} />
      </div>

      {metrics.top_features && (
        <div className="mt-6">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Top 5 Important Features
          </h4>
          <div className="space-y-2">
            {metrics.top_features.slice(0, 5).map((feature, i) => (
              <div key={i} className="flex items-center">
                <span className="text-xs text-gray-600 dark:text-gray-400 w-32 truncate">
                  {feature.name}
                </span>
                <div className="flex-1 ml-3">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all"
                      style={{ width: `${feature.importance * 100}%` }}
                    ></div>
                  </div>
                </div>
                <span className="text-xs text-gray-600 dark:text-gray-400 ml-2 w-12 text-right">
                  {(feature.importance * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

interface MetricCardProps {
  label: string;
  value: string;
  isGood: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ label, value, isGood }) => {
  return (
    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">{label}</p>
      <p className={`text-lg font-bold ${isGood ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
        {value}
      </p>
    </div>
  );
};

export default TrainingMetrics;
