// ModelPerformance.tsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ModelPerformanceProps, ModelGrade, PerformanceColor } from '../types';

interface ChartRow {
  symbol: string;
  r2: number;
  rmse: number;
  direction: number;
  hasR2: boolean;
  hasRmse: boolean;
  hasDirection: boolean;
  color: PerformanceColor;
}

type ChartMetricKey = 'r2' | 'rmse' | 'direction';
type AvailabilityKey = 'hasR2' | 'hasRmse' | 'hasDirection';

interface MetricConfig {
  id: string;
  title: string;
  subtitle: string;
  dataKey: ChartMetricKey;
  availabilityKey: AvailabilityKey;
  domain: [number, number];
  formatter: (value: number) => string;
  tickFormatter?: (value: number) => string;
  color: string | ((row: ChartRow) => string);
}

const getColor = (r2?: number | null): PerformanceColor => {
  const value = typeof r2 === 'number' ? r2 : -1;
  if (value > 0.8) return '#10B981';
  if (value > 0.5) return '#F59E0B';
  if (value > 0) return '#FCD34D';
  return '#EF4444';
};

const safeMetric = (value?: number | null): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const formatValue = (hasValue: boolean, value: number, digits: number): string =>
  hasValue ? value.toFixed(digits) : 'N/A';

const ModelPerformance: React.FC<ModelPerformanceProps> = ({ symbols }) => {
  if (!Array.isArray(symbols) || symbols.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
          No model performance data available
        </div>
      </div>
    );
  }

  const data: ChartRow[] = symbols.map(s => {
    const r2 = safeMetric(s.test_r2);
    const rmse = safeMetric(s.test_rmse);
    const direction = safeMetric(s.test_direction_accuracy);

    return {
      symbol: s.symbol,
      r2: r2 ?? 0,
      rmse: rmse ?? 0,
      direction: direction ?? 0,
      hasR2: r2 !== null,
      hasRmse: rmse !== null,
      hasDirection: direction !== null,
      color: getColor(r2)
    };
  });

  const rmseValues = data.filter(row => row.hasRmse).map(row => row.rmse);
  const rmseMax = rmseValues.length ? Math.max(...rmseValues) : 0;
  const rmseDomain: [number, number] = [0, rmseMax ? Math.ceil(rmseMax * 1.1) : 1];

  const metricConfigs: MetricConfig[] = [
    {
      id: 'r2',
      title: 'R^2 Score',
      subtitle: 'Higher values indicate better fit',
      dataKey: 'r2',
      availabilityKey: 'hasR2',
      domain: [-1, 1],
      formatter: value => value.toFixed(4),
      tickFormatter: value => value.toFixed(1),
      color: (row: ChartRow) => row.color,
    },
    {
      id: 'rmse',
      title: 'RMSE ($)',
      subtitle: 'Lower values indicate less error',
      dataKey: 'rmse',
      availabilityKey: 'hasRmse',
      domain: rmseDomain,
      formatter: value => `$${value.toFixed(2)}`,
      tickFormatter: value => value.toFixed(0),
      color: '#8B5CF6',
    },
    {
      id: 'direction',
      title: 'Direction Accuracy',
      subtitle: 'Higher values indicate better directional calls',
      dataKey: 'direction',
      availabilityKey: 'hasDirection',
      domain: [0, 100],
      formatter: value => `${value.toFixed(1)}%`,
      tickFormatter: value => `${value.toFixed(0)}%`,
      color: '#0EA5E9',
    },
  ];

  return (
    <div>
      <div className="space-y-6">
        {metricConfigs.map(metric => {
          const availableData = data.filter(row => row[metric.availabilityKey]);

          if (availableData.length === 0) {
            return (
              <div key={metric.id} className="bg-gray-50 dark:bg-gray-800/40 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {metric.title}
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  No data available for this metric.
                </p>
              </div>
            );
          }

          const resolveColor = (row: ChartRow): string =>
            typeof metric.color === 'function' ? metric.color(row) : metric.color;

          return (
            <div key={metric.id}>
              <div className="flex items-baseline justify-between mb-2">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {metric.title}
                </h3>
                <span className="text-xs text-gray-500 dark:text-gray-400">{metric.subtitle}</span>
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={availableData} layout="vertical" margin={{ left: 60, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.08} />
                  <XAxis
                    type="number"
                    domain={metric.domain}
                    stroke="#9CA3AF"
                    tick={{ fontSize: 12 }}
                    tickFormatter={metric.tickFormatter ?? (value => value.toString())}
                  />
                  <YAxis
                    type="category"
                    dataKey="symbol"
                    stroke="#9CA3AF"
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: 'none',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                    formatter={(value: number) =>
                      metric.formatter(typeof value === 'number' ? value : Number(value))
                    }
                  />
                  <Bar dataKey={metric.dataKey} radius={[0, 4, 4, 0]}>
                    {availableData.map((entry, index) => (
                      <Cell key={`cell-${metric.id}-${index}`} fill={resolveColor(entry)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>

      {/* Performance Summary Table */}
      <div className="overflow-x-auto mt-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-2 px-3 text-gray-700 dark:text-gray-300 font-medium">Symbol</th>
              <th className="text-right py-2 px-3 text-gray-700 dark:text-gray-300 font-medium">R^2</th>
              <th className="text-right py-2 px-3 text-gray-700 dark:text-gray-300 font-medium">RMSE</th>
              <th className="text-right py-2 px-3 text-gray-700 dark:text-gray-300 font-medium">Direction</th>
              <th className="text-center py-2 px-3 text-gray-700 dark:text-gray-300 font-medium">Grade</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr
                key={i}
                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50"
              >
                <td className="py-2 px-3 font-medium text-gray-900 dark:text-white">
                  {row.symbol}
                </td>
                <td className="py-2 px-3 text-right">
                  <span style={{ color: row.color }} className="font-semibold">
                    {formatValue(row.hasR2, row.r2, 4)}
                  </span>
                </td>
                <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                  {row.hasRmse ? `%${row.rmse.toFixed(2)}` : 'N/A'}
                </td>
                <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                  {row.hasDirection ? `${row.direction.toFixed(1)}%` : 'N/A'}
                </td>
                <td className="py-2 px-3 text-center">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getGradeClass(row.hasR2 ? row.r2 : null)}`}>
                    {getGrade(row.hasR2 ? row.r2 : null)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 mt-4 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
          <span className="text-gray-600 dark:text-gray-400">Excellent (R^2 &gt; 0.8)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-orange-500 mr-2"></div>
          <span className="text-gray-600 dark:text-gray-400">Decent (R^2 0.5-0.8)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-yellow-400 mr-2"></div>
          <span className="text-gray-600 dark:text-gray-400">Poor (R^2 0-0.5)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
          <span className="text-gray-600 dark:text-gray-400">Bad (R^2 &lt; 0)</span>
        </div>
      </div>
    </div>
  );
};

const getGrade = (r2?: number | null): ModelGrade => {
  const value = typeof r2 === 'number' ? r2 : -1;
  if (value > 0.9) return 'A';
  if (value > 0.8) return 'A-';
  if (value > 0.7) return 'B';
  if (value > 0.5) return 'C';
  if (value > 0) return 'D';
  return 'F';
};

const getGradeClass = (r2?: number | null): string => {
  const value = typeof r2 === 'number' ? r2 : -1;
  if (value > 0.8) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
  if (value > 0.5) return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400';
  if (value > 0) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
  return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
};

export default ModelPerformance;
