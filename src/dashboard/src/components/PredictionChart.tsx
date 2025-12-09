import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { fetchPredictions } from '../services/api';
import type { PredictionChartProps, Prediction, TimeRange } from '../types';

const PredictionChart: React.FC<PredictionChartProps> = ({ symbol }) => {
  const [data, setData] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [timeRange, setTimeRange] = useState<TimeRange>(100);
  const [yearFilter, _setYearFilter] = useState<string>('all');

  const loadPredictions = useCallback(async (): Promise<void> => {
    setLoading(true);
    try {
      const predictions = await fetchPredictions(symbol, timeRange);

      // Filter data by year if a specific year is selected
      let filteredData = predictions;
      if (yearFilter !== 'all') {
        filteredData = predictions.filter(pred => {
          const year = new Date(pred.date).getFullYear();
          return year.toString() === yearFilter;
        });
      }

      setData(filteredData);
    } catch (error) {
      console.error('Error loading predictions:', error);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeRange, yearFilter]);

  useEffect(() => {
    loadPredictions();
  }, [loadPredictions]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  const timeRanges: { days: TimeRange; label: string }[] = [
    { days: 30, label: '1M' },
    { days: 60, label: '2M' },
    { days: 100, label: '100D' },
    { days: 180, label: '6M' },
    { days: 365, label: '1Y' },
  ];

  const formatCurrency = (value?: number | null): string => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return `$${value.toFixed(2)}`;
    }
    return 'N/A';
  };

  const tooltipFormatter = (value: ValueType, name?: NameType): [string, string] => {
    const formattedValue = typeof value === 'number' ? `$${value.toFixed(2)}` : 'N/A';
    return [formattedValue, typeof name === 'undefined' ? '' : String(name)];
  };

  return (
    <div>
      {/* Time Range Selector and Year Filter */}
      <div className="mb-4">
        <div className="flex items-center justify-between gap-4 mb-3">
          {/* Time Range Buttons */}
          <div className="flex space-x-2 flex-wrap gap-2">
            {timeRanges.map(({ days, label }) => (
              <button
                key={days}
                onClick={() => setTimeRange(days)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  timeRange === days
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Year Filter Dropdown 
          <div className="flex items-center gap-2">
            <label htmlFor="year-filter" className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
              Filter by Year:
            </label>
            <select
              id="year-filter"
              value={yearFilter}
              onChange={(e) => setYearFilter(e.target.value)}
              className="px-3 py-2 rounded-lg text-sm font-medium bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:border-blue-500 dark:hover:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 transition-colors cursor-pointer"
            >
              <option value="all">All Years</option>
              <option value="2025">2025</option>
              <option value="2024">2024</option>
              <option value="2023">2023</option>
              <option value="2022">2022</option>
              <option value="2021">2021</option>
              <option value="2020">2020</option>
            </select>
          </div>

          */}
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
          <XAxis
            dataKey="date"
            stroke="#9CA3AF"
            tick={{ fontSize: 12 }}
            tickFormatter={(date: string) => {
              const d = new Date(date);
              // For ranges > 180 days, show month and year
              if (timeRange > 180) {
                return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
              }
              // For ranges <= 180 days, show month and day
              return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }}
          />
          <YAxis
            stroke="#9CA3AF"
            tick={{ fontSize: 12 }}
            tickFormatter={(value: number) => `$${value.toFixed(0)}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: 'none',
              borderRadius: '8px',
              color: '#fff'
            }}
            formatter={tooltipFormatter}
            labelFormatter={(date: string) => new Date(date).toLocaleDateString()}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
            name="Actual Price"
          />
          <Line
            type="monotone"
            dataKey="agentVD"
            stroke="#F97316"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
            name="XGBoost Prediction"
          />
          <Line
            type="monotone"
            dataKey="gpt"
            stroke="#10B981"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
            name="GPT Prediction"
          />
          <Line
            type="monotone"
            dataKey="claude"
            stroke="#A855F7"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
            name="Claude Prediction"
          />
          <Line
            type="monotone"
            dataKey="deepseek"
            stroke="#EC4899"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={false}
            name="DeepSeek Prediction"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
        <StatBox
          label="Latest Actual"
          value={formatCurrency(data[data.length - 1]?.actual)}
          color="blue"
        />
        <StatBox
          label="XGBoost"
          value={formatCurrency(data[data.length - 1]?.agentVD)}
          color="orange"
        />
        <StatBox
          label="GPT"
          value={formatCurrency(data[data.length - 1]?.gpt)}
          color="green"
        />
        <StatBox
          label="Claude"
          value={formatCurrency(data[data.length - 1]?.claude)}
          color="purple"
        />
        <StatBox
          label="DeepSeek"
          value={formatCurrency(data[data.length - 1]?.deepseek)}
          color="pink"
        />
      </div>
    </div>
  );
};

interface StatBoxProps {
  label: string;
  value: string;
  color: 'blue' | 'red' | 'purple' | 'green' | 'orange' | 'pink';
}

const StatBox: React.FC<StatBoxProps> = ({ label, value, color }) => {
  const colors: Record<StatBoxProps['color'], string> = {
    blue: 'text-blue-600 dark:text-blue-400',
    red: 'text-red-600 dark:text-red-400',
    purple: 'text-purple-600 dark:text-purple-400',
    green: 'text-green-600 dark:text-green-400',
    orange: 'text-orange-500 dark:text-orange-300',
    pink: 'text-pink-600 dark:text-pink-400',
  };

  return (
    <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">{label}</p>
      <p className={`text-lg font-bold ${colors[color]}`}>{value}</p>
    </div>
  );
};

export default PredictionChart;
