import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, PercentIcon, Target, BarChart3, GitBranch } from 'lucide-react';
import ModelPerformance from './ModelPerformance';
import PredictionChart from './PredictionChart';
// import TrainingMetrics from './TrainingMetrics'; // HIDDEN FOR DEMO
import { SymbolSelector } from './SymbolSelector';
import { LivePredictions } from './LivePredictions';
import PipelineVisualization from './PipelineVisualization';
import { fetchSummary, fetchSymbols } from '../services/api';
import type { Summary, SymbolMetrics, StatCardProps } from '../types';

type TabType = 'dashboard' | 'pipeline';

const Dashboard: React.FC = () => {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [symbols, setSymbols] = useState<SymbolMetrics[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BMO.TO');
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async (): Promise<void> => {
    setLoading(true);
    try {
      const [summaryData, symbolsData] = await Promise.all([
        fetchSummary(),
        fetchSymbols()
      ]);
      setSummary(summaryData);
      setSymbols(symbolsData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Tab Navigation */}
      <div className="mb-6 border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`${
              activeTab === 'dashboard'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition-colors`}
          >
            <BarChart3 className="w-5 h-5" />
            Performance Dashboard
          </button>
          <button
            onClick={() => setActiveTab('pipeline')}
            className={`${
              activeTab === 'pipeline'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition-colors`}
          >
            <GitBranch className="w-5 h-5" />
            Pipeline Visualization
          </button>
        </nav>
      </div>

      {/* Render Pipeline View */}
      {activeTab === 'pipeline' && <PipelineVisualization />}

      {/* Render Dashboard View */}
      {activeTab === 'dashboard' && (
        <>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={<Activity className="w-6 h-6" />}
          title="Models Trained"
          value={summary?.total_models || 0}
          color="blue"
        />
        <StatCard
          icon={<TrendingUp className="w-6 h-6" />}
          title="Avg R² Score"
          value={(summary?.avg_r2 || 0).toFixed(3)}
          color="green"
        />
        <StatCard
          icon={<PercentIcon className="w-6 h-6" />}
          title="Avg RMSE"
          value={`$${(summary?.avg_rmse || 0).toFixed(2)}`}
          color="purple"
        />
        <StatCard
          icon={<Target className="w-6 h-6" />}
          title="Avg Direction"
          value={`${(summary?.avg_direction || 0).toFixed(1)}%`}
          color="orange"
        />
      </div>

      {/* Symbol Selector */}
      <div className="mb-8">
        <SymbolSelector
          symbols={symbols}
          selectedSymbol={selectedSymbol}
          onSelectSymbol={setSelectedSymbol}
        />
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 gap-6 mb-8">
        {/* Model Performance */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Model Performance Comparison
          </h2>
          <ModelPerformance symbols={symbols} />
        </div>

        {/* Training Metrics - TEMPORARILY HIDDEN FOR DEMO */}
        {/* <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Training Metrics
          </h2>
          <TrainingMetrics symbol={selectedSymbol} />
        </div> */}
      </div>

      {/* Prediction Chart - Full Width */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Actual vs Predicted Prices
        </h2>
        <PredictionChart symbol={selectedSymbol} />
      </div>

          {/* Multi Agent Forecast */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Multi Agent Forecast
            </h2>
            <LivePredictions symbol={selectedSymbol} />
          </div>
        </>
      )}
    </div>
  );
};

const StatCard: React.FC<StatCardProps> = ({ icon, title, value, color }) => {
  const colorClasses: Record<StatCardProps['color'], string> = {
    blue: 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400',
    green: 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400',
    purple: 'bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400',
    orange: 'bg-orange-100 text-orange-600 dark:bg-orange-900/30 dark:text-orange-400',
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
