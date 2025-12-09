/**
 * Type definitions for AI Trading Arena
 *
 * This file contains all TypeScript interfaces, types, and enums
 * used throughout the application for type safety and code clarity.
 */

// ============================================================================
// API Response Types
// ============================================================================

export interface Summary {
  total_models: number;
  avg_r2: number;
  avg_rmse: number;
  avg_mae: number;
  avg_direction: number;
  avg_mape?: number;
  data_source?: 'database' | 'mock_data';
  symbols_count?: number;
  total_records?: number;
  date_range?: {
    start: string;
    end: string;
  };
}

export interface SymbolMetrics {
  symbol: string;
  test_r2: number;
  test_rmse: number;
  test_mae: number;
  test_mape: number;
  test_direction_accuracy: number;
  train_samples: number;
  test_samples: number;
  features: number;
  top_feature?: string;
}

export type AgentName = 'agentVD' | 'gpt' | 'claude' | 'deepseek';
export type AgentPredictionMap = Partial<Record<AgentName, number | null>>;

export interface ModelPredictionPoint {
  timestamp: string;
  actual?: number | null;
  predicted?: number | null;
  error?: number | null;
  error_pct?: number | null;
  prediction_type?: string;
}

export interface ModelPredictionSeries {
  model_id: number;
  model_name: string;
  model_type: string;
  horizon?: number | null;
  r2_score?: number | null;
  rmse?: number | null;
  mape?: number | null;
  prediction_count: number;
  predictions: ModelPredictionPoint[];
}

export interface MultiModelPredictionResponse {
  symbol: string;
  models: ModelPredictionSeries[];
}

export interface Prediction extends AgentPredictionMap {
  date: string;
  actual: number | null;
}

export interface Feature {
  name: string;
  importance: number;
}

export interface TrainingMetrics extends SymbolMetrics {
  top_features?: Feature[];
}

export interface LivePrediction {
  date: string;
  actual?: number;
  predicted: number;
  confidence?: number;
  change?: number;
  action?: 'buy' | 'sell' | 'hold';
  signal?: string;
  agent?: string;
  symbol?: string;
  timestamp?: string;
  current_price?: number;
  predicted_price?: number;
  change_pct?: number;
}

export interface ChartData {
  date: string;
  actual: number;
  predicted: number;
}

// ============================================================================
// Component Props Types
// ============================================================================

export interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  color: 'blue' | 'green' | 'purple' | 'orange';
}

export interface SymbolSelectorProps {
  symbols: SymbolMetrics[];
  selectedSymbol: string;
  onSelectSymbol: (symbol: string) => void;
}

export interface ModelPerformanceProps {
  symbols: SymbolMetrics[];
}

export interface PredictionChartProps {
  symbol: string;
}

export interface TrainingMetricsProps {
  symbol: string;
}

export interface LivePredictionsProps {
  symbol: string;
}

// ============================================================================
// Utility Types & Enums
// ============================================================================

export type TimeRange = 30 | 60 | 100 | 180 | 365 | 1825;

export type ModelGrade = 'A' | 'A-' | 'B' | 'C' | 'D' | 'F';

export type PerformanceColor = '#10B981' | '#F59E0B' | '#FCD34D' | '#EF4444';

export type Theme = 'light' | 'dark' | 'system';

// ============================================================================
// API Status Types
// ============================================================================

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  timestamp?: string;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

// ============================================================================
// Utility Helper Types
// ============================================================================

// Make all properties optional
export type Optional<T> = {
  [P in keyof T]?: T[P];
};

// Make all properties readonly
export type Immutable<T> = {
  readonly [P in keyof T]: T[P];
};

// Extract specific keys from a type
export type PickPartial<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// Nullable type
export type Nullable<T> = T | null;

// Create a type with required keys
export type RequireKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

// ============================================================================
// Validation & Status Types
// ============================================================================

export enum LoadingState {
  IDLE = 'idle',
  LOADING = 'loading',
  SUCCESS = 'success',
  ERROR = 'error',
}

export interface AsyncState<T> {
  data: Nullable<T>;
  loading: boolean;
  error: Nullable<string>;
  status: LoadingState;
}

// ============================================================================
// Feature Flag Types (for future use)
// ============================================================================

export interface FeatureFlags {
  enableAdvancedCharts?: boolean;
  enableRealTimeUpdates?: boolean;
  enableExportData?: boolean;
  enableNotifications?: boolean;
}

// ============================================================================
// Type Guards
// ============================================================================

export const isSymbolMetrics = (obj: unknown): obj is SymbolMetrics => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'symbol' in obj &&
    'test_r2' in obj &&
    'test_rmse' in obj
  );
};

export const isPrediction = (obj: unknown): obj is Prediction => {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'date' in obj &&
    'actual' in obj
  );
};
