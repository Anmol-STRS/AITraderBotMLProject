import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig, AxiosResponse } from 'axios';
import type {
  Summary,
  SymbolMetrics,
  Prediction,
  TrainingMetrics,
  LivePrediction,
  MultiModelPredictionResponse,
  ModelPredictionSeries,
  AgentName,
} from '../types';

// Extend AxiosRequestConfig to include metadata
interface ExtendedAxiosRequestConfig extends InternalAxiosRequestConfig {
  metadata?: {
    startTime: number;
  };
  _retry?: number;
}

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;

// Toggle between real and mock data (set to true for fake data)
const USE_MOCK_DATA = true;
const MOCK_PREFIX = USE_MOCK_DATA ? '/mock' : '';

// Mock data generators (client-side)
const SYMBOLS = ['BMO.TO', 'BNS.TO', 'CM.TO', 'CNQ.TO', 'ENB.TO', 'RY.TO', 'SHOP.TO', 'SU.TO', 'TD.TO', 'TRP.TO'];

// Performance grades for different symbols (XGBoost/agentVD performance)
const SYMBOL_GRADES: Record<string, { r2_min: number; r2_max: number; grade: string }> = {
  'RY.TO': { r2_min: 0.85, r2_max: 0.92, grade: 'A' },      // Grade A
  'TD.TO': { r2_min: 0.82, r2_max: 0.88, grade: 'A-' },     // Grade A-
  'BMO.TO': { r2_min: 0.72, r2_max: 0.78, grade: 'B' },     // Grade B
  'BNS.TO': { r2_min: 0.68, r2_max: 0.74, grade: 'B' },     // Grade B
  'CM.TO': { r2_min: 0.58, r2_max: 0.64, grade: 'D' },      // Grade C
  'ENB.TO': { r2_min: 0.52, r2_max: 0.58, grade: 'D' },     // Grade C
  'CNQ.TO': { r2_min: 0.15, r2_max: 0.25, grade: 'C' },     // Grade D
  'SU.TO': { r2_min: 0.08, r2_max: 0.15, grade: 'C' },      // Grade D
  'TRP.TO': { r2_min: -0.05, r2_max: 0.05, grade: 'F' },    // Grade F
  'SHOP.TO': { r2_min: -0.15, r2_max: -0.05, grade: 'F' },  // Grade F
};

const generateMockSymbolMetrics = (symbol: string) => {
  const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const rng = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  // Get performance grade for this symbol
  const gradeInfo = SYMBOL_GRADES[symbol] || { r2_min: 0.0, r2_max: 0.3, grade: 'D' };
  const r2 = gradeInfo.r2_min + rng(hash + 1) * (gradeInfo.r2_max - gradeInfo.r2_min);

  // Better models have lower RMSE
  const baseRmse = r2 > 0.7 ? 0.008 : r2 > 0.5 ? 0.010 : r2 > 0.2 ? 0.012 : 0.015;
  const rmseVariance = r2 > 0.7 ? 0.002 : r2 > 0.5 ? 0.003 : 0.005;

  return {
    symbol,
    test_r2: Math.round(r2 * 1000000) / 1000000,
    test_rmse: Math.round((baseRmse + rng(hash + 3) * rmseVariance) * 1000000) / 1000000,
    test_mae: Math.round((baseRmse * 0.75 + rng(hash + 4) * 0.003) * 1000000) / 1000000,
    test_mape: Math.round((100 + rng(hash + 5) * 60) * 100) / 100,
    test_direction_accuracy: Math.round((48 + rng(hash + 6) * 7) * 100) / 100,
    train_samples: Math.floor(800 + rng(hash + 7) * 200),
    test_samples: Math.floor(150 + rng(hash + 8) * 50),
    features: 38,
    top_feature: 'momentum_20'
  };
};

// Create axios instance with enhanced configuration
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor - Add request ID and timing
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const extendedConfig = config as ExtendedAxiosRequestConfig;

    // Add request timestamp for performance monitoring
    extendedConfig.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    extendedConfig.metadata = { startTime: new Date().getTime() };

    // Log request in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[API Request] ${extendedConfig.method?.toUpperCase()} ${extendedConfig.url}`);
    }

    return extendedConfig;
  },
  (error: AxiosError) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// Response interceptor - Handle errors and logging
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Calculate request duration
    const config = response.config as ExtendedAxiosRequestConfig;
    if (config.metadata?.startTime) {
      const duration = new Date().getTime() - config.metadata.startTime;

      if (process.env.NODE_ENV === 'development') {
        console.log(
          `[API Response] ${config.method?.toUpperCase()} ${config.url} - ${response.status} (${duration}ms)`
        );
      }
    }

    return response;
  },
  async (error: AxiosError) => {
    const config = error.config as ExtendedAxiosRequestConfig;

    // Log error details
    if (error.response) {
      // Server responded with error status
      console.error(
        `[API Error] ${error.response.status} - ${error.config?.method?.toUpperCase()} ${error.config?.url}`,
        error.response.data
      );
    } else if (error.request) {
      // Request made but no response received
      console.error('[API Error] No response received', error.message);
    } else {
      // Error in request setup
      console.error('[API Error] Request setup failed', error.message);
    }

    // Retry logic for network errors and 5xx errors
    if (
      config &&
      (!config._retry || config._retry < MAX_RETRIES) &&
      (error.code === 'ECONNABORTED' ||
        error.code === 'ERR_NETWORK' ||
        (error.response && error.response.status >= 500))
    ) {
      config._retry = (config._retry || 0) + 1;

      // Exponential backoff
      const delay = Math.min(1000 * Math.pow(2, config._retry - 1), 10000);

      console.log(`[API] Retrying request (${config._retry}/${MAX_RETRIES}) after ${delay}ms...`);

      await new Promise(resolve => setTimeout(resolve, delay));

      return api(config);
    }

    return Promise.reject(error);
  }
);

// API Health Check
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    await api.get('/health', { timeout: 5000 });
    return true;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};

const normalizeAgentName = (
  model: Pick<ModelPredictionSeries, 'model_name' | 'model_type'>
): AgentName | null => {
  const source = `${model.model_name || ''} ${model.model_type || ''}`.toLowerCase();
  if (!source.trim()) {
    return null;
  }

  const patternMap: Record<AgentName, string[]> = {
    agentVD: ['agentvd', 'xgboost', 'xgb', 'baseline'],
    gpt: ['gpt', 'openai', 'chatgpt'],
    claude: ['claude', 'anthropic'],
    deepseek: ['deepseek', 'reasoner'],
  };

  for (const [agent, patterns] of Object.entries(patternMap) as [AgentName, string[]][]) {
    if (patterns.some(pattern => source.includes(pattern))) {
      return agent;
    }
  }

  return null;
};

const extractDate = (timestamp?: string | null): string | null => {
  if (!timestamp) {
    return null;
  }
  if (timestamp.includes('T')) {
    const [datePart] = timestamp.split('T');
    return datePart || null;
  }
  if (timestamp.includes(' ')) {
    const [datePart] = timestamp.split(' ');
    return datePart || null;
  }
  return timestamp;
};

const normalizeMultiModelPredictions = (models: ModelPredictionSeries[]): Prediction[] => {
  const timeline = new Map<string, Prediction>();

  if (!Array.isArray(models)) {
    return [];
  }

  models.forEach(model => {
    const agentKey = normalizeAgentName(model);
    if (!agentKey) {
      return;
    }

    model.predictions.forEach(pred => {
      const date = extractDate(pred.timestamp);
      if (!date) {
        return;
      }

      if (!timeline.has(date)) {
        timeline.set(date, {
          date,
          actual: typeof pred.actual === 'number' ? pred.actual : null,
        });
      }

      const point = timeline.get(date)!;

      if (point.actual == null && typeof pred.actual === 'number') {
        point.actual = pred.actual;
      }

      point[agentKey] = typeof pred.predicted === 'number' ? pred.predicted : null;
    });
  });

  return Array.from(timeline.entries())
    .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
    .map(([, point]) => point);
};

// Summary statistics
export const fetchSummary = async (): Promise<Summary> => {
  if (USE_MOCK_DATA) {
    const rows = SYMBOLS.map(generateMockSymbolMetrics);

    const avg = (values: number[]) =>
      values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;

    const round = (n: number, decimals = 3) => {
      const f = Math.pow(10, decimals);
      return Math.round(n * f) / f;
    };

    return Promise.resolve({
      total_models: rows.length,
      symbols_count: rows.length,
      total_records: rows.length, // mock: keep consistent, don't invent 12550
      avg_r2: round(avg(rows.map(r => r.test_r2)), 3),
      avg_rmse: round(avg(rows.map(r => r.test_rmse)), 3),
      avg_mae: round(avg(rows.map(r => r.test_mae)), 3),
      avg_direction: round(avg(rows.map(r => r.test_direction_accuracy)), 1),
      avg_mape: round(avg(rows.map(r => r.test_mape)), 2),
      data_source: 'mock_data' as const,
      date_range: {
        start: '2020-12-07',
        end: '2025-12-05',
      },
    });
  }

  const response = await api.get<Summary>(`${MOCK_PREFIX}/summary`);
  return response.data;
};


// All symbols with metrics
export const fetchSymbols = async (): Promise<SymbolMetrics[]> => {
  if (USE_MOCK_DATA) {
    // Return mock data directly
    return Promise.resolve(SYMBOLS.map(generateMockSymbolMetrics));
  }
  const response = await api.get<SymbolMetrics[]>(`${MOCK_PREFIX}/symbols`);
  return response.data;
};

// Predictions for a symbol
export const fetchPredictions = async (
  symbol: string,
  days: number = 100
): Promise<Prediction[]> => {
  if (USE_MOCK_DATA) {
    // Generate mock predictions
    const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const rng = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    const basePrice = 50 + rng(hash) * 150;
    const predictions: Prediction[] = [];

    // Generate predictions for the requested number of days
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      const dateStr = date.toISOString().split('T')[0];

      const actual = basePrice + rng(hash + i) * 10 - 5;

      // XGBoost/agentVD: Best performance (±2 error range)
      const predicted = actual + rng(hash + i + 1000) * 4 - 2;

      // GPT: 2x worse than XGBoost (±4 error range)
      const gptPredicted = actual + rng(hash + i + 2000) * 8 - 4;

      // Claude: ~1.75x worse than XGBoost (±3.5 error range)
      const claudePredicted = actual + rng(hash + i + 3000) * 7 - 3.5;

      // DeepSeek: ~2.25x worse than XGBoost (±4.5 error range)
      const deepseekPredicted = actual + rng(hash + i + 4000) * 9 - 4.5;

      predictions.push({
        date: dateStr,
        actual: Math.round(actual * 100) / 100,
        agentVD: Math.round(predicted * 100) / 100,
        gpt: Math.round(gptPredicted * 100) / 100,
        claude: Math.round(claudePredicted * 100) / 100,
        deepseek: Math.round(deepseekPredicted * 100) / 100,
      });
    }

    return Promise.resolve(predictions);
  }

  const response = await api.get<MultiModelPredictionResponse>(`${MOCK_PREFIX}/predictions/compare`, {
    params: { symbol, limit: days },
  });

  if (!response.data || !response.data.models) {
    console.warn('No models data in prediction response:', {
      hasData: !!response.data,
      hasModels: !!(response.data && response.data.models),
      symbol,
      days
    });
    return [];
  }

  const normalized = normalizeMultiModelPredictions(response.data.models);

  if (!normalized.length) {
    console.warn('No predictions after normalization - no recognized agents found');
    return [];
  }

  return normalized;
};

// Training metrics for a symbol
export const fetchTrainingMetrics = async (symbol: string): Promise<TrainingMetrics> => {
  if (USE_MOCK_DATA) {
    const metrics = generateMockSymbolMetrics(symbol);
    return Promise.resolve({
      ...metrics,
      top_features: []
    });
  }
  const response = await api.get<TrainingMetrics>(`${MOCK_PREFIX}/metrics/${symbol}`);
  return response.data;
};

// Live predictions
export const fetchLivePredictions = async (symbol: string): Promise<LivePrediction[]> => {
  if (USE_MOCK_DATA) {
    const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const rng = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    const basePrice = 50 + rng(hash) * 150;
    const agents = ['agentVD', 'gpt', 'claude', 'deepseek'];

    return Promise.resolve(agents.map((agent, i) => {
      const currentPrice = basePrice + rng(hash + i) * 5 - 2.5;
      const predictedPrice = currentPrice + rng(hash + i + 100) * 6 - 3;
      const change = ((predictedPrice - currentPrice) / currentPrice) * 100;

      return {
        agent,
        symbol,
        timestamp: new Date().toISOString(),
        current_price: Math.round(currentPrice * 100) / 100,
        predicted_price: Math.round(predictedPrice * 100) / 100,
        predicted: Math.round(predictedPrice * 100) / 100,
        actual: Math.round(currentPrice * 100) / 100,
        change_pct: Math.round(change * 100) / 100,
        change: Math.round(change * 100) / 100,
        confidence: 0.6 + rng(hash + i + 200) * 0.3,
        action: ['BUY', 'HOLD', 'SELL'][Math.floor(rng(hash + i + 300) * 3)] as any,
        signal: ['bullish', 'neutral', 'bearish'][Math.floor(rng(hash + i + 400) * 3)] as any,
        date: new Date().toISOString()
      };
    }));
  }

  const response = await api.get<any[]>(`${MOCK_PREFIX}/live/${symbol}`);

  // Map the backend response to LivePrediction format
  return response.data.map((pred: any) => {
    // Extract predicted price from various sources
    let predictedPrice = pred.predicted_price || pred.predicted;

    // For LLM agents, try to get from decision.entry_price or decision.exit_price
    if (!predictedPrice && pred.decision) {
      predictedPrice = pred.decision.entry_price || pred.decision.exit_price;
    }

    // For LLM agents, also try from analysis
    if (!predictedPrice && pred.analysis) {
      predictedPrice = pred.analysis.entry_price || pred.analysis.exit_price;
    }

    // Calculate change percentage if we have both prices
    let changePercentage = pred.change_pct || pred.change;
    if (!changePercentage && predictedPrice && pred.current_price) {
      changePercentage = ((predictedPrice - pred.current_price) / pred.current_price) * 100;
    }

    return {
      date: pred.timestamp || new Date().toISOString(),
      actual: pred.current_price,
      predicted: predictedPrice,
      confidence: pred.confidence || pred.decision?.confidence,
      change: changePercentage,
      action: pred.action || pred.decision?.action,
      signal: pred.signal || pred.analysis?.signal || pred.analysis?.sentiment,
      agent: pred.agent,
      symbol: pred.symbol,
      timestamp: pred.timestamp,
      current_price: pred.current_price,
      predicted_price: predictedPrice,
      change_pct: changePercentage,
    };
  });
};

export default api;
