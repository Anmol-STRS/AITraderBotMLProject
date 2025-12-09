import React from 'react';
import type { SymbolSelectorProps } from '../types';

const getGradeColor = (r2?: number | null): string => {
  if (typeof r2 === 'number' && r2 > 0.8) return 'bg-green-500';
  if (typeof r2 === 'number' && r2 > 0.5) return 'bg-orange-500';
  if (typeof r2 === 'number' && r2 > 0) return 'bg-yellow-500';
  return 'bg-red-500';
};

const formatR2 = (r2?: number | null): string =>
  typeof r2 === 'number' && Number.isFinite(r2) ? r2.toFixed(2) : 'N/A';

export const SymbolSelector: React.FC<SymbolSelectorProps> = ({ symbols, selectedSymbol, onSelectSymbol }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
        Select Symbol
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {!Array.isArray(symbols) || symbols.length === 0 ? (
          <div className="col-span-full text-center py-8 text-gray-500 dark:text-gray-400">
            No symbols available
          </div>
        ) : (
          symbols.map((s) => (
            <button
              key={s.symbol}
              onClick={() => onSelectSymbol(s.symbol)}
              className={`relative p-4 rounded-lg border-2 transition-all ${
                selectedSymbol === s.symbol
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className={`absolute top-2 right-2 w-2 h-2 rounded-full ${getGradeColor(s.test_r2)}`}></div>
              <p className="font-semibold text-gray-900 dark:text-white">{s.symbol}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                R²: {formatR2(s.test_r2)}
              </p>
            </button>
          ))
        )}
      </div>
    </div>
  );
};

export default SymbolSelector;
