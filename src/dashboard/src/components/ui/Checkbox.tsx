import { forwardRef, InputHTMLAttributes } from 'react';

export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  helperText?: string;
  error?: string;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ label, helperText, error, className = '', disabled, ...props }, ref) => {
    return (
      <div className="flex items-start">
        <div className="flex items-center h-5">
          <input
            ref={ref}
            type="checkbox"
            disabled={disabled}
            className={`
              w-5 h-5 rounded border-gray-300 dark:border-gray-600
              text-blue-600 focus:ring-2 focus:ring-blue-500/20
              transition-all duration-200 cursor-pointer
              disabled:opacity-60 disabled:cursor-not-allowed
              bg-white dark:bg-gray-800
              ${error ? 'border-red-500 dark:border-red-400' : ''}
              ${className}
            `}
            {...props}
          />
        </div>

        {(label || helperText || error) && (
          <div className="ml-3">
            {label && (
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 cursor-pointer">
                {label}
              </label>
            )}
            {(helperText || error) && (
              <p
                className={`text-sm ${
                  error
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-gray-500 dark:text-gray-400'
                }`}
              >
                {error || helperText}
              </p>
            )}
          </div>
        )}
      </div>
    );
  }
);

Checkbox.displayName = 'Checkbox';
