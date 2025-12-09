import { forwardRef, TextareaHTMLAttributes } from 'react';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
  fullWidth?: boolean;
  resize?: 'none' | 'vertical' | 'horizontal' | 'both';
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      helperText,
      fullWidth = false,
      resize = 'vertical',
      className = '',
      disabled,
      ...props
    },
    ref
  ) => {
    const resizeStyles = {
      none: 'resize-none',
      vertical: 'resize-y',
      horizontal: 'resize-x',
      both: 'resize',
    };

    return (
      <div className={`${fullWidth ? 'w-full' : ''}`}>
        {label && (
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {label}
            {props.required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}

        <textarea
          ref={ref}
          disabled={disabled}
          className={`
            px-4 py-2 rounded-lg transition-all duration-200 outline-none focus:ring-2
            bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
            focus:border-blue-500 dark:focus:border-blue-400 focus:ring-blue-500/20
            text-gray-900 dark:text-gray-100
            placeholder:text-gray-400 dark:placeholder:text-gray-500
            ${disabled ? 'opacity-60 cursor-not-allowed bg-gray-50 dark:bg-gray-900' : ''}
            ${error ? 'border-red-500 dark:border-red-400 focus:border-red-500 focus:ring-red-500/20' : ''}
            ${fullWidth ? 'w-full' : ''}
            ${resizeStyles[resize]}
            ${className}
          `}
          {...props}
        />

        {(error || helperText) && (
          <p
            className={`mt-1.5 text-sm ${
              error
                ? 'text-red-600 dark:text-red-400'
                : 'text-gray-500 dark:text-gray-400'
            }`}
          >
            {error || helperText}
          </p>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';
