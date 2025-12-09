import React from 'react';
import { Input, Select, Button, Checkbox, Textarea } from './ui';
import { useForm } from '../hooks/useForm';
import { validators } from '../utils/validation';

// Form data interface
interface ContactFormData {
  name: string;
  email: string;
  symbol: string;
  days: number;
  message: string;
  subscribe: boolean;
}

export const ExampleForm: React.FC = () => {
  const {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    resetForm,
  } = useForm<ContactFormData>({
    initialValues: {
      name: '',
      email: '',
      symbol: '',
      days: 30,
      message: '',
      subscribe: false,
    },
    validationRules: {
      name: {
        required: 'Name is required',
        minLength: { value: 2, message: 'Name must be at least 2 characters' },
        maxLength: { value: 50, message: 'Name must be at most 50 characters' },
      },
      email: {
        required: 'Email is required',
        custom: validators.email,
      },
      symbol: {
        required: 'Please select a symbol',
      },
      days: {
        required: 'Days is required',
        min: { value: 1, message: 'Must be at least 1 day' },
        max: { value: 365, message: 'Must be at most 365 days' },
      },
      message: {
        maxLength: { value: 500, message: 'Message must be at most 500 characters' },
      },
    },
    onSubmit: async (data) => {
      // Simulate API call
      console.log('Form submitted:', data);
      await new Promise((resolve) => setTimeout(resolve, 2000));
      alert('Form submitted successfully!');
      resetForm();
    },
  });

  const symbolOptions = [
    { value: '', label: 'Select a symbol', disabled: true },
    { value: 'BMO.TO', label: 'BMO.TO - Bank of Montreal' },
    { value: 'BNS.TO', label: 'BNS.TO - Bank of Nova Scotia' },
    { value: 'CM.TO', label: 'CM.TO - CIBC' },
    { value: 'RY.TO', label: 'RY.TO - Royal Bank' },
    { value: 'TD.TO', label: 'TD.TO - TD Bank' },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Example Form
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Request custom analysis or contact us for more information
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Name Input */}
        <Input
          name="name"
          label="Full Name"
          placeholder="Enter your name"
          value={values.name}
          onChange={handleChange}
          onBlur={handleBlur}
          error={touched.name ? errors.name : undefined}
          required
          fullWidth
          leftIcon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
              />
            </svg>
          }
        />

        {/* Email Input */}
        <Input
          name="email"
          type="email"
          label="Email Address"
          placeholder="you@example.com"
          value={values.email}
          onChange={handleChange}
          onBlur={handleBlur}
          error={touched.email ? errors.email : undefined}
          required
          fullWidth
          leftIcon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              />
            </svg>
          }
        />

        {/* Symbol Select */}
        <Select
          name="symbol"
          label="Trading Symbol"
          options={symbolOptions}
          value={values.symbol}
          onChange={handleChange}
          onBlur={handleBlur}
          error={touched.symbol ? errors.symbol : undefined}
          required
          fullWidth
          placeholder="Select a symbol"
        />

        {/* Days Input */}
        <Input
          name="days"
          type="number"
          label="Analysis Period (Days)"
          placeholder="30"
          value={values.days}
          onChange={handleChange}
          onBlur={handleBlur}
          error={touched.days ? errors.days : undefined}
          helperText="Number of days to analyze (1-365)"
          required
          fullWidth
          leftIcon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          }
        />

        {/* Message Textarea */}
        <Textarea
          name="message"
          label="Additional Notes"
          placeholder="Tell us more about your analysis requirements..."
          value={values.message}
          onChange={handleChange}
          onBlur={handleBlur}
          error={touched.message ? errors.message : undefined}
          helperText={`${values.message.length}/500 characters`}
          rows={4}
          fullWidth
        />

        {/* Subscribe Checkbox */}
        <Checkbox
          name="subscribe"
          label="Subscribe to newsletter"
          helperText="Receive updates about market trends and model performance"
          checked={values.subscribe}
          onChange={handleChange}
        />

        {/* Form Actions */}
        <div className="flex gap-4 pt-4">
          <Button
            type="submit"
            variant="primary"
            size="lg"
            loading={isSubmitting}
            fullWidth
          >
            Submit Request
          </Button>

          <Button
            type="button"
            variant="outline"
            size="lg"
            onClick={resetForm}
            disabled={isSubmitting}
          >
            Reset
          </Button>
        </div>
      </form>

      {/* Form Debug Info (Development Only) */}
      {process.env.NODE_ENV === 'development' && (
        <details className="mt-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300">
            Debug Info (Dev Only)
          </summary>
          <pre className="mt-2 text-xs text-gray-600 dark:text-gray-400 overflow-auto">
            {JSON.stringify({ values, errors, touched }, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
};
