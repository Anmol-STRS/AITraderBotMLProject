import { useState, useCallback, ChangeEvent, FormEvent } from 'react';

// Validation rules
export interface ValidationRule<T = any> {
  required?: boolean | string;
  minLength?: { value: number; message: string };
  maxLength?: { value: number; message: string };
  min?: { value: number; message: string };
  max?: { value: number; message: string };
  pattern?: { value: RegExp; message: string };
  custom?: (value: T) => string | undefined;
}

export type ValidationRules<T> = {
  [K in keyof T]?: ValidationRule<T[K]>;
};

// Form state
export interface FormState<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
  isValid: boolean;
}

// Hook options
export interface UseFormOptions<T> {
  initialValues: T;
  validationRules?: ValidationRules<T>;
  onSubmit: (values: T) => void | Promise<void>;
  validateOnChange?: boolean;
  validateOnBlur?: boolean;
}

export function useForm<T extends Record<string, any>>({
  initialValues,
  validationRules = {},
  onSubmit,
  validateOnChange = true,
  validateOnBlur = true,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Validate a single field
  const validateField = useCallback(
    (name: keyof T, value: any): string | undefined => {
      const rules = validationRules[name];
      if (!rules) return undefined;

      // Required validation
      if (rules.required) {
        const isEmpty =
          value === undefined ||
          value === null ||
          value === '' ||
          (Array.isArray(value) && value.length === 0);

        if (isEmpty) {
          return typeof rules.required === 'string'
            ? rules.required
            : `${String(name)} is required`;
        }
      }

      // Skip other validations if value is empty and not required
      if (value === undefined || value === null || value === '') {
        return undefined;
      }

      // Min length validation
      if (rules.minLength && typeof value === 'string') {
        if (value.length < rules.minLength.value) {
          return rules.minLength.message;
        }
      }

      // Max length validation
      if (rules.maxLength && typeof value === 'string') {
        if (value.length > rules.maxLength.value) {
          return rules.maxLength.message;
        }
      }

      // Min value validation
      if (rules.min !== undefined && typeof value === 'number') {
        if (value < rules.min.value) {
          return rules.min.message;
        }
      }

      // Max value validation
      if (rules.max !== undefined && typeof value === 'number') {
        if (value > rules.max.value) {
          return rules.max.message;
        }
      }

      // Pattern validation
      if (rules.pattern && typeof value === 'string') {
        if (!rules.pattern.value.test(value)) {
          return rules.pattern.message;
        }
      }

      // Custom validation
      if (rules.custom) {
        return rules.custom(value);
      }

      return undefined;
    },
    [validationRules]
  );

  // Validate all fields
  const validateForm = useCallback((): boolean => {
    const newErrors: Partial<Record<keyof T, string>> = {};
    let isValid = true;

    Object.keys(validationRules).forEach((key) => {
      const fieldName = key as keyof T;
      const error = validateField(fieldName, values[fieldName]);
      if (error) {
        newErrors[fieldName] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  }, [values, validationRules, validateField]);

  // Handle input change
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
      const { name, value, type } = e.target;
      const fieldName = name as keyof T;

      let fieldValue: any = value;

      // Handle checkbox
      if (type === 'checkbox') {
        fieldValue = (e.target as HTMLInputElement).checked;
      }

      // Handle number input
      if (type === 'number') {
        fieldValue = value === '' ? '' : Number(value);
      }

      setValues((prev) => ({ ...prev, [fieldName]: fieldValue }));

      // Validate on change if enabled
      if (validateOnChange && touched[fieldName]) {
        const error = validateField(fieldName, fieldValue);
        setErrors((prev) => ({ ...prev, [fieldName]: error }));
      }
    },
    [touched, validateOnChange, validateField]
  );

  // Handle blur
  const handleBlur = useCallback(
    (e: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
      const fieldName = e.target.name as keyof T;

      setTouched((prev) => ({ ...prev, [fieldName]: true }));

      // Validate on blur if enabled
      if (validateOnBlur) {
        const error = validateField(fieldName, values[fieldName]);
        setErrors((prev) => ({ ...prev, [fieldName]: error }));
      }
    },
    [values, validateOnBlur, validateField]
  );

  // Handle form submission
  const handleSubmit = useCallback(
    async (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();

      // Mark all fields as touched
      const allTouched = Object.keys(values).reduce(
        (acc, key) => ({ ...acc, [key]: true }),
        {} as Partial<Record<keyof T, boolean>>
      );
      setTouched(allTouched);

      // Validate form
      const isValid = validateForm();

      if (!isValid) {
        return;
      }

      setIsSubmitting(true);

      try {
        await onSubmit(values);
      } catch (error) {
        console.error('Form submission error:', error);
      } finally {
        setIsSubmitting(false);
      }
    },
    [values, validateForm, onSubmit]
  );

  // Set a specific field value
  const setFieldValue = useCallback((name: keyof T, value: any) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  // Set a specific field error
  const setFieldError = useCallback((name: keyof T, error: string) => {
    setErrors((prev) => ({ ...prev, [name]: error }));
  }, []);

  // Reset form
  const resetForm = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  // Check if form is valid
  const isValid = Object.keys(errors).length === 0;

  return {
    values,
    errors,
    touched,
    isSubmitting,
    isValid,
    handleChange,
    handleBlur,
    handleSubmit,
    setFieldValue,
    setFieldError,
    resetForm,
    validateForm,
  };
}
