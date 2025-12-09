/**
 * Common validation utilities and regex patterns
 */

// Email validation regex
export const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// Password validation (at least 8 chars, 1 uppercase, 1 lowercase, 1 number)
export const PASSWORD_REGEX = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/;

// Phone number validation (flexible format)
export const PHONE_REGEX = /^[\d\s\-\+\(\)]+$/;

// URL validation
export const URL_REGEX = /^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/;

// Alphanumeric validation
export const ALPHANUMERIC_REGEX = /^[a-zA-Z0-9]+$/;

// Validation helper functions
export const validators = {
  email: (value: string): string | undefined => {
    if (!EMAIL_REGEX.test(value)) {
      return 'Please enter a valid email address';
    }
    return undefined;
  },

  password: (value: string): string | undefined => {
    if (!PASSWORD_REGEX.test(value)) {
      return 'Password must be at least 8 characters with uppercase, lowercase, and number';
    }
    return undefined;
  },

  phone: (value: string): string | undefined => {
    if (!PHONE_REGEX.test(value)) {
      return 'Please enter a valid phone number';
    }
    return undefined;
  },

  url: (value: string): string | undefined => {
    if (!URL_REGEX.test(value)) {
      return 'Please enter a valid URL';
    }
    return undefined;
  },

  alphanumeric: (value: string): string | undefined => {
    if (!ALPHANUMERIC_REGEX.test(value)) {
      return 'Only letters and numbers are allowed';
    }
    return undefined;
  },

  minValue: (min: number) => (value: number): string | undefined => {
    if (value < min) {
      return `Value must be at least ${min}`;
    }
    return undefined;
  },

  maxValue: (max: number) => (value: number): string | undefined => {
    if (value > max) {
      return `Value must be at most ${max}`;
    }
    return undefined;
  },

  minLength: (min: number) => (value: string): string | undefined => {
    if (value.length < min) {
      return `Must be at least ${min} characters`;
    }
    return undefined;
  },

  maxLength: (max: number) => (value: string): string | undefined => {
    if (value.length > max) {
      return `Must be at most ${max} characters`;
    }
    return undefined;
  },

  match: (otherValue: string, fieldName: string) => (value: string): string | undefined => {
    if (value !== otherValue) {
      return `Must match ${fieldName}`;
    }
    return undefined;
  },
};
