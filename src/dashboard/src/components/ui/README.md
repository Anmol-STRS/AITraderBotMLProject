# UI Components Library

Modern, accessible, and reusable form components for the AI Trading Arena application.

## Components

### Input

A flexible input component with support for icons, validation, and multiple variants.

```tsx
import { Input } from './components/ui';

<Input
  name="email"
  type="email"
  label="Email Address"
  placeholder="you@example.com"
  value={value}
  onChange={handleChange}
  error={error}
  helperText="We'll never share your email"
  required
  fullWidth
  leftIcon={<EmailIcon />}
  rightIcon={<CheckIcon />}
  variant="outlined"
/>
```

**Props:**
- `label`: Optional label text
- `error`: Error message to display
- `helperText`: Helper text below input
- `variant`: 'default' | 'filled' | 'outlined'
- `fullWidth`: Makes input full width
- `leftIcon`: Icon to display on the left
- `rightIcon`: Icon to display on the right
- All standard HTML input attributes

### Select

A styled select dropdown with custom arrow and validation.

```tsx
import { Select } from './components/ui';

<Select
  name="symbol"
  label="Trading Symbol"
  options={[
    { value: 'BMO.TO', label: 'Bank of Montreal' },
    { value: 'RY.TO', label: 'Royal Bank', disabled: true }
  ]}
  value={value}
  onChange={handleChange}
  error={error}
  placeholder="Select a symbol"
  required
  fullWidth
/>
```

**Props:**
- `options`: Array of `{ value, label, disabled? }`
- `placeholder`: Placeholder text
- All Input props except icons

### Button

A versatile button component with multiple variants, sizes, and loading state.

```tsx
import { Button } from './components/ui';

<Button
  variant="primary"
  size="lg"
  loading={isLoading}
  fullWidth
  leftIcon={<SaveIcon />}
  onClick={handleClick}
>
  Save Changes
</Button>
```

**Props:**
- `variant`: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost' | 'outline'
- `size`: 'sm' | 'md' | 'lg'
- `loading`: Shows loading spinner
- `leftIcon`: Icon on the left
- `rightIcon`: Icon on the right
- `fullWidth`: Makes button full width

### Checkbox

A styled checkbox with label and helper text.

```tsx
import { Checkbox } from './components/ui';

<Checkbox
  name="subscribe"
  label="Subscribe to newsletter"
  helperText="Get weekly updates"
  checked={value}
  onChange={handleChange}
  error={error}
/>
```

**Props:**
- `label`: Label text
- `helperText`: Helper text
- `error`: Error message
- All standard checkbox attributes

### Textarea

A multi-line text input with resize options.

```tsx
import { Textarea } from './components/ui';

<Textarea
  name="message"
  label="Message"
  placeholder="Enter your message..."
  value={value}
  onChange={handleChange}
  error={error}
  rows={4}
  resize="vertical"
  fullWidth
/>
```

**Props:**
- `resize`: 'none' | 'vertical' | 'horizontal' | 'both'
- All Input props except icons

## Form Hook

The `useForm` hook provides complete form management with validation.

```tsx
import { useForm } from '../hooks/useForm';
import { validators } from '../utils/validation';

interface FormData {
  email: string;
  password: string;
  age: number;
}

const { values, errors, touched, isSubmitting, handleChange, handleBlur, handleSubmit, resetForm } = useForm<FormData>({
  initialValues: {
    email: '',
    password: '',
    age: 0
  },
  validationRules: {
    email: {
      required: 'Email is required',
      custom: validators.email
    },
    password: {
      required: 'Password is required',
      minLength: { value: 8, message: 'At least 8 characters' }
    },
    age: {
      min: { value: 18, message: 'Must be 18+' },
      max: { value: 100, message: 'Invalid age' }
    }
  },
  onSubmit: async (data) => {
    await api.post('/submit', data);
  },
  validateOnChange: true,
  validateOnBlur: true
});
```

### Validation Rules

- `required`: boolean | string
- `minLength`: { value: number, message: string }
- `maxLength`: { value: number, message: string }
- `min`: { value: number, message: string }
- `max`: { value: number, message: string }
- `pattern`: { value: RegExp, message: string }
- `custom`: (value: any) => string | undefined

### Built-in Validators

```tsx
import { validators } from '../utils/validation';

// Email validation
validators.email(value)

// Password validation (8+ chars, uppercase, lowercase, number)
validators.password(value)

// Phone validation
validators.phone(value)

// URL validation
validators.url(value)

// Custom validators
validators.minValue(18)(value)
validators.maxLength(100)(value)
validators.match(otherValue, 'field name')(value)
```

## Features

- **Fully Typed**: Complete TypeScript support
- **Accessible**: ARIA attributes and keyboard navigation
- **Dark Mode**: Automatic dark mode support
- **Responsive**: Mobile-friendly by default
- **Validation**: Built-in validation with custom rules
- **Icons**: Support for left and right icons
- **Loading States**: Built-in loading indicators
- **Error Handling**: Automatic error display
- **Customizable**: Easy to extend and customize

## Styling

All components use Tailwind CSS and support:
- Light/dark mode
- Focus states
- Hover effects
- Disabled states
- Error states
- Smooth transitions

## Example Usage

See [ExampleForm.tsx](../ExampleForm.tsx) for a complete working example.
