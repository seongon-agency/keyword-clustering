# UI/UX Design System Guide

A comprehensive design system inspired by GitHub's interface patterns, featuring a robust light/dark theme system, component library, and styling conventions. Use this as a reference prompt for AI agents implementing consistent UI/UX across projects.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Technology Stack](#technology-stack)
3. [Theme System](#theme-system)
4. [Color Palette](#color-palette)
5. [Typography](#typography)
6. [Component Library](#component-library)
7. [Animations & Transitions](#animations--transitions)
8. [Layout Patterns](#layout-patterns)
9. [Icons](#icons)
10. [Accessibility](#accessibility)
11. [Implementation Examples](#implementation-examples)

---

## Design Philosophy

### Core Principles

1. **GitHub-Inspired Aesthetics**: Clean, professional interface with subtle shadows, rounded corners (6px standard), and clear visual hierarchy
2. **Functional Minimalism**: Every element serves a purpose; avoid decorative clutter
3. **Theme Consistency**: Full light/dark mode support with seamless transitions
4. **Subtle Feedback**: Micro-interactions that feel responsive but not distracting
5. **Information Density**: Efficient use of space while maintaining readability

### Visual Characteristics

- **Border Radius**: 6px for cards/buttons, 4px for inner elements, 2em for pills/badges
- **Shadows**: Minimal, layered shadows that respond to theme
- **Spacing**: Consistent 4px grid system (4, 8, 12, 16, 24, 32px)
- **Transitions**: 0.15s-0.3s with ease-out or cubic-bezier curves

---

## Technology Stack

```
Framework:      Next.js 14+ (App Router)
Styling:        Tailwind CSS + CSS Custom Properties
Components:     React 18+ with TypeScript
Icons:          Lucide React
State:          React Context for theme management
```

---

## Theme System

### Implementation Architecture

```tsx
// context/ThemeContext.tsx
"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Check localStorage first, then system preference
    const stored = localStorage.getItem("theme") as Theme | null;
    if (stored) {
      setTheme(stored);
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setTheme("dark");
    }
  }, []);

  useEffect(() => {
    if (mounted) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("theme", theme);
    }
  }, [theme, mounted]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
```

### Theme Toggle Component

```tsx
import { Sun, Moon } from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="gh-btn gh-btn-sm"
      aria-label="Toggle theme"
    >
      {theme === "light" ? (
        <Moon className="w-4 h-4" />
      ) : (
        <Sun className="w-4 h-4" />
      )}
    </button>
  );
}
```

---

## Color Palette

### CSS Custom Properties

```css
/* GitHub-inspired Design System */
:root,
[data-theme="light"] {
  /* Background colors */
  --color-canvas-default: #ffffff;
  --color-canvas-subtle: #f6f8fa;
  --color-canvas-inset: #eff2f5;

  /* Foreground colors */
  --color-fg-default: #1f2328;
  --color-fg-muted: #656d76;
  --color-fg-subtle: #6e7781;

  /* Border colors */
  --color-border-default: #d0d7de;
  --color-border-muted: #d8dee4;
  --color-border-subtle: rgba(31, 35, 40, 0.15);

  /* Accent colors (Blue) */
  --color-accent-fg: #0969da;
  --color-accent-emphasis: #0969da;
  --color-accent-muted: rgba(9, 105, 218, 0.4);
  --color-accent-subtle: #ddf4ff;

  /* Success colors (Green) */
  --color-success-fg: #1a7f37;
  --color-success-emphasis: #1f883d;
  --color-success-subtle: #dafbe1;

  /* Warning colors (Yellow/Orange) */
  --color-warning-fg: #9a6700;
  --color-warning-emphasis: #bf8700;
  --color-warning-subtle: #fff8c5;

  /* Danger colors (Red) */
  --color-danger-fg: #d1242f;
  --color-danger-emphasis: #cf222e;
  --color-danger-subtle: #ffebe9;

  /* Neutral colors */
  --color-neutral-emphasis: #6e7781;
  --color-neutral-muted: rgba(175, 184, 193, 0.2);
  --color-neutral-subtle: rgba(234, 238, 242, 0.5);

  /* Button colors */
  --color-btn-bg: #f6f8fa;
  --color-btn-hover-bg: #f3f4f6;
  --color-btn-active-bg: #ebecf0;
  --color-btn-border: rgba(31, 35, 40, 0.15);
  --color-btn-primary-bg: #1f883d;
  --color-btn-primary-hover-bg: #1a7f37;

  /* Shadows */
  --color-shadow-small: 0 1px 0 rgba(31, 35, 40, 0.04);
  --color-shadow-medium: 0 3px 6px rgba(140, 149, 159, 0.15);
  --color-shadow-large: 0 8px 24px rgba(140, 149, 159, 0.2);
}

[data-theme="dark"] {
  /* Background colors */
  --color-canvas-default: #0d1117;
  --color-canvas-subtle: #161b22;
  --color-canvas-inset: #010409;

  /* Foreground colors */
  --color-fg-default: #e6edf3;
  --color-fg-muted: #8d96a0;
  --color-fg-subtle: #6e7681;

  /* Border colors */
  --color-border-default: #30363d;
  --color-border-muted: #21262d;
  --color-border-subtle: rgba(240, 246, 252, 0.1);

  /* Accent colors (Blue - brighter for dark mode) */
  --color-accent-fg: #58a6ff;
  --color-accent-emphasis: #1f6feb;
  --color-accent-muted: rgba(56, 139, 253, 0.4);
  --color-accent-subtle: rgba(56, 139, 253, 0.15);

  /* Success colors */
  --color-success-fg: #3fb950;
  --color-success-emphasis: #238636;
  --color-success-subtle: rgba(46, 160, 67, 0.15);

  /* Warning colors */
  --color-warning-fg: #d29922;
  --color-warning-emphasis: #9e6a03;
  --color-warning-subtle: rgba(187, 128, 9, 0.15);

  /* Danger colors */
  --color-danger-fg: #f85149;
  --color-danger-emphasis: #da3633;
  --color-danger-subtle: rgba(248, 81, 73, 0.15);

  /* Neutral colors */
  --color-neutral-emphasis: #6e7681;
  --color-neutral-muted: rgba(110, 118, 129, 0.4);
  --color-neutral-subtle: rgba(110, 118, 129, 0.1);

  /* Button colors */
  --color-btn-bg: #21262d;
  --color-btn-hover-bg: #30363d;
  --color-btn-active-bg: #383e47;
  --color-btn-border: rgba(240, 246, 252, 0.1);
  --color-btn-primary-bg: #238636;
  --color-btn-primary-hover-bg: #2ea043;

  /* Shadows (darker/invisible in dark mode) */
  --color-shadow-small: 0 0 transparent;
  --color-shadow-medium: 0 3px 6px #010409;
  --color-shadow-large: 0 8px 24px #010409;
}

/* Color scheme for native elements */
html {
  color-scheme: light;
}

[data-theme="dark"] {
  color-scheme: dark;
}
```

### Utility Classes

```css
/* Background utilities */
.bg-canvas-default { background: var(--color-canvas-default); }
.bg-canvas-subtle { background: var(--color-canvas-subtle); }
.bg-canvas-inset { background: var(--color-canvas-inset); }

/* Text utilities */
.text-fg-default { color: var(--color-fg-default); }
.text-fg-muted { color: var(--color-fg-muted); }
.text-fg-subtle { color: var(--color-fg-subtle); }

/* Border utilities */
.border-default { border-color: var(--color-border-default); }
.border-muted { border-color: var(--color-border-muted); }
.border-subtle { border-color: var(--color-border-subtle); }

/* Semantic color utilities */
.text-accent { color: var(--color-accent-fg); }
.text-success { color: var(--color-success-fg); }
.text-warning { color: var(--color-warning-fg); }
.text-danger { color: var(--color-danger-fg); }
```

---

## Typography

### Font Stack

```css
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans',
               Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Monospace for code/technical content */
.font-mono {
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas,
               'Liberation Mono', monospace;
}
```

### Font Sizes & Weights

| Purpose | Size | Weight | Line Height |
|---------|------|--------|-------------|
| Body text | 14px | 400 | 20px |
| Small text | 12px | 400 | 18px |
| Tiny text | 10-11px | 500 | 16px |
| Headings | 14-16px | 600 | 20-24px |
| Labels | 12px | 500 | 18px |

---

## Component Library

### Box / Card

```css
.gh-box {
  background: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
}

.gh-box-subtle {
  background: var(--color-canvas-subtle);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
}
```

### Buttons

```css
/* Base Button */
.gh-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 5px 16px;
  font-size: 14px;
  font-weight: 500;
  line-height: 20px;
  color: var(--color-fg-default);
  background-color: var(--color-btn-bg);
  border: 1px solid var(--color-btn-border);
  border-radius: 6px;
  box-shadow: var(--color-shadow-small);
  cursor: pointer;
  transition: 0.2s cubic-bezier(0.3, 0, 0.5, 1);
  transition-property: color, background-color, border-color;
}

.gh-btn:hover {
  background-color: var(--color-btn-hover-bg);
  border-color: var(--color-border-default);
}

.gh-btn:active {
  background-color: var(--color-btn-active-bg);
}

/* Primary Button (Green) */
.gh-btn-primary {
  color: #ffffff;
  background-color: var(--color-btn-primary-bg);
  border-color: rgba(31, 35, 40, 0.15);
}

.gh-btn-primary:hover {
  background-color: var(--color-btn-primary-hover-bg);
}

/* Small Button */
.gh-btn-sm {
  padding: 3px 12px;
  font-size: 12px;
  line-height: 20px;
}
```

### Labels / Badges

```css
.gh-label {
  display: inline-flex;
  align-items: center;
  padding: 0 7px;
  font-size: 12px;
  font-weight: 500;
  line-height: 18px;
  border-radius: 2em;
  white-space: nowrap;
}

.gh-label-accent {
  color: var(--color-accent-fg);
  background: var(--color-accent-subtle);
}

.gh-label-success {
  color: var(--color-success-fg);
  background: var(--color-success-subtle);
}

.gh-label-warning {
  color: var(--color-warning-fg);
  background: var(--color-warning-subtle);
}

.gh-label-danger {
  color: var(--color-danger-fg);
  background: var(--color-danger-subtle);
}

.gh-label-neutral {
  color: var(--color-fg-muted);
  background: var(--color-neutral-subtle);
}
```

### Counter Badge

```css
.gh-counter {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  padding: 0 6px;
  font-size: 12px;
  font-weight: 500;
  line-height: 18px;
  color: var(--color-fg-default);
  background: var(--color-neutral-muted);
  border-radius: 2em;
}
```

### Input Fields

```css
.gh-input {
  width: 100%;
  padding: 5px 12px;
  font-size: 14px;
  line-height: 20px;
  color: var(--color-fg-default);
  background-color: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  box-shadow: var(--color-shadow-small);
  transition: border-color 0.2s ease;
}

.gh-input:focus {
  outline: none;
  border-color: var(--color-accent-emphasis);
  box-shadow: 0 0 0 3px var(--color-accent-muted);
}

.gh-input::placeholder {
  color: var(--color-fg-subtle);
}
```

### Segmented Control

```css
.gh-segmented {
  display: inline-flex;
  background: var(--color-canvas-subtle);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  padding: 2px;
}

.gh-segmented-btn {
  padding: 4px 12px;
  font-size: 12px;
  font-weight: 500;
  color: var(--color-fg-muted);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.gh-segmented-btn:hover {
  color: var(--color-fg-default);
}

.gh-segmented-btn.active {
  color: var(--color-fg-default);
  background: var(--color-canvas-default);
  box-shadow: var(--color-shadow-small);
}
```

### Header

```css
.gh-header {
  background: var(--color-canvas-subtle);
  border-bottom: 1px solid var(--color-border-muted);
}
```

### Dropdown Menu

```css
.gh-dropdown {
  position: absolute;
  z-index: 100;
  min-width: 160px;
  padding: 4px 0;
  background: var(--color-canvas-default);
  border: 1px solid var(--color-border-default);
  border-radius: 6px;
  box-shadow: var(--color-shadow-large);
}

.gh-dropdown-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 16px;
  font-size: 14px;
  color: var(--color-fg-default);
  text-align: left;
  cursor: pointer;
  transition: background-color 0.1s ease;
}

.gh-dropdown-item:hover {
  background: var(--color-accent-subtle);
}
```

### Table

```css
.gh-table {
  width: 100%;
  font-size: 14px;
  border-collapse: collapse;
}

.gh-table th {
  padding: 8px 16px;
  font-weight: 600;
  text-align: left;
  color: var(--color-fg-default);
  background: var(--color-canvas-subtle);
  border-bottom: 1px solid var(--color-border-default);
}

.gh-table td {
  padding: 8px 16px;
  color: var(--color-fg-default);
  border-bottom: 1px solid var(--color-border-muted);
}

.gh-table tr:hover td {
  background: var(--color-canvas-subtle);
}
```

### Navigation Item

```css
.gh-nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  font-size: 14px;
  font-weight: 500;
  color: var(--color-fg-default);
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.1s ease;
}

.gh-nav-item:hover {
  background: var(--color-neutral-muted);
}

.gh-nav-item.active {
  background: var(--color-accent-subtle);
  color: var(--color-accent-fg);
}
```

### Divider

```css
.gh-divider {
  height: 1px;
  background: var(--color-border-muted);
  border: 0;
}
```

---

## Animations & Transitions

### Keyframe Animations

```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* Utility classes */
.animate-fade-in {
  animation: fadeIn 0.2s ease-out forwards;
}

.animate-fade-in-up {
  animation: fadeInUp 0.3s ease-out forwards;
}

.animate-scale-in {
  animation: scaleIn 0.15s ease-out forwards;
}
```

### Transition Standards

| Element | Duration | Easing |
|---------|----------|--------|
| Colors/backgrounds | 0.1s-0.2s | ease |
| Transforms | 0.15s-0.3s | ease-out |
| Layout changes | 0.2s-0.3s | ease-out |
| Theme transitions | 0.2s | ease |

### Loading Spinner

```tsx
// Simple CSS spinner
<div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />

// Or use Lucide's Loader2
import { Loader2 } from "lucide-react";
<Loader2 className="w-4 h-4 animate-spin" />
```

---

## Layout Patterns

### Container

```tsx
<div className="container mx-auto px-4 max-w-[1600px]">
  {/* Content */}
</div>
```

### Header Layout

```tsx
<header className="gh-header sticky top-0 z-50">
  <div className="container mx-auto px-4 max-w-[1600px]">
    <div className="flex items-center justify-between h-14">
      {/* Logo & Brand */}
      <div className="flex items-center gap-3">
        <Icon className="w-5 h-5 text-fg-default" />
        <h1 className="font-semibold text-fg-default text-base">
          App Name
        </h1>
        <span className="gh-label gh-label-accent text-[11px]">
          Beta
        </span>
      </div>

      {/* Right side actions */}
      <div className="flex items-center gap-3">
        <ThemeToggle />
      </div>
    </div>
  </div>
</header>
```

### Card with Header

```tsx
<div className="gh-box">
  {/* Header */}
  <div className="px-4 py-3 border-b border-default">
    <div className="flex items-center gap-2">
      <Icon className="w-5 h-5 text-fg-muted" />
      <div>
        <h2 className="font-semibold text-fg-default text-sm">Card Title</h2>
        <p className="text-xs text-fg-muted">Description text</p>
      </div>
    </div>
  </div>

  {/* Content */}
  <div className="p-4">
    {/* ... */}
  </div>

  {/* Optional Footer */}
  <div className="px-4 py-3 border-t border-default bg-canvas-subtle">
    {/* ... */}
  </div>
</div>
```

---

## Icons

### Recommended Library

Use **Lucide React** for consistent, well-designed icons.

```bash
npm install lucide-react
```

### Common Icon Sizes

| Context | Size |
|---------|------|
| Inline with text | w-3.5 h-3.5 |
| Button icons | w-4 h-4 |
| Card header icons | w-5 h-5 |
| Large/decorative | w-8 h-8 or larger |

### Usage Pattern

```tsx
import {
  Sun, Moon,           // Theme
  Upload, File,        // Files
  Search, Filter,      // Actions
  Check, X, AlertCircle, // Status
  ChevronDown, ChevronRight, // Navigation
  Loader2              // Loading
} from "lucide-react";

<Icon className="w-4 h-4 text-fg-muted" />
```

---

## Accessibility

### Focus States

```css
*:focus-visible {
  outline: 2px solid var(--color-accent-emphasis);
  outline-offset: 2px;
}
```

### Selection Color

```css
::selection {
  background: var(--color-accent-muted);
}
```

### Scrollbar Styling

```css
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--color-border-default);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-fg-subtle);
}
```

### ARIA Labels

Always include appropriate ARIA labels for interactive elements:

```tsx
<button
  onClick={toggleTheme}
  className="gh-btn gh-btn-sm"
  aria-label="Toggle theme"
>
  <Moon className="w-4 h-4" />
</button>
```

---

## Implementation Examples

### File Upload Dropzone

```tsx
<div
  className={`
    relative border-2 border-dashed rounded-md p-4 text-center
    cursor-pointer transition-all
    ${isDragActive
      ? "border-[var(--color-accent-emphasis)] bg-[var(--color-accent-subtle)]"
      : "border-default hover:border-[var(--color-accent-emphasis)] hover:bg-canvas-subtle"
    }
    ${isDisabled ? "opacity-50 cursor-not-allowed" : ""}
  `}
>
  <div className="flex items-center justify-center gap-3">
    <div className="w-10 h-10 rounded-md bg-canvas-subtle flex items-center justify-center">
      <Upload className="w-5 h-5 text-fg-muted" />
    </div>
    <div className="text-left">
      <p className="text-fg-default font-medium text-sm">
        {isDragActive ? "Drop here..." : "Drop file or click"}
      </p>
      <p className="text-xs text-fg-muted">.xlsx, .csv</p>
    </div>
  </div>
</div>
```

### Status Step Indicator

```tsx
<div className={`flex items-center gap-3 px-3 py-2 rounded-md transition-all ${
  isActive
    ? "bg-[var(--color-accent-subtle)] border border-[var(--color-accent-emphasis)]"
    : isCompleted
    ? "bg-[var(--color-success-subtle)]"
    : isError
    ? "bg-[var(--color-danger-subtle)] border border-[var(--color-danger-emphasis)]"
    : "opacity-50"
}`}>
  <div className={`w-7 h-7 rounded-md flex items-center justify-center ${
    isCompleted
      ? "bg-[var(--color-success-emphasis)] text-white"
      : isActive
      ? "bg-[var(--color-accent-emphasis)] text-white"
      : "bg-[var(--color-neutral-muted)] text-fg-subtle"
  }`}>
    {isCompleted && <Check className="w-3.5 h-3.5" />}
    {isActive && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
  </div>
  <span className="text-xs font-medium">{label}</span>
</div>
```

### Keyword/Tag Chips

```tsx
<div className="flex flex-wrap gap-1.5">
  {items.slice(0, 6).map((item, i) => (
    <span
      key={i}
      className="px-2 py-0.5 bg-canvas-default border border-default
                 text-fg-muted rounded text-[11px]"
    >
      {item.length > 20 ? `${item.slice(0, 20)}...` : item}
    </span>
  ))}
  {items.length > 6 && (
    <span className="gh-counter text-[11px]">
      +{items.length - 6}
    </span>
  )}
</div>
```

---

## Tailwind Configuration

```typescript
// tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
    },
  },
  plugins: [],
};

export default config;
```

---

## Quick Reference Checklist

When implementing this design system:

- [ ] Set up ThemeProvider wrapping the app
- [ ] Import globals.css with all CSS custom properties
- [ ] Use `data-theme` attribute on `<html>` for theme switching
- [ ] Apply `gh-*` classes for consistent component styling
- [ ] Use CSS variables (`var(--color-*)`) for all colors
- [ ] Include Lucide React for icons
- [ ] Add transition classes for smooth interactions
- [ ] Ensure focus-visible styles are present
- [ ] Test both light and dark modes thoroughly

---

## Summary

This design system provides a professional, GitHub-inspired interface with:

1. **Robust theming** via CSS custom properties and React Context
2. **Consistent components** using the `gh-*` class naming convention
3. **Accessible defaults** with proper focus states and ARIA support
4. **Smooth animations** for polished micro-interactions
5. **Semantic color system** for status indicators (success, warning, danger, accent)

Apply these patterns consistently across your application for a cohesive, professional user experience.
