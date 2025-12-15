"use client";

import { useState } from "react";
import { Settings2, ChevronDown, ChevronUp } from "lucide-react";

export interface ClusteringConfig {
  target_clusters: number;
  granularity: number;
  min_keywords_per_cluster: number;
  cluster_coherence: number;
  cluster_method: "eom" | "leaf";
  merge_clusters: boolean;
  assign_outliers: boolean;
}

// Simplified presets with clear, distinct configurations
export const CLUSTERING_PRESETS: Record<string, { name: string; hint: string; config: ClusteringConfig }> = {
  recommended: {
    name: "Balanced",
    hint: "Default",
    config: {
      target_clusters: 0,
      granularity: 5,
      min_keywords_per_cluster: 8,
      cluster_coherence: 5,
      cluster_method: "eom",
      merge_clusters: true,
      assign_outliers: true
    }
  },
  few_large: {
    name: "Broad",
    hint: "Fewer clusters",
    config: {
      target_clusters: 0,
      granularity: 2,
      min_keywords_per_cluster: 20,
      cluster_coherence: 3,
      cluster_method: "eom",
      merge_clusters: true,
      assign_outliers: true
    }
  },
  many_small: {
    name: "Granular",
    hint: "More clusters",
    config: {
      target_clusters: 0,
      granularity: 9,
      min_keywords_per_cluster: 5,
      cluster_coherence: 7,
      cluster_method: "leaf",
      merge_clusters: false,
      assign_outliers: true
    }
  },
  strict_quality: {
    name: "Strict",
    hint: "High similarity",
    config: {
      target_clusters: 0,
      granularity: 6,
      min_keywords_per_cluster: 10,
      cluster_coherence: 9,
      cluster_method: "leaf",
      merge_clusters: false,
      assign_outliers: false
    }
  }
};

interface ClusteringConfigPanelProps {
  config: ClusteringConfig;
  onChange: (config: ClusteringConfig) => void;
  disabled?: boolean;
  compact?: boolean;
}

// Clean slider component
function Slider({
  label,
  value,
  onChange,
  min,
  max,
  disabled,
  leftHint,
  rightHint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
  leftHint: string;
  rightHint: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-fg-muted">{label}</span>
        <span className="text-xs font-mono font-medium text-fg-default bg-canvas-subtle px-1.5 py-0.5 rounded">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full h-1.5 bg-[var(--color-border-default)] rounded-full appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:h-3.5
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-[var(--color-accent-emphasis)]
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:transition-transform
          [&::-webkit-slider-thumb]:hover:scale-110
          [&::-moz-range-thumb]:w-3.5
          [&::-moz-range-thumb]:h-3.5
          [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:bg-[var(--color-accent-emphasis)]
          [&::-moz-range-thumb]:border-0
          [&::-moz-range-thumb]:cursor-pointer"
      />
      <div className="flex justify-between text-[10px] text-fg-subtle">
        <span>{leftHint}</span>
        <span>{rightHint}</span>
      </div>
    </div>
  );
}

// Toggle component for binary choices
function Toggle({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-center justify-between py-1 ${disabled ? "opacity-50" : "cursor-pointer"}`}>
      <span className="text-xs text-fg-muted">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
          checked ? "bg-[var(--color-accent-emphasis)]" : "bg-[var(--color-border-default)]"
        } ${disabled ? "cursor-not-allowed" : "cursor-pointer"}`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow-sm transition-transform ${
            checked ? "translate-x-4" : "translate-x-1"
          }`}
        />
      </button>
    </label>
  );
}

// Segmented control for method selection
function MethodToggle({
  value,
  onChange,
  disabled,
}: {
  value: "eom" | "leaf";
  onChange: (value: "eom" | "leaf") => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <span className="text-xs text-fg-muted">Detection method</span>
      <div className="flex gap-1 p-0.5 bg-canvas-subtle rounded-md">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("eom")}
          className={`flex-1 px-2 py-1 text-xs font-medium rounded transition-all ${
            value === "eom"
              ? "bg-canvas-default text-fg-default shadow-sm"
              : "text-fg-muted hover:text-fg-default"
          } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
        >
          Balanced
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("leaf")}
          className={`flex-1 px-2 py-1 text-xs font-medium rounded transition-all ${
            value === "leaf"
              ? "bg-canvas-default text-fg-default shadow-sm"
              : "text-fg-muted hover:text-fg-default"
          } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
        >
          Tight
        </button>
      </div>
    </div>
  );
}

export default function ClusteringConfigPanel({
  config,
  onChange,
  disabled = false,
  compact = false,
}: ClusteringConfigPanelProps) {
  const [showTuning, setShowTuning] = useState(false);

  const handlePresetChange = (presetKey: string) => {
    const preset = CLUSTERING_PRESETS[presetKey];
    if (preset) {
      onChange(preset.config);
    }
  };

  const handleConfigChange = <K extends keyof ClusteringConfig>(key: K, value: ClusteringConfig[K]) => {
    onChange({ ...config, [key]: value });
  };

  // Check if current config matches a preset
  const getMatchingPreset = () => {
    for (const [key, preset] of Object.entries(CLUSTERING_PRESETS)) {
      const p = preset.config;
      if (
        p.granularity === config.granularity &&
        p.min_keywords_per_cluster === config.min_keywords_per_cluster &&
        p.cluster_coherence === config.cluster_coherence &&
        p.cluster_method === config.cluster_method &&
        p.merge_clusters === config.merge_clusters &&
        p.assign_outliers === config.assign_outliers
      ) {
        return key;
      }
    }
    return "custom";
  };

  const currentPreset = getMatchingPreset();

  // Unified design for both modes
  return (
    <div className="space-y-3">
      {/* Header - only show in non-compact mode */}
      {!compact && (
        <div className="flex items-center gap-2 mb-1">
          <Settings2 className="w-4 h-4 text-fg-muted" />
          <span className="text-sm font-medium text-fg-default">Clustering Mode</span>
          {currentPreset === "custom" && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-canvas-subtle text-fg-muted">Custom</span>
          )}
        </div>
      )}

      {/* Preset buttons - pill style */}
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(CLUSTERING_PRESETS).map(([key, preset]) => (
          <button
            key={key}
            onClick={() => handlePresetChange(key)}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
              currentPreset === key
                ? "bg-[var(--color-accent-emphasis)] text-white shadow-sm"
                : "bg-canvas-subtle text-fg-muted hover:bg-canvas-inset hover:text-fg-default"
            } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Tuning toggle */}
      <button
        onClick={() => setShowTuning(!showTuning)}
        disabled={disabled}
        className="flex items-center gap-1.5 text-xs text-fg-muted hover:text-fg-default transition-colors disabled:opacity-50"
      >
        {showTuning ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        <span>{showTuning ? "Hide options" : "Fine-tune"}</span>
      </button>

      {/* Advanced options */}
      {showTuning && (
        <div className={`space-y-4 pt-3 border-t border-default animate-fade-in ${compact ? "" : "pb-1"}`}>
          {/* Sliders */}
          <Slider
            label="Granularity"
            value={config.granularity}
            onChange={(v) => handleConfigChange("granularity", v)}
            min={1}
            max={10}
            disabled={disabled}
            leftHint="Broad"
            rightHint="Fine"
          />
          <Slider
            label="Min cluster size"
            value={config.min_keywords_per_cluster}
            onChange={(v) => handleConfigChange("min_keywords_per_cluster", v)}
            min={3}
            max={30}
            disabled={disabled}
            leftHint="3"
            rightHint="30"
          />
          <Slider
            label="Coherence"
            value={config.cluster_coherence}
            onChange={(v) => handleConfigChange("cluster_coherence", v)}
            min={1}
            max={10}
            disabled={disabled}
            leftHint="Loose"
            rightHint="Strict"
          />

          {/* Method toggle */}
          <MethodToggle
            value={config.cluster_method}
            onChange={(v) => handleConfigChange("cluster_method", v)}
            disabled={disabled}
          />

          {/* Boolean toggles */}
          <div className="space-y-1 pt-2 border-t border-default">
            <Toggle
              label="Merge similar clusters"
              checked={config.merge_clusters}
              onChange={(v) => handleConfigChange("merge_clusters", v)}
              disabled={disabled}
            />
            <Toggle
              label="Assign outliers to nearest"
              checked={config.assign_outliers}
              onChange={(v) => handleConfigChange("assign_outliers", v)}
              disabled={disabled}
            />
          </div>
        </div>
      )}
    </div>
  );
}
