"use client";

import { useState } from "react";
import { Settings2, Sparkles, Target, Layers, ChevronDown, ChevronUp, HelpCircle } from "lucide-react";

export interface ClusteringConfig {
  cluster_strictness: number;
  min_keywords_per_cluster: number;
  quality_vs_coverage: number;
}

// Presets matching backend
export const CLUSTERING_PRESETS: Record<string, { name: string; description: string; config: ClusteringConfig }> = {
  recommended: {
    name: "Recommended",
    description: "Balanced settings that work well for most keyword sets",
    config: { cluster_strictness: 5, min_keywords_per_cluster: 10, quality_vs_coverage: 5 }
  },
  strict_quality: {
    name: "Strict Quality",
    description: "Only group very similar keywords together",
    config: { cluster_strictness: 8, min_keywords_per_cluster: 15, quality_vs_coverage: 8 }
  },
  more_clusters: {
    name: "More Clusters",
    description: "Create more granular clusters with smaller groups",
    config: { cluster_strictness: 3, min_keywords_per_cluster: 5, quality_vs_coverage: 3 }
  },
  balanced: {
    name: "Balanced",
    description: "Good balance between cluster size and quality",
    config: { cluster_strictness: 5, min_keywords_per_cluster: 8, quality_vs_coverage: 5 }
  }
};

interface ClusteringConfigPanelProps {
  config: ClusteringConfig;
  onChange: (config: ClusteringConfig) => void;
  disabled?: boolean;
  compact?: boolean; // For showing in results re-cluster mode
}

// Tooltip component
function Tooltip({ text }: { text: string }) {
  return (
    <div className="group relative inline-block">
      <HelpCircle className="w-3.5 h-3.5 text-fg-muted cursor-help" />
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-[var(--color-neutral-emphasis)] text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50 max-w-[250px] whitespace-normal text-center">
        {text}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[var(--color-neutral-emphasis)]" />
      </div>
    </div>
  );
}

// Custom slider component
function Slider({
  value,
  onChange,
  min,
  max,
  disabled,
  lowLabel,
  highLabel,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
  lowLabel: string;
  highLabel: string;
}) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-1">
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className="w-full h-2 bg-canvas-subtle rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:bg-[var(--color-accent-emphasis)]
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:hover:scale-110
            [&::-webkit-slider-thumb]:shadow-md
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:bg-[var(--color-accent-emphasis)]
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:border-0"
          style={{
            background: `linear-gradient(to right, var(--color-accent-emphasis) 0%, var(--color-accent-emphasis) ${percentage}%, var(--color-canvas-subtle) ${percentage}%, var(--color-canvas-subtle) 100%)`
          }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-fg-muted">
        <span>{lowLabel}</span>
        <span className="font-medium text-fg-default">{value}</span>
        <span>{highLabel}</span>
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
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>("recommended");

  const handlePresetChange = (presetKey: string) => {
    setSelectedPreset(presetKey);
    const preset = CLUSTERING_PRESETS[presetKey];
    if (preset) {
      onChange(preset.config);
    }
  };

  const handleConfigChange = (key: keyof ClusteringConfig, value: number) => {
    setSelectedPreset("custom");
    onChange({ ...config, [key]: value });
  };

  // Check if current config matches a preset
  const getMatchingPreset = () => {
    for (const [key, preset] of Object.entries(CLUSTERING_PRESETS)) {
      if (
        preset.config.cluster_strictness === config.cluster_strictness &&
        preset.config.min_keywords_per_cluster === config.min_keywords_per_cluster &&
        preset.config.quality_vs_coverage === config.quality_vs_coverage
      ) {
        return key;
      }
    }
    return "custom";
  };

  const currentPreset = getMatchingPreset();

  if (compact) {
    // Compact mode for re-clustering
    return (
      <div className="space-y-3">
        {/* Quick Presets */}
        <div className="flex flex-wrap gap-2">
          {Object.entries(CLUSTERING_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => handlePresetChange(key)}
              disabled={disabled}
              className={`px-3 py-1.5 text-xs rounded-md border transition-all ${
                currentPreset === key
                  ? "bg-[var(--color-accent-subtle)] border-[var(--color-accent-emphasis)] text-[var(--color-accent-fg)]"
                  : "bg-canvas-subtle border-default text-fg-muted hover:border-[var(--color-accent-emphasis)]"
              } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {preset.name}
            </button>
          ))}
        </div>

        {/* Expand for sliders */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1 text-xs text-fg-muted hover:text-fg-default transition-colors"
        >
          {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          {showAdvanced ? "Hide fine-tuning" : "Fine-tune settings"}
        </button>

        {showAdvanced && (
          <div className="space-y-4 pt-2 animate-fade-in">
            <SliderControl
              icon={<Target className="w-4 h-4" />}
              label="Cluster Strictness"
              tooltip="Higher = only very similar keywords grouped together"
              value={config.cluster_strictness}
              onChange={(v) => handleConfigChange("cluster_strictness", v)}
              min={1}
              max={10}
              disabled={disabled}
              lowLabel="Loose"
              highLabel="Strict"
            />
            <SliderControl
              icon={<Layers className="w-4 h-4" />}
              label="Min Keywords per Cluster"
              tooltip="Minimum number of keywords required to form a cluster"
              value={config.min_keywords_per_cluster}
              onChange={(v) => handleConfigChange("min_keywords_per_cluster", v)}
              min={3}
              max={30}
              disabled={disabled}
              lowLabel="3"
              highLabel="30"
            />
            <SliderControl
              icon={<Sparkles className="w-4 h-4" />}
              label="Quality vs Coverage"
              tooltip="Higher = better quality but some keywords may be uncategorized"
              value={config.quality_vs_coverage}
              onChange={(v) => handleConfigChange("quality_vs_coverage", v)}
              min={1}
              max={10}
              disabled={disabled}
              lowLabel="Coverage"
              highLabel="Quality"
            />
          </div>
        )}
      </div>
    );
  }

  // Full mode for initial clustering
  return (
    <div className="gh-box p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Settings2 className="w-5 h-5 text-fg-muted" />
          <div>
            <h2 className="font-semibold text-fg-default text-sm">Clustering Settings</h2>
            <p className="text-xs text-fg-muted">Adjust how keywords are grouped</p>
          </div>
        </div>
        {currentPreset !== "custom" && (
          <span className="gh-label gh-label-accent text-[10px]">
            {CLUSTERING_PRESETS[currentPreset]?.name}
          </span>
        )}
      </div>

      {/* Preset Buttons */}
      <div className="mb-4">
        <p className="text-xs text-fg-muted mb-2">Quick Presets</p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(CLUSTERING_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => handlePresetChange(key)}
              disabled={disabled}
              className={`p-2.5 text-left rounded-md border transition-all ${
                currentPreset === key
                  ? "bg-[var(--color-accent-subtle)] border-[var(--color-accent-emphasis)]"
                  : "bg-canvas-subtle border-default hover:border-[var(--color-accent-emphasis)]"
              } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <p className={`text-xs font-medium ${currentPreset === key ? "text-[var(--color-accent-fg)]" : "text-fg-default"}`}>
                {preset.name}
              </p>
              <p className="text-[10px] text-fg-muted mt-0.5 line-clamp-1">
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        disabled={disabled}
        className="flex items-center justify-between w-full py-2 px-3 bg-canvas-subtle rounded-md text-xs text-fg-muted hover:text-fg-default transition-colors disabled:opacity-50"
      >
        <span className="flex items-center gap-2">
          <Settings2 className="w-3.5 h-3.5" />
          Fine-tune Settings
        </span>
        {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {/* Advanced Sliders */}
      {showAdvanced && (
        <div className="mt-4 space-y-5 animate-fade-in">
          <SliderControl
            icon={<Target className="w-4 h-4" />}
            label="Cluster Strictness"
            tooltip="How strict should keyword grouping be? Higher values mean only very similar keywords will be grouped together."
            value={config.cluster_strictness}
            onChange={(v) => handleConfigChange("cluster_strictness", v)}
            min={1}
            max={10}
            disabled={disabled}
            lowLabel="Loose"
            highLabel="Strict"
          />

          <SliderControl
            icon={<Layers className="w-4 h-4" />}
            label="Min Keywords per Cluster"
            tooltip="The minimum number of keywords required to form a cluster. Smaller values allow more niche clusters."
            value={config.min_keywords_per_cluster}
            onChange={(v) => handleConfigChange("min_keywords_per_cluster", v)}
            min={3}
            max={30}
            disabled={disabled}
            lowLabel="Small OK"
            highLabel="Larger only"
          />

          <SliderControl
            icon={<Sparkles className="w-4 h-4" />}
            label="Quality vs Coverage"
            tooltip="Balance between cluster quality and keyword coverage. Higher values prioritize quality but may leave some keywords uncategorized."
            value={config.quality_vs_coverage}
            onChange={(v) => handleConfigChange("quality_vs_coverage", v)}
            min={1}
            max={10}
            disabled={disabled}
            lowLabel="More coverage"
            highLabel="Higher quality"
          />
        </div>
      )}
    </div>
  );
}

// Helper component for slider with label
function SliderControl({
  icon,
  label,
  tooltip,
  value,
  onChange,
  min,
  max,
  disabled,
  lowLabel,
  highLabel,
}: {
  icon: React.ReactNode;
  label: string;
  tooltip: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
  lowLabel: string;
  highLabel: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-fg-muted">{icon}</span>
        <span className="text-xs font-medium text-fg-default">{label}</span>
        <Tooltip text={tooltip} />
      </div>
      <Slider
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        disabled={disabled}
        lowLabel={lowLabel}
        highLabel={highLabel}
      />
    </div>
  );
}
