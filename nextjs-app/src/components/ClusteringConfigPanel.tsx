"use client";

import { useState } from "react";
import { Settings2, Layers, Target, Sparkles, ChevronDown, ChevronUp, HelpCircle, Grid3X3, LayoutGrid, Grid2X2 } from "lucide-react";

export interface ClusteringConfig {
  target_clusters: number;
  granularity: number;
  min_keywords_per_cluster: number;
  cluster_coherence: number;
}

// Presets matching backend - now with very different settings for visible impact
export const CLUSTERING_PRESETS: Record<string, { name: string; description: string; icon: React.ReactNode; config: ClusteringConfig }> = {
  recommended: {
    name: "Recommended",
    description: "Balanced settings for most keyword sets",
    icon: <Grid2X2 className="w-5 h-5" />,
    config: { target_clusters: 0, granularity: 5, min_keywords_per_cluster: 8, cluster_coherence: 5 }
  },
  few_large: {
    name: "Few Large",
    description: "Fewer, broader clusters",
    icon: <LayoutGrid className="w-5 h-5" />,
    config: { target_clusters: 0, granularity: 2, min_keywords_per_cluster: 20, cluster_coherence: 3 }
  },
  many_small: {
    name: "Many Small",
    description: "More granular clusters",
    icon: <Grid3X3 className="w-5 h-5" />,
    config: { target_clusters: 0, granularity: 9, min_keywords_per_cluster: 5, cluster_coherence: 7 }
  },
  strict_quality: {
    name: "Strict Quality",
    description: "Only very similar keywords",
    icon: <Target className="w-5 h-5" />,
    config: { target_clusters: 0, granularity: 6, min_keywords_per_cluster: 10, cluster_coherence: 9 }
  }
};

interface ClusteringConfigPanelProps {
  config: ClusteringConfig;
  onChange: (config: ClusteringConfig) => void;
  disabled?: boolean;
  compact?: boolean;
}

// Visual slider with gradient track
function VisualSlider({
  value,
  onChange,
  min,
  max,
  disabled,
  leftIcon,
  rightIcon,
  leftLabel,
  rightLabel,
  accentColor = "accent",
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  leftLabel: string;
  rightLabel: string;
  accentColor?: "accent" | "success" | "warning";
}) {
  const percentage = ((value - min) / (max - min)) * 100;

  const colorVar = accentColor === "accent"
    ? "var(--color-accent-emphasis)"
    : accentColor === "success"
    ? "var(--color-success-emphasis)"
    : "var(--color-warning-emphasis)";

  // Dynamic class for thumb color based on accent
  const thumbBgClass = accentColor === "accent"
    ? "[&::-webkit-slider-thumb]:bg-[var(--color-accent-emphasis)] [&::-moz-range-thumb]:bg-[var(--color-accent-emphasis)]"
    : accentColor === "success"
    ? "[&::-webkit-slider-thumb]:bg-[var(--color-success-emphasis)] [&::-moz-range-thumb]:bg-[var(--color-success-emphasis)]"
    : "[&::-webkit-slider-thumb]:bg-[var(--color-warning-emphasis)] [&::-moz-range-thumb]:bg-[var(--color-warning-emphasis)]";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-fg-muted">
        <span className="flex items-center gap-1.5">
          {leftIcon}
          {leftLabel}
        </span>
        <span className="flex items-center gap-1.5">
          {rightLabel}
          {rightIcon}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className={`w-full h-2 bg-canvas-subtle rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-5
            [&::-webkit-slider-thumb]:h-5
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-all
            [&::-webkit-slider-thumb]:hover:scale-110
            [&::-webkit-slider-thumb]:shadow-lg
            [&::-webkit-slider-thumb]:border-2
            [&::-webkit-slider-thumb]:border-white
            [&::-moz-range-thumb]:w-5
            [&::-moz-range-thumb]:h-5
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:border-2
            [&::-moz-range-thumb]:border-white
            ${thumbBgClass}`}
          style={{
            background: `linear-gradient(to right, ${colorVar} 0%, ${colorVar} ${percentage}%, var(--color-canvas-subtle) ${percentage}%, var(--color-canvas-subtle) 100%)`,
          }}
        />
        {/* Value badge */}
        <div
          className="absolute -top-1 transform -translate-x-1/2 pointer-events-none"
          style={{ left: `${percentage}%` }}
        >
          <span
            className="inline-block px-1.5 py-0.5 text-[10px] font-bold text-white rounded shadow-sm"
            style={{ backgroundColor: colorVar }}
          >
            {value}
          </span>
        </div>
      </div>
    </div>
  );
}

// Tooltip component
function Tooltip({ text }: { text: string }) {
  return (
    <div className="group relative inline-block">
      <HelpCircle className="w-3.5 h-3.5 text-fg-muted cursor-help" />
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-[var(--color-neutral-emphasis)] text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 max-w-[200px] text-center whitespace-normal">
        {text}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[var(--color-neutral-emphasis)]" />
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

  const handlePresetChange = (presetKey: string) => {
    const preset = CLUSTERING_PRESETS[presetKey];
    if (preset) {
      onChange(preset.config);
    }
  };

  const handleConfigChange = (key: keyof ClusteringConfig, value: number) => {
    onChange({ ...config, [key]: value });
  };

  // Check if current config matches a preset
  const getMatchingPreset = () => {
    for (const [key, preset] of Object.entries(CLUSTERING_PRESETS)) {
      if (
        preset.config.granularity === config.granularity &&
        preset.config.min_keywords_per_cluster === config.min_keywords_per_cluster &&
        preset.config.cluster_coherence === config.cluster_coherence
      ) {
        return key;
      }
    }
    return "custom";
  };

  const currentPreset = getMatchingPreset();

  // Compact mode for re-clustering in results view - side by side layout
  if (compact) {
    return (
      <div className="flex gap-6">
        {/* Left: Presets */}
        <div className="flex-shrink-0">
          <p className="text-[10px] text-fg-muted mb-2 font-medium uppercase tracking-wide">Quick Presets</p>
          <div className="grid grid-cols-2 gap-1.5 w-[200px]">
            {Object.entries(CLUSTERING_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => handlePresetChange(key)}
                disabled={disabled}
                className={`relative p-2 rounded-md border transition-all ${
                  currentPreset === key
                    ? "bg-[var(--color-accent-subtle)] border-[var(--color-accent-emphasis)]"
                    : "bg-canvas-subtle border-transparent hover:border-[var(--color-border-default)]"
                } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
              >
                <div className="flex items-center gap-1.5">
                  <span className={`${currentPreset === key ? "text-[var(--color-accent-fg)]" : "text-fg-muted"}`}>
                    {preset.icon}
                  </span>
                  <span className={`text-[11px] font-medium ${currentPreset === key ? "text-[var(--color-accent-fg)]" : "text-fg-default"}`}>
                    {preset.name}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Right: Fine-tune sliders */}
        <div className="flex-1 min-w-0">
          <p className="text-[10px] text-fg-muted mb-2 font-medium uppercase tracking-wide">Fine-tune</p>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Layers className="w-3.5 h-3.5 text-fg-muted" />
                <span className="text-[11px] font-medium text-fg-default">Granularity</span>
              </div>
              <VisualSlider
                value={config.granularity}
                onChange={(v) => handleConfigChange("granularity", v)}
                min={1}
                max={10}
                disabled={disabled}
                leftLabel="Few"
                rightLabel="Many"
                accentColor="accent"
              />
            </div>

            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Target className="w-3.5 h-3.5 text-fg-muted" />
                <span className="text-[11px] font-medium text-fg-default">Min Size</span>
              </div>
              <VisualSlider
                value={config.min_keywords_per_cluster}
                onChange={(v) => handleConfigChange("min_keywords_per_cluster", v)}
                min={3}
                max={30}
                disabled={disabled}
                leftLabel="3"
                rightLabel="30"
                accentColor="success"
              />
            </div>

            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Sparkles className="w-3.5 h-3.5 text-fg-muted" />
                <span className="text-[11px] font-medium text-fg-default">Coherence</span>
              </div>
              <VisualSlider
                value={config.cluster_coherence}
                onChange={(v) => handleConfigChange("cluster_coherence", v)}
                min={1}
                max={10}
                disabled={disabled}
                leftLabel="Loose"
                rightLabel="Strict"
                accentColor="warning"
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Full mode for initial clustering setup
  return (
    <div className="gh-box p-4 border-2 border-[var(--color-accent-emphasis)] bg-[var(--color-accent-subtle)]/30">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-md bg-[var(--color-accent-emphasis)]">
            <Settings2 className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-fg-default text-sm">Clustering Style</h2>
            <p className="text-[11px] text-fg-muted">Choose how keywords are grouped</p>
          </div>
        </div>
        {currentPreset !== "custom" && (
          <span className="gh-label gh-label-accent text-[10px] font-medium">
            {CLUSTERING_PRESETS[currentPreset]?.name}
          </span>
        )}
      </div>

      {/* Preset Cards - Prominent visual grid */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {Object.entries(CLUSTERING_PRESETS).map(([key, preset]) => (
          <button
            key={key}
            onClick={() => handlePresetChange(key)}
            disabled={disabled}
            className={`relative p-3 rounded-lg border-2 transition-all text-left ${
              currentPreset === key
                ? "bg-white dark:bg-[var(--color-canvas-default)] border-[var(--color-accent-emphasis)] shadow-lg ring-2 ring-[var(--color-accent-emphasis)]/20"
                : "bg-canvas-subtle border-transparent hover:border-[var(--color-border-default)] hover:shadow-sm"
            } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
          >
            <div className="flex items-start gap-2">
              <span className={`mt-0.5 ${currentPreset === key ? "text-[var(--color-accent-fg)]" : "text-fg-muted"}`}>
                {preset.icon}
              </span>
              <div className="flex-1 min-w-0">
                <p className={`text-xs font-semibold ${currentPreset === key ? "text-[var(--color-accent-fg)]" : "text-fg-default"}`}>
                  {preset.name}
                </p>
                <p className="text-[10px] text-fg-muted line-clamp-2 mt-0.5">
                  {preset.description}
                </p>
              </div>
            </div>
            {currentPreset === key && (
              <div className="absolute top-2 right-2 w-2.5 h-2.5 rounded-full bg-[var(--color-accent-emphasis)] shadow-sm" />
            )}
          </button>
        ))}
      </div>

      {/* Advanced Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        disabled={disabled}
        className="flex items-center justify-between w-full py-2 px-3 bg-canvas-default rounded-md text-xs text-fg-muted hover:text-fg-default transition-colors disabled:opacity-50 border border-default"
      >
        <span className="flex items-center gap-2">
          <Settings2 className="w-3.5 h-3.5" />
          Fine-tune Settings
        </span>
        {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {/* Advanced Sliders */}
      {showAdvanced && (
        <div className="mt-4 space-y-6 animate-fade-in p-3 bg-canvas-default rounded-md border border-default">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-[var(--color-accent-fg)]" />
              <span className="text-xs font-semibold text-fg-default">Cluster Granularity</span>
              <Tooltip text="How fine-grained should the clusters be? Lower = fewer large clusters, Higher = many smaller clusters." />
            </div>
            <VisualSlider
              value={config.granularity}
              onChange={(v) => handleConfigChange("granularity", v)}
              min={1}
              max={10}
              disabled={disabled}
              leftIcon={<LayoutGrid className="w-3.5 h-3.5" />}
              rightIcon={<Grid3X3 className="w-3.5 h-3.5" />}
              leftLabel="Few large clusters"
              rightLabel="Many small clusters"
              accentColor="accent"
            />
          </div>

          <div>
            <div className="flex items-center gap-2 mb-3">
              <Target className="w-4 h-4 text-[var(--color-success-fg)]" />
              <span className="text-xs font-semibold text-fg-default">Minimum Keywords per Cluster</span>
              <Tooltip text="The minimum number of keywords required to form a cluster. Smaller values allow more niche clusters." />
            </div>
            <VisualSlider
              value={config.min_keywords_per_cluster}
              onChange={(v) => handleConfigChange("min_keywords_per_cluster", v)}
              min={3}
              max={30}
              disabled={disabled}
              leftLabel="Small OK (3+)"
              rightLabel="Larger only (30+)"
              accentColor="success"
            />
          </div>

          <div>
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-4 h-4 text-[var(--color-warning-fg)]" />
              <span className="text-xs font-semibold text-fg-default">Cluster Coherence</span>
              <Tooltip text="How similar must keywords be to belong to the same cluster? Higher = stricter grouping, more outliers possible." />
            </div>
            <VisualSlider
              value={config.cluster_coherence}
              onChange={(v) => handleConfigChange("cluster_coherence", v)}
              min={1}
              max={10}
              disabled={disabled}
              leftLabel="Loose (include more)"
              rightLabel="Strict (only similar)"
              accentColor="warning"
            />
          </div>
        </div>
      )}
    </div>
  );
}
