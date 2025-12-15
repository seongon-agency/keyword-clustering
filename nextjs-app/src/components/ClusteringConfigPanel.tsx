"use client";

import { useState } from "react";
import { Settings2, ChevronDown, ChevronUp, Lightbulb } from "lucide-react";

export interface ClusteringConfig {
  target_clusters: number;
  granularity: number;
  min_keywords_per_cluster: number;
  cluster_coherence: number;
  cluster_method: "eom" | "leaf";
  merge_clusters: boolean;
  assign_outliers: boolean;
}

// Presets with user-friendly descriptions
const PRESETS = [
  {
    key: "recommended",
    name: "Cân bằng",
    shortDesc: "Phù hợp cho hầu hết trường hợp",
    longDesc: "Tạo các cụm có kích thước vừa phải, cân bằng giữa số lượng cụm và độ chi tiết.",
    config: {
      target_clusters: 0,
      granularity: 5,
      min_keywords_per_cluster: 8,
      cluster_coherence: 5,
      cluster_method: "eom" as const,
      merge_clusters: true,
      assign_outliers: true
    }
  },
  {
    key: "few_large",
    name: "Tổng quan",
    shortDesc: "Ít cụm, dễ quản lý",
    longDesc: "Gộp từ khóa thành ít cụm lớn hơn. Phù hợp khi bạn muốn nhìn bức tranh tổng thể.",
    config: {
      target_clusters: 0,
      granularity: 2,
      min_keywords_per_cluster: 20,
      cluster_coherence: 3,
      cluster_method: "eom" as const,
      merge_clusters: true,
      assign_outliers: true
    }
  },
  {
    key: "many_small",
    name: "Chi tiết",
    shortDesc: "Nhiều cụm nhỏ, phân loại kỹ",
    longDesc: "Tạo nhiều cụm nhỏ để phân loại chi tiết hơn. Phù hợp khi bạn muốn phân tích sâu.",
    config: {
      target_clusters: 0,
      granularity: 9,
      min_keywords_per_cluster: 5,
      cluster_coherence: 7,
      cluster_method: "leaf" as const,
      merge_clusters: false,
      assign_outliers: true
    }
  },
  {
    key: "strict_quality",
    name: "Chất lượng",
    shortDesc: "Chỉ gộp từ thực sự giống nhau",
    longDesc: "Yêu cầu độ tương đồng cao. Từ khóa không phù hợp sẽ được giữ riêng để bạn xem xét.",
    config: {
      target_clusters: 0,
      granularity: 6,
      min_keywords_per_cluster: 10,
      cluster_coherence: 9,
      cluster_method: "leaf" as const,
      merge_clusters: false,
      assign_outliers: false
    }
  }
];

interface ClusteringConfigPanelProps {
  config: ClusteringConfig;
  onChange: (config: ClusteringConfig) => void;
  disabled?: boolean;
  compact?: boolean;
}

// Slider with dynamic explanation
function SmartSlider({
  label,
  value,
  onChange,
  min,
  max,
  disabled,
  lowLabel,
  highLabel,
  getExplanation,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
  lowLabel: string;
  highLabel: string;
  getExplanation: (value: number, min: number, max: number) => string;
}) {
  const percentage = ((value - min) / (max - min)) * 100;
  const isLow = percentage < 35;
  const isHigh = percentage > 65;

  return (
    <div className="p-4 rounded-lg border border-default bg-canvas-default space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-fg-default">{label}</span>
        <span className="text-sm font-mono text-fg-muted bg-canvas-subtle px-2 py-0.5 rounded">
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
        className="w-full h-2 bg-[var(--color-border-default)] rounded-full appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-4
          [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-[var(--color-accent-emphasis)]
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:shadow-md
          [&::-webkit-slider-thumb]:hover:scale-110
          [&::-webkit-slider-thumb]:transition-transform
          [&::-moz-range-thumb]:w-4
          [&::-moz-range-thumb]:h-4
          [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:bg-[var(--color-accent-emphasis)]
          [&::-moz-range-thumb]:border-0
          [&::-moz-range-thumb]:cursor-pointer"
      />

      <div className="flex justify-between text-xs">
        <span className={isLow ? "text-fg-default font-medium" : "text-fg-muted"}>
          {lowLabel}
        </span>
        <span className={isHigh ? "text-fg-default font-medium" : "text-fg-muted"}>
          {highLabel}
        </span>
      </div>

      {/* Dynamic explanation */}
      <div className="flex items-start gap-2 text-xs text-fg-muted bg-canvas-subtle rounded-md px-3 py-2">
        <Lightbulb className="w-3.5 h-3.5 text-warning flex-shrink-0 mt-0.5" />
        <span>{getExplanation(value, min, max)}</span>
      </div>
    </div>
  );
}

// Toggle with explanation
function SmartToggle({
  label,
  checked,
  onChange,
  disabled,
  onExplanation,
  offExplanation,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  onExplanation: string;
  offExplanation: string;
}) {
  return (
    <div className="p-4 rounded-lg border border-default bg-canvas-default space-y-3">
      <div className={`flex items-center justify-between ${disabled ? "opacity-50" : ""}`}>
        <span className="text-sm font-medium text-fg-default">{label}</span>
        <button
          type="button"
          role="switch"
          aria-checked={checked}
          disabled={disabled}
          onClick={() => onChange(!checked)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            checked ? "bg-[var(--color-accent-emphasis)]" : "bg-[var(--color-border-default)]"
          } ${disabled ? "cursor-not-allowed" : "cursor-pointer"}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform ${
              checked ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
      </div>
      <p className="text-xs text-fg-muted bg-canvas-subtle rounded-md px-3 py-2">
        {checked ? onExplanation : offExplanation}
      </p>
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

  const handlePresetChange = (preset: typeof PRESETS[0]) => {
    onChange(preset.config);
  };

  const handleConfigChange = <K extends keyof ClusteringConfig>(key: K, value: ClusteringConfig[K]) => {
    onChange({ ...config, [key]: value });
  };

  // Check if current config matches a preset
  const getMatchingPreset = () => {
    for (const preset of PRESETS) {
      const p = preset.config;
      if (
        p.granularity === config.granularity &&
        p.min_keywords_per_cluster === config.min_keywords_per_cluster &&
        p.cluster_coherence === config.cluster_coherence &&
        p.cluster_method === config.cluster_method &&
        p.merge_clusters === config.merge_clusters &&
        p.assign_outliers === config.assign_outliers
      ) {
        return preset.key;
      }
    }
    return null;
  };

  const currentPreset = getMatchingPreset();

  return (
    <div className="space-y-4">
      {/* Header */}
      {!compact && (
        <div className="flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-fg-muted" />
          <span className="text-sm font-semibold text-fg-default">Cách phân cụm</span>
        </div>
      )}

      {/* Presets - Card style */}
      <div className="grid grid-cols-2 gap-2">
        {PRESETS.map((preset) => {
          const isSelected = currentPreset === preset.key;
          return (
            <button
              key={preset.key}
              onClick={() => handlePresetChange(preset)}
              disabled={disabled}
              title={preset.longDesc}
              className={`text-left p-3 rounded-lg border-2 transition-all ${
                isSelected
                  ? "border-[var(--color-accent-emphasis)] bg-[var(--color-accent-subtle)]"
                  : "border-default bg-canvas-default hover:border-[var(--color-border-emphasis)] hover:bg-canvas-subtle"
              } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <div className="mb-1">
                <span className="font-medium text-sm text-fg-default">{preset.name}</span>
              </div>
              <p className="text-xs text-fg-muted leading-relaxed">{preset.shortDesc}</p>
            </button>
          );
        })}
      </div>

      {/* Custom indicator */}
      {currentPreset === null && (
        <div className="px-3 py-2 bg-[var(--color-attention-subtle)] border border-[var(--color-attention-muted)] rounded-lg">
          <span className="text-xs text-fg-default">Đang dùng cài đặt tùy chỉnh</span>
        </div>
      )}

      {/* Advanced toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        disabled={disabled}
        className="flex items-center gap-2 text-sm text-fg-muted hover:text-fg-default transition-colors disabled:opacity-50 w-full"
      >
        {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        <span>{showAdvanced ? "Ẩn tùy chỉnh chi tiết" : "Tùy chỉnh chi tiết"}</span>
        <div className="flex-1 h-px bg-[var(--color-border-default)]" />
      </button>

      {/* Advanced options */}
      {showAdvanced && (
        <div className="space-y-3 animate-fade-in">
          {/* Tip box */}
          <div className="bg-[var(--color-accent-subtle)] border border-[var(--color-accent-muted)] rounded-lg p-3">
            <p className="text-xs text-fg-muted leading-relaxed">
              <strong className="text-fg-default">Mẹo:</strong> Kéo thanh trượt để điều chỉnh.
              Phần giải thích bên dưới mỗi thanh sẽ cho bạn biết cài đặt này có tác dụng gì.
            </p>
          </div>

          {/* Sliders with explanations */}
          <SmartSlider
            label="Số lượng cụm"
            value={config.granularity}
            onChange={(v) => handleConfigChange("granularity", v)}
            min={1}
            max={10}
            disabled={disabled}
            lowLabel="Ít cụm"
            highLabel="Nhiều cụm"
            getExplanation={(value, min, max) => {
              const pct = ((value - min) / (max - min)) * 100;
              if (pct < 35) return "Sẽ tạo ít cụm hơn, mỗi cụm chứa nhiều từ khóa. Phù hợp khi bạn muốn nhìn tổng quan.";
              if (pct > 65) return "Sẽ tạo nhiều cụm nhỏ hơn, phân loại chi tiết hơn. Phù hợp khi bạn muốn phân tích sâu.";
              return "Mức cân bằng - tạo số lượng cụm vừa phải, phù hợp cho hầu hết trường hợp.";
            }}
          />

          <SmartSlider
            label="Kích thước cụm tối thiểu"
            value={config.min_keywords_per_cluster}
            onChange={(v) => handleConfigChange("min_keywords_per_cluster", v)}
            min={3}
            max={30}
            disabled={disabled}
            lowLabel="Cho phép cụm nhỏ"
            highLabel="Yêu cầu cụm lớn"
            getExplanation={(value) => {
              if (value <= 8) return `Cho phép tạo cụm từ ${value} từ khóa trở lên. Sẽ có nhiều cụm nhỏ, bao gồm các nhóm ngách.`;
              if (value >= 20) return `Mỗi cụm phải có ít nhất ${value} từ khóa. Các nhóm nhỏ sẽ bị gộp vào cụm lớn hơn.`;
              return `Mỗi cụm cần ít nhất ${value} từ khóa để được tạo thành. Đây là mức cân bằng tốt.`;
            }}
          />

          <SmartSlider
            label="Độ tương đồng yêu cầu"
            value={config.cluster_coherence}
            onChange={(v) => handleConfigChange("cluster_coherence", v)}
            min={1}
            max={10}
            disabled={disabled}
            lowLabel="Linh hoạt"
            highLabel="Nghiêm ngặt"
            getExplanation={(value, min, max) => {
              const pct = ((value - min) / (max - min)) * 100;
              if (pct < 35) return "Từ khóa không cần quá giống nhau để được xếp chung cụm. Kết quả sẽ đa dạng hơn.";
              if (pct > 65) return "Chỉ những từ khóa thực sự giống nhau mới được xếp chung. Các cụm sẽ đồng nhất hơn.";
              return "Mức yêu cầu vừa phải - từ khóa cần có sự liên quan rõ ràng để được xếp chung.";
            }}
          />

          {/* Toggles section */}
          <div className="space-y-3 pt-3 border-t border-default">
            <p className="text-xs font-medium text-fg-muted uppercase tracking-wide">Tùy chọn nâng cao</p>

            <SmartToggle
              label="Tự động gộp cụm giống nhau"
              checked={config.merge_clusters}
              onChange={(v) => handleConfigChange("merge_clusters", v)}
              disabled={disabled}
              onExplanation="Bật: Các cụm có nội dung tương tự sẽ được tự động gộp lại để tránh trùng lặp."
              offExplanation="Tắt: Giữ nguyên tất cả cụm, kể cả khi có những cụm giống nhau. Hữu ích khi bạn muốn xem chi tiết."
            />

            <SmartToggle
              label="Xếp từ khóa lẻ vào cụm gần nhất"
              checked={config.assign_outliers}
              onChange={(v) => handleConfigChange("assign_outliers", v)}
              disabled={disabled}
              onExplanation="Bật: Những từ khóa không hoàn toàn phù hợp sẽ được xếp vào cụm gần nhất."
              offExplanation="Tắt: Từ khóa không phù hợp sẽ được giữ riêng, giúp bạn nhận ra những từ khóa độc đáo hoặc bất thường."
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Export presets for use in other components
export const CLUSTERING_PRESETS = PRESETS.reduce((acc, preset) => {
  acc[preset.key] = { name: preset.name, hint: preset.shortDesc, config: preset.config };
  return acc;
}, {} as Record<string, { name: string; hint: string; config: ClusteringConfig }>);
