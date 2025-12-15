"use client";

import { Check, Loader2, Circle, AlertCircle, Sparkles, Database, Brain, Tags, BarChart3 } from "lucide-react";
import { useEffect, useState } from "react";

export interface ProcessingStep {
  id: string;
  label: string;
  status: "pending" | "processing" | "completed" | "error";
  message?: string;
  icon?: React.ReactNode;
}

interface ProcessingStatusProps {
  steps: ProcessingStep[];
  startTime?: number;
}

const STEP_ICONS: Record<string, React.ReactNode> = {
  preprocess: <Sparkles className="w-3.5 h-3.5" />,
  embed: <Database className="w-3.5 h-3.5" />,
  cluster: <Brain className="w-3.5 h-3.5" />,
  label: <Tags className="w-3.5 h-3.5" />,
  visualize: <BarChart3 className="w-3.5 h-3.5" />,
};

const STEP_TECH_DETAILS: Record<string, string> = {
  preprocess: "Tách từ NLP",
  embed: "text-embedding-3-large",
  cluster: "HDBSCAN + UMAP",
  label: "GPT-4o-mini",
  visualize: "Chiếu 3D",
};

export default function ProcessingStatus({ steps, startTime }: ProcessingStatusProps) {
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (!startTime) return;

    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  const completedSteps = steps.filter((s) => s.status === "completed").length;
  const currentStep = steps.find((s) => s.status === "processing");
  const progress = (completedSteps / steps.length) * 100;

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}p ${secs}s` : `${secs}s`;
  };

  return (
    <div className="gh-box animate-fade-in h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-default">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-8 h-8 rounded-md bg-[var(--color-accent-subtle)] flex items-center justify-center">
                <Loader2 className="w-4 h-4 text-accent animate-spin" />
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-sm text-fg-default">Đang xử lý</h3>
              <p className="text-xs text-fg-muted truncate max-w-[140px]">
                {currentStep ? currentStep.label : "Đang chuẩn bị..."}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-fg-default">{Math.round(progress)}%</div>
            <div className="text-[10px] text-fg-muted font-medium">{formatTime(elapsedTime)}</div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mt-3 h-1.5 bg-[var(--color-neutral-muted)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--color-accent-emphasis)] transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Steps List */}
      <div className="flex-1 p-3 space-y-1">
        {steps.map((step, index) => {
          const isActive = step.status === "processing";
          const isCompleted = step.status === "completed";
          const isError = step.status === "error";
          const isPending = step.status === "pending";

          return (
            <div
              key={step.id}
              className={`flex items-center gap-3 px-3 py-2 rounded-md transition-all ${
                isActive
                  ? "bg-[var(--color-accent-subtle)] border border-[var(--color-accent-emphasis)]"
                  : isCompleted
                  ? "bg-[var(--color-success-subtle)]"
                  : isError
                  ? "bg-[var(--color-danger-subtle)] border border-[var(--color-danger-emphasis)]"
                  : "opacity-50"
              }`}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              {/* Step indicator */}
              <div className={`flex-shrink-0 w-7 h-7 rounded-md flex items-center justify-center ${
                isCompleted
                  ? "bg-[var(--color-success-emphasis)] text-white"
                  : isActive
                  ? "bg-[var(--color-accent-emphasis)] text-white"
                  : isError
                  ? "bg-[var(--color-danger-emphasis)] text-white"
                  : "bg-[var(--color-neutral-muted)] text-fg-subtle"
              }`}>
                {isCompleted && <Check className="w-3.5 h-3.5" />}
                {isActive && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                {isError && <AlertCircle className="w-3.5 h-3.5" />}
                {isPending && (STEP_ICONS[step.id] || <Circle className="w-3 h-3" />)}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <p className={`text-xs font-medium truncate ${
                  isCompleted
                    ? "text-success"
                    : isActive
                    ? "text-accent"
                    : isError
                    ? "text-danger"
                    : "text-fg-subtle"
                }`}>
                  {step.label}
                </p>
                {/* Tech detail label */}
                {(isActive || isCompleted) && STEP_TECH_DETAILS[step.id] && (
                  <p className="text-[10px] text-fg-muted font-mono mt-0.5">
                    {STEP_TECH_DETAILS[step.id]}
                  </p>
                )}
              </div>

              {/* Status indicator */}
              {isActive && (
                <div className="flex gap-1">
                  <span className="w-1 h-1 rounded-full bg-[var(--color-accent-fg)] animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-1 h-1 rounded-full bg-[var(--color-accent-fg)] animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-1 h-1 rounded-full bg-[var(--color-accent-fg)] animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              )}
              {isCompleted && (
                <span className="gh-label gh-label-success text-[9px]">
                  Xong
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Tech Stack Footer */}
      <div className="px-4 py-3 border-t border-default bg-canvas-subtle">
        <div className="flex items-center justify-between text-[10px] text-fg-muted">
          <span className="font-mono">OpenAI + HDBSCAN + UMAP</span>
          <span className="font-mono">vector 3,072 chiều</span>
        </div>
      </div>
    </div>
  );
}
