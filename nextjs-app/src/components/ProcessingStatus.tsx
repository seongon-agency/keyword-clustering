"use client";

import { Check, Loader2, Circle, AlertCircle } from "lucide-react";
import type { ProcessingStep } from "@/app/page";

interface ProcessingStatusProps {
  steps: ProcessingStep[];
}

export default function ProcessingStatus({ steps }: ProcessingStatusProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-slate-800 dark:text-white mb-4">
        Processing...
      </h3>

      <div className="space-y-3">
        {steps.map((step) => (
          <div
            key={step.id}
            className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
              step.status === "processing"
                ? "bg-blue-50 dark:bg-blue-900/30"
                : step.status === "completed"
                ? "bg-green-50 dark:bg-green-900/30"
                : step.status === "error"
                ? "bg-red-50 dark:bg-red-900/30"
                : "bg-slate-50 dark:bg-slate-700/50"
            }`}
          >
            {/* Status Icon */}
            <div className="flex-shrink-0">
              {step.status === "completed" && (
                <Check className="w-5 h-5 text-green-600" />
              )}
              {step.status === "processing" && (
                <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
              )}
              {step.status === "pending" && (
                <Circle className="w-5 h-5 text-slate-400" />
              )}
              {step.status === "error" && (
                <AlertCircle className="w-5 h-5 text-red-600" />
              )}
            </div>

            {/* Label */}
            <div className="flex-1">
              <p
                className={`font-medium ${
                  step.status === "completed"
                    ? "text-green-700 dark:text-green-400"
                    : step.status === "processing"
                    ? "text-blue-700 dark:text-blue-400"
                    : step.status === "error"
                    ? "text-red-700 dark:text-red-400"
                    : "text-slate-500 dark:text-slate-400"
                }`}
              >
                {step.label}
              </p>
              {step.message && (
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                  {step.message}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
