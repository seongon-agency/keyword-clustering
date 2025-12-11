"use client";

import { Key, Globe, Sliders } from "lucide-react";

interface SettingsProps {
  apiKey: string;
  setApiKey: (key: string) => void;
  language: "Vietnamese" | "English";
  setLanguage: (lang: "Vietnamese" | "English") => void;
  clusteringBlocks: number;
  setClusteringBlocks: (blocks: number) => void;
  disabled: boolean;
}

export default function Settings({
  apiKey,
  setApiKey,
  language,
  setLanguage,
  clusteringBlocks,
  setClusteringBlocks,
  disabled,
}: SettingsProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 space-y-6">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Sliders className="w-5 h-5" />
        Settings
      </h2>

      {/* API Key */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
          <Key className="w-4 h-4" />
          OpenAI API Key
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          disabled={disabled}
          placeholder="sk-..."
          className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg
                   bg-white dark:bg-slate-700 text-slate-800 dark:text-white
                   focus:ring-2 focus:ring-blue-500 focus:border-transparent
                   disabled:opacity-50 disabled:cursor-not-allowed"
        />
        {apiKey ? (
          <p className="text-sm text-green-600 dark:text-green-400">
            API key configured
          </p>
        ) : (
          <p className="text-sm text-amber-600 dark:text-amber-400">
            Required for clustering
          </p>
        )}
      </div>

      {/* Language Selection */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
          <Globe className="w-4 h-4" />
          Language
        </label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value as "Vietnamese" | "English")}
          disabled={disabled}
          className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg
                   bg-white dark:bg-slate-700 text-slate-800 dark:text-white
                   focus:ring-2 focus:ring-blue-500 focus:border-transparent
                   disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="Vietnamese">Vietnamese</option>
          <option value="English">English</option>
        </select>
      </div>

      {/* Clustering Blocks */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Clustering Blocks: {clusteringBlocks}
        </label>
        <input
          type="range"
          min={100}
          max={2000}
          step={100}
          value={clusteringBlocks}
          onChange={(e) => setClusteringBlocks(Number(e.target.value))}
          disabled={disabled}
          className="w-full h-2 bg-slate-200 dark:bg-slate-600 rounded-lg appearance-none cursor-pointer
                   disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Higher = lower memory usage
        </p>
      </div>

      {/* Cost Estimate */}
      <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-4">
        <p className="text-sm text-blue-700 dark:text-blue-300">
          <strong>Estimated cost:</strong><br />
          $0.50-$2.00 per 1000 keywords
        </p>
      </div>
    </div>
  );
}
