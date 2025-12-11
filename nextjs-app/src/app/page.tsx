"use client";

import { useState } from "react";
import FileUpload from "@/components/FileUpload";
import Settings from "@/components/Settings";
import Results from "@/components/Results";
import ProcessingStatus from "@/components/ProcessingStatus";

export interface ClusterResult {
  keywords: string[];
  segmented: string[];
  clusters: number[];
  clusterLabels: Record<number, string>;
  embeddings2D: [number, number][];
}

export interface ProcessingStep {
  id: string;
  label: string;
  status: "pending" | "processing" | "completed" | "error";
  message?: string;
}

export default function Home() {
  const [apiKey, setApiKey] = useState("");
  const [language, setLanguage] = useState<"Vietnamese" | "English">("Vietnamese");
  const [clusteringBlocks, setClusteringBlocks] = useState(1000);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<ClusterResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);

  const updateStep = (stepId: string, updates: Partial<ProcessingStep>) => {
    setProcessingSteps((prev) =>
      prev.map((step) =>
        step.id === stepId ? { ...step, ...updates } : step
      )
    );
  };

  const handleProcess = async (keywords: string[]) => {
    if (!apiKey) {
      setError("Please enter your OpenAI API key");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults(null);

    const steps: ProcessingStep[] = [
      { id: "preprocess", label: "Preprocessing keywords", status: "pending" },
      { id: "segment", label: "Word segmentation", status: "pending" },
      { id: "embed", label: "Generating embeddings", status: "pending" },
      { id: "cluster", label: "Clustering with HDBSCAN", status: "pending" },
      { id: "label", label: "Generating cluster labels", status: "pending" },
      { id: "visualize", label: "Creating visualizations", status: "pending" },
    ];
    setProcessingSteps(steps);

    try {
      // Step 1: Preprocess
      updateStep("preprocess", { status: "processing" });

      const response = await fetch("/api/python/cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          keywords,
          api_key: apiKey,
          language,
          clustering_blocks: clusteringBlocks,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Clustering failed");
      }

      // Use SSE or polling for progress - for simplicity, we'll wait for full response
      const data = await response.json();

      // Update all steps to completed
      steps.forEach((step) => updateStep(step.id, { status: "completed" }));

      setResults({
        keywords: data.keywords,
        segmented: data.segmented,
        clusters: data.clusters,
        clusterLabels: data.cluster_labels,
        embeddings2D: data.embeddings_2d,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      setProcessingSteps((prev) =>
        prev.map((step) =>
          step.status === "processing"
            ? { ...step, status: "error" }
            : step
        )
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 dark:text-white mb-2">
            Keyword Clustering Tool
          </h1>
          <p className="text-slate-600 dark:text-slate-300">
            AI-Powered Clustering with OpenAI & GPT-4o
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <Settings
              apiKey={apiKey}
              setApiKey={setApiKey}
              language={language}
              setLanguage={setLanguage}
              clusteringBlocks={clusteringBlocks}
              setClusteringBlocks={setClusteringBlocks}
              disabled={isProcessing}
            />
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* File Upload */}
            <FileUpload
              onProcess={handleProcess}
              isProcessing={isProcessing}
              disabled={!apiKey}
            />

            {/* Processing Status */}
            {isProcessing && (
              <ProcessingStatus steps={processingSteps} />
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Results */}
            {results && <Results data={results} />}
          </div>
        </div>
      </div>
    </main>
  );
}
