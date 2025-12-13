"use client";

import { useState, useRef } from "react";
import Header from "@/components/Header";
import FileUpload from "@/components/FileUpload";
import Results from "@/components/Results";
import ProcessingStatus from "@/components/ProcessingStatus";
import { AlertCircle, GitBranch, ArrowRight } from "lucide-react";

export interface ClusterResult {
  keywords: string[];
  segmented: string[];
  clusters: number[];
  clusterLabels: Record<number, string>;
  embeddings2D: [number, number][];
  embeddings3D: [number, number, number][];
}

export interface ProcessingStep {
  id: string;
  label: string;
  status: "pending" | "processing" | "completed" | "error";
  message?: string;
}

// Fixed to highest quality clustering
const CLUSTERING_BLOCKS = 100;

export default function Home() {
  const [language, setLanguage] = useState<"Vietnamese" | "English">("Vietnamese");
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<ClusterResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [startTime, setStartTime] = useState<number | undefined>(undefined);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleProcess = async (keywords: string[]) => {
    setIsProcessing(true);
    setError(null);
    setResults(null);
    setStartTime(Date.now());

    const initialSteps: ProcessingStep[] = [
      { id: "preprocess", label: "Preprocessing keywords", status: "pending" },
      { id: "embed", label: "Analyzing semantics", status: "pending" },
      { id: "cluster", label: "Grouping keywords", status: "pending" },
      { id: "label", label: "Generating labels", status: "pending" },
      { id: "visualize", label: "Creating visualization", status: "pending" },
    ];
    setProcessingSteps(initialSteps);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch("/api/cluster/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          keywords,
          language,
          clustering_blocks: CLUSTERING_BLOCKS,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Clustering failed");
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const eventData = JSON.parse(line.slice(6));

              if (eventData.type === "step") {
                setProcessingSteps((prev) =>
                  prev.map((step) =>
                    step.id === eventData.step
                      ? {
                          ...step,
                          status: eventData.status,
                          message: eventData.message
                        }
                      : step
                  )
                );
              } else if (eventData.type === "complete") {
                const result = eventData.result;
                console.log("API Result:", result);
                console.log("3D embeddings:", result.embeddings_3d?.length, result.embeddings_3d?.[0]);
                setResults({
                  keywords: result.keywords,
                  segmented: result.segmented,
                  clusters: result.clusters,
                  clusterLabels: result.cluster_labels,
                  embeddings2D: result.embeddings_2d,
                  embeddings3D: result.embeddings_3d,
                });
              } else if (eventData.type === "error") {
                throw new Error(eventData.message);
              }
            } catch (parseError) {
              if (parseError instanceof SyntaxError) continue;
              throw parseError;
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        return;
      }
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
      setStartTime(undefined);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-canvas-default">
      <Header />

      <main className="flex-1 py-6">
        <div className="container mx-auto px-4 max-w-[1600px]">
          {/* Hero - Only show when idle */}
          {!results && !isProcessing && (
            <div className="text-center mb-8 animate-fade-in">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 gh-label gh-label-accent mb-4">
                <GitBranch className="w-3 h-3" />
                AI-Powered Clustering
              </div>
              <h2 className="text-2xl md:text-3xl font-bold text-fg-default mb-2">
                Organize Keywords Intelligently
              </h2>
              <p className="text-fg-muted max-w-lg mx-auto text-sm">
                Transform keyword lists into semantic clusters for SEO and content strategy.
              </p>
            </div>
          )}

          {/* Main Layout */}
          {!results ? (
            // Input & Processing Layout
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
              {/* File Upload */}
              <div className={isProcessing ? "lg:col-span-7" : "lg:col-span-12"}>
                <div className="max-w-2xl mx-auto">
                  <FileUpload
                    onProcess={handleProcess}
                    isProcessing={isProcessing}
                    disabled={false}
                    language={language}
                    setLanguage={setLanguage}
                  />
                </div>
              </div>

              {/* Processing Status */}
              {isProcessing && (
                <div className="lg:col-span-5">
                  <ProcessingStatus steps={processingSteps} startTime={startTime} />
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="lg:col-span-12">
                  <div className="max-w-2xl mx-auto gh-box p-4 animate-fade-in" style={{ borderColor: 'var(--color-danger-emphasis)' }}>
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-danger flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-semibold text-danger">Something went wrong</p>
                        <p className="text-sm text-fg-muted mt-1">{error}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            // Results Layout
            <div className="space-y-3">
              {/* New Analysis Button */}
              <div className="flex justify-end">
                <button
                  onClick={() => setResults(null)}
                  className="gh-btn group"
                >
                  <span>New Analysis</span>
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
                </button>
              </div>
              <Results data={results} />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 border-t border-default mt-auto bg-canvas-subtle">
        <div className="container mx-auto px-4 max-w-[1600px]">
          <div className="flex items-center justify-between text-xs text-fg-muted">
            <span>Keyword Clustering Tool</span>
            <div className="flex items-center gap-3">
              <span>Semantic Analysis</span>
              <span className="w-1 h-1 rounded-full bg-[var(--color-border-default)]" />
              <span>AI Labels</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
