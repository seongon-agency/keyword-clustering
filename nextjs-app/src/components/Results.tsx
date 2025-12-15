"use client";

import { useState, useMemo, useCallback, memo } from "react";
import {
  Download,
  ChevronDown,
  FileSpreadsheet,
  Table,
  FileJson,
  Search,
  ChevronLeft,
  ChevronRight,
  Copy,
  Check,
  BarChart3,
  Box,
  Layers,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import * as XLSX from "xlsx";
import type { ClusterResult } from "@/app/page";
import ClusterScatterInteractive from "./ClusterScatterInteractive";
import ClusterScatter3D from "./ClusterScatter3D";
import ClusteringConfigPanel, { ClusteringConfig } from "./ClusteringConfigPanel";

interface ResultsProps {
  data: ClusterResult;
  onNewAnalysis: () => void;
  onRecluster: (config: ClusteringConfig) => void;
  currentConfig: ClusteringConfig;
  onConfigChange: (config: ClusteringConfig) => void;
  isProcessing: boolean;
}

const COLORS = [
  "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
  "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
  "#14B8A6", "#A855F7", "#EAB308", "#22C55E", "#0EA5E9",
];

// Cluster Grid component
const ClusterGrid = memo(function ClusterGrid({
  keywords,
  clusters,
  clusterLabels,
  selectedCluster,
  onClusterSelect,
}: {
  keywords: string[];
  clusters: number[];
  clusterLabels: Record<number, string>;
  selectedCluster: number | null;
  onClusterSelect: (cluster: number | null) => void;
}) {
  const [copiedCluster, setCopiedCluster] = useState<number | null>(null);

  const clusterData = useMemo(() => {
    const data: Record<number, string[]> = {};
    clusters.forEach((c, i) => { if (!data[c]) data[c] = []; data[c].push(keywords[i]); });

    const totalKeywords = keywords.length;

    return Object.entries(data)
      .map(([cluster, kws]) => ({
        cluster: Number(cluster),
        label: clusterLabels[Number(cluster)] || `Cluster ${cluster}`,
        keywords: kws,
        count: kws.length,
        percentage: Math.round((kws.length / totalKeywords) * 100),
      }))
      .sort((a, b) => b.count - a.count);
  }, [clusters, keywords, clusterLabels]);

  const maxCount = Math.max(...clusterData.map(c => c.count));

  const copyKeywords = async (e: React.MouseEvent, cluster: number, kws: string[]) => {
    e.stopPropagation();
    await navigator.clipboard.writeText(kws.join("\n"));
    setCopiedCluster(cluster);
    setTimeout(() => setCopiedCluster(null), 2000);
  };

  return (
    <div className="space-y-1 max-h-[500px] overflow-y-auto pr-1">
      {clusterData.map(item => (
        <div
          key={item.cluster}
          onClick={() => onClusterSelect(selectedCluster === item.cluster ? null : item.cluster)}
          className={`group relative p-2.5 rounded-md cursor-pointer transition-all border ${
            selectedCluster === item.cluster
              ? "bg-[var(--color-accent-subtle)] border-[var(--color-accent-emphasis)]"
              : selectedCluster !== null
                ? "bg-canvas-subtle border-transparent opacity-50"
                : "bg-canvas-default border-default hover:bg-canvas-subtle"
          }`}
        >
          {/* Top row */}
          <div className="flex items-center gap-2 mb-1.5">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: COLORS[item.cluster % COLORS.length] }}
            />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-fg-default truncate">{item.label}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="gh-counter text-[11px]">
                {item.count}
              </span>
              <button
                onClick={(e) => copyKeywords(e, item.cluster, item.keywords)}
                className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-[var(--color-neutral-muted)] text-fg-muted hover:text-fg-default transition-all"
                title="Copy all keywords"
              >
                {copiedCluster === item.cluster ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
              </button>
            </div>
          </div>

          {/* Distribution bar */}
          <div className="h-1 bg-[var(--color-neutral-muted)] rounded-full overflow-hidden mb-1.5">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${(item.count / maxCount) * 100}%`,
                backgroundColor: COLORS[item.cluster % COLORS.length],
                opacity: selectedCluster !== null && selectedCluster !== item.cluster ? 0.3 : 1
              }}
            />
          </div>

          {/* Sample keywords */}
          <div className="flex flex-wrap gap-1">
            {item.keywords.slice(0, 3).map((kw, i) => (
              <span
                key={i}
                className="px-1.5 py-0.5 bg-[var(--color-neutral-subtle)] text-fg-muted rounded text-[10px] truncate max-w-[100px]"
                title={kw}
              >
                {kw.length > 15 ? `${kw.slice(0, 15)}...` : kw}
              </span>
            ))}
            {item.keywords.length > 3 && (
              <span className="px-1.5 py-0.5 text-fg-subtle text-[10px]">
                +{item.keywords.length - 3}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
});

// Keyword table component
const KeywordTableCompact = memo(function KeywordTableCompact({
  keywords, clusters, clusterLabels, selectedCluster, onClusterSelect,
}: {
  keywords: string[];
  clusters: number[];
  clusterLabels: Record<number, string>;
  selectedCluster: number | null;
  onClusterSelect: (cluster: number | null) => void;
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 20;

  const effectiveFilter = selectedCluster;

  const tableData = useMemo(() => {
    let data = keywords.map((kw, i) => ({ index: i, keyword: kw, cluster: clusters[i] }));
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      data = data.filter(item => item.keyword.toLowerCase().includes(query));
    }
    if (effectiveFilter !== null) {
      data = data.filter(item => item.cluster === effectiveFilter);
    }
    return data.sort((a, b) => a.cluster - b.cluster);
  }, [keywords, clusters, searchQuery, effectiveFilter]);

  const totalPages = Math.ceil(tableData.length / pageSize);
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return tableData.slice(start, start + pageSize);
  }, [tableData, currentPage]);

  useMemo(() => setCurrentPage(1), [searchQuery, effectiveFilter]);

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-fg-subtle" />
          <input
            type="text"
            placeholder="Search keywords..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="gh-input pl-8 py-1.5 text-xs"
          />
        </div>
        <div className="gh-counter">
          {tableData.length}
        </div>
      </div>
      <div className="overflow-auto rounded-md border border-default max-h-[420px] bg-canvas-default">
        <table className="gh-table text-xs">
          <thead>
            <tr>
              <th className="px-3 py-2">Keyword</th>
              <th className="px-3 py-2 w-20">Cluster</th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((item) => (
              <tr key={item.index}>
                <td className="px-3 py-1.5 truncate max-w-[200px]">{item.keyword}</td>
                <td className="px-3 py-1.5">
                  <button
                    onClick={() => onClusterSelect(selectedCluster === item.cluster ? null : item.cluster)}
                    className="gh-label text-[10px]"
                    style={{
                      backgroundColor: `${COLORS[item.cluster % COLORS.length]}20`,
                      color: COLORS[item.cluster % COLORS.length],
                    }}
                  >
                    {item.cluster}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-fg-muted">Page {currentPage} of {totalPages}</span>
          <div className="flex gap-1">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="gh-btn gh-btn-sm disabled:opacity-30"
            >
              <ChevronLeft className="w-3 h-3" />
            </button>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="gh-btn gh-btn-sm disabled:opacity-30"
            >
              <ChevronRight className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
});


export default function Results({
  data,
  onNewAnalysis,
  onRecluster,
  currentConfig,
  onConfigChange,
  isProcessing,
}: ResultsProps) {
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [viewMode, setViewMode] = useState<"2d" | "3d">("3d");

  const stats = useMemo(() => {
    const uniqueClusters = new Set(data.clusters);
    const clusterCounts: Record<number, number> = {};
    data.clusters.forEach(c => { clusterCounts[c] = (clusterCounts[c] || 0) + 1; });
    const counts = Object.values(clusterCounts);
    return {
      totalKeywords: data.keywords.length,
      numClusters: uniqueClusters.size,
      avgPerCluster: Math.round(data.keywords.length / uniqueClusters.size),
      maxCluster: Math.max(...counts),
    };
  }, [data]);

  const exportToExcel = useCallback((clusterFilter?: number) => {
    const exportData = data.keywords
      .map((kw, i) => ({ Keyword: kw, Segmented: data.segmented[i], "Cluster ID": data.clusters[i], "Cluster Label": data.clusterLabels[data.clusters[i]] || "Unknown" }))
      .filter(row => clusterFilter === undefined || row["Cluster ID"] === clusterFilter);
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.json_to_sheet(exportData);
    XLSX.utils.book_append_sheet(wb, ws, "Clusters");
    XLSX.writeFile(wb, `clusters_${new Date().toISOString().slice(0, 10)}.xlsx`);
  }, [data]);

  const exportToCSV = useCallback(() => {
    const headers = ["Keyword", "Segmented", "ClusterID", "ClusterLabel"];
    const csvContent = [headers.join(","), ...data.keywords.map((kw, i) =>
      [`"${kw.replace(/"/g, '""')}"`, `"${data.segmented[i].replace(/"/g, '""')}"`, data.clusters[i], `"${(data.clusterLabels[data.clusters[i]] || "").replace(/"/g, '""')}"`].join(",")
    )].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `clusters_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
  }, [data]);

  const exportToJSON = useCallback(() => {
    const clusterCounts: Record<number, number> = {};
    data.clusters.forEach(c => { clusterCounts[c] = (clusterCounts[c] || 0) + 1; });
    const exportData = {
      metadata: { exportedAt: new Date().toISOString(), totalKeywords: data.keywords.length, totalClusters: stats.numClusters },
      clusters: Object.entries(clusterCounts).map(([cluster, count]) => ({
        id: Number(cluster), label: data.clusterLabels[Number(cluster)] || "Unknown", count,
        keywords: data.keywords.filter((_, i) => data.clusters[i] === Number(cluster)),
      })),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `clusters_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
  }, [data, stats.numClusters]);

  return (
    <div className="space-y-2 animate-fade-in">
      {/* Top Action Bar - Simple */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-fg-muted">
          <span>Analysis complete</span>
          <span className="w-1 h-1 rounded-full bg-[var(--color-success-emphasis)]" />
          <span className="text-fg-default font-medium">{stats.numClusters} clusters found</span>
        </div>
        <button
          onClick={onNewAnalysis}
          disabled={isProcessing}
          className="gh-btn group"
        >
          <span>New Keywords</span>
          <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
        </button>
      </div>

      {/* Stats Bar */}
      <div className="gh-box px-4 py-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-fg-muted">Keywords</span>
              <span className="font-semibold text-fg-default">{stats.totalKeywords.toLocaleString()}</span>
            </div>
            <div className="w-px h-4 bg-[var(--color-border-muted)]" />
            <div className="flex items-center gap-2 text-sm">
              <span className="text-fg-muted">Clusters</span>
              <span className="font-semibold text-fg-default">{stats.numClusters}</span>
            </div>
            <div className="w-px h-4 bg-[var(--color-border-muted)]" />
            <div className="flex items-center gap-2 text-sm">
              <span className="text-fg-muted">Avg Size</span>
              <span className="font-semibold text-fg-default">{stats.avgPerCluster}</span>
            </div>
          </div>

          <div className="relative">
          <button
            onClick={() => setShowExportMenu(!showExportMenu)}
            className="gh-btn gh-btn-sm"
          >
            <Download className="w-4 h-4" />
            Export
            <ChevronDown className={`w-3 h-3 transition-transform ${showExportMenu ? "rotate-180" : ""}`} />
          </button>
          {showExportMenu && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowExportMenu(false)} />
              <div className="gh-dropdown right-0 mt-2 w-48 z-50 animate-scale-in">
                <button onClick={() => { exportToExcel(); setShowExportMenu(false); }} className="gh-dropdown-item">
                  <FileSpreadsheet className="w-4 h-4 text-success" /> Export to Excel
                </button>
                <button onClick={() => { exportToCSV(); setShowExportMenu(false); }} className="gh-dropdown-item">
                  <Table className="w-4 h-4 text-accent" /> Export to CSV
                </button>
                <button onClick={() => { exportToJSON(); setShowExportMenu(false); }} className="gh-dropdown-item">
                  <FileJson className="w-4 h-4 text-warning" /> Export to JSON
                </button>
              </div>
            </>
          )}
        </div>
        </div>

        {/* Technical Details Row */}
        <div className="flex items-center gap-4 mt-2 pt-2 border-t border-default">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-success-fg)]" />
            <span className="text-[10px] font-mono text-fg-muted">text-embedding-3-large</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent-fg)]" />
            <span className="text-[10px] font-mono text-fg-muted">3,072-dim vectors</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-warning-fg)]" />
            <span className="text-[10px] font-mono text-fg-muted">HDBSCAN clustering</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-done-fg)]" />
            <span className="text-[10px] font-mono text-fg-muted">UMAP projection</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-sponsors-fg)]" />
            <span className="text-[10px] font-mono text-fg-muted">GPT-4o-mini labels</span>
          </div>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-12 gap-2">
        {/* Cluster Map - Left side */}
        <div className="col-span-5 gh-box p-3">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {viewMode === "2d" ? <Layers className="w-4 h-4 text-fg-muted" /> : <Box className="w-4 h-4 text-fg-muted" />}
              <h3 className="text-sm font-semibold text-fg-default">Visualization</h3>
            </div>
            {/* 2D/3D Toggle */}
            <div className="gh-segmented">
              <button
                onClick={() => setViewMode("2d")}
                className={`gh-segmented-btn ${viewMode === "2d" ? "active" : ""}`}
              >
                2D
              </button>
              <button
                onClick={() => setViewMode("3d")}
                className={`gh-segmented-btn ${viewMode === "3d" ? "active" : ""}`}
              >
                3D
              </button>
            </div>
          </div>
          <div>
            {viewMode === "2d" ? (
              <ClusterScatterInteractive
                embeddings2D={data.embeddings2D}
                clusters={data.clusters}
                keywords={data.keywords}
                clusterLabels={data.clusterLabels}
                selectedCluster={selectedCluster}
                onClusterSelect={setSelectedCluster}
              />
            ) : (
              <ClusterScatter3D
                embeddings3D={data.embeddings3D}
                clusters={data.clusters}
                keywords={data.keywords}
                clusterLabels={data.clusterLabels}
                selectedCluster={selectedCluster}
                onClusterSelect={setSelectedCluster}
              />
            )}
          </div>
        </div>

        {/* Right side - Clusters Overview + Keywords Explorer */}
        <div className="col-span-7 grid grid-cols-2 gap-2">
          {/* Clusters Overview */}
          <div className="gh-box p-3">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-fg-muted" />
                <h3 className="text-sm font-semibold text-fg-default">Clusters</h3>
              </div>
              <span className="gh-counter">{stats.numClusters}</span>
            </div>
            <ClusterGrid
              keywords={data.keywords}
              clusters={data.clusters}
              clusterLabels={data.clusterLabels}
              selectedCluster={selectedCluster}
              onClusterSelect={setSelectedCluster}
            />
          </div>

          {/* Keywords Explorer */}
          <div className="gh-box p-3">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Table className="w-4 h-4 text-fg-muted" />
                <h3 className="text-sm font-semibold text-fg-default">Keywords</h3>
              </div>
              {selectedCluster !== null && (
                <button
                  onClick={() => setSelectedCluster(null)}
                  className="gh-btn gh-btn-sm"
                >
                  Clear filter
                </button>
              )}
            </div>
            <KeywordTableCompact
              keywords={data.keywords}
              clusters={data.clusters}
              clusterLabels={data.clusterLabels}
              selectedCluster={selectedCluster}
              onClusterSelect={setSelectedCluster}
            />
          </div>
        </div>
      </div>

      {/* Re-clustering Panel - Always visible at bottom */}
      <div className="gh-box p-4 border-2 border-[var(--color-accent-emphasis)]/50 bg-gradient-to-r from-[var(--color-accent-subtle)]/30 to-transparent">
        <div className="flex items-start gap-6">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <div className="p-1.5 rounded-md bg-[var(--color-accent-emphasis)]">
                <RefreshCw className={`w-4 h-4 text-white ${isProcessing ? 'animate-spin' : ''}`} />
              </div>
              <div>
                <h3 className="font-semibold text-fg-default text-sm">Not happy with the results?</h3>
                <p className="text-xs text-fg-muted">Adjust settings and re-cluster with the same keywords</p>
              </div>
            </div>
            <div className="mt-4">
              <ClusteringConfigPanel
                config={currentConfig}
                onChange={onConfigChange}
                disabled={isProcessing}
                compact={true}
              />
            </div>
          </div>
          <div className="pt-1">
            <button
              onClick={() => onRecluster(currentConfig)}
              disabled={isProcessing}
              className="gh-btn gh-btn-primary whitespace-nowrap"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  Re-cluster Now
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
