"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight, Copy, Check, TrendingUp, Hash, Layers } from "lucide-react";

interface ClusterExplorerProps {
  keywords: string[];
  clusters: number[];
  clusterLabels: Record<number, string>;
  onClusterSelect?: (cluster: number | null) => void;
  selectedCluster?: number | null;
}

const COLORS = [
  "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
  "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
  "#14B8A6", "#A855F7", "#EAB308", "#22C55E", "#0EA5E9",
];

export default function ClusterExplorer({
  keywords,
  clusters,
  clusterLabels,
  onClusterSelect,
  selectedCluster,
}: ClusterExplorerProps) {
  const [expandedClusters, setExpandedClusters] = useState<Set<number>>(new Set());
  const [copiedCluster, setCopiedCluster] = useState<number | null>(null);
  const [sortBy, setSortBy] = useState<"size" | "id">("size");

  const clusterData = useMemo(() => {
    const data: Record<number, { keywords: string[]; count: number }> = {};

    clusters.forEach((c, i) => {
      if (!data[c]) {
        data[c] = { keywords: [], count: 0 };
      }
      data[c].keywords.push(keywords[i]);
      data[c].count++;
    });

    return Object.entries(data)
      .map(([cluster, info]) => ({
        cluster: Number(cluster),
        label: clusterLabels[Number(cluster)] || `Cluster ${cluster}`,
        keywords: info.keywords,
        count: info.count,
        percentage: ((info.count / keywords.length) * 100).toFixed(1),
      }))
      .sort((a, b) => {
        if (sortBy === "size") return b.count - a.count;
        return a.cluster - b.cluster;
      });
  }, [clusters, keywords, clusterLabels, sortBy]);

  const toggleExpand = (cluster: number) => {
    const newExpanded = new Set(expandedClusters);
    if (newExpanded.has(cluster)) {
      newExpanded.delete(cluster);
    } else {
      newExpanded.add(cluster);
    }
    setExpandedClusters(newExpanded);
  };

  const copyKeywords = async (cluster: number, clusterKeywords: string[]) => {
    await navigator.clipboard.writeText(clusterKeywords.join("\n"));
    setCopiedCluster(cluster);
    setTimeout(() => setCopiedCluster(null), 2000);
  };

  const handleClusterClick = (cluster: number) => {
    if (onClusterSelect) {
      onClusterSelect(selectedCluster === cluster ? null : cluster);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h3 className="text-sm font-semibold text-slate-700">
            {clusterData.length} Clusters
          </h3>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500">Sort:</span>
            <button
              onClick={() => setSortBy("size")}
              className={`px-2 py-1 rounded ${
                sortBy === "size"
                  ? "bg-indigo-100 text-indigo-700"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              By Size
            </button>
            <button
              onClick={() => setSortBy("id")}
              className={`px-2 py-1 rounded ${
                sortBy === "id"
                  ? "bg-indigo-100 text-indigo-700"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              By ID
            </button>
          </div>
        </div>
        <button
          onClick={() => {
            if (expandedClusters.size === clusterData.length) {
              setExpandedClusters(new Set());
            } else {
              setExpandedClusters(new Set(clusterData.map((c) => c.cluster)));
            }
          }}
          className="text-xs text-indigo-600 hover:text-indigo-700 font-medium"
        >
          {expandedClusters.size === clusterData.length ? "Collapse All" : "Expand All"}
        </button>
      </div>

      {/* Cluster Cards */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
        {clusterData.map((item) => (
          <div
            key={item.cluster}
            className={`border rounded-xl transition-all ${
              selectedCluster === item.cluster
                ? "border-indigo-300 bg-indigo-50/50 shadow-sm"
                : "border-slate-200 hover:border-slate-300 bg-white"
            }`}
          >
            {/* Cluster Header */}
            <div
              className="flex items-center gap-3 p-3 cursor-pointer"
              onClick={() => toggleExpand(item.cluster)}
            >
              {/* Expand Icon */}
              <button className="text-slate-400 hover:text-slate-600 transition-colors">
                {expandedClusters.has(item.cluster) ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
              </button>

              {/* Color Dot */}
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: COLORS[item.cluster % COLORS.length] }}
              />

              {/* Label */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-slate-800 truncate">
                    {item.label}
                  </span>
                  <span
                    className="text-xs px-1.5 py-0.5 rounded-full bg-slate-100 text-slate-500"
                  >
                    #{item.cluster}
                  </span>
                </div>
              </div>

              {/* Stats */}
              <div className="flex items-center gap-3 text-sm">
                <div className="flex items-center gap-1 text-slate-500">
                  <Hash className="w-3.5 h-3.5" />
                  <span className="font-medium">{item.count}</span>
                </div>
                <div className="flex items-center gap-1 text-slate-400 min-w-[50px] justify-end">
                  <TrendingUp className="w-3.5 h-3.5" />
                  <span>{item.percentage}%</span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleClusterClick(item.cluster);
                  }}
                  className={`p-1.5 rounded-lg transition-colors ${
                    selectedCluster === item.cluster
                      ? "bg-indigo-200 text-indigo-700"
                      : "hover:bg-slate-100 text-slate-400 hover:text-slate-600"
                  }`}
                  title="Highlight in scatter plot"
                >
                  <Layers className="w-4 h-4" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    copyKeywords(item.cluster, item.keywords);
                  }}
                  className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                  title="Copy all keywords"
                >
                  {copiedCluster === item.cluster ? (
                    <Check className="w-4 h-4 text-green-600" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>

            {/* Expanded Content */}
            {expandedClusters.has(item.cluster) && (
              <div className="px-4 pb-3 pt-1 border-t border-slate-100">
                {/* Progress Bar */}
                <div className="mb-3">
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${item.percentage}%`,
                        backgroundColor: COLORS[item.cluster % COLORS.length],
                      }}
                    />
                  </div>
                </div>

                {/* Keywords Preview */}
                <div className="space-y-1">
                  <div className="text-xs font-medium text-slate-500 mb-2">
                    Sample Keywords ({Math.min(10, item.keywords.length)} of {item.count})
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {item.keywords.slice(0, 10).map((kw, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center px-2 py-1 rounded-md text-xs
                                 bg-slate-100 text-slate-700 hover:bg-slate-200 transition-colors"
                      >
                        {kw.length > 30 ? `${kw.slice(0, 30)}...` : kw}
                      </span>
                    ))}
                    {item.keywords.length > 10 && (
                      <span className="inline-flex items-center px-2 py-1 rounded-md text-xs
                                     bg-slate-50 text-slate-400">
                        +{item.keywords.length - 10} more
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-3 pt-4 border-t border-slate-200">
        <div className="text-center p-3 bg-slate-50 rounded-lg">
          <div className="text-lg font-bold text-slate-800">
            {Math.round(keywords.length / clusterData.length)}
          </div>
          <div className="text-xs text-slate-500">Avg. per cluster</div>
        </div>
        <div className="text-center p-3 bg-slate-50 rounded-lg">
          <div className="text-lg font-bold text-slate-800">
            {clusterData[0]?.count || 0}
          </div>
          <div className="text-xs text-slate-500">Largest cluster</div>
        </div>
        <div className="text-center p-3 bg-slate-50 rounded-lg">
          <div className="text-lg font-bold text-slate-800">
            {clusterData[clusterData.length - 1]?.count || 0}
          </div>
          <div className="text-xs text-slate-500">Smallest cluster</div>
        </div>
      </div>
    </div>
  );
}
