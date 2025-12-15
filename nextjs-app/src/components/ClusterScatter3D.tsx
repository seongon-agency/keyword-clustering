"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Box, ChevronDown, ChevronUp } from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface ClusterScatter3DProps {
  embeddings3D: [number, number, number][];
  clusters: number[];
  keywords: string[];
  clusterLabels: Record<number, string>;
  selectedCluster?: number | null;
  onClusterSelect?: (cluster: number | null) => void;
}

const COLORS = [
  "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
  "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
  "#14B8A6", "#A855F7", "#EAB308", "#22C55E", "#0EA5E9",
];

export default function ClusterScatter3D({
  embeddings3D,
  clusters,
  keywords,
  clusterLabels,
  selectedCluster,
  onClusterSelect,
}: ClusterScatter3DProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [showLegend, setShowLegend] = useState(true);

  const traces = useMemo(() => {
    // Guard against undefined data
    if (!embeddings3D || embeddings3D.length === 0) {
      return [];
    }

    // Group data by cluster
    const clusterGroups: Record<number, {
      x: number[];
      y: number[];
      z: number[];
      text: string[];
    }> = {};

    embeddings3D.forEach((coord, i) => {
      const cluster = clusters[i];
      if (!clusterGroups[cluster]) {
        clusterGroups[cluster] = { x: [], y: [], z: [], text: [] };
      }
      clusterGroups[cluster].x.push(coord[0]);
      clusterGroups[cluster].y.push(coord[1]);
      clusterGroups[cluster].z.push(coord[2]);
      clusterGroups[cluster].text.push(keywords[i]);
    });

    // Create a trace for each cluster
    return Object.entries(clusterGroups).map(([clusterId, data]) => {
      const cluster = Number(clusterId);
      const isSelected = selectedCluster === cluster;
      const isOtherSelected = selectedCluster !== null && selectedCluster !== cluster;

      return {
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: clusterLabels[cluster] || `Cluster ${cluster}`,
        x: data.x,
        y: data.y,
        z: data.z,
        text: data.text,
        hovertemplate: "<b>%{text}</b><br>" + (clusterLabels[cluster] || `Cluster ${cluster}`) + "<extra></extra>",
        marker: {
          size: isSelected ? 6 : 4,
          color: COLORS[cluster % COLORS.length],
          opacity: isOtherSelected ? 0.15 : isSelected ? 1 : 0.7,
          line: {
            color: isSelected ? "#fff" : "transparent",
            width: isSelected ? 1 : 0,
          },
        },
      };
    });
  }, [embeddings3D, clusters, keywords, clusterLabels, selectedCluster]);

  // Get unique clusters with their counts for the legend
  const clusterInfo = useMemo(() => {
    const counts: Record<number, number> = {};
    clusters.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
    return Object.entries(counts)
      .map(([id, count]) => ({
        id: Number(id),
        label: clusterLabels[Number(id)] || `Cluster ${id}`,
        count,
        color: COLORS[Number(id) % COLORS.length],
      }))
      .sort((a, b) => b.count - a.count);
  }, [clusters, clusterLabels]);

  const layout = useMemo(() => ({
    autosize: true,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: {
        showgrid: true,
        gridcolor: isDark ? "#30363d" : "#e2e8f0",
        showticklabels: false,
        title: { text: "" },
        zeroline: false,
        showspikes: false,
      },
      yaxis: {
        showgrid: true,
        gridcolor: isDark ? "#30363d" : "#e2e8f0",
        showticklabels: false,
        title: { text: "" },
        zeroline: false,
        showspikes: false,
      },
      zaxis: {
        showgrid: true,
        gridcolor: isDark ? "#30363d" : "#e2e8f0",
        showticklabels: false,
        title: { text: "" },
        zeroline: false,
        showspikes: false,
      },
      bgcolor: isDark ? "#161b22" : "#f8fafc",
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 },
      },
    },
    showlegend: false, // Hide Plotly legend - we'll show our own below
    hovermode: "closest" as const,
    modebar: {
      bgcolor: isDark ? "rgba(22, 27, 34, 0.8)" : "rgba(248, 250, 252, 0.8)",
      color: isDark ? "#8b949e" : "#94a3b8",
      activecolor: "#6366f1",
      orientation: "v" as const,
    },
  }), [isDark]);

  const config = useMemo(() => ({
    displayModeBar: true,
    modeBarButtons: [["resetCameraDefault3d"]] as any,
    displaylogo: false,
    responsive: true,
  }), []);

  // Show loading state if no 3D data available
  if (!embeddings3D || embeddings3D.length === 0) {
    return (
      <div className="w-full h-[500px] bg-canvas-subtle rounded-md flex items-center justify-center">
        <div className="text-center text-fg-muted">
          <Box className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p className="text-sm font-medium">3D data not available</p>
          <p className="text-xs mt-1">Run a new analysis to generate 3D visualization</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* 3D Plot */}
      <div className="w-full h-[400px] bg-canvas-subtle rounded-md overflow-hidden">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
          onClick={(event: any) => {
            if (event.points && event.points.length > 0) {
              const pointData = event.points[0];
              const traceIndex = pointData.curveNumber;
              const clusterId = Number(Object.keys(
                clusters.reduce((acc: Record<number, boolean>, c) => {
                  acc[c] = true;
                  return acc;
                }, {})
              ).sort((a, b) => Number(a) - Number(b))[traceIndex]);

              if (onClusterSelect) {
                onClusterSelect(selectedCluster === clusterId ? null : clusterId);
              }
            }
          }}
        />
      </div>

      {/* Legend Section */}
      <div className="mt-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-fg-muted">{clusterInfo.length} clusters</span>
          <button
            onClick={() => setShowLegend(!showLegend)}
            className="text-xs text-fg-muted hover:text-fg-default transition-colors"
          >
            {showLegend ? "Hide" : "Show"}
          </button>
        </div>
        {showLegend && (
          <div className="flex flex-wrap gap-2">
            {clusterInfo.map((cluster) => (
              <button
                key={cluster.id}
                onClick={() => onClusterSelect?.(selectedCluster === cluster.id ? null : cluster.id)}
                className={`inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md border transition-all ${
                  selectedCluster === cluster.id
                    ? "border-[var(--color-accent-emphasis)] bg-[var(--color-accent-subtle)]"
                    : selectedCluster !== null
                    ? "border-transparent bg-canvas-subtle opacity-40 hover:opacity-70"
                    : "border-transparent bg-canvas-subtle hover:bg-canvas-default"
                }`}
                title={`${cluster.label} (${cluster.count} keywords)`}
              >
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: cluster.color }}
                />
                <span className="text-xs text-fg-default font-medium truncate max-w-[120px]">
                  {cluster.label}
                </span>
                <span className="text-xs text-fg-muted">{cluster.count}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
