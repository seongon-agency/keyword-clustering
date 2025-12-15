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
      {/* 3D Plot - full height when legend hidden */}
      <div className={`w-full ${showLegend ? 'h-[420px]' : 'h-[500px]'} bg-canvas-subtle rounded-t-md overflow-hidden transition-all`}>
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

      {/* Compact Legend Bar */}
      <div className="bg-canvas-default border-t border-default rounded-b-md px-2 py-1.5">
        <div className="flex items-center gap-2">
          {/* Toggle */}
          <button
            onClick={() => setShowLegend(!showLegend)}
            className="flex items-center gap-1 text-[10px] text-fg-muted hover:text-fg-default transition-colors flex-shrink-0"
          >
            {showLegend ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            <span className="font-medium">{clusterInfo.length}</span>
          </button>

          {/* Horizontal scrollable legend */}
          {showLegend && (
            <div className="flex-1 overflow-x-auto scrollbar-thin">
              <div className="flex gap-1">
                {clusterInfo.map((cluster) => (
                  <button
                    key={cluster.id}
                    onClick={() => onClusterSelect?.(selectedCluster === cluster.id ? null : cluster.id)}
                    className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] whitespace-nowrap transition-all flex-shrink-0 ${
                      selectedCluster === cluster.id
                        ? "ring-1 ring-[var(--color-accent-emphasis)]"
                        : selectedCluster !== null
                        ? "opacity-30 hover:opacity-60"
                        : "hover:opacity-80"
                    }`}
                    style={{
                      backgroundColor: `${cluster.color}20`,
                    }}
                    title={`${cluster.label} (${cluster.count} keywords)`}
                  >
                    <span
                      className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: cluster.color }}
                    />
                    <span className="text-fg-default font-medium max-w-[80px] truncate">
                      {cluster.label}
                    </span>
                    <span className="text-fg-muted">{cluster.count}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
