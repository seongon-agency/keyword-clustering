"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { Box } from "lucide-react";
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
        title: "",
        zeroline: false,
        showspikes: false,
      },
      yaxis: {
        showgrid: true,
        gridcolor: isDark ? "#30363d" : "#e2e8f0",
        showticklabels: false,
        title: "",
        zeroline: false,
        showspikes: false,
      },
      zaxis: {
        showgrid: true,
        gridcolor: isDark ? "#30363d" : "#e2e8f0",
        showticklabels: false,
        title: "",
        zeroline: false,
        showspikes: false,
      },
      bgcolor: isDark ? "#161b22" : "#f8fafc",
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 },
      },
    },
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      bgcolor: isDark ? "rgba(22, 27, 34, 0.9)" : "rgba(255,255,255,0.9)",
      bordercolor: isDark ? "#30363d" : "#e2e8f0",
      borderwidth: 1,
      font: { size: 10, color: isDark ? "#e6edf3" : "#1f2328" },
      itemclick: "toggle" as const,
      itemdoubleclick: "toggleothers" as const,
    },
    hovermode: "closest" as const,
    modebar: {
      bgcolor: isDark ? "rgba(22, 27, 34, 0.8)" : "rgba(248, 250, 252, 0.8)",
      color: isDark ? "#8b949e" : "#94a3b8",
      activecolor: "#6366f1",
      orientation: "v",
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
    <div className="w-full h-[500px] bg-canvas-subtle rounded-md overflow-hidden">
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
  );
}
