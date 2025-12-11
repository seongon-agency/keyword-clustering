"use client";

import { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ClusterScatterProps {
  embeddings2D: [number, number][];
  clusters: number[];
  keywords: string[];
  clusterLabels: Record<number, string>;
}

// Color palette for clusters
const COLORS = [
  "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
  "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
  "#14B8A6", "#A855F7", "#EAB308", "#22C55E", "#0EA5E9",
];

export default function ClusterScatter({
  embeddings2D,
  clusters,
  keywords,
  clusterLabels,
}: ClusterScatterProps) {
  const chartData = useMemo(() => {
    return embeddings2D.map((coord, i) => ({
      x: coord[0],
      y: coord[1],
      cluster: clusters[i],
      keyword: keywords[i],
      label: clusterLabels[clusters[i]] || `Cluster ${clusters[i]}`,
    }));
  }, [embeddings2D, clusters, keywords, clusterLabels]);

  const uniqueClusters = useMemo(() => {
    return [...new Set(clusters)].sort((a, b) => a - b);
  }, [clusters]);

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof chartData[0] }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-slate-800 shadow-lg rounded-lg p-3 border border-slate-200 dark:border-slate-700 max-w-xs">
          <p className="font-medium text-slate-800 dark:text-white truncate">
            {data.keyword}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Cluster {data.cluster}: {data.label}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      <div className="h-[500px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <XAxis
              type="number"
              dataKey="x"
              name="UMAP 1"
              tick={{ fill: "#64748b", fontSize: 12 }}
              axisLine={{ stroke: "#e2e8f0" }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="UMAP 2"
              tick={{ fill: "#64748b", fontSize: 12 }}
              axisLine={{ stroke: "#e2e8f0" }}
            />
            <ZAxis range={[40, 40]} />
            <Tooltip content={<CustomTooltip />} />
            <Scatter name="Keywords" data={chartData}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[entry.cluster % COLORS.length]}
                  fillOpacity={0.7}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-2 justify-center">
        {uniqueClusters.slice(0, 15).map((cluster) => (
          <div
            key={cluster}
            className="flex items-center gap-2 px-2 py-1 bg-slate-50 dark:bg-slate-700 rounded text-sm"
          >
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: COLORS[cluster % COLORS.length] }}
            />
            <span className="text-slate-600 dark:text-slate-300 truncate max-w-[150px]">
              {clusterLabels[cluster] || `Cluster ${cluster}`}
            </span>
          </div>
        ))}
        {uniqueClusters.length > 15 && (
          <span className="text-sm text-slate-500 dark:text-slate-400 px-2 py-1">
            +{uniqueClusters.length - 15} more
          </span>
        )}
      </div>
    </div>
  );
}
