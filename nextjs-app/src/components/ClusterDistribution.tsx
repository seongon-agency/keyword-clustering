"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ClusterDistributionProps {
  clusters: number[];
  clusterLabels: Record<number, string>;
}

const COLORS = [
  "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
  "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
  "#14B8A6", "#A855F7", "#EAB308", "#22C55E", "#0EA5E9",
];

export default function ClusterDistribution({
  clusters,
  clusterLabels,
}: ClusterDistributionProps) {
  const chartData = useMemo(() => {
    const counts: Record<number, number> = {};
    clusters.forEach((c) => {
      counts[c] = (counts[c] || 0) + 1;
    });

    return Object.entries(counts)
      .map(([cluster, count]) => ({
        cluster: Number(cluster),
        count,
        label: clusterLabels[Number(cluster)] || `Cluster ${cluster}`,
      }))
      .sort((a, b) => b.count - a.count);
  }, [clusters, clusterLabels]);

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof chartData[0] }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-slate-800 shadow-lg rounded-lg p-3 border border-slate-200 dark:border-slate-700">
          <p className="font-medium text-slate-800 dark:text-white">
            {data.label}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {data.count.toLocaleString()} keywords
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
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 20, right: 30, left: 150, bottom: 20 }}
          >
            <XAxis
              type="number"
              tick={{ fill: "#64748b", fontSize: 12 }}
              axisLine={{ stroke: "#e2e8f0" }}
            />
            <YAxis
              type="category"
              dataKey="label"
              tick={{ fill: "#64748b", fontSize: 12 }}
              axisLine={{ stroke: "#e2e8f0" }}
              width={140}
              tickFormatter={(value) =>
                value.length > 20 ? `${value.slice(0, 20)}...` : value
              }
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[entry.cluster % COLORS.length]}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-slate-200 dark:border-slate-700">
        <div className="text-center">
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {chartData.length}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Total Clusters
          </p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {chartData[0]?.count.toLocaleString() || 0}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Largest Cluster
          </p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {chartData[chartData.length - 1]?.count.toLocaleString() || 0}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Smallest Cluster
          </p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {Math.round(
              clusters.length / chartData.length
            ).toLocaleString()}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Avg per Cluster
          </p>
        </div>
      </div>
    </div>
  );
}
