"use client";

import { useMemo, useState } from "react";
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
import { ZoomIn, ZoomOut, RotateCcw, Eye, EyeOff } from "lucide-react";

interface ClusterScatterInteractiveProps {
  embeddings2D: [number, number][];
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

interface ChartDataPoint {
  x: number;
  y: number;
  cluster: number;
  keyword: string;
  label: string;
}

export default function ClusterScatterInteractive({
  embeddings2D,
  clusters,
  keywords,
  clusterLabels,
  selectedCluster,
  onClusterSelect,
}: ClusterScatterInteractiveProps) {
  const [hiddenClusters, setHiddenClusters] = useState<Set<number>>(new Set());
  const [zoomArea, setZoomArea] = useState<{ x1: number; x2: number; y1: number; y2: number } | null>(null);

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

  // Calculate domain based on zoom
  const domain = useMemo(() => {
    if (zoomArea) {
      return {
        x: [Math.min(zoomArea.x1, zoomArea.x2), Math.max(zoomArea.x1, zoomArea.x2)] as [number, number],
        y: [Math.min(zoomArea.y1, zoomArea.y2), Math.max(zoomArea.y1, zoomArea.y2)] as [number, number],
      };
    }

    const xValues = chartData.map((d) => d.x);
    const yValues = chartData.map((d) => d.y);
    const padding = 0.1;

    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const xPad = (xMax - xMin) * padding;
    const yPad = (yMax - yMin) * padding;

    return {
      x: [xMin - xPad, xMax + xPad] as [number, number],
      y: [yMin - yPad, yMax + yPad] as [number, number],
    };
  }, [chartData, zoomArea]);

  // Filter data based on hidden clusters
  const filteredData = useMemo(() => {
    return chartData.filter((d) => !hiddenClusters.has(d.cluster));
  }, [chartData, hiddenClusters]);

  const resetZoom = () => setZoomArea(null);

  const zoomIn = () => {
    const currentDomain = domain;
    const xCenter = (currentDomain.x[0] + currentDomain.x[1]) / 2;
    const yCenter = (currentDomain.y[0] + currentDomain.y[1]) / 2;
    const xRange = (currentDomain.x[1] - currentDomain.x[0]) * 0.35;
    const yRange = (currentDomain.y[1] - currentDomain.y[0]) * 0.35;

    setZoomArea({
      x1: xCenter - xRange,
      x2: xCenter + xRange,
      y1: yCenter - yRange,
      y2: yCenter + yRange,
    });
  };

  const zoomOut = () => {
    const currentDomain = domain;
    const xCenter = (currentDomain.x[0] + currentDomain.x[1]) / 2;
    const yCenter = (currentDomain.y[0] + currentDomain.y[1]) / 2;
    const xRange = (currentDomain.x[1] - currentDomain.x[0]) * 0.75;
    const yRange = (currentDomain.y[1] - currentDomain.y[0]) * 0.75;

    setZoomArea({
      x1: xCenter - xRange,
      x2: xCenter + xRange,
      y1: yCenter - yRange,
      y2: yCenter + yRange,
    });
  };

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ payload: ChartDataPoint }>;
  }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-canvas-default shadow-xl rounded-lg p-3 border border-default max-w-xs z-50">
          <p className="font-medium text-fg-default break-words">{data.keyword}</p>
          <div className="flex items-center gap-2 mt-2">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: COLORS[data.cluster % COLORS.length] }}
            />
            <p className="text-sm text-fg-muted">
              {data.label}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <button
            onClick={zoomIn}
            className="gh-btn gh-btn-sm"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={zoomOut}
            className="gh-btn gh-btn-sm"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={resetZoom}
            className="gh-btn gh-btn-sm"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <div className="h-5 w-px bg-[var(--color-border-muted)] mx-1" />
          <button
            onClick={() => {
              if (hiddenClusters.size > 0) {
                setHiddenClusters(new Set());
              } else {
                setHiddenClusters(new Set(uniqueClusters));
              }
            }}
            className="gh-btn gh-btn-sm"
          >
            {hiddenClusters.size > 0 ? (
              <>
                <Eye className="w-3.5 h-3.5" />
                Show All
              </>
            ) : (
              <>
                <EyeOff className="w-3.5 h-3.5" />
                Hide All
              </>
            )}
          </button>
        </div>
        <div className="text-xs text-fg-muted">
          {filteredData.length.toLocaleString()} / {chartData.length.toLocaleString()} điểm
        </div>
      </div>

      {/* Chart */}
      <div className="h-[400px] bg-canvas-subtle rounded-md p-2">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <XAxis
              type="number"
              dataKey="x"
              domain={domain.x}
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={domain.y}
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <ZAxis range={[20, 40]} />
            <Tooltip content={<CustomTooltip />} />
            <Scatter
              name="Keywords"
              data={filteredData}
              isAnimationActive={false}
            >
              {filteredData.map((entry, index) => {
                const isSelected = selectedCluster === entry.cluster;
                const isOtherSelected = selectedCluster !== null && selectedCluster !== entry.cluster;

                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[entry.cluster % COLORS.length]}
                    fillOpacity={isOtherSelected ? 0.15 : isSelected ? 1 : 0.7}
                    stroke={isSelected ? COLORS[entry.cluster % COLORS.length] : "transparent"}
                    strokeWidth={isSelected ? 2 : 0}
                    style={{ cursor: "pointer" }}
                    onClick={() => onClusterSelect?.(entry.cluster)}
                  />
                );
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
}
