"use client";

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

interface ClusterScatterCanvasProps {
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

export default function ClusterScatterCanvas({
  embeddings2D,
  clusters,
  keywords,
  clusterLabels,
  selectedCluster,
  onClusterSelect,
}: ClusterScatterCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; keyword: string; cluster: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Calculate bounds
  const bounds = useMemo(() => {
    const xValues = embeddings2D.map(d => d[0]);
    const yValues = embeddings2D.map(d => d[1]);
    const padding = 0.1;
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const xPad = (xMax - xMin) * padding;
    const yPad = (yMax - yMin) * padding;
    return {
      xMin: xMin - xPad,
      xMax: xMax + xPad,
      yMin: yMin - yPad,
      yMax: yMax + yPad,
    };
  }, [embeddings2D]);

  // Transform data coordinates to canvas coordinates
  const toCanvasCoords = useCallback((x: number, y: number, width: number, height: number) => {
    const margin = 40;
    const plotWidth = width - margin * 2;
    const plotHeight = height - margin * 2;

    const xRange = bounds.xMax - bounds.xMin;
    const yRange = bounds.yMax - bounds.yMin;

    const cx = margin + ((x - bounds.xMin) / xRange) * plotWidth;
    const cy = margin + ((bounds.yMax - y) / yRange) * plotHeight; // Flip Y

    // Apply zoom and pan
    const centerX = width / 2;
    const centerY = height / 2;
    const zoomedX = centerX + (cx - centerX) * zoom + pan.x;
    const zoomedY = centerY + (cy - centerY) * zoom + pan.y;

    return { cx: zoomedX, cy: zoomedY };
  }, [bounds, zoom, pan]);

  // Draw the chart
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    const gridSize = 50 * zoom;
    for (let x = 40; x < width - 40; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x + pan.x % gridSize, 40);
      ctx.lineTo(x + pan.x % gridSize, height - 40);
      ctx.stroke();
    }
    for (let y = 40; y < height - 40; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(40, y + pan.y % gridSize);
      ctx.lineTo(width - 40, y + pan.y % gridSize);
      ctx.stroke();
    }

    // Sort points: draw non-selected first, then selected cluster on top
    const sortedIndices = [...Array(embeddings2D.length).keys()].sort((a, b) => {
      const aSelected = clusters[a] === selectedCluster;
      const bSelected = clusters[b] === selectedCluster;
      if (aSelected && !bSelected) return 1;
      if (!aSelected && bSelected) return -1;
      return 0;
    });

    // Draw points
    const pointSize = Math.max(3, 5 / Math.sqrt(embeddings2D.length / 500)) * zoom;

    for (const i of sortedIndices) {
      const [x, y] = embeddings2D[i];
      const cluster = clusters[i];
      const { cx, cy } = toCanvasCoords(x, y, width, height);

      // Skip points outside visible area
      if (cx < 0 || cx > width || cy < 0 || cy > height) continue;

      const isSelected = selectedCluster === cluster;
      const isOtherSelected = selectedCluster !== null && selectedCluster !== cluster;

      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? pointSize * 1.3 : pointSize, 0, Math.PI * 2);

      const color = COLORS[cluster % COLORS.length];
      ctx.fillStyle = isOtherSelected ? `${color}30` : isSelected ? color : `${color}BB`;
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

  }, [embeddings2D, clusters, selectedCluster, zoom, pan, toCanvasCoords]);

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Find nearest point
    let nearestDist = Infinity;
    let nearestIdx = -1;
    const threshold = 15;

    for (let i = 0; i < embeddings2D.length; i++) {
      const [x, y] = embeddings2D[i];
      const { cx, cy } = toCanvasCoords(x, y, rect.width, rect.height);
      const dist = Math.sqrt((cx - mouseX) ** 2 + (cy - mouseY) ** 2);
      if (dist < threshold && dist < nearestDist) {
        nearestDist = dist;
        nearestIdx = i;
      }
    }

    if (nearestIdx >= 0) {
      setTooltip({
        x: e.clientX,
        y: e.clientY,
        keyword: keywords[nearestIdx],
        cluster: clusters[nearestIdx],
      });
    } else {
      setTooltip(null);
    }
  }, [embeddings2D, keywords, clusters, toCanvasCoords]);

  // Handle click to select cluster
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || isDragging) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Find clicked point
    const threshold = 15;
    for (let i = 0; i < embeddings2D.length; i++) {
      const [x, y] = embeddings2D[i];
      const { cx, cy } = toCanvasCoords(x, y, rect.width, rect.height);
      const dist = Math.sqrt((cx - mouseX) ** 2 + (cy - mouseY) ** 2);
      if (dist < threshold) {
        onClusterSelect?.(clusters[i]);
        return;
      }
    }
  }, [embeddings2D, clusters, onClusterSelect, toCanvasCoords, isDragging]);

  // Pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleDrag = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  // Cluster counts for legend
  const clusterCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    clusters.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
    return counts;
  }, [clusters]);

  const uniqueClusters = useMemo(() => [...new Set(clusters)].sort((a, b) => a - b), [clusters]);

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(z => Math.min(z * 1.5, 10))}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom(z => Math.max(z / 1.5, 0.5))}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
        <div className="text-sm text-slate-500">
          {embeddings2D.length.toLocaleString()} points | Zoom: {zoom.toFixed(1)}x
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="relative h-[400px] bg-slate-50 rounded-xl overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-full cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { setTooltip(null); handleMouseUp(); }}
          onMouseMoveCapture={handleDrag}
          onClick={handleClick}
        />
        {tooltip && (
          <div
            className="fixed z-50 bg-white shadow-xl rounded-lg p-3 border border-slate-200 max-w-xs pointer-events-none"
            style={{ left: tooltip.x + 10, top: tooltip.y + 10 }}
          >
            <p className="font-medium text-slate-800 break-words">{tooltip.keyword}</p>
            <div className="flex items-center gap-2 mt-2">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: COLORS[tooltip.cluster % COLORS.length] }}
              />
              <p className="text-sm text-slate-500">
                {clusterLabels[tooltip.cluster] || `Cluster ${tooltip.cluster}`}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Compact Legend */}
      <div className="flex flex-wrap gap-1.5 max-h-[80px] overflow-y-auto">
        {uniqueClusters.map(cluster => {
          const isSelected = selectedCluster === cluster;
          return (
            <button
              key={cluster}
              onClick={() => onClusterSelect?.(selectedCluster === cluster ? null : cluster)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-all
                        ${isSelected ? "bg-slate-200 ring-2 ring-indigo-400" : "bg-slate-100 hover:bg-slate-200"}`}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: COLORS[cluster % COLORS.length] }}
              />
              <span className="text-slate-600 truncate max-w-[100px]">
                {clusterLabels[cluster] || `Cluster ${cluster}`}
              </span>
              <span className="text-slate-400">({clusterCounts[cluster]})</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
