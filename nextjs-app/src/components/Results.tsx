"use client";

import { useState, useMemo } from "react";
import { Download, BarChart3, Scatter, Table } from "lucide-react";
import * as XLSX from "xlsx";
import type { ClusterResult } from "@/app/page";
import ClusterScatter from "./ClusterScatter";
import ClusterDistribution from "./ClusterDistribution";

interface ResultsProps {
  data: ClusterResult;
}

export default function Results({ data }: ResultsProps) {
  const [activeTab, setActiveTab] = useState<"scatter" | "distribution" | "table">(
    "scatter"
  );

  const stats = useMemo(() => {
    const uniqueClusters = new Set(data.clusters);
    const clusterCounts: Record<number, number> = {};
    data.clusters.forEach((c) => {
      clusterCounts[c] = (clusterCounts[c] || 0) + 1;
    });

    return {
      totalKeywords: data.keywords.length,
      numClusters: uniqueClusters.size,
      clusterCounts,
    };
  }, [data]);

  const handleDownload = () => {
    const exportData = data.keywords.map((kw, i) => ({
      keywords: kw,
      segmented: data.segmented[i],
      Cluster: data.clusters[i],
      "Cluster Label": data.clusterLabels[data.clusters[i]] || "Unknown",
    }));

    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.json_to_sheet(exportData);
    XLSX.utils.book_append_sheet(wb, ws, "Clusters");

    const timestamp = new Date().toISOString().slice(0, 16).replace(/[:-]/g, "");
    XLSX.writeFile(wb, `cluster_${timestamp}.xlsx`);
  };

  const tabs = [
    { id: "scatter" as const, label: "Cluster Map", icon: Scatter },
    { id: "distribution" as const, label: "Distribution", icon: BarChart3 },
    { id: "table" as const, label: "Keywords", icon: Table },
  ];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-white">
          Clustering Results
        </h2>
        <button
          onClick={handleDownload}
          className="flex items-center justify-center gap-2 px-4 py-2
                   bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg
                   transition-colors"
        >
          <Download className="w-4 h-4" />
          Download Excel
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
          <p className="text-sm text-slate-500 dark:text-slate-400">Total Keywords</p>
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {stats.totalKeywords.toLocaleString()}
          </p>
        </div>
        <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4">
          <p className="text-sm text-slate-500 dark:text-slate-400">Clusters</p>
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {stats.numClusters}
          </p>
        </div>
        <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-4 col-span-2 md:col-span-1">
          <p className="text-sm text-slate-500 dark:text-slate-400">Unique Labels</p>
          <p className="text-2xl font-bold text-slate-800 dark:text-white">
            {Object.keys(data.clusterLabels).length}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex gap-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 border-b-2 transition-colors ${
                activeTab === tab.id
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === "scatter" && (
          <ClusterScatter
            embeddings2D={data.embeddings2D}
            clusters={data.clusters}
            keywords={data.keywords}
            clusterLabels={data.clusterLabels}
          />
        )}

        {activeTab === "distribution" && (
          <ClusterDistribution
            clusters={data.clusters}
            clusterLabels={data.clusterLabels}
          />
        )}

        {activeTab === "table" && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 dark:bg-slate-700">
                  <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">
                    Keyword
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">
                    Cluster
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-slate-700 dark:text-slate-300">
                    Label
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {data.keywords.slice(0, 100).map((kw, i) => (
                  <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-700/50">
                    <td className="px-4 py-3 text-slate-800 dark:text-slate-200">
                      {kw}
                    </td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-400">
                      {data.clusters[i]}
                    </td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-400">
                      {data.clusterLabels[data.clusters[i]] || "Unknown"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data.keywords.length > 100 && (
              <p className="text-center text-sm text-slate-500 dark:text-slate-400 py-4">
                Showing 100 of {data.keywords.length} keywords. Download for full data.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
