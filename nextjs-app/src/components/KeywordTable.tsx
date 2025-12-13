"use client";

import { useState, useMemo } from "react";
import { Search, ChevronLeft, ChevronRight, ArrowUpDown, Filter, X } from "lucide-react";

interface KeywordTableProps {
  keywords: string[];
  segmented: string[];
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

type SortField = "keyword" | "cluster";
type SortDirection = "asc" | "desc";

export default function KeywordTable({
  keywords,
  segmented,
  clusters,
  clusterLabels,
  onClusterSelect,
  selectedCluster,
}: KeywordTableProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterCluster, setFilterCluster] = useState<number | "all">("all");
  const [sortField, setSortField] = useState<SortField>("cluster");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  const uniqueClusters = useMemo(() => {
    return [...new Set(clusters)].sort((a, b) => a - b);
  }, [clusters]);

  const tableData = useMemo(() => {
    let data = keywords.map((kw, i) => ({
      index: i,
      keyword: kw,
      segmented: segmented[i],
      cluster: clusters[i],
      label: clusterLabels[clusters[i]] || `Cluster ${clusters[i]}`,
    }));

    // Filter by search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      data = data.filter(
        (item) =>
          item.keyword.toLowerCase().includes(query) ||
          item.label.toLowerCase().includes(query)
      );
    }

    // Filter by cluster
    if (filterCluster !== "all") {
      data = data.filter((item) => item.cluster === filterCluster);
    }

    // Sort
    data.sort((a, b) => {
      let comparison = 0;
      if (sortField === "keyword") {
        comparison = a.keyword.localeCompare(b.keyword);
      } else {
        comparison = a.cluster - b.cluster;
      }
      return sortDirection === "asc" ? comparison : -comparison;
    });

    return data;
  }, [keywords, segmented, clusters, clusterLabels, searchQuery, filterCluster, sortField, sortDirection]);

  // Pagination
  const totalPages = Math.ceil(tableData.length / pageSize);
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return tableData.slice(start, start + pageSize);
  }, [tableData, currentPage, pageSize]);

  // Reset to page 1 when filters change
  useMemo(() => {
    setCurrentPage(1);
  }, [searchQuery, filterCluster]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const handleClusterClick = (cluster: number) => {
    if (onClusterSelect) {
      onClusterSelect(selectedCluster === cluster ? null : cluster);
    }
  };

  return (
    <div className="space-y-4">
      {/* Search and Filter Bar */}
      <div className="flex flex-col sm:flex-row gap-3">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search keywords or labels..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 border-2 border-slate-200 rounded-xl bg-white
                     text-slate-800 placeholder-slate-400
                     focus:ring-2 focus:ring-indigo-500 focus:border-indigo-300
                     transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Cluster Filter */}
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <select
            value={filterCluster}
            onChange={(e) => setFilterCluster(e.target.value === "all" ? "all" : Number(e.target.value))}
            className="pl-10 pr-8 py-2.5 border-2 border-slate-200 rounded-xl bg-white
                     text-slate-800 appearance-none cursor-pointer min-w-[200px]
                     focus:ring-2 focus:ring-indigo-500 focus:border-indigo-300
                     transition-all"
          >
            <option value="all">All Clusters ({uniqueClusters.length})</option>
            {uniqueClusters.map((c) => (
              <option key={c} value={c}>
                {clusterLabels[c] || `Cluster ${c}`} ({clusters.filter(x => x === c).length})
              </option>
            ))}
          </select>
        </div>

        {/* Page Size */}
        <select
          value={pageSize}
          onChange={(e) => {
            setPageSize(Number(e.target.value));
            setCurrentPage(1);
          }}
          className="px-3 py-2.5 border-2 border-slate-200 rounded-xl bg-white
                   text-slate-800 cursor-pointer
                   focus:ring-2 focus:ring-indigo-500 focus:border-indigo-300
                   transition-all"
        >
          <option value={25}>25 rows</option>
          <option value={50}>50 rows</option>
          <option value={100}>100 rows</option>
          <option value={250}>250 rows</option>
        </select>
      </div>

      {/* Results count */}
      <div className="flex items-center justify-between text-sm text-slate-500">
        <span>
          Showing {paginatedData.length} of {tableData.length} keywords
          {searchQuery && ` matching "${searchQuery}"`}
          {filterCluster !== "all" && ` in cluster ${filterCluster}`}
        </span>
        {(searchQuery || filterCluster !== "all") && (
          <button
            onClick={() => {
              setSearchQuery("");
              setFilterCluster("all");
            }}
            className="text-indigo-600 hover:text-indigo-700 font-medium"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-200">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50">
              <th className="px-4 py-3 text-left font-semibold text-slate-700 border-b border-slate-200 w-12">
                #
              </th>
              <th
                className="px-4 py-3 text-left font-semibold text-slate-700 border-b border-slate-200 cursor-pointer hover:bg-slate-100 transition-colors"
                onClick={() => handleSort("keyword")}
              >
                <div className="flex items-center gap-2">
                  Keyword
                  <ArrowUpDown className={`w-4 h-4 ${sortField === "keyword" ? "text-indigo-600" : "text-slate-400"}`} />
                </div>
              </th>
              <th
                className="px-4 py-3 text-left font-semibold text-slate-700 border-b border-slate-200 cursor-pointer hover:bg-slate-100 transition-colors"
                onClick={() => handleSort("cluster")}
              >
                <div className="flex items-center gap-2">
                  Cluster
                  <ArrowUpDown className={`w-4 h-4 ${sortField === "cluster" ? "text-indigo-600" : "text-slate-400"}`} />
                </div>
              </th>
              <th className="px-4 py-3 text-left font-semibold text-slate-700 border-b border-slate-200">
                Label
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {paginatedData.map((item, idx) => (
              <tr
                key={item.index}
                className={`hover:bg-slate-50 transition-colors ${
                  selectedCluster === item.cluster ? "bg-indigo-50" : ""
                }`}
              >
                <td className="px-4 py-3 text-slate-400 text-xs">
                  {(currentPage - 1) * pageSize + idx + 1}
                </td>
                <td className="px-4 py-3 text-slate-800 font-medium">
                  {searchQuery ? (
                    <HighlightText text={item.keyword} query={searchQuery} />
                  ) : (
                    item.keyword
                  )}
                </td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => handleClusterClick(item.cluster)}
                    className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium transition-all
                              ${selectedCluster === item.cluster
                                ? "ring-2 ring-offset-1 ring-indigo-500"
                                : "hover:ring-2 hover:ring-offset-1 hover:ring-slate-300"
                              }`}
                    style={{
                      backgroundColor: `${COLORS[item.cluster % COLORS.length]}20`,
                      color: COLORS[item.cluster % COLORS.length],
                    }}
                  >
                    <span
                      className="w-2 h-2 rounded-full mr-1.5"
                      style={{ backgroundColor: COLORS[item.cluster % COLORS.length] }}
                    />
                    {item.cluster}
                  </button>
                </td>
                <td className="px-4 py-3 text-slate-600 max-w-xs truncate">
                  {searchQuery ? (
                    <HighlightText text={item.label} query={searchQuery} />
                  ) : (
                    item.label
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {paginatedData.length === 0 && (
          <div className="text-center py-12 text-slate-500">
            <Search className="w-12 h-12 mx-auto mb-3 text-slate-300" />
            <p className="font-medium">No keywords found</p>
            <p className="text-sm mt-1">Try adjusting your search or filters</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-slate-500">
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              className="px-3 py-1.5 text-sm font-medium text-slate-600 bg-slate-100 rounded-lg
                       hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              First
            </button>
            <button
              onClick={() => setCurrentPage(currentPage - 1)}
              disabled={currentPage === 1}
              className="p-1.5 text-slate-600 bg-slate-100 rounded-lg
                       hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            {/* Page numbers */}
            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }
                return (
                  <button
                    key={pageNum}
                    onClick={() => setCurrentPage(pageNum)}
                    className={`w-8 h-8 text-sm font-medium rounded-lg transition-colors ${
                      currentPage === pageNum
                        ? "bg-indigo-600 text-white"
                        : "text-slate-600 hover:bg-slate-100"
                    }`}
                  >
                    {pageNum}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="p-1.5 text-slate-600 bg-slate-100 rounded-lg
                       hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
              className="px-3 py-1.5 text-sm font-medium text-slate-600 bg-slate-100 rounded-lg
                       hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Last
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper component to highlight search matches
function HighlightText({ text, query }: { text: string; query: string }) {
  if (!query) return <>{text}</>;

  const parts = text.split(new RegExp(`(${query})`, "gi"));
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <mark key={i} className="bg-yellow-200 text-yellow-900 rounded px-0.5">
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </>
  );
}
