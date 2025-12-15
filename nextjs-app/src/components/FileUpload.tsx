"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileSpreadsheet, X, Type, File, Wand2, Globe } from "lucide-react";
import * as XLSX from "xlsx";
import ClusteringConfigPanel, { ClusteringConfig } from "./ClusteringConfigPanel";

interface FileUploadProps {
  onProcess: (keywords: string[]) => void;
  isProcessing: boolean;
  disabled: boolean;
  language: "Vietnamese" | "English";
  setLanguage: (lang: "Vietnamese" | "English") => void;
  clusteringConfig: ClusteringConfig;
  setClusteringConfig: (config: ClusteringConfig) => void;
}

type InputMode = "paste" | "file";

function parseKeywordsFromText(text: string): string[] {
  if (!text || !text.trim()) return [];

  const lines = text.trim().split("\n");
  const keywords: string[] = [];

  for (const line of lines) {
    const trimmedLine = line.trim();
    if (!trimmedLine) continue;

    if (trimmedLine.includes("\t")) {
      const parts = trimmedLine.split("\t");
      keywords.push(...parts.map((p) => p.trim()).filter((p) => p));
    } else if (trimmedLine.includes(",") && trimmedLine.split(",").length > 1) {
      const parts = trimmedLine.split(",");
      keywords.push(...parts.map((p) => p.trim()).filter((p) => p));
    } else {
      keywords.push(trimmedLine);
    }
  }

  const seen = new Set<string>();
  return keywords.filter((kw) => {
    const lower = kw.toLowerCase();
    if (seen.has(lower)) return false;
    seen.add(lower);
    return true;
  });
}

export default function FileUpload({
  onProcess,
  isProcessing,
  disabled,
  language,
  setLanguage,
  clusteringConfig,
  setClusteringConfig,
}: FileUploadProps) {
  const [inputMode, setInputMode] = useState<InputMode>("paste");
  const [pasteText, setPasteText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [sheets, setSheets] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string>("");
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [preview, setPreview] = useState<string[]>([]);
  const [workbook, setWorkbook] = useState<XLSX.WorkBook | null>(null);

  const parsedKeywords = parseKeywordsFromText(pasteText);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setFile(file);
    setSelectedSheet("");
    setSelectedColumn("");
    setPreview([]);

    const reader = new FileReader();
    reader.onload = (e) => {
      const data = new Uint8Array(e.target?.result as ArrayBuffer);
      const wb = XLSX.read(data, { type: "array" });
      setWorkbook(wb);
      setSheets(wb.SheetNames);
      if (wb.SheetNames.length === 1) {
        handleSheetSelect(wb.SheetNames[0], wb);
      }
    };
    reader.readAsArrayBuffer(file);
  }, []);

  const handleSheetSelect = (sheetName: string, wb?: XLSX.WorkBook) => {
    const book = wb || workbook;
    if (!book) return;

    setSelectedSheet(sheetName);
    setSelectedColumn("");
    setPreview([]);

    const sheet = book.Sheets[sheetName];
    const jsonData = XLSX.utils.sheet_to_json(sheet, { header: 1 }) as string[][];

    if (jsonData.length > 0) {
      const headers = jsonData[0].map((h, i) => h?.toString() || `Column ${i + 1}`);
      setColumns(headers);
    }
  };

  const handleColumnSelect = (column: string) => {
    if (!workbook || !selectedSheet) return;

    setSelectedColumn(column);

    const sheet = workbook.Sheets[selectedSheet];
    const jsonData = XLSX.utils.sheet_to_json(sheet) as Record<string, unknown>[];

    const keywords = jsonData
      .map((row) => row[column]?.toString() || "")
      .filter((k) => k.trim() !== "")
      .slice(0, 10);

    setPreview(keywords);
  };

  const handleProcess = () => {
    if (inputMode === "paste") {
      if (parsedKeywords.length >= 2) {
        onProcess(parsedKeywords);
      }
    } else {
      if (!workbook || !selectedSheet || !selectedColumn) return;

      const sheet = workbook.Sheets[selectedSheet];
      const jsonData = XLSX.utils.sheet_to_json(sheet) as Record<string, unknown>[];

      const keywords = jsonData
        .map((row) => row[selectedColumn]?.toString() || "")
        .filter((k) => k.trim() !== "");

      onProcess(keywords);
    }
  };

  const clearFile = () => {
    setFile(null);
    setWorkbook(null);
    setSheets([]);
    setSelectedSheet("");
    setColumns([]);
    setSelectedColumn("");
    setPreview([]);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
    maxFiles: 1,
    disabled: isProcessing,
  });

  const canProcess =
    inputMode === "paste"
      ? parsedKeywords.length >= 2
      : !!selectedColumn;

  return (
    <div className="gh-box p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileSpreadsheet className="w-5 h-5 text-fg-muted" />
          <div>
            <h2 className="font-semibold text-fg-default text-sm">Input Keywords</h2>
            <p className="text-xs text-fg-muted">Paste or upload your keywords</p>
          </div>
        </div>
        {parsedKeywords.length > 0 && inputMode === "paste" && (
          <span className="gh-counter">{parsedKeywords.length.toLocaleString()}</span>
        )}
      </div>

      {/* Input Mode Tabs */}
      <div className="gh-segmented w-full mb-4">
        <button
          onClick={() => setInputMode("paste")}
          disabled={isProcessing}
          className={`gh-segmented-btn flex-1 flex items-center justify-center gap-1.5 ${inputMode === "paste" ? "active" : ""}`}
        >
          <Type className="w-3.5 h-3.5" />
          Paste Text
        </button>
        <button
          onClick={() => setInputMode("file")}
          disabled={isProcessing}
          className={`gh-segmented-btn flex-1 flex items-center justify-center gap-1.5 ${inputMode === "file" ? "active" : ""}`}
        >
          <File className="w-3.5 h-3.5" />
          Upload File
        </button>
      </div>

      {/* Paste Mode */}
      {inputMode === "paste" && (
        <div className="space-y-3 animate-fade-in">
          <div className="relative">
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              disabled={isProcessing}
              placeholder="Enter keywords here (one per line, or paste from Excel)..."
              className="gh-input h-32 font-mono text-xs resize-none"
            />
            {pasteText && (
              <button
                onClick={() => setPasteText("")}
                className="absolute top-2 right-2 p-1 text-fg-subtle hover:text-fg-default hover:bg-[var(--color-neutral-muted)] rounded transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Preview */}
          {parsedKeywords.length > 0 && (
            <div className="bg-canvas-subtle rounded-md p-2.5 animate-fade-in">
              <div className="flex flex-wrap gap-1.5">
                {parsedKeywords.slice(0, 6).map((kw, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-canvas-default border border-default text-fg-muted rounded text-[11px]"
                  >
                    {kw.length > 20 ? `${kw.slice(0, 20)}...` : kw}
                  </span>
                ))}
                {parsedKeywords.length > 6 && (
                  <span className="gh-counter text-[11px]">
                    +{parsedKeywords.length - 6}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* File Mode */}
      {inputMode === "file" && (
        <div className="space-y-3 animate-fade-in">
          {!file ? (
            <div
              {...getRootProps()}
              className={`
                relative border-2 border-dashed rounded-md p-4 text-center cursor-pointer transition-all
                ${isDragActive
                  ? "border-[var(--color-accent-emphasis)] bg-[var(--color-accent-subtle)]"
                  : "border-default hover:border-[var(--color-accent-emphasis)] hover:bg-canvas-subtle"
                }
                ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}
              `}
            >
              <input {...getInputProps()} />
              <div className="flex items-center justify-center gap-3">
                <div className="w-10 h-10 rounded-md bg-canvas-subtle flex items-center justify-center">
                  <Upload className="w-5 h-5 text-fg-muted" />
                </div>
                <div className="text-left">
                  <p className="text-fg-default font-medium text-sm">
                    {isDragActive ? "Drop here..." : "Drop Excel file or click"}
                  </p>
                  <p className="text-xs text-fg-muted">.xlsx, .xls</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {/* File Info */}
              <div className="flex items-center justify-between bg-[var(--color-success-subtle)] rounded-md p-2.5 border border-[var(--color-success-emphasis)]">
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-success" />
                  <div>
                    <p className="font-medium text-fg-default text-xs">{file.name}</p>
                    <p className="text-[10px] text-fg-muted">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <button
                  onClick={clearFile}
                  disabled={isProcessing}
                  className="p-1 text-fg-muted hover:text-danger rounded transition-all"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Sheet Selection */}
              {sheets.length > 1 && (
                <select
                  value={selectedSheet}
                  onChange={(e) => handleSheetSelect(e.target.value)}
                  disabled={isProcessing}
                  className="gh-input py-1.5 text-xs"
                >
                  <option value="">Select sheet...</option>
                  {sheets.map((sheet) => (
                    <option key={sheet} value={sheet}>{sheet}</option>
                  ))}
                </select>
              )}

              {/* Column Selection */}
              {columns.length > 0 && (
                <select
                  value={selectedColumn}
                  onChange={(e) => handleColumnSelect(e.target.value)}
                  disabled={isProcessing}
                  className="gh-input py-1.5 text-xs"
                >
                  <option value="">Select keyword column...</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              )}

              {/* Preview */}
              {preview.length > 0 && (
                <div className="bg-canvas-subtle rounded-md p-2 animate-fade-in">
                  <div className="flex flex-wrap gap-1">
                    {preview.slice(0, 5).map((kw, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-canvas-default border border-default text-fg-muted rounded text-[10px] truncate max-w-[100px]">
                        {kw}
                      </span>
                    ))}
                    {preview.length > 5 && (
                      <span className="gh-counter text-[10px]">
                        +{preview.length - 5}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Language Selection */}
      <div className="flex items-center justify-between mt-4 mb-3 px-1">
        <div className="flex items-center gap-2">
          <Globe className="w-3.5 h-3.5 text-fg-muted" />
          <span className="text-xs text-fg-muted">Language</span>
        </div>
        <div className="gh-segmented">
          {([
            { code: "Vietnamese", label: "VN" },
            { code: "English", label: "EN" },
          ] as const).map((lang) => (
            <button
              key={lang.code}
              onClick={() => setLanguage(lang.code)}
              disabled={isProcessing}
              className={`gh-segmented-btn ${language === lang.code ? "active" : ""}`}
            >
              {lang.label}
            </button>
          ))}
        </div>
      </div>

      {/* Clustering Settings - Always visible */}
      <div className="mt-4 mb-4">
        <ClusteringConfigPanel
          config={clusteringConfig}
          onChange={setClusteringConfig}
          disabled={isProcessing}
        />
      </div>

      {/* Process Button */}
      <button
        onClick={handleProcess}
        disabled={disabled || !canProcess || isProcessing}
        className="gh-btn gh-btn-primary w-full py-2.5 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isProcessing ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            <span>Analyzing keywords...</span>
          </>
        ) : (
          <>
            <Wand2 className="w-4 h-4" />
            <span>Start Analysis</span>
            {canProcess && (
              <span className="px-2 py-0.5 bg-white/20 rounded-full text-xs font-medium">
                {inputMode === "paste" ? parsedKeywords.length.toLocaleString() : "Ready"}
              </span>
            )}
          </>
        )}
      </button>
    </div>
  );
}
