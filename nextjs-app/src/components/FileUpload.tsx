"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileSpreadsheet, Play, X } from "lucide-react";
import * as XLSX from "xlsx";

interface FileUploadProps {
  onProcess: (keywords: string[]) => void;
  isProcessing: boolean;
  disabled: boolean;
}

export default function FileUpload({
  onProcess,
  isProcessing,
  disabled,
}: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [sheets, setSheets] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string>("");
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [preview, setPreview] = useState<string[]>([]);
  const [workbook, setWorkbook] = useState<XLSX.WorkBook | null>(null);

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
    if (!workbook || !selectedSheet || !selectedColumn) return;

    const sheet = workbook.Sheets[selectedSheet];
    const jsonData = XLSX.utils.sheet_to_json(sheet) as Record<string, unknown>[];

    const keywords = jsonData
      .map((row) => row[selectedColumn]?.toString() || "")
      .filter((k) => k.trim() !== "");

    onProcess(keywords);
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

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 space-y-4">
      <h2 className="text-lg font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <FileSpreadsheet className="w-5 h-5" />
        Upload Keywords
      </h2>

      {!file ? (
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all
            ${isDragActive
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
              : "border-slate-300 dark:border-slate-600 hover:border-blue-400 hover:bg-slate-50 dark:hover:bg-slate-700/50"
            }
            ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}
          `}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 mx-auto text-slate-400 mb-4" />
          <p className="text-slate-600 dark:text-slate-300">
            {isDragActive
              ? "Drop the Excel file here..."
              : "Drag & drop an Excel file, or click to select"}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
            Supports .xlsx and .xls files
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {/* File Info */}
          <div className="flex items-center justify-between bg-slate-50 dark:bg-slate-700 rounded-lg p-3">
            <div className="flex items-center gap-3">
              <FileSpreadsheet className="w-8 h-8 text-green-600" />
              <div>
                <p className="font-medium text-slate-800 dark:text-white">
                  {file.name}
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            <button
              onClick={clearFile}
              disabled={isProcessing}
              className="p-2 text-slate-500 hover:text-red-500 transition-colors disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Sheet Selection */}
          {sheets.length > 1 && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Select Sheet
              </label>
              <select
                value={selectedSheet}
                onChange={(e) => handleSheetSelect(e.target.value)}
                disabled={isProcessing}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg
                         bg-white dark:bg-slate-700 text-slate-800 dark:text-white
                         focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Choose a sheet...</option>
                {sheets.map((sheet) => (
                  <option key={sheet} value={sheet}>
                    {sheet}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Column Selection */}
          {columns.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Select Keyword Column
              </label>
              <select
                value={selectedColumn}
                onChange={(e) => handleColumnSelect(e.target.value)}
                disabled={isProcessing}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg
                         bg-white dark:bg-slate-700 text-slate-800 dark:text-white
                         focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Choose a column...</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Preview */}
          {preview.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Preview (first 10 keywords)
              </label>
              <div className="bg-slate-50 dark:bg-slate-700 rounded-lg p-3 max-h-48 overflow-y-auto">
                <ul className="space-y-1">
                  {preview.map((kw, i) => (
                    <li
                      key={i}
                      className="text-sm text-slate-600 dark:text-slate-300 truncate"
                    >
                      {kw}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Process Button */}
          <button
            onClick={handleProcess}
            disabled={disabled || !selectedColumn || isProcessing}
            className="w-full flex items-center justify-center gap-2 px-4 py-3
                     bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg
                     transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start Clustering
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
