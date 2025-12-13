"use client";

import { Globe, Zap, Sparkles } from "lucide-react";

interface SettingsProps {
  language: "Vietnamese" | "English";
  setLanguage: (lang: "Vietnamese" | "English") => void;
  disabled: boolean;
}

export default function Settings({
  language,
  setLanguage,
  disabled,
}: SettingsProps) {
  return (
    <div className="space-y-3">
      {/* Language Selection Card */}
      <div className="card !p-4 hover:shadow-xl transition-shadow duration-300">
        <div className="flex items-center gap-2.5 mb-3">
          <div className="relative p-1.5 bg-gradient-to-br from-purple-600 via-violet-600 to-fuchsia-500 rounded-xl shadow-lg shadow-purple-500/25">
            <Globe className="w-3.5 h-3.5 text-white" />
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
          </div>
          <div>
            <h2 className="font-semibold text-slate-800 text-sm">Language</h2>
            <p className="text-[10px] text-slate-400">Select keyword language</p>
          </div>
        </div>

        <div className="space-y-2">
          {([
            { code: "Vietnamese", label: "VN", name: "Vietnamese" },
            { code: "English", label: "EN", name: "English" },
          ] as const).map((lang) => (
            <button
              key={lang.code}
              onClick={() => setLanguage(lang.code)}
              disabled={disabled}
              className={`group w-full px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 ${
                language === lang.code
                  ? "bg-gradient-to-r from-purple-600 via-violet-600 to-fuchsia-600 text-white shadow-lg shadow-purple-500/30"
                  : "bg-slate-50 text-slate-600 hover:bg-slate-100 border border-slate-200 hover:border-purple-200"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <span className="flex items-center justify-center gap-2">
                <span className={`font-bold uppercase tracking-wide ${
                  language === lang.code ? "text-white" : "text-purple-600"
                }`}>
                  {lang.label}
                </span>
                <span className={`text-[10px] ${
                  language === lang.code ? "text-white/80" : "text-slate-400"
                }`}>
                  {lang.name}
                </span>
                {language === lang.code && (
                  <Sparkles className="w-3 h-3 ml-1" />
                )}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Quick Info */}
      <div className="card !p-3.5 bg-gradient-to-br from-emerald-50 to-teal-50 border-emerald-100 hover:shadow-lg transition-shadow duration-300">
        <div className="flex items-start gap-2.5">
          <div className="p-1.5 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-lg shadow-sm shadow-emerald-500/30">
            <Zap className="w-3 h-3 text-white" />
          </div>
          <div>
            <p className="font-semibold text-emerald-800 text-xs">Fast & Accurate</p>
            <p className="text-[10px] text-emerald-700 mt-0.5 leading-relaxed">
              Process thousands of keywords in seconds with AI-powered semantic clustering.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
