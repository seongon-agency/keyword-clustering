"use client";

import { GitBranch, Sun, Moon } from "lucide-react";
import { useTheme } from "@/context/ThemeContext";

export default function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="gh-header sticky top-0 z-50">
      <div className="container mx-auto px-4 max-w-[1600px]">
        <div className="flex items-center justify-between h-14">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <GitBranch className="w-5 h-5 text-fg-default" />
              <h1 className="font-semibold text-fg-default text-base">
                Phân cụm từ khóa
              </h1>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="gh-btn gh-btn-sm"
              aria-label="Chuyển đổi giao diện"
            >
              {theme === "light" ? (
                <Moon className="w-4 h-4" />
              ) : (
                <Sun className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
