/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Command-centre palette: near-black base, cool slate panels, and a
        // single restrained risk ramp. Risk colours are reserved for risk -
        // never used decoratively, so a red on screen always means something.
        base:   { 900: "#080B10", 800: "#0D1117", 700: "#141A22", 600: "#1C242E" },
        line:   { DEFAULT: "#243040", soft: "#1A2330" },
        ink:    { 100: "#E8EDF4", 300: "#9FB0C4", 500: "#647A93" },
        accent: { DEFAULT: "#38BDF8", dim: "#0EA5E9" },
        risk:   { low: "#22C55E", med: "#EAB308", high: "#F97316", crit: "#EF4444" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: { panel: "0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.4)" },
    },
  },
  plugins: [],
};
