/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Command-centre palette: near-black base, cool slate panels, and a
        // single restrained risk ramp. Risk colours are reserved for risk -
        // never used decoratively, so a red on screen always means something.
        base:   { 900: "#050F17", 800: "#0A1822", 700: "#0F2230", 600: "#163040" },
        line:   { DEFAULT: "#1F3A4C", soft: "#15293A" },
        ink:    { 100: "#E6F1F7", 300: "#9DB4C4", 500: "#5F7A8E" },
        accent: { DEFAULT: "#22D3EE", dim: "#06B6D4" },
        risk:   { low: "#22C55E", med: "#EAB308", high: "#F97316", crit: "#EF4444" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 rgba(255,255,255,0.05) inset, 0 10px 30px rgba(0,0,0,0.45)",
        glow: "0 0 24px rgba(34,211,238,0.35)",
      },
    },
  },
  plugins: [],
};
