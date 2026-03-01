import { useState, useEffect } from 'react'
import FlightForm from './components/FlightForm'

export default function App() {
  const [isDark, setIsDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches
  )

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDark])

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">✈</span>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                Flight Delay Predictor
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                LightGBM · SHAP Explanations · T-24h Rule
              </p>
            </div>
          </div>
          <button
            onClick={() => setIsDark(!isDark)}
            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            aria-label="Toggle dark mode"
          >
            {isDark ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        <FlightForm />
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-xs text-gray-400 dark:text-gray-600">
        Binary classification · ≥15 min arrival delay · No post-departure data used
      </footer>
    </div>
  )
}
