import GaugeMeter from './GaugeMeter'
import FactorsChart from './FactorsChart'

const RISK_CONFIG = {
  Low:    { bg: 'bg-green-100 dark:bg-green-900/40',  text: 'text-green-700 dark:text-green-400',  icon: '🟢', border: 'border-green-200 dark:border-green-700' },
  Medium: { bg: 'bg-yellow-100 dark:bg-yellow-900/40', text: 'text-yellow-700 dark:text-yellow-400', icon: '🟡', border: 'border-yellow-200 dark:border-yellow-700' },
  High:   { bg: 'bg-red-100 dark:bg-red-900/40',    text: 'text-red-700 dark:text-red-400',    icon: '🔴', border: 'border-red-200 dark:border-red-700' },
}

export default function PredictionResult({ result }) {
  const { delay_probability, risk_level, top_factors, threshold_used } = result
  const risk = RISK_CONFIG[risk_level] || RISK_CONFIG.Medium

  return (
    <div className="mt-6 bg-white dark:bg-gray-800 rounded-2xl shadow-lg overflow-hidden">
      {/* Top section: gauge + risk badge */}
      <div className="p-6 border-b border-gray-100 dark:border-gray-700">
        <div className="flex flex-col sm:flex-row items-center gap-6">
          {/* Gauge */}
          <div className="flex-shrink-0">
            <GaugeMeter probability={delay_probability} size={180} />
          </div>

          {/* Risk info */}
          <div className="flex-1 text-center sm:text-left">
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Risk Level</p>
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border ${risk.bg} ${risk.border}`}>
              <span className="text-xl">{risk.icon}</span>
              <span className={`text-xl font-bold ${risk.text}`}>{risk_level.toUpperCase()}</span>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg px-3 py-2">
                <p className="text-gray-400 text-xs">Probability</p>
                <p className="font-bold text-gray-900 dark:text-white">
                  {(delay_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg px-3 py-2">
                <p className="text-gray-400 text-xs">Threshold</p>
                <p className="font-bold text-gray-900 dark:text-white">
                  {(threshold_used * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            <p className="mt-3 text-xs text-gray-400 dark:text-gray-500">
              {risk_level === 'High'
                ? 'High chance of a 15+ min arrival delay.'
                : risk_level === 'Medium'
                ? 'Moderate delay risk. Monitor before departure.'
                : 'Low delay risk based on historical patterns.'}
            </p>
          </div>
        </div>
      </div>

      {/* Bottom section: SHAP factors */}
      <div className="p-6">
        <FactorsChart factors={top_factors} />
      </div>
    </div>
  )
}
