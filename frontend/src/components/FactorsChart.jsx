import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts'

const cleanFeatureName = (name) =>
  name
    .replace(/_te$/, ' (encoded)')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(' Sin', ' (sin)')
    .replace(' Cos', ' (cos)')

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const { feature, impact } = payload[0].payload
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-3 py-2 shadow-lg text-sm">
      <p className="font-medium text-gray-900 dark:text-white">{cleanFeatureName(feature)}</p>
      <p style={{ color: impact >= 0 ? '#ef4444' : '#3b82f6' }}>
        Impact: {impact >= 0 ? '+' : ''}{(impact * 100).toFixed(2)}%
      </p>
    </div>
  )
}

export default function FactorsChart({ factors }) {
  if (!factors || factors.length === 0) {
    return (
      <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-4">
        No SHAP factors available.
      </p>
    )
  }

  const data = factors.map((f) => ({ ...f, absImpact: Math.abs(f.impact) }))

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 uppercase tracking-wide">
        Key Factors
      </h3>
      <ResponsiveContainer width="100%" height={Math.max(160, factors.length * 36)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 60, left: 8, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" />
          <XAxis
            type="number"
            tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tickFormatter={cleanFeatureName}
            width={150}
            tick={{ fontSize: 12, fill: '#6b7280' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="absImpact" radius={[0, 4, 4, 0]} label={{ position: 'right', formatter: (v) => `${(v * 100).toFixed(1)}%`, fill: '#9ca3af', fontSize: 11 }}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.impact >= 0 ? '#ef4444' : '#3b82f6'}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: '#ef4444' }} />
          Increases delay risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: '#3b82f6' }} />
          Decreases delay risk
        </span>
      </div>
    </div>
  )
}
