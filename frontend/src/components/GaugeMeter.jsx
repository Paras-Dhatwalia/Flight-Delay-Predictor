import { RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts'

const getColor = (probability) => {
  if (probability < 0.3) return '#22c55e'   // green
  if (probability <= 0.6) return '#eab308'  // yellow
  return '#ef4444'                           // red
}

export default function GaugeMeter({ probability, size = 200 }) {
  const pct = Math.round(probability * 100)
  const color = getColor(probability)
  const data = [{ value: pct, fill: color }]

  return (
    <div className="relative flex items-center justify-center">
      <RadialBarChart
        width={size}
        height={size}
        cx={size / 2}
        cy={size / 2}
        innerRadius={size * 0.35}
        outerRadius={size * 0.48}
        barSize={size * 0.08}
        data={data}
        startAngle={90}
        endAngle={-270}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
        <RadialBar
          dataKey="value"
          cornerRadius={size * 0.04}
          background={{ fill: '#e5e7eb' }}
        />
      </RadialBarChart>
      {/* Center label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="text-3xl font-bold"
          style={{ color }}
        >
          {pct}%
        </span>
        <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Delay Prob.
        </span>
      </div>
    </div>
  )
}
