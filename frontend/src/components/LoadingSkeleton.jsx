export default function LoadingSkeleton() {
  return (
    <div className="mt-6 bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 animate-pulse">
      <div className="flex items-center justify-between mb-6">
        <div className="h-6 w-32 bg-gray-200 dark:bg-gray-700 rounded-lg" />
        <div className="h-8 w-20 bg-gray-200 dark:bg-gray-700 rounded-full" />
      </div>

      {/* Gauge placeholder */}
      <div className="flex justify-center mb-6">
        <div className="w-48 h-48 rounded-full bg-gray-200 dark:bg-gray-700" />
      </div>

      {/* Bar chart placeholder */}
      <div className="space-y-3">
        {[80, 60, 45, 30, 20].map((w, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="h-3 w-28 bg-gray-200 dark:bg-gray-700 rounded" />
            <div
              className="h-5 bg-gray-200 dark:bg-gray-700 rounded"
              style={{ width: `${w}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  )
}
