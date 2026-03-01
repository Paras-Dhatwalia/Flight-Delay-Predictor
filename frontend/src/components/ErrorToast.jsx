import { useEffect } from 'react'

export default function ErrorToast({ message, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 5000)
    return () => clearTimeout(timer)
  }, [onDismiss])

  return (
    <div className="fixed top-4 right-4 z-50 flex items-start gap-3 bg-red-600 text-white px-4 py-3 rounded-xl shadow-xl max-w-sm animate-slide-in">
      <span className="text-lg mt-0.5">⚠️</span>
      <p className="flex-1 text-sm font-medium">{message}</p>
      <button
        onClick={onDismiss}
        className="text-red-200 hover:text-white transition-colors text-lg leading-none"
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  )
}
