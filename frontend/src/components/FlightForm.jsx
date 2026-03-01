import { useState, useCallback } from 'react'
import { predictDelay } from '../utils/api'
import PredictionResult from './PredictionResult'
import LoadingSkeleton from './LoadingSkeleton'
import ErrorToast from './ErrorToast'

const INITIAL_FORM = {
  airline:             '',
  origin:              '',
  destination:         '',
  scheduled_departure: '',
  tail_number:         '',
}

const InputField = ({ label, id, required, ...props }) => (
  <div>
    <label htmlFor={id} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
      {label}{required && <span className="text-red-500 ml-1">*</span>}
    </label>
    <input
      id={id}
      {...props}
      className="w-full px-3 py-2.5 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors text-sm"
    />
  </div>
)

export default function FlightForm() {
  const [formData, setFormData]   = useState(INITIAL_FORM)
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await predictDelay(formData)
      setResult(data)
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.error ||
        err.message ||
        'Prediction failed. Please try again.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const dismissError = useCallback(() => setError(null), [])

  return (
    <div>
      {error && <ErrorToast message={error} onDismiss={dismissError} />}

      {/* Form card */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-5">
          Flight Details
        </h2>
        <form onSubmit={handleSubmit} noValidate>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <InputField
              label="Airline Code"
              id="airline"
              name="airline"
              value={formData.airline}
              onChange={handleChange}
              placeholder="AA"
              maxLength={3}
              required
            />
            <InputField
              label="Origin Airport"
              id="origin"
              name="origin"
              value={formData.origin}
              onChange={handleChange}
              placeholder="JFK"
              maxLength={4}
              required
            />
            <InputField
              label="Destination Airport"
              id="destination"
              name="destination"
              value={formData.destination}
              onChange={handleChange}
              placeholder="LAX"
              maxLength={4}
              required
            />
            <div>
              <label htmlFor="scheduled_departure" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Scheduled Departure <span className="text-red-500 ml-1">*</span>
              </label>
              <input
                type="datetime-local"
                id="scheduled_departure"
                name="scheduled_departure"
                value={formData.scheduled_departure}
                onChange={handleChange}
                required
                className="w-full px-3 py-2.5 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors text-sm"
              />
            </div>
            <InputField
              label="Tail Number (optional)"
              id="tail_number"
              name="tail_number"
              value={formData.tail_number}
              onChange={handleChange}
              placeholder="N123AA"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold rounded-xl transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Predicting…
              </span>
            ) : (
              'Predict Delay ✈'
            )}
          </button>
        </form>
      </div>

      {/* Results */}
      {loading && <LoadingSkeleton />}
      {!loading && result && <PredictionResult result={result} />}
    </div>
  )
}
