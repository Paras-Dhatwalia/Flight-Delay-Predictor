import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

export const predictDelay = async (formData) => {
  const payload = {
    airline:              formData.airline.toUpperCase().trim(),
    origin:               formData.origin.toUpperCase().trim(),
    destination:          formData.destination.toUpperCase().trim(),
    scheduled_departure:  formData.scheduled_departure,
    tail_number:          formData.tail_number?.trim() || null,
  }
  const response = await api.post('/predict', payload)
  return response.data
}

export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}
