import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './theme/tokens.css'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
            <App />
        </BrowserRouter>
    </StrictMode>,
)
