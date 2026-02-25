"""
Monitoring and Observability

Provides OpenTelemetry tracing, metrics, and dashboard functionality
for monitoring the trading agent pipeline and performance.
"""

from .otel_tracer import setup_tracing, get_tracer, TraceConfig
from .dashboard import CLIDashboard, DashboardData

__all__ = [
    'setup_tracing',
    'get_tracer', 
    'TraceConfig',
    'CLIDashboard',
    'DashboardData'
]