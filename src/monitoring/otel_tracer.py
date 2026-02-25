"""
OpenTelemetry Tracing Setup

Configures OpenTelemetry tracing with console export by default.
Provides trace spans for each pipeline step: data fetch → strategy eval → signal generation.
"""

import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

@dataclass
class TraceConfig:
    """Configuration for OpenTelemetry tracing"""
    service_name: str = "claude-enterprise-trading"
    service_version: str = "1.0.0"
    environment: str = "development"
    
    # Exporter configuration
    export_to_console: bool = True
    export_to_jaeger: bool = False
    export_to_otlp: bool = False
    
    # Jaeger settings
    jaeger_endpoint: Optional[str] = None
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    
    # OTLP settings  
    otlp_endpoint: Optional[str] = None
    otlp_headers: Optional[Dict[str, str]] = None
    
    # Sampling configuration
    sampling_ratio: float = 1.0  # 100% sampling by default
    
    # Additional resource attributes
    additional_attributes: Optional[Dict[str, str]] = None

class TradingTracer:
    """
    Enhanced tracer for trading operations with domain-specific spans
    """
    
    def __init__(self, tracer_name: str = "claude_trading"):
        self.tracer = trace.get_tracer(tracer_name)
        self.logger = logging.getLogger(f'claude_trading.tracing.{tracer_name}')
    
    def start_strategy_execution_span(self, strategy_id: str, strategy_name: str):
        """Start a span for strategy execution"""
        span = self.tracer.start_span("strategy_execution")
        span.set_attribute("strategy.id", strategy_id)
        span.set_attribute("strategy.name", strategy_name)
        span.set_attribute("component", "strategy_execution")
        return span
    
    def start_data_fetch_span(self, symbols: list, data_source: str = "unknown"):
        """Start a span for market data fetching"""
        span = self.tracer.start_span("market_data_fetch")
        span.set_attribute("data.symbols", ",".join(symbols))
        span.set_attribute("data.source", data_source)
        span.set_attribute("data.symbol_count", len(symbols))
        span.set_attribute("component", "data_fetch")
        return span
    
    def start_indicator_calculation_span(self, symbol: str, indicators: list):
        """Start a span for technical indicator calculation"""
        span = self.tracer.start_span("indicator_calculation")
        span.set_attribute("symbol", symbol)
        span.set_attribute("indicators", ",".join(indicators))
        span.set_attribute("indicator_count", len(indicators))
        span.set_attribute("component", "indicator_calculation")
        return span
    
    def start_condition_evaluation_span(self, node_id: str, condition_count: int):
        """Start a span for strategy condition evaluation"""
        span = self.tracer.start_span("condition_evaluation")
        span.set_attribute("node.id", node_id)
        span.set_attribute("condition.count", condition_count)
        span.set_attribute("component", "condition_evaluation")
        return span
    
    def start_signal_generation_span(self, signal_type: str, symbol: str):
        """Start a span for trading signal generation"""
        span = self.tracer.start_span("signal_generation")
        span.set_attribute("signal.type", signal_type)
        span.set_attribute("signal.symbol", symbol)
        span.set_attribute("component", "signal_generation")
        return span
    
    def start_backtest_span(self, strategy_id: str, start_date: str, end_date: str):
        """Start a span for backtesting"""
        span = self.tracer.start_span("backtest_execution")
        span.set_attribute("strategy.id", strategy_id)
        span.set_attribute("backtest.start_date", start_date)
        span.set_attribute("backtest.end_date", end_date)
        span.set_attribute("component", "backtest")
        return span
    
    def start_circuit_breaker_span(self, check_type: str):
        """Start a span for circuit breaker checks"""
        span = self.tracer.start_span("circuit_breaker_check")
        span.set_attribute("check.type", check_type)
        span.set_attribute("component", "safety")
        return span
    
    def record_strategy_metrics(self, span, metrics: Dict[str, Any]):
        """Record strategy-related metrics on a span"""
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                span.set_attribute(f"metrics.{key}", value)
            elif isinstance(value, str):
                span.set_attribute(f"metrics.{key}", value)
    
    def record_performance_metrics(self, span, total_return: float, sharpe_ratio: float, max_drawdown: float):
        """Record performance metrics on a span"""
        span.set_attribute("performance.total_return", total_return)
        span.set_attribute("performance.sharpe_ratio", sharpe_ratio)
        span.set_attribute("performance.max_drawdown", max_drawdown)
    
    def record_error(self, span, error: Exception, component: str = "unknown"):
        """Record an error on a span"""
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        span.set_attribute("error.type", type(error).__name__)
        span.set_attribute("error.message", str(error))
        span.set_attribute("error.component", component)
        
        # Log the error as well
        self.logger.error(f"Error in {component}: {error}")

def setup_tracing(config: Optional[TraceConfig] = None) -> TracerProvider:
    """
    Setup OpenTelemetry tracing with the specified configuration
    
    Returns:
        TracerProvider: Configured tracer provider
    """
    if config is None:
        config = TraceConfig()
    
    logger = logging.getLogger('claude_trading.tracing.setup')
    
    # Create resource with service information
    resource_attributes = {
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "environment": config.environment,
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python"
    }
    
    # Add additional attributes if provided
    if config.additional_attributes:
        resource_attributes.update(config.additional_attributes)
    
    resource = Resource.create(resource_attributes)
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Setup exporters
    span_processors = []
    
    # Console exporter (default)
    if config.export_to_console:
        console_exporter = ConsoleSpanExporter()
        span_processors.append(BatchSpanProcessor(console_exporter))
        logger.info("Enabled console span export")
    
    # Jaeger exporter
    if config.export_to_jaeger:
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name=config.jaeger_agent_host,
                agent_port=config.jaeger_agent_port,
                collector_endpoint=config.jaeger_endpoint
            )
            span_processors.append(BatchSpanProcessor(jaeger_exporter))
            logger.info(f"Enabled Jaeger export to {config.jaeger_agent_host}:{config.jaeger_agent_port}")
        except Exception as e:
            logger.warning(f"Failed to setup Jaeger exporter: {e}")
    
    # OTLP exporter
    if config.export_to_otlp and config.otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.otlp_endpoint,
                headers=config.otlp_headers or {}
            )
            span_processors.append(BatchSpanProcessor(otlp_exporter))
            logger.info(f"Enabled OTLP export to {config.otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to setup OTLP exporter: {e}")
    
    # Add all span processors to tracer provider
    for processor in span_processors:
        tracer_provider.add_span_processor(processor)
    
    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    logger.info(f"OpenTelemetry tracing setup complete for {config.service_name}")
    
    return tracer_provider

def get_tracer(name: str = "claude_trading") -> trace.Tracer:
    """
    Get a tracer instance
    
    Args:
        name: Tracer name
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)

def get_trading_tracer(name: str = "claude_trading") -> TradingTracer:
    """
    Get enhanced trading tracer with domain-specific functionality
    
    Args:
        name: Tracer name
        
    Returns:
        TradingTracer instance
    """
    return TradingTracer(name)

def setup_auto_instrumentation():
    """
    Setup automatic instrumentation for common libraries
    """
    try:
        # Auto-instrument HTTP requests
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        
        # Auto-instrument aiohttp
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
        
        # Auto-instrument pandas operations (if available)
        try:
            from opentelemetry.instrumentation.pandas import PandasInstrumentor
            PandasInstrumentor().instrument()
        except ImportError:
            pass
        
        logging.getLogger('claude_trading.tracing').info("Auto-instrumentation setup complete")
        
    except ImportError as e:
        logging.getLogger('claude_trading.tracing').warning(f"Some auto-instrumentation packages not available: {e}")

def create_trace_config_from_env() -> TraceConfig:
    """
    Create TraceConfig from environment variables
    """
    return TraceConfig(
        service_name=os.getenv("OTEL_SERVICE_NAME", "claude-enterprise-trading"),
        service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        environment=os.getenv("ENVIRONMENT", "development"),
        
        export_to_console=os.getenv("OTEL_EXPORT_CONSOLE", "true").lower() == "true",
        export_to_jaeger=os.getenv("OTEL_EXPORT_JAEGER", "false").lower() == "true",
        export_to_otlp=os.getenv("OTEL_EXPORT_OTLP", "false").lower() == "true",
        
        jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
        jaeger_agent_host=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        jaeger_agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
        
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        
        sampling_ratio=float(os.getenv("OTEL_SAMPLING_RATIO", "1.0"))
    )

# Context managers for common tracing patterns
class trace_strategy_execution:
    """Context manager for strategy execution tracing"""
    
    def __init__(self, strategy_id: str, strategy_name: str, tracer: Optional[TradingTracer] = None):
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.tracer = tracer or get_trading_tracer()
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_strategy_execution_span(self.strategy_id, self.strategy_name)
        self.span.__enter__()
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracer.record_error(self.span, exc_val, "strategy_execution")
        return self.span.__exit__(exc_type, exc_val, exc_tb)

class trace_data_fetch:
    """Context manager for data fetching tracing"""
    
    def __init__(self, symbols: list, data_source: str = "unknown", tracer: Optional[TradingTracer] = None):
        self.symbols = symbols
        self.data_source = data_source
        self.tracer = tracer or get_trading_tracer()
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_data_fetch_span(self.symbols, self.data_source)
        self.span.__enter__()
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracer.record_error(self.span, exc_val, "data_fetch")
        return self.span.__exit__(exc_type, exc_val, exc_tb)

class trace_backtest:
    """Context manager for backtesting tracing"""
    
    def __init__(self, strategy_id: str, start_date: str, end_date: str, tracer: Optional[TradingTracer] = None):
        self.strategy_id = strategy_id
        self.start_date = start_date
        self.end_date = end_date
        self.tracer = tracer or get_trading_tracer()
        self.span = None
    
    def __enter__(self):
        self.span = self.tracer.start_backtest_span(self.strategy_id, self.start_date, self.end_date)
        self.span.__enter__()
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracer.record_error(self.span, exc_val, "backtest")
        return self.span.__exit__(exc_type, exc_val, exc_tb)

# Initialize tracing on module import if environment variable is set
if os.getenv("AUTO_SETUP_TRACING", "false").lower() == "true":
    config = create_trace_config_from_env()
    setup_tracing(config)
    setup_auto_instrumentation()
    logging.getLogger('claude_trading.tracing').info("Auto-setup tracing from environment variables")