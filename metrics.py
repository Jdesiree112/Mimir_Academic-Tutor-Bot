import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import threading

@dataclass
class InteractionMetric:
    """Single interaction metrics"""
    timestamp: str
    query_length: int
    response_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    streaming_chunks: int
    provider_latency: float
    error_occurred: bool
    error_message: Optional[str] = None

class MimirMetrics:
    """Metrics collection and analysis for Mimir"""
    
    def __init__(self, save_file: str = "Mimir_metrics.json"):
        self.metrics: List[InteractionMetric] = []
        self.save_file = save_file
        self.lock = threading.Lock()  # Thread-safe for concurrent requests
        
        # Load existing metrics if file exists
        self.load_metrics()
    
    def start_timing(self) -> dict:
        """Start timing an interaction - returns timing context"""
        return {
            'start_time': time.time(),
            'provider_start': None,
            'chunk_count': 0,
            'chunks_timing': []
        }
    
    def mark_provider_start(self, timing_context: dict):
        """Mark when provider API call starts"""
        timing_context['provider_start'] = time.time()
    
    def mark_provider_end(self, timing_context: dict):
        """Mark when provider API call ends and calculate latency"""
        if timing_context['provider_start']:
            timing_context['provider_latency'] = time.time() - timing_context['provider_start']
        else:
            timing_context['provider_latency'] = 0.0
    
    def record_chunk(self, timing_context: dict):
        """Record a streaming chunk"""
        timing_context['chunk_count'] += 1
        timing_context['chunks_timing'].append(time.time())
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting (approximation)"""
        # Rough approximation: 1 token ≈ 4 characters for most models
        return len(text) // 4
    
    def log_interaction(self, 
                       query: str,
                       response: str,
                       timing_context: dict,
                       error_occurred: bool = False,
                       error_message: str = None):
        """Log a complete interaction with all metrics"""
        
        end_time = time.time()
        response_time = end_time - timing_context['start_time']
        
        # Count tokens
        input_tokens = self.count_tokens(query)
        output_tokens = self.count_tokens(response)
        total_tokens = input_tokens + output_tokens
        
        # Get provider latency
        provider_latency = timing_context.get('provider_latency', 0.0)
        
        # Create metric record
        metric = InteractionMetric(
            timestamp=datetime.now().isoformat(),
            query_length=len(query),
            response_time=response_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            streaming_chunks=timing_context['chunk_count'],
            provider_latency=provider_latency,
            error_occurred=error_occurred,
            error_message=error_message
        )
        
        # Thread-safe append
        with self.lock:
            self.metrics.append(metric)
        
        # Auto-save every 10 interactions
        if len(self.metrics) % 2 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            with self.lock:
                data = [
                    {
                        'timestamp': m.timestamp,
                        'query_length': m.query_length,
                        'response_time': m.response_time,
                        'input_tokens': m.input_tokens,
                        'output_tokens': m.output_tokens,
                        'total_tokens': m.total_tokens,
                        'streaming_chunks': m.streaming_chunks,
                        'provider_latency': m.provider_latency,
                        'error_occurred': m.error_occurred,
                        'error_message': m.error_message
                    }
                    for m in self.metrics
                ]
            
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def load_metrics(self):
        """Load existing metrics from file"""
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
            
            self.metrics = [
                InteractionMetric(**item) for item in data
            ]
            
        except FileNotFoundError:
            # File doesn't exist yet, start fresh
            self.metrics = []
        except Exception as e:
            print(f"Error loading metrics: {e}")
            self.metrics = []
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics"""
        if not self.metrics:
            return {"message": "No metrics recorded yet"}
        
        response_times = [m.response_time for m in self.metrics]
        provider_latencies = [m.provider_latency for m in self.metrics]
        token_counts = [m.total_tokens for m in self.metrics]
        chunk_counts = [m.streaming_chunks for m in self.metrics]
        error_count = sum(1 for m in self.metrics if m.error_occurred)
        
        return {
            "total_interactions": len(self.metrics),
            "error_rate": (error_count / len(self.metrics)) * 100,
            "avg_response_time": sum(response_times) / len(response_times),
            "avg_provider_latency": sum(provider_latencies) / len(provider_latencies),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "avg_chunks": sum(chunk_counts) / len(chunk_counts),
        }
    
    def export_csv(self, filename: str = None):
        """Export metrics to CSV format"""
        if filename is None:
            filename = f"Mimir_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'query_length', 'response_time',
                    'input_tokens', 'output_tokens', 'total_tokens',
                    'streaming_chunks', 'provider_latency', 'error_occurred'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for metric in self.metrics:
                    writer.writerow({
                        'timestamp': metric.timestamp,
                        'query_length': metric.query_length,
                        'response_time': metric.response_time,
                        'input_tokens': metric.input_tokens,
                        'output_tokens': metric.output_tokens,
                        'total_tokens': metric.total_tokens,
                        'streaming_chunks': metric.streaming_chunks,
                        'provider_latency': metric.provider_latency,
                        'error_occurred': metric.error_occurred
                    })
            
            return f"Metrics exported to {filename}"
            
        except Exception as e:
            return f"Error exporting CSV: {e}"