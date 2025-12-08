"""
Timer utility for performance tracking.
"""
import time
from typing import Optional
from contextlib import contextmanager


class Timer:
    """
    Timer class for measuring execution time.
    
    Usage:
        timer = Timer()
        timer.start()
        # ... do something ...
        elapsed = timer.stop()
        print(f"Elapsed: {elapsed:.2f}s")
        
        # Or as context manager:
        with Timer("Training") as t:
            # ... do training ...
        # Automatically prints elapsed time
    """
    
    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Optional name for the timer
            verbose: Whether to print timing info
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None
        
        if self.verbose and self.name:
            print(f"⏱️  Starting: {self.name}")
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.verbose:
            self._print_elapsed()
        
        return self.elapsed
    
    def _print_elapsed(self):
        """Print elapsed time in human-readable format."""
        if self.elapsed is None:
            return
        
        name_str = f"{self.name}: " if self.name else ""
        
        if self.elapsed < 60:
            print(f"⏱️  {name_str}{self.elapsed:.2f} seconds")
        elif self.elapsed < 3600:
            minutes = self.elapsed / 60
            print(f"⏱️  {name_str}{minutes:.2f} minutes")
        else:
            hours = self.elapsed / 3600
            print(f"⏱️  {name_str}{hours:.2f} hours")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format time in human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.1f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)}h {int(minutes)}m"


@contextmanager
def timed_block(name: str = None, verbose: bool = True):
    """
    Context manager for timing a block of code.
    
    Usage:
        with timed_block("Training"):
            # ... do training ...
    """
    timer = Timer(name=name, verbose=verbose)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
