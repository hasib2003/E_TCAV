import os
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_file:str=None, timestamp=True):

        assert log_file is not None, f"Log file is required log_file: {log_file}"

        self.log_file = log_file
        self.timestamp = timestamp
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create log file if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write(f"Log started at {datetime.now()}\n")
    
    def write(self, text):
        # Write to original stdout (console)
        self.original_stdout.write(text)
        
        # Write to log file with optional timestamp
        if text.strip():  # Only log non-empty lines
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if self.timestamp:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {text}\n")
                else:
                    f.write(f"{text}\n")
        
        # force flush after each write
        self.flush()
    
    def flush(self):
        # Flush both outputs
        self.original_stdout.flush()
    
    def enable(self):
        """Enable logging by redirecting stdout & stderr"""
        sys.stdout = self
        sys.stderr = self
    
    def disable(self):
        """Disable logging by restoring original stdout & stderr"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr