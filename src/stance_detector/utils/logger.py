# @Author: Rachith Aiyappa
# @Date: 2025-04-30

import logging
import sys
from pathlib import Path
from datetime import datetime

class CustomLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory at project root
        log_dir = Path(__file__).parents[3] / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create handlers
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        file_handler = logging.FileHandler(
            log_dir / f"{name}_{current_time}.log"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set levels
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def get_logger(self):
        return self.logger