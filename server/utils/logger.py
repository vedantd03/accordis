import os
import sys
import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name: str = None) -> logging.Logger:
    """Setup root logger and uvicorn loggers with console and file handlers, returning the named logger."""
    
    root = logging.getLogger()
    
    # Avoid duplicate handlers if already configured
    if not getattr(root, '_custom_configured', False):
        root.setLevel(logging.DEBUG)

        # Resolve logs relative to the project root so uvicorn/script cwd does not matter.
        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "accordis.log"
        
        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        # Timed Rotating File Handler (rotates daily, keeps 30 days)
        fh = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # Attach to root
        root.handlers.clear()
        root.addHandler(ch)
        root.addHandler(fh)
        root._custom_configured = True
        
        # Attach to uvicorn/fastapi/openenv to ensure they use our formatting and don't double log
        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "openenv", "fastapi"]:
            l = logging.getLogger(logger_name)
            l.handlers.clear()
            l.addHandler(ch)
            l.addHandler(fh)
            l.propagate = False
            
    return logging.getLogger(name)


logger = setup_logger(__name__)
