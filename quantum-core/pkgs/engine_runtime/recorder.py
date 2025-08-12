"""Enhanced recording with JSONL/Parquet support."""

import csv
import os
import json
from typing import Dict, List, Any, Optional

class SimpleRecorder:
    """General purpose recorder for dictionary-based logs with multiple output formats."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.rows: List[Dict[str, Any]] = []

    def log(self, row: Dict[str, Any]):
        """Log a dictionary row, cleaning values for serialization."""
        if not self.enabled:
            return
            
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, (int, float)):
                clean_row[k] = float(v)
            elif hasattr(v, 'item'):  # torch.Tensor or numpy scalar
                clean_row[k] = v.item() if hasattr(v, 'numel') and v.numel() == 1 else v.cpu().numpy().tolist()
            elif hasattr(v, 'cpu'):  # torch.Tensor
                clean_row[k] = v.cpu().numpy().tolist() 
            elif hasattr(v, 'tolist'):  # numpy array
                clean_row[k] = v.tolist()
            else:
                clean_row[k] = str(v)
        self.rows.append(clean_row)

    def dump_csv(self, path: str):
        """Dump logs to CSV file."""
        if not self.enabled or not self.rows:
            return
            
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keys = sorted(self.rows[0].keys())
        
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
            
    def dump_jsonl(self, path: str):
        """Dump logs to JSONL file."""
        if not self.enabled or not self.rows:
            return
            
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        with open(path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
                
    def dump_parquet(self, path: str):
        """Dump logs to Parquet file (requires pandas/pyarrow)."""
        if not self.enabled or not self.rows:
            return
            
        try:
            import pandas as pd
            df = pd.DataFrame(self.rows)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df.to_parquet(path, index=False)
        except ImportError:
            # Fallback to JSONL if pandas not available
            self.dump_jsonl(path.replace('.parquet', '.jsonl'))
            
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent n entries."""
        return self.rows[-n:] if len(self.rows) >= n else self.rows.copy()
        
    def clear(self):
        """Clear all logged entries."""
        self.rows.clear()