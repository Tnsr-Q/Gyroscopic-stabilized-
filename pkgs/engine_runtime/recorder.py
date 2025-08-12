"""
Enhanced recording capabilities for engine runtime with multiple output formats.

Provides SimpleRecorder with CSV, JSONL, and Parquet output support while
maintaining backward compatibility with existing CSV functionality.
"""
import csv
import json
import os
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
import numpy as np
import torch

logger = logging.getLogger('QuantumCore')


class SimpleRecorder:
    """Enhanced general purpose recorder with multiple output format support."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.rows: List[Dict] = []
        self._metadata = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'git_commit': None,  # Could be injected
            'config_hash': None  # Could be injected
        }

    def set_metadata(self, **kwargs):
        """Set metadata that will be included in recordings."""
        self._metadata.update(kwargs)

    def log(self, row: Dict):
        """Log a dictionary row with automatic type conversion."""
        if not self.enabled:
            return
            
        clean_row = {}
        # Add timestamp
        clean_row['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Clean and convert values
        for k, v in row.items():
            if isinstance(v, (int, float, np.number)):
                clean_row[k] = float(v)
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    clean_row[k] = v.item()
                else:
                    # For larger tensors, store as numpy array for JSONL/Parquet
                    clean_row[k] = v.cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                if v.size == 1:
                    clean_row[k] = float(v.item())
                else:
                    clean_row[k] = v.tolist()
            else:
                clean_row[k] = str(v)
        
        self.rows.append(clean_row)

    def dump_csv(self, path: str):
        """Dump to CSV format (original functionality for backward compatibility)."""
        if not self.enabled or not self.rows:
            return
            
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Flatten any list values for CSV compatibility
        csv_rows = []
        for row in self.rows:
            csv_row = {}
            for k, v in row.items():
                if isinstance(v, list):
                    # Convert lists to string representation for CSV
                    csv_row[k] = str(v)
                else:
                    csv_row[k] = v
            csv_rows.append(csv_row)
        
        if csv_rows:
            keys = sorted(csv_rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(csv_rows)
            logger.info(f"Saved {len(csv_rows)} rows to CSV: {path}")

    def dump_jsonl(self, path: str):
        """Dump to JSONL format for structured event logging."""
        if not self.enabled or not self.rows:
            return
            
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        with open(path, 'w') as f:
            # Write metadata as first line
            f.write(json.dumps({'_metadata': self._metadata}) + '\n')
            
            # Write each row as a JSON line
            for row in self.rows:
                f.write(json.dumps(row) + '\n')
        
        logger.info(f"Saved {len(self.rows)} rows to JSONL: {path}")

    def dump_parquet(self, path: str):
        """Dump to Parquet format for efficient metrics storage."""
        if not self.enabled or not self.rows:
            return
            
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pandas/pyarrow not available, skipping Parquet export")
            return
            
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Convert to DataFrame, handling nested structures
        df_rows = []
        for row in self.rows:
            df_row = {}
            for k, v in row.items():
                if isinstance(v, list) and len(v) > 0:
                    # For small lists, create separate columns
                    if len(v) <= 10 and all(isinstance(x, (int, float)) for x in v):
                        for i, val in enumerate(v):
                            df_row[f"{k}_{i}"] = val
                    else:
                        # For larger/complex lists, store as string
                        df_row[k] = str(v)
                else:
                    df_row[k] = v
            df_rows.append(df_row)
        
        df = pd.DataFrame(df_rows)
        
        # Add metadata to Parquet file metadata
        metadata = {'metadata': json.dumps(self._metadata)}
        table = pa.Table.from_pandas(df)
        table = table.replace_schema_metadata(metadata)
        
        pq.write_table(table, path)
        logger.info(f"Saved {len(df)} rows to Parquet: {path}")

    def dump_all_formats(self, base_path: str):
        """Convenience method to dump to all supported formats."""
        base_dir = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        
        self.dump_csv(os.path.join(base_dir, f"{base_name}.csv"))
        self.dump_jsonl(os.path.join(base_dir, f"{base_name}.jsonl"))
        self.dump_parquet(os.path.join(base_dir, f"{base_name}.parquet"))

    def clear(self):
        """Clear all logged rows."""
        self.rows.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged data."""
        if not self.rows:
            return {'row_count': 0}
        
        summary = {
            'row_count': len(self.rows),
            'first_timestamp': self.rows[0].get('timestamp'),
            'last_timestamp': self.rows[-1].get('timestamp'),
            'columns': list(self.rows[0].keys()) if self.rows else []
        }
        
        # Basic stats for numeric columns
        numeric_cols = {}
        for row in self.rows:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != 'timestamp':
                    if k not in numeric_cols:
                        numeric_cols[k] = []
                    numeric_cols[k].append(v)
        
        summary['numeric_stats'] = {}
        for col, values in numeric_cols.items():
            if values:
                summary['numeric_stats'][col] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary