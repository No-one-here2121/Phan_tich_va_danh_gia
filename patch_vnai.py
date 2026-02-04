#!/usr/bin/env python3
import pathlib

# Path to the problematic file
file_path = pathlib.Path('/usr/local/lib/python3.10/site-packages/vnai/scope/lc_integration.py')

if file_path.exists():
    content = file_path.read_text()
    
    # Check if we need to add the import
    if 'from typing import Any' not in content:
        # Add the import at the beginning
        new_content = 'from typing import Any, Dict\n' + content
        file_path.write_text(new_content)
        print("Successfully patched vnai/scope/lc_integration.py")
    else:
        print("File already has the correct imports")
else:
    print("File not found, skipping patch")
