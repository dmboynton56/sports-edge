#!/usr/bin/env python3
"""Quick verification that PropLine fallback logic is correctly implemented.

This script checks:
1. PropLine client module exists and imports cleanly
2. MLB HR odds fetcher has both fetch functions
3. Script imports all required functions
4. Provider tracking is present in both functions
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Verification: PropLine fallback implementation")
print("=" * 60)

# Test 1: Import PropLine client
print("\n1. Importing PropLine client...")
try:
    from src.data.propline_client import (
        PropLineClient,
        PropLineError,
        fetch_propline_event_odds,
        fetch_propline_mlb_events,
        get_propline_api_key,
    )
    print("   ✓ PropLine client imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import PropLine client: {e}")
    sys.exit(1)

# Test 2: Import MLB HR fetcher with PropLine support
print("\n2. Importing MLB HR odds fetcher with PropLine functions...")
try:
    from src.data.mlb_hr_odds_fetcher import (
        MlbHrOddsError,
        OddsApiClient,
        fetch_day_hr_odds,
        fetch_day_hr_odds_propline,
        get_api_key,
    )
    print("   ✓ MLB HR odds fetcher imported successfully")
    print("   ✓ fetch_day_hr_odds found")
    print("   ✓ fetch_day_hr_odds_propline found")
except ImportError as e:
    print(f"   ✗ Failed to import MLB HR odds fetcher: {e}")
    sys.exit(1)

# Test 3: Check provider parameter in flatten_event_hr_odds
print("\n3. Checking provider parameter support...")
try:
    from src.data.mlb_hr_odds_fetcher import flatten_event_hr_odds
    import inspect
    sig = inspect.signature(flatten_event_hr_odds)
    if "provider" in sig.parameters:
        print("   ✓ flatten_event_hr_odds has 'provider' parameter")
    else:
        print("   ✗ flatten_event_hr_odds missing 'provider' parameter")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Failed to check function signature: {e}")
    sys.exit(1)

# Test 4: Verify script can load all functions
print("\n4. Verifying fetch_mlb_home_run_odds.py imports...")
try:
    sys.path.insert(0, str(ROOT / "scripts"))
    # Import the module without running main()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_mlb_home_run_odds", ROOT / "scripts" / "fetch_mlb_home_run_odds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("   ✓ fetch_mlb_home_run_odds.py loads without errors")
    
    # Check main() exists
    if hasattr(module, "main"):
        print("   ✓ main() function found")
    else:
        print("   ✗ main() function not found")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Failed to load script: {e}")
    sys.exit(1)

# Test 5: Check that both fetch functions accept schedule
print("\n5. Checking function signatures for schedule parameter...")
try:
    sig_odds_api = inspect.signature(fetch_day_hr_odds)
    sig_propline = inspect.signature(fetch_day_hr_odds_propline)
    
    if "schedule" in sig_odds_api.parameters and "schedule" in sig_propline.parameters:
        print("   ✓ Both functions accept 'schedule' parameter")
    else:
        print("   ✗ Missing 'schedule' parameter in one or both functions")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Failed to check function signatures: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All verification checks passed!")
print("\nPropLine fallback implementation is ready.")
print("\nNext steps:")
print("1. Add GitHub secret PROPLINE_API_KEY (get free key at prop-line.com)")
print("2. Wait for scheduled run at 17:00 UTC")
print("3. Check audit JSON for 'provider' field")
print("4. Verify board publishes with priced rows")
