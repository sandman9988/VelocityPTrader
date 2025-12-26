#!/usr/bin/env python3
"""
Quick System Test - Verify all components work
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "core"))

def test_symbol_manager():
    """Test symbol manager"""
    print("📊 Testing Symbol Manager...")
    try:
        from core.symbol_manager import ProductionSymbolManager
        manager = ProductionSymbolManager()
        status = manager.get_system_status()
        print(f"   ✅ {status['total_instruments']} instruments loaded")
        print(f"   ✅ {status['ecn_instruments']} ECN instruments")
        print(f"   ✅ {status['tradeable_pairs']} tradeable pairs")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_data_pipeline():
    """Test MT5 data pipeline"""
    print("📡 Testing MT5 Data Pipeline...")
    try:
        import asyncio
        from core.mt5_data_pipeline import ProductionMT5Pipeline
        
        async def test_pipeline():
            pipeline = ProductionMT5Pipeline(['EURUSD', 'EURUSD+', 'BTCUSD'])
            await pipeline.initialize()
            
            status = pipeline.get_status_report()
            print(f"   ✅ Active source: {status['active_source']}")
            print(f"   ✅ {status['total_sources']} data sources")
            
            # Test single update
            data = await pipeline.get_live_data()
            print(f"   ✅ {len(data)} symbols with live data")
            
            for symbol, tick in list(data.items())[:2]:
                print(f"      {symbol}: {tick.bid:.5f}/{tick.ask:.5f}")
            
            await pipeline.stop()
            return len(data) > 0
        
        return asyncio.run(test_pipeline())
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run system tests"""
    print("🧪 PRODUCTION SYSTEM COMPONENT TESTS")
    print("=" * 50)
    
    results = []
    
    # Test components
    results.append(test_symbol_manager())
    results.append(test_data_pipeline())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ ALL COMPONENTS WORKING")
        print("🚀 Ready to start: ./run_system.sh")
    else:
        print("❌ SOME COMPONENTS FAILED")
        print("Check error messages above")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)