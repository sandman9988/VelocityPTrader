#!/usr/bin/env python3
"""
COMPREHENSIVE TEST FRAMEWORK
Ensures no regression throughout phased development
"""

import unittest
import time
import json
import numpy as np
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, List
import sys
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """Test result container"""
    phase: str
    test_name: str
    passed: bool
    duration: float
    error_message: str = ""
    performance_metrics: Dict[str, Any] = None

class TestFramework:
    """Comprehensive testing framework"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.baseline_metrics = {}
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup isolated test environment"""
        self.test_dir = Path("test_environment")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test data
        self.create_test_data()
        
        print("🧪 Test Framework initialized")
        print(f"   📁 Test directory: {self.test_dir}")
        print(f"   🔬 Test data created")
    
    def create_test_data(self):
        """Create realistic test data"""
        test_symbols = {
            "EURUSD+": {
                "bid": 1.17756,
                "ask": 1.17769,
                "digits": 5,
                "spread_pips": 1.3,
                "category": "FOREX"
            },
            "BTCUSD+": {
                "bid": 95120.50,
                "ask": 95122.50,
                "digits": 2,
                "spread_pips": 20.0,
                "category": "CRYPTO"
            },
            "SPX500+": {
                "bid": 5890.25,
                "ask": 5890.75,
                "digits": 2,
                "spread_pips": 5.0,
                "category": "INDEX"
            }
        }
        
        test_data = {
            "success": True,
            "account": {
                "server": "VantageInternational-Demo",
                "login": 10916362,
                "balance": 1007.86
            },
            "symbols": test_symbols
        }
        
        # Save test data
        test_file = self.test_dir / "test_mt5_symbols.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
    
    def run_test(self, phase: str, test_name: str, test_function, *args, **kwargs) -> TestResult:
        """Run individual test with metrics collection"""
        print(f"\n🔬 Running {phase} - {test_name}")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent(interval=None)
        
        try:
            # Run the test
            result = test_function(*args, **kwargs)
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = psutil.virtual_memory().used
            end_cpu = psutil.cpu_percent(interval=None)
            
            performance_metrics = {
                "duration_seconds": duration,
                "memory_delta_mb": (end_memory - start_memory) / 1024 / 1024,
                "cpu_usage_percent": end_cpu,
                "timestamp": datetime.now().isoformat()
            }
            
            test_result = TestResult(
                phase=phase,
                test_name=test_name,
                passed=True,
                duration=duration,
                performance_metrics=performance_metrics
            )
            
            print(f"   ✅ PASSED ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                phase=phase,
                test_name=test_name,
                passed=False,
                duration=duration,
                error_message=str(e)
            )
            
            print(f"   ❌ FAILED: {e}")
        
        self.results.append(test_result)
        return test_result
    
    def establish_baseline(self, phase: str):
        """Establish performance baseline for phase"""
        print(f"\n📊 Establishing baseline for {phase}")
        
        # System metrics
        self.baseline_metrics[phase] = {
            "cpu_cores": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "disk_free_gb": psutil.disk_usage('/').free / 1024**3,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"   💾 Memory: {self.baseline_metrics[phase]['memory_available_gb']:.1f}GB available")
        print(f"   🔥 CPU: {self.baseline_metrics[phase]['cpu_cores']} cores")
        print(f"   💽 Disk: {self.baseline_metrics[phase]['disk_free_gb']:.1f}GB free")
    
    def check_regression(self, phase: str, current_metrics: Dict[str, Any]) -> bool:
        """Check for performance regression"""
        if phase not in self.baseline_metrics:
            return True  # No baseline to compare
        
        baseline = self.baseline_metrics[phase]
        
        # Check memory regression (allow 10% increase)
        memory_regression = (
            current_metrics.get("memory_delta_mb", 0) > 
            baseline.get("memory_available_gb", 0) * 1024 * 0.1
        )
        
        if memory_regression:
            print(f"⚠️ Memory regression detected in {phase}")
            return False
        
        return True
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report = f"""
# TEST EXECUTION REPORT
Generated: {datetime.now().isoformat()}

## SUMMARY
- Total Tests: {total_tests}
- Passed: {passed_tests}
- Failed: {total_tests - passed_tests}
- Pass Rate: {pass_rate:.1%}

## PHASE RESULTS
"""
        
        phases = {}
        for result in self.results:
            if result.phase not in phases:
                phases[result.phase] = {'passed': 0, 'failed': 0, 'tests': []}
            
            if result.passed:
                phases[result.phase]['passed'] += 1
            else:
                phases[result.phase]['failed'] += 1
            
            phases[result.phase]['tests'].append(result)
        
        for phase, data in phases.items():
            total = data['passed'] + data['failed']
            rate = data['passed'] / total if total > 0 else 0
            
            report += f"\n### {phase}\n"
            report += f"- Tests: {total} ({rate:.1%} pass rate)\n"
            report += f"- Passed: {data['passed']}\n"
            report += f"- Failed: {data['failed']}\n"
            
            # Failed tests details
            if data['failed'] > 0:
                report += "\n**Failed Tests:**\n"
                for test in data['tests']:
                    if not test.passed:
                        report += f"- {test.test_name}: {test.error_message}\n"
        
        report += "\n## PERFORMANCE METRICS\n"
        for phase, metrics in self.baseline_metrics.items():
            report += f"\n### {phase} Baseline\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    report += f"- {key}: {value:.2f}\n"
                else:
                    report += f"- {key}: {value}\n"
        
        return report
    
    def save_test_report(self):
        """Save test report to file"""
        report = self.generate_test_report()
        
        report_file = self.test_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Test report saved: {report_file}")
        return report_file

# PHASE 1 TESTS: Hardware Validation
class Phase1HardwareTests(unittest.TestCase):
    """Phase 1: Hardware validation tests"""
    
    def setUp(self):
        self.framework = TestFramework()
    
    def test_cpu_detection(self):
        """Test CPU detection and core count"""
        cpu_count = psutil.cpu_count()
        physical_cores = psutil.cpu_count(logical=False)
        
        self.assertGreaterEqual(cpu_count, 16, "Should detect 16+ logical cores")
        self.assertGreaterEqual(physical_cores, 8, "Should detect 8+ physical cores")
        
        print(f"   💻 Detected: {cpu_count} logical cores, {physical_cores} physical")
        return {"logical_cores": cpu_count, "physical_cores": physical_cores}
    
    def test_memory_detection(self):
        """Test memory detection and availability"""
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
        
        self.assertGreaterEqual(total_gb, 32, "Should have 32GB+ total memory")
        self.assertGreaterEqual(available_gb, 16, "Should have 16GB+ available memory")
        
        print(f"   💾 Memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        return {"total_gb": total_gb, "available_gb": available_gb}
    
    def test_disk_performance(self):
        """Test NVMe SSD performance"""
        test_file = Path("test_disk_performance.bin")
        
        # Write test (1MB)
        test_data = b"0" * 1024 * 1024
        start_time = time.time()
        with open(test_file, 'wb') as f:
            f.write(test_data)
            f.flush()
            os.fsync(f.fileno())
        write_time = time.time() - start_time
        
        # Read test
        start_time = time.time()
        with open(test_file, 'rb') as f:
            read_data = f.read()
        read_time = time.time() - start_time
        
        # Cleanup
        test_file.unlink()
        
        write_mbps = 1 / write_time  # MB/s
        read_mbps = 1 / read_time    # MB/s
        
        self.assertGreater(write_mbps, 100, "Write speed should be > 100 MB/s")
        self.assertGreater(read_mbps, 500, "Read speed should be > 500 MB/s")
        
        print(f"   💽 Disk: {write_mbps:.0f} MB/s write, {read_mbps:.0f} MB/s read")
        return {"write_mbps": write_mbps, "read_mbps": read_mbps}
    
    def test_gpu_detection(self):
        """Test GPU detection and capabilities"""
        gpu_available = False
        gpu_info = {}
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    "compute_capability": torch.cuda.get_device_properties(0).major
                }
                print(f"   🎯 GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
        except ImportError:
            try:
                import pyopencl as cl
                platforms = cl.get_platforms()
                for platform in platforms:
                    if 'AMD' in platform.name or 'NVIDIA' in platform.name:
                        devices = platform.get_devices()
                        if devices:
                            gpu_available = True
                            gpu_info = {"name": devices[0].name, "platform": platform.name}
                            print(f"   🎯 GPU: {gpu_info['name']} via OpenCL")
                            break
            except ImportError:
                pass
        
        if not gpu_available:
            print("   ⚠️ No GPU acceleration detected")
        
        return {"gpu_available": gpu_available, "gpu_info": gpu_info}
    
    def test_python_performance(self):
        """Test Python performance and optimizations"""
        # NumPy performance test
        start_time = time.time()
        large_array = np.random.random((1000, 1000))
        result = np.dot(large_array, large_array.T)
        numpy_time = time.time() - start_time
        
        # Thread performance test
        def cpu_intensive_task():
            return sum(range(100000))
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(32)]
            results = [f.result() for f in futures]
        threading_time = time.time() - start_time
        
        self.assertLess(numpy_time, 1.0, "NumPy operations should be fast")
        self.assertLess(threading_time, 2.0, "Threading should be efficient")
        
        print(f"   🐍 NumPy: {numpy_time:.3f}s, Threading: {threading_time:.3f}s")
        return {"numpy_time": numpy_time, "threading_time": threading_time}

# Phase 1 Test Runner
def run_phase1_tests():
    """Run Phase 1 hardware validation tests"""
    framework = TestFramework()
    framework.establish_baseline("PHASE1_HARDWARE")
    
    print("\n" + "="*60)
    print("🔬 PHASE 1: HARDWARE VALIDATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Phase1HardwareTests)
    
    # Custom test runner that captures our metrics
    class CustomTestRunner:
        def __init__(self, framework):
            self.framework = framework
            self.results = []
        
        def run(self, test_suite):
            for test in test_suite:
                test_name = test._testMethodName
                
                try:
                    test.setUp()
                    test_method = getattr(test, test_name)
                    
                    result = self.framework.run_test(
                        "PHASE1_HARDWARE",
                        test_name,
                        test_method
                    )
                    self.results.append(result)
                    
                except Exception as e:
                    result = TestResult(
                        phase="PHASE1_HARDWARE",
                        test_name=test_name,
                        passed=False,
                        duration=0,
                        error_message=str(e)
                    )
                    self.results.append(result)
                    self.framework.results.append(result)
    
    # Run tests
    runner = CustomTestRunner(framework)
    runner.run(suite)
    
    # Generate report
    framework.save_test_report()
    
    # Check if all tests passed
    all_passed = all(r.passed for r in framework.results)
    
    if all_passed:
        print("\n✅ PHASE 1 COMPLETE - All hardware tests passed")
        print("   Ready to proceed to Phase 2: Core Data Pipeline")
    else:
        print("\n❌ PHASE 1 FAILED - Fix hardware issues before proceeding")
        failed_tests = [r for r in framework.results if not r.passed]
        for test in failed_tests:
            print(f"   - {test.test_name}: {test.error_message}")
    
    return all_passed

if __name__ == "__main__":
    # Import required modules
    try:
        from concurrent.futures import ThreadPoolExecutor
        import torch
        import pyopencl
    except ImportError as e:
        print(f"⚠️ Missing dependencies: {e}")
        print("Install with: pip install torch pyopencl")
    
    success = run_phase1_tests()
    sys.exit(0 if success else 1)