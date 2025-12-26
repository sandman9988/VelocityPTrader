#!/usr/bin/env python3
"""
Test dashboard connectivity
"""

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_dashboard():
    """Test if dashboard is accessible"""
    
    print("🧪 TESTING DASHBOARD CONNECTIVITY")
    print("=" * 50)
    
    base_url = "https://localhost:8443"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", verify=False, timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test main page
    try:
        response = requests.get(base_url, verify=False, timeout=5)
        print(f"Main page: {response.status_code} - {len(response.text)} bytes")
        if "Enterprise Trading Dashboard" in response.text:
            print("✅ Dashboard HTML contains expected content")
        else:
            print("⚠️ Dashboard HTML may be incomplete")
    except Exception as e:
        print(f"Main page failed: {e}")
        return False
    
    # Test metrics endpoint
    try:
        metrics_response = requests.get("http://localhost:9090/metrics", timeout=5)
        print(f"Metrics: {metrics_response.status_code} - {len(metrics_response.text)} bytes")
        if "# HELP" in metrics_response.text:
            print("✅ Metrics endpoint working")
        else:
            print("⚠️ Metrics may not be working")
    except Exception as e:
        print(f"Metrics failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 DASHBOARD ACCESS INFORMATION:")
    print(f"🌐 Dashboard URL: https://localhost:8443")
    print(f"📊 Metrics URL:  http://localhost:9090")
    print("🔑 Default login: admin / secure_password")
    print()
    print("📋 BROWSER ACCESS STEPS:")
    print("1. Open browser and go to: https://localhost:8443")
    print("2. Accept the self-signed certificate warning")
    print("3. You should see the Enterprise Trading Dashboard")
    print("4. Use login: admin / secure_password")
    
    return True

if __name__ == "__main__":
    test_dashboard()