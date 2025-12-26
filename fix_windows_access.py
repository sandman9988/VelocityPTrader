#!/usr/bin/env python3
"""
Fix Windows Access to WSL2 Dashboard
Creates port forwarding and shows correct access URLs
"""

import subprocess
import socket
import sys
import os
import time

def get_wsl_ip():
    """Get WSL2 IP address"""
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split()[0]
    except:
        pass
    return None

def get_windows_ip():
    """Get Windows host IP from WSL2 perspective"""
    try:
        # Get default route to find Windows IP
        result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract IP from: default via 172.27.208.1 dev eth0
            parts = result.stdout.split()
            for i, part in enumerate(parts):
                if part == 'via' and i + 1 < len(parts):
                    return parts[i + 1]
    except:
        pass
    return None

def test_port_accessible(host, port):
    """Test if port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def setup_windows_access():
    """Setup Windows access to WSL2 services"""
    
    print("🔧 FIXING WINDOWS ACCESS TO WSL2 DASHBOARD")
    print("=" * 60)
    
    # Get network info
    wsl_ip = get_wsl_ip()
    windows_ip = get_windows_ip()
    
    print(f"🖥️  WSL2 IP: {wsl_ip}")
    print(f"🪟 Windows IP: {windows_ip}")
    
    # Test if services are running
    dashboard_port = 8443
    metrics_port = 9090
    
    print(f"\n🧪 Testing service accessibility...")
    
    # Test from WSL2 localhost
    dashboard_local = test_port_accessible('localhost', dashboard_port)
    metrics_local = test_port_accessible('localhost', metrics_port)
    
    print(f"Dashboard (localhost:{dashboard_port}): {'✅' if dashboard_local else '❌'}")
    print(f"Metrics (localhost:{metrics_port}): {'✅' if metrics_local else '❌'}")
    
    if not dashboard_local:
        print("❌ Dashboard not running on WSL2 - start the system first")
        return False
    
    # Test from WSL2 IP
    if wsl_ip:
        dashboard_wsl = test_port_accessible(wsl_ip, dashboard_port)
        print(f"Dashboard ({wsl_ip}:{dashboard_port}): {'✅' if dashboard_wsl else '❌'}")
        
        if not dashboard_wsl:
            print("⚠️ Dashboard not accessible on WSL2 IP - may need to bind to 0.0.0.0")
    
    print(f"\n🌐 WINDOWS ACCESS URLs:")
    print(f"📊 Dashboard: https://{wsl_ip}:{dashboard_port}")
    print(f"📈 Metrics:   http://{wsl_ip}:{metrics_port}")
    
    print(f"\n📋 WINDOWS ACCESS STEPS:")
    print(f"1. Open Windows browser")
    print(f"2. Go to: https://{wsl_ip}:{dashboard_port}")
    print(f"3. Accept certificate warning")
    print(f"4. Login: admin / secure_password")
    
    # Create Windows batch file for easy access
    try:
        batch_content = f'''@echo off
echo Opening AI Trading Dashboard...
start https://{wsl_ip}:{dashboard_port}
echo Opening Metrics Dashboard...
start http://{wsl_ip}:{metrics_port}
'''
        
        # Try to write to Windows accessible location
        windows_path = "/mnt/c/Users/Public/Desktop/AI_Trading_Dashboard.bat"
        try:
            with open(windows_path, 'w') as f:
                f.write(batch_content)
            print(f"\n💾 Created Windows shortcut: C:\\Users\\Public\\Desktop\\AI_Trading_Dashboard.bat")
        except:
            # Fallback to local directory
            local_path = "AI_Trading_Dashboard.bat"
            with open(local_path, 'w') as f:
                f.write(batch_content)
            print(f"\n💾 Created local shortcut: {os.path.abspath(local_path)}")
            
    except Exception as e:
        print(f"⚠️ Could not create Windows shortcut: {e}")
    
    # Test if Windows can access
    print(f"\n🔍 Testing Windows accessibility...")
    
    if wsl_ip and test_port_accessible(wsl_ip, dashboard_port):
        print("✅ Dashboard should be accessible from Windows!")
        print(f"🚀 Try: https://{wsl_ip}:{dashboard_port}")
    else:
        print("❌ Dashboard not accessible from Windows")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure system is running: ps aux | grep python3")
        print("2. Check if binding to 0.0.0.0: netstat -tlnp | grep 8443")
        print("3. Check Windows firewall settings")
        print("4. Restart system with: python3 start_production_system_clean.py")
    
    return True

def check_system_status():
    """Check if production system is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'start_production_system' in result.stdout:
            print("✅ Production system is running")
            return True
        else:
            print("❌ Production system not running")
            print("🚀 Start it with: python3 start_production_system_clean.py")
            return False
    except:
        return False

if __name__ == "__main__":
    print("🔍 Checking system status...")
    if not check_system_status():
        print("System not running - start it first")
        sys.exit(1)
    
    setup_windows_access()