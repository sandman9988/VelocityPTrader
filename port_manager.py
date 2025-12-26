#!/usr/bin/env python3
"""
Production Port Manager
Handles proper port cleanup and reuse for enterprise applications
"""

import os
import sys
import time
import signal
import socket
import psutil
import subprocess
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortManager:
    """Production-grade port management with process cleanup"""
    
    def __init__(self, pid_dir: str = "pids"):
        self.pid_dir = Path(pid_dir)
        self.pid_dir.mkdir(exist_ok=True)
        
        # Standard ports for our system
        self.system_ports = {
            'dashboard': 8443,
            'dashboard_alt': 8444, 
            'metrics': 9090,
            'redis': 6379,
            'websocket': 8080
        }
        
        # PID files for tracking
        self.pid_files = {
            'dashboard': self.pid_dir / "dashboard.pid",
            'production_system': self.pid_dir / "production_system.pid",
            'metrics_server': self.pid_dir / "metrics.pid"
        }
    
    def find_processes_on_port(self, port: int) -> List[Dict]:
        """Find all processes using a specific port"""
        processes = []
        
        try:
            # Method 1: Using psutil (cross-platform)
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr and conn.laddr.port == port:
                    try:
                        process = psutil.Process(conn.pid)
                        processes.append({
                            'pid': conn.pid,
                            'name': process.name(),
                            'cmdline': ' '.join(process.cmdline()),
                            'status': process.status(),
                            'port': port,
                            'connection': conn
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
        except Exception as e:
            logger.warning(f"psutil method failed: {e}")
            
            # Method 2: Fallback to netstat
            try:
                result = subprocess.run(
                    ['netstat', '-tlnp'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                for line in result.stdout.split('\n'):
                    if f':{port} ' in line and 'LISTEN' in line:
                        parts = line.split()
                        if len(parts) >= 7:
                            pid_program = parts[-1]
                            if '/' in pid_program and pid_program != '-':
                                pid = pid_program.split('/')[0]
                                if pid.isdigit():
                                    processes.append({
                                        'pid': int(pid),
                                        'name': pid_program.split('/')[-1],
                                        'cmdline': f'netstat detection on port {port}',
                                        'status': 'running',
                                        'port': port,
                                        'connection': None
                                    })
            except Exception as e:
                logger.warning(f"netstat fallback failed: {e}")
        
        return processes
    
    def kill_port(self, port: int, force: bool = False) -> bool:
        """Kill all processes using a specific port"""
        logger.info(f"Cleaning up port {port}...")
        
        processes = self.find_processes_on_port(port)
        
        if not processes:
            logger.info(f"Port {port} is already free")
            return True
        
        killed_count = 0
        
        for proc_info in processes:
            pid = proc_info['pid']
            name = proc_info['name']
            
            logger.info(f"Found process on port {port}: PID {pid} ({name})")
            
            try:
                process = psutil.Process(pid)
                
                # First try graceful termination
                if not force:
                    logger.info(f"Sending SIGTERM to PID {pid}")
                    process.terminate()
                    
                    # Wait up to 5 seconds for graceful shutdown
                    try:
                        process.wait(timeout=5)
                        logger.info(f"Process {pid} terminated gracefully")
                        killed_count += 1
                        continue
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {pid} did not terminate gracefully")
                
                # Force kill if graceful failed or force=True
                logger.warning(f"Force killing PID {pid}")
                process.kill()
                
                # Verify it's dead
                try:
                    process.wait(timeout=2)
                    logger.info(f"Process {pid} force killed")
                    killed_count += 1
                except psutil.TimeoutExpired:
                    logger.error(f"Failed to kill process {pid}")
                    
            except psutil.NoSuchProcess:
                logger.info(f"Process {pid} already terminated")
                killed_count += 1
            except psutil.AccessDenied:
                logger.error(f"Access denied to kill process {pid} (try with sudo)")
            except Exception as e:
                logger.error(f"Error killing process {pid}: {e}")
        
        # Verify port is now free
        time.sleep(1)  # Brief pause
        remaining_processes = self.find_processes_on_port(port)
        
        if not remaining_processes:
            logger.info(f"✅ Port {port} successfully cleaned up")
            return True
        else:
            logger.error(f"❌ Port {port} still has {len(remaining_processes)} processes")
            return False
    
    def is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0  # 0 means connection successful (port occupied)
        except Exception:
            return True  # Assume free if we can't check
    
    def find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        """Find a free port in the given range"""
        for port in range(start_port, end_port):
            if self.is_port_free(port):
                return port
        raise Exception(f"No free ports found in range {start_port}-{end_port}")
    
    def save_pid(self, service: str, pid: int):
        """Save PID to file for tracking"""
        pid_file = self.pid_files.get(service)
        if pid_file:
            pid_file.write_text(str(pid))
            logger.info(f"Saved PID {pid} for service '{service}'")
    
    def load_pid(self, service: str) -> Optional[int]:
        """Load PID from file"""
        pid_file = self.pid_files.get(service)
        if pid_file and pid_file.exists():
            try:
                return int(pid_file.read_text().strip())
            except (ValueError, FileNotFoundError):
                pass
        return None
    
    def kill_service_by_name(self, service: str) -> bool:
        """Kill service by name (using saved PID)"""
        pid = self.load_pid(service)
        if pid:
            try:
                process = psutil.Process(pid)
                logger.info(f"Killing service '{service}' (PID {pid})")
                process.terminate()
                process.wait(timeout=5)
                
                # Remove PID file
                pid_file = self.pid_files.get(service)
                if pid_file and pid_file.exists():
                    pid_file.unlink()
                    
                logger.info(f"✅ Service '{service}' stopped successfully")
                return True
                
            except (psutil.NoSuchProcess, psutil.TimeoutExpired, FileNotFoundError):
                logger.warning(f"Service '{service}' PID {pid} not found or already dead")
                # Clean up stale PID file
                pid_file = self.pid_files.get(service)
                if pid_file and pid_file.exists():
                    pid_file.unlink()
                return True
            except Exception as e:
                logger.error(f"Error killing service '{service}': {e}")
                return False
        else:
            logger.warning(f"No PID found for service '{service}'")
            return True  # Consider it "stopped" if no PID
    
    def cleanup_all_system_ports(self) -> Dict[int, bool]:
        """Clean up all known system ports"""
        logger.info("🧹 Cleaning up all system ports...")
        results = {}
        
        for name, port in self.system_ports.items():
            logger.info(f"Cleaning {name} port {port}...")
            results[port] = self.kill_port(port)
        
        return results
    
    def cleanup_system_services(self) -> Dict[str, bool]:
        """Clean up all known system services"""
        logger.info("🛑 Stopping all system services...")
        results = {}
        
        for service in self.pid_files.keys():
            results[service] = self.kill_service_by_name(service)
        
        return results
    
    def get_port_status(self) -> Dict[str, Dict]:
        """Get status of all system ports"""
        status = {}
        
        for name, port in self.system_ports.items():
            processes = self.find_processes_on_port(port)
            status[name] = {
                'port': port,
                'free': len(processes) == 0,
                'processes': processes
            }
        
        return status
    
    def force_cleanup_python_processes(self):
        """Force cleanup of any hanging Python processes from our system"""
        logger.info("🔥 Force cleaning Python processes...")
        
        python_processes = []
        keywords = ['start_production_system', 'enterprise_dashboard', 'mt5_data_pipeline']
        
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if process.info['name'] in ['python', 'python3']:
                    cmdline = ' '.join(process.info['cmdline'])
                    if any(keyword in cmdline for keyword in keywords):
                        python_processes.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        for process in python_processes:
            try:
                logger.info(f"Force killing Python process: PID {process.pid}")
                process.kill()
                process.wait(timeout=3)
            except Exception as e:
                logger.warning(f"Could not kill Python process {process.pid}: {e}")
        
        if python_processes:
            logger.info(f"✅ Cleaned up {len(python_processes)} Python processes")
        else:
            logger.info("No hanging Python processes found")

def main():
    """Main port management interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Port Manager")
    parser.add_argument('--port', '-p', type=int, help="Specific port to clean")
    parser.add_argument('--force', '-f', action='store_true', help="Force kill (no graceful shutdown)")
    parser.add_argument('--all', '-a', action='store_true', help="Clean all system ports")
    parser.add_argument('--services', '-s', action='store_true', help="Stop all services")
    parser.add_argument('--status', action='store_true', help="Show port status")
    parser.add_argument('--python', action='store_true', help="Force kill hanging Python processes")
    
    args = parser.parse_args()
    
    port_manager = PortManager()
    
    if args.status:
        print("📊 PORT STATUS:")
        print("=" * 50)
        status = port_manager.get_port_status()
        for name, info in status.items():
            port = info['port']
            free = "🟢 FREE" if info['free'] else f"🔴 OCCUPIED ({len(info['processes'])} processes)"
            print(f"{name:15} | Port {port:5} | {free}")
            
            if not info['free']:
                for proc in info['processes']:
                    print(f"                  └── PID {proc['pid']} ({proc['name']})")
    
    elif args.port:
        success = port_manager.kill_port(args.port, force=args.force)
        if success:
            print(f"✅ Port {args.port} cleaned successfully")
        else:
            print(f"❌ Failed to clean port {args.port}")
            sys.exit(1)
    
    elif args.all:
        results = port_manager.cleanup_all_system_ports()
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"📊 Cleaned {success_count}/{total_count} ports successfully")
        
        if success_count == total_count:
            print("✅ All system ports cleaned")
        else:
            print("⚠️  Some ports could not be cleaned")
            for port, success in results.items():
                if not success:
                    print(f"   ❌ Port {port} failed")
    
    elif args.services:
        results = port_manager.cleanup_system_services()
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"📊 Stopped {success_count}/{total_count} services successfully")
        
        if success_count == total_count:
            print("✅ All services stopped")
        else:
            print("⚠️  Some services could not be stopped")
    
    elif args.python:
        port_manager.force_cleanup_python_processes()
        
    else:
        print("🧹 PRODUCTION PORT MANAGER")
        print("=" * 40)
        print("Usage examples:")
        print("  python3 port_manager.py --status          # Show port status")
        print("  python3 port_manager.py --all             # Clean all system ports")
        print("  python3 port_manager.py --services        # Stop all services")
        print("  python3 port_manager.py --port 8443       # Clean specific port")
        print("  python3 port_manager.py --python          # Kill hanging Python processes")
        print("  python3 port_manager.py --all --force     # Force clean all ports")
        print("")
        print("Current system ports:")
        for name, port in port_manager.system_ports.items():
            print(f"  {name}: {port}")

if __name__ == "__main__":
    main()