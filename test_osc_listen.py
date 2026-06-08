"""OSC 监听测试 - 在端口9001监听，动捕发送到9001"""
import socket, time, struct, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.settimeout(2.0)
sock.bind(('0.0.0.0', PORT))
print(f'Listening on 0.0.0.0:{PORT} for OSC packets...')

count = 0
start = time.time()
try:
    while time.time() - start < 10:
        try:
            data, addr = sock.recvfrom(4096)
            count += 1
            if count <= 10:
                addr_end = data.find(b'\x00')
                if addr_end > 0:
                    osc_addr = data[:addr_end].decode('utf-8', errors='replace')
                    print(f'  [{count}] from {addr}: {osc_addr} ({len(data)} bytes)')
        except socket.timeout:
            if count == 0:
                print('  No data yet...')
            continue
except KeyboardInterrupt:
    pass

sock.close()
print(f'Total packets received in 10s: {count}')
