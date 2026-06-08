"""OSC 抓包测试 - 解析完整的 OSC 消息内容"""
import socket, time, struct, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9002
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 6

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.settimeout(2.0)
sock.bind(('0.0.0.0', PORT))

def parse_osc_message(data):
    """解析 OSC 消息，返回 (地址, 类型标签, 值列表)"""
    # 解析地址
    addr_end = data.find(b'\x00')
    if addr_end < 0:
        return None, None, None
    address = data[:addr_end].decode('utf-8', errors='replace')

    # 跳过地址的 padding
    pos = addr_end + 1
    while pos < len(data) and pos % 4 != 0:
        pos += 1

    # 解析类型标签
    if pos >= len(data) or data[pos] != ord(','):
        return address, "", []
    type_start = pos
    type_end = data.find(b'\x00', type_start)
    if type_end < 0:
        return address, "", []
    type_tags = data[type_start+1:type_end].decode('utf-8', errors='replace')

    # 跳过类型标签的 padding
    pos = type_end + 1
    while pos < len(data) and pos % 4 != 0:
        pos += 1

    # 解析值
    values = []
    for tag in type_tags:
        if tag == 'f' and pos + 4 <= len(data):
            val = struct.unpack('!f', data[pos:pos+4])[0]
            values.append(round(val, 4))
            pos += 4
        elif tag == 'i' and pos + 4 <= len(data):
            val = struct.unpack('!i', data[pos:pos+4])[0]
            values.append(val)
            pos += 4
        elif tag == 's':
            s_end = data.find(b'\x00', pos)
            if s_end >= 0:
                values.append(data[pos:s_end].decode('utf-8', errors='replace'))
                pos = s_end + 1
                while pos % 4 != 0:
                    pos += 1
            else:
                values.append("?")
                break
        else:
            values.append(f"?({tag})")
            break

    return address, type_tags, values

print(f'Listening on 0.0.0.0:{PORT} for {DURATION}s...')
count = 0
tracker_msgs = 0
param_msgs = 0
other_msgs = 0
start = time.time()

try:
    while time.time() - start < DURATION:
        try:
            data, addr = sock.recvfrom(4096)
            count += 1
            address, type_tags, values = parse_osc_message(data)
            if address is None:
                continue

            if '/tracking/trackers/' in address:
                tracker_msgs += 1
                if tracker_msgs <= 25:
                    print(f'  TRACKER [{tracker_msgs}]: {address} ,{type_tags} -> {values}')
            elif '/avatar/parameters/' in address:
                param_msgs += 1
                if param_msgs <= 5:
                    print(f'  PARAM  [{param_msgs}]: {address} ,{type_tags} -> {values}')
            else:
                other_msgs += 1
                if other_msgs <= 5:
                    print(f'  OTHER  [{other_msgs}]: {address} ,{type_tags} -> {values}')

        except socket.timeout:
            if count == 0:
                print('  No data yet...')
            continue
except KeyboardInterrupt:
    pass

sock.close()
print(f'\n=== Summary ===')
print(f'Total packets: {count}')
print(f'  Tracker messages: {tracker_msgs}')
print(f'  Param messages: {param_msgs}')
print(f'  Other messages: {other_msgs}')
