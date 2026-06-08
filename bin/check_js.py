import re
html = open(r'c:\Users\Autumnsentiment\Documents\YOLO26n-VRC-Mocap\bin\static\index.html', encoding='utf-8').read()
onclicks = re.findall(r'onclick="(\w+)\(', html)
funcs = re.findall(r'function\s+(\w+)\s*\(', html)
missing = [f for f in set(onclicks) if f not in set(funcs)]
if missing:
    print('MISSING functions:', missing)
else:
    print(f'All {len(set(onclicks))} onclick functions are defined')
    print('Functions:', sorted(set(onclicks)))
