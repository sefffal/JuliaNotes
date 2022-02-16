import time
start = time.time()
for i in range(0,1_000_000):
    i * i
end = time.time()
elapsed = (end - start)

instrs = 3.5e9 * elapsed/1_000_000

print(f"Est. instructions per Python multiply: {instrs}")
