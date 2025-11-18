# test_job.py
import time

print("Job started!")

# Simulate some work
for i in range(5):
    print(f"Step {i+1}/5")
    time.sleep(1)  # wait 1 second

print("Job finished successfully!")