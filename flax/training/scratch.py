import checkpoints
import time
from tqdm import tqdm

state=None
save_times=80 # save every ~ 10 epochs
compute=5 # ~ 10 epochs of compute on v3-2048 it should be "4.6875" but ~ should be good enough to size the problem. 
readable_bucket_path="gs://flax_public/examples/imagenet/tpu_v3_32"

writeable_bucket="gsoc_bucket_eu4"
writable_path=f"gs://{writeable_bucket}/perf"

print("Reading parallel GCS","="*30)
state=checkpoints.restore_checkpoint(readable_bucket_path,state,parallel=True)


print("Starting blocking writes to bucket")

total_write=0

# # Writing in blocking mode
# tic_total = time.time()

# for step in tqdm(range(save_times)):
#     tic_write = time.time()
#     checkpoints.save_checkpoint(writable_path,state,step=step,keep=3)
#     toc_write = time.time()
#     total_write += (toc_write-tic_write)
#     # simulating a compute load of 5 secs because otherwise the writes will be get caught in racing condition. 
#     # and sometimes they do anyway because of the non deterministic nature of the GCS access time (which in its own a tell on how slow GCS can be!)
#     time.sleep(compute)
# toc_total = time.tic = time.time()
# blocking_total = toc_total - tic_total
# print("Total time on blocked GCS write:",blocking_total)
# print("Total write time on blocked GCS write:",total_write)

# =======================================
blocked_write=170.1
blocking_total=blocked_write+(compute*80)
blocked_write=blocked_write/16
blocking_total=blocking_total/16
print("Starting non-blocking writes to bucket")

blocked_write=total_write


total_write=0
ts=[]

# Writing in non-blocking mode

tic_total = time.time()
# for step in tqdm(range(save_times,save_times*2)):
for step in tqdm(range(5)):
    tic_write = time.time()
    thread=checkpoints.save_checkpoint(writable_path,state,step=step,keep=3,blocking=False)
    toc_write = time.time()
    total_write += (toc_write-tic_write)
    # simulating a compute load of 5 secs same as above.
    ts.append(thread)
    time.sleep(compute)
    
    


print("Threaded on GCS before join:",total_write)

tsr=[t.result() for t in ts]

toc_total= time.time()
non_blocking_total = toc_total - tic_total
non_bocked_write=total_write
print("Time on non-blocked GCS write",non_blocking_total)
print(f"Total speedup ={blocking_total/non_blocking_total}X")
print(f"Total write speedup ={blocked_write/non_bocked_write}X")
