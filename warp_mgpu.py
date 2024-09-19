import warp as wp


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid]+1.0

#get all cude device 
devices = wp.get_cuda_device()
device_count = len(devices)

#number of launces 
iters = 1000

#list of arrays, one per device 

#loop over all devices 
for device in devices:
    #use a ScopedDevice to set the target device 
    with wp.ScopedDevice(device):
        #allocate array
        a = wp.zeros(1024, dtype=float)
        arrs.append(a)
        
        #launch 
        for _ in range(iters):
            wp.launch(inc, dim=a.size, inputs=[a])
            
wp.synchronize()

#print 
for i in range(device_count):
    print(f"(arrs[i].device) -> {arrs[i].numpy()}") 