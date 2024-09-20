import warp as wp
import numpy as np

#a = wp.zeros(1024, dtype=wp.vec3, device="cuda", requires_grad=True)
#
#tape = wp.Tape()
#
##forward pass 
#with tape: 
#    wp.launch(kernel=compute1, inputs=[a,b], device="cuda")
#    wp.launch(kernel=compute2, inputs=[c, d], device="cuda")
#    wp.launch(kernel=loss, inputs=[d, l], device="cuda")
#    
#
##revser pass 
#tape.backward(l)

@wp.kernel
def double(x:wp.array(dtype=float), y:wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] * 2.0
    

x = wp.array(np.arange(3), dtype= float, requires_grad=True)
y = wp.zeros_like(x)
z = wp.zeros_like(x)

print(x)

tape = wp.Tape()
with tape:
    wp.launch(double, dim = 3, inputs=[x,y])
    wp.copy(z,y)
    
tape.backward(grads={z:wp.ones_like(x)})

print(x.grad)
print(x)
print(y)
print(z)



