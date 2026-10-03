"""Numerical checks for short/non-aligned CUDA token batches and packed embedding."""
import torch
from mixed_int8 import MixedInt8Linear, Int8Embedding

def main():
    torch.manual_seed(12)
    for dtype in (torch.float32,torch.bfloat16,torch.float16):
        m=MixedInt8Linear(32,24,True,dtype).cuda()
        q=torch.randint(-127,128,(32,24),dtype=torch.int8,device='cuda')
        m.weight_q_t.copy_(q);m.weight_scale.fill_(.003);m.weight_sum.copy_(q.sum(0,dtype=torch.int32))
        m.activation_scale.fill_(.025);m.activation_zero.fill_(-7);m.bias.fill_(.02)
        for rows in (1,7,8,17,50):
            x=torch.randn(1,rows,32,device='cuda',dtype=dtype)
            iq=(x.float()/.025).round().add(-7).clamp(-128,127)
            ref=(((iq+7)@q.float())*(m.activation_scale*m.weight_scale)+m.bias).to(dtype)
            torch.testing.assert_close(m(x),ref,rtol=0,atol=0)
        e=Int8Embedding(100,32,dtype).cuda()
        e.weight_q.random_(-127,128);e.weight_scale.fill_(.003)
        ids=torch.tensor([[1,2,1,0]],device='cuda')
        ref=(e.weight_q.float()*e.weight_scale[:,None])[ids].to(dtype)
        torch.testing.assert_close(e(ids),ref,rtol=0,atol=0)
    print('PASS: 15 CUDA integer GEMM cases + 3 embedding dtype cases')

if __name__=='__main__':main()
