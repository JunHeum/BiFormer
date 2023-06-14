import cupy
import torch
import re
from math import sqrt

'''
Apply sliding bilateral attention weights to Value.

'''

apply_attn_inv_forward = '''
    #define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

    extern "C" __global__ void apply_attn_inv_forward(
        const int n,
        const float* attn,
        const float* value,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        float fltOutput = 0.0;
    
        const int intN  = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY  = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX  = ( intIndex                                                    ) % SIZE_3(output);

        for (int intD = 0; intD < SIZE_1(attn); intD +=1) {
            int k = (intD % F_SIZE) - F_SIZE_H;
            int l = (intD / F_SIZE) - F_SIZE_H;
            
            int intX1 = intX - k;
            int intY1 = intY - l;
            
            if (WITHIN_BOUNDS(intX1, intY1, SIZE_2(output), SIZE_3(output))){
                fltOutput += VALUE_4(attn, intN, intD, intY, intX) * VALUE_4(value, intN, intC, intY1, intX1);
            }
        }

        output[intIndex] = fltOutput;

    } }
'''

apply_attn_inv_backward = '''
    #define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

    extern "C" __global__ void apply_attn_inv_backward(
        const int n,
        const float* gradLoss,
        const float* attn,
        const float* value,
        float* gradattn,
        float* gradvalue
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        const int intN  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss) / SIZE_1(gradLoss) ) % SIZE_0(gradLoss);
        const int intC  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss)                    ) % SIZE_1(gradLoss);
        const int intY  = ( intIndex / SIZE_3(gradLoss)                                       ) % SIZE_2(gradLoss);
        const int intX  = ( intIndex                                                          ) % SIZE_3(gradLoss);

        float fltLoss = VALUE_4(gradLoss, intN, intC, intY, intX);
        
        for (int intD = 0; intD < SIZE_1(attn); intD +=1) {
            int k = (intD % F_SIZE) - F_SIZE_H;
            int l = (intD / F_SIZE) - F_SIZE_H;
            
            int intX1 = intX - k;
            int intY1 = intY - l;
            
            if (WITHIN_BOUNDS(intX1, intY1, SIZE_2(gradLoss), SIZE_3(gradLoss))){
                atomicAdd(&gradattn[OFFSET_4(gradattn, intN, intD, intY,intX)], VALUE_4(value, intN, intC, intY1, intX1)* fltLoss);
                atomicAdd(&gradvalue[OFFSET_4(gradvalue, intN, intC, intY1, intX1)], VALUE_4(attn, intN, intD, intY, intX)* fltLoss);
            }
        }
        
    } }
'''

def cupy_kernel(strFunction, intWindowSize, objectVariables):
    strKernel = globals()[strFunction]

    strKernel = strKernel.replace('F_SIZE_H', str((intWindowSize - 1) // 2))
    strKernel = strKernel.replace('F_SIZE', str(intWindowSize))

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))
        
        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')


    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


class apply_attn_inv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attn, value):
        ctx.save_for_backward(attn, value)
        
        intWindowLength = attn.size(1)
        intWindowSize = int(sqrt(intWindowLength))
        assert intWindowLength == intWindowSize ** 2
        
        # intInputBatch, intInputChannel, intInputHeight, intInputWidth = value.size()
        # intWindowSize = (2 * ctx.md + 1)

        # output = feature1.new_zeros(intInputBatch, intWindowSize ** 2, intInputHeight, intInputWidth)
        output = value.new_zeros(value.size())

        assert (attn.size(-2) == value.size(-2)) and (attn.size(-1) == value.size(-1))
        assert (attn.is_contiguous() == True) and (value.is_contiguous() == True)
        assert attn.device == value.device

        if attn.is_cuda == True and value.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('apply_attn_inv_forward',
                        cupy_kernel('apply_attn_inv_forward', intWindowSize, {
                            'attn': attn,
                            'value': value,
                            'output': output
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, attn.data_ptr(), value.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        # end
        return output


    @staticmethod
    def backward(ctx, gradOutput):
        attn, value = ctx.saved_tensors
        
        intWindowLength = attn.size(1)
        intWindowSize = int(sqrt(intWindowLength))
        assert intWindowLength == intWindowSize ** 2       

        # intInputBatch, _, intInputHeight, intInputWidth = SBM.size() # (+) feature1 -> SBM
        # intInputChannel = feature1.size(1) # (+) intInputChannel is a channel size of the feature map
        
        gradattn = attn.new_zeros(attn.size()) if \
            ctx.needs_input_grad[0] == True else None
        gradvalue = value.new_zeros(value.size()) if \
            ctx.needs_input_grad[1] == True else None
        
        # gradOutput = gradOutput / torch.sqrt(torch.tensor(intInputChannel).float())

        if attn.is_cuda == True and value.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # weight grad
            n = gradOutput.nelement()
            cupy_launch('apply_attn_inv_backward',
                        cupy_kernel('apply_attn_inv_backward', intWindowSize, {
                            'gradLoss': gradOutput,
                            'attn': attn,
                            'value': value,
                            'gradattn': gradattn,
                            'gradvalue': gradvalue
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), attn.data_ptr(), value.data_ptr(),
                      gradattn.data_ptr(), gradvalue.data_ptr()],
                stream=Stream
            )

        # end

        return gradattn, gradvalue, None, None

# end
