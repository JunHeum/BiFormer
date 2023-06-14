import cupy
import torch
import re

'''
Multi-head bilateral cross attention(MBCA-A) cuda version
'''


bilateral_cross_attn_forward = '''
    #define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

    extern "C" __global__ void bilateral_cross_attn_forward(
        const int n,
        const float* query,
        const float* key,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        float fltOutput = 0.0;
    
        const int intN  = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY  = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX  = ( intIndex                                                    ) % SIZE_3(output);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;
        
        int intX1 = intX - k;
        int intY1 = intY - l;
        int intX2 = intX + k;
        int intY2 = intY + l;

        if (WITHIN_BOUNDS(intX1, intY1, SIZE_2(output), SIZE_3(output)) & WITHIN_BOUNDS(intX2, intY2, SIZE_2(output), SIZE_3(output))){
            for (int intChannel = 0; intChannel < SIZE_1(query); intChannel += 1) {
                fltOutput += VALUE_4(query, intN, intChannel, intY1, intX1) * VALUE_4(key, intN, intChannel, intY2, intX2);                
            }
        }

        output[intIndex] = fltOutput;

    } }
'''

bilateral_cross_attn_backward = '''
    #define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

    extern "C" __global__ void bilateral_cross_attn_backward(
        const int n,
        const float* gradLoss,
        const float* query,
        const float* key,
        float* gradquery,
        float* gradkey
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        const int intN  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss) / SIZE_1(gradLoss) ) % SIZE_0(gradLoss);
        const int intC  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss)                    ) % SIZE_1(gradLoss);
        const int intY  = ( intIndex / SIZE_3(gradLoss)                                       ) % SIZE_2(gradLoss);
        const int intX  = ( intIndex                                                          ) % SIZE_3(gradLoss);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;

        int intX1 = intX - k;
        int intY1 = intY - l;
        int intX2 = intX + k;
        int intY2 = intY + l;

        if (WITHIN_BOUNDS(intX1, intY1, SIZE_2(gradLoss), SIZE_3(gradLoss)) & WITHIN_BOUNDS(intX2, intY2, SIZE_2(gradLoss), SIZE_3(gradLoss))){
            float fltLoss = VALUE_4(gradLoss, intN, intC, intY, intX);
            for (int intChannel = 0; intChannel < SIZE_1(query); intChannel += 1) {
                atomicAdd(&gradquery[OFFSET_4(gradquery, intN, intChannel, intY1, intX1)], VALUE_4(key, intN, intChannel, intY2, intX2) * fltLoss);
                atomicAdd(&gradkey[OFFSET_4(gradkey, intN, intChannel, intY2, intX2)], VALUE_4(query, intN, intChannel, intY1, intX1) * fltLoss);
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


class bi_attn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query, key, md):
        ctx.save_for_backward(query, key)
        ctx.md = md

        intInputBatch, intInputChannel, intInputHeight, intInputWidth = query.size()
        intWindowSize = (2 * ctx.md + 1)

        output = query.new_zeros(intInputBatch, intWindowSize ** 2, intInputHeight, intInputWidth)

        assert query.size() == key.size()
        assert (query.is_contiguous() == True) and (key.is_contiguous() == True)
        assert query.device == key.device

        if query.is_cuda == True and key.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('bilateral_cross_attn_forward',
                        cupy_kernel('bilateral_cross_attn_forward', intWindowSize, {
                            'query': query,
                            'key': key,
                            'output': output
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, query.data_ptr(), key.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        # end
        return output


    @staticmethod
    def backward(ctx, gradOutput):
        query, key = ctx.saved_tensors

        # intInputBatch, _, intInputHeight, intInputWidth = SBM.size() # (+) feature1 -> SBM
        intWindowSize = (2 * ctx.md + 1)

        gradquery = query.new_zeros(query.size()) if \
            ctx.needs_input_grad[0] == True else None
        gradkey = key.new_zeros(key.size()) if \
            ctx.needs_input_grad[1] == True else None
                
        if query.is_cuda == True and key.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # weight grad
            n = gradOutput.nelement()
            cupy_launch('bilateral_cross_attn_backward',
                        cupy_kernel('bilateral_cross_attn_backward', intWindowSize, {
                            'gradLoss': gradOutput,
                            'query': query,
                            'key': key,
                            'gradquery': gradquery,
                            'gradkey': gradkey
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), query.data_ptr(), key.data_ptr(),
                      gradquery.data_ptr(), gradkey.data_ptr()],
                stream=Stream
            )

        # end

        return gradquery, gradkey, None, None

# end
